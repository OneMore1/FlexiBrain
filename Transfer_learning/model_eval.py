import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NiftiTxtDataset
from main import custom_collate_fn, prepare_batch_data, build_model  # type: ignore


# ---------------- Utilities ---------------- #

def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(__name__ + ".eval")
    logger.setLevel(logging.DEBUG)
    # reset handlers to avoid duplication across runs
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(os.path.join(log_dir, 'eval.log'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def ensure_model_defaults(a):
    """
    Ensure args has all hyperparams that build_model expects.
    Keep defaults consistent with your training script.
    """
    safe = dict(
        # core
        model_type='mamba',
        embed_dim=512,
        depth=24,
        predictor_depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        rms_norm=False,
        fused_add_norm=True,
        residual_in_fp32=True,
        predictor_dropout=0.0,
        predictor_mlp_ratio=4.0,
        # mamba/bimamba related
        bimamba_type='none',
        if_bimamba=False,
        mixer_type='mamba',
        if_devide_out=True,
        predictor_hidden=None,
        # training-time model opts
        momentum=0.992,
        norm_target=True,
        use_res_cond=False,
    )
    for k, v in safe.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    return a


def load_checkpoint_only_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)  # allow raw state dict
    model.load_state_dict(state, strict=True)
    return ckpt


def build_test_loader(args) -> DataLoader:
    test_set = NiftiTxtDataset(
        txt_files=args.test_list,
        return_torch=True,
        memory_map=args.memory_map,
        cache_meta=True,
        T_prime=args.T_prime,
        tau_seconds=args.tau_seconds,
    )
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )
    return loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args, logger: logging.Logger) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    num_batches = 0
    batch_losses = []

    for i, batch in enumerate(loader, 1):
        x, meta, orig_Ts, affines = prepare_batch_data(batch, device)
        with torch.cuda.amp.autocast(enabled=args.amp and torch.cuda.is_available()):
            loss, _, _, _ = model(
                x,
                mask_ratio=args.mask_ratio,
                meta=meta,
                orig_Ts=orig_Ts,
                affines=affines,
            )
        li = float(loss.item())
        total_loss += li
        num_batches += 1
        batch_losses.append(li)
        if args.log_interval > 0 and i % args.log_interval == 0:
            logger.info(f"Batch {i}/{len(loader)}  Loss {li:.6f}  Avg {(total_loss/num_batches):.6f}")

    avg = total_loss / max(1, num_batches)
    logger.info(f"Average loss on test set: {avg:.6f}")
    return {"avg_loss": avg, "batch_losses": batch_losses}


def make_tag(ckpt_path: str, test_list: str) -> str:
    return f"{Path(ckpt_path).stem}__on__{Path(test_list).stem}"


def apply_overrides(base_args, overrides: Dict[str, Any], keys):
    """Temporarily override a subset of args fields from JSON run."""
    snapshot = {k: getattr(base_args, k) for k in keys}
    for k in keys:
        if k in overrides:
            setattr(base_args, k, overrides[k])
    return snapshot


# ---------------- Per-run pipeline ---------------- #

def run_one(base_args, device: torch.device, job: Dict[str, Any]):
    # Required fields in job
    test_list = job["test_list"]
    checkpoint = job["checkpoint"]

    # Build naming from plan.json (priority) or auto tag
    if "output_dir" in job and str(job["output_dir"]).strip():
        output_dir = Path(job["output_dir"]).resolve()
        base = output_dir.name
        result_json = output_dir / f"{base}.eval.json"
        per_batch_json = output_dir / f"{base}.per_batch.json"
        log_dir = Path(base_args.log_root) / base
        run_name = base
    else:
        tag = make_tag(checkpoint, test_list)
        output_dir = (Path(base_args.results_root) / tag).resolve()
        result_json = output_dir / f"{tag}.eval.json"
        per_batch_json = output_dir / f"{tag}.per_batch.json"
        log_dir = Path(base_args.log_root) / tag
        run_name = tag

    logger = setup_logging(str(log_dir))
    logger.info(f"=== Eval run: {run_name} ===")
    logger.info(f"test_list = {test_list}")
    logger.info(f"checkpoint = {checkpoint}")
    logger.info(f"output_dir = {output_dir}")

    # Prepare args for this run
    args = argparse.Namespace(**vars(base_args))  # shallow copy
    args.test_list = test_list
    args.checkpoint = checkpoint

    # Allow a few common overrides from plan.json per run
    override_keys = [
        "batch_size", "num_workers", "memory_map", "T_prime", "tau_seconds",
        "mask_ratio", "amp", "model_type", "embed_dim", "depth",
        "predictor_depth", "drop_path_rate", "rms_norm", "fused_add_norm",
        "residual_in_fp32", "bimamba_type", "if_bimamba", "mixer_type",
        "if_devide_out", "predictor_hidden", "num_heads", "mlp_ratio",
        "momentum", "norm_target", "use_res_cond"
    ]
    snap = apply_overrides(args, job, override_keys)

    # Sanity checks
    if not Path(args.test_list).is_file():
        raise FileNotFoundError(f"Test list not found: {args.test_list}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Build model & load weights
    ensure_model_defaults(args)
    model = build_model(args, device).to(device)
    logger.info("Building model & loading checkpoint ...")
    ckpt = load_checkpoint_only_model(model, args.checkpoint, device)
    if 'epoch' in ckpt:
        logger.info(f"Checkpoint epoch: {ckpt['epoch']}  best_loss: {ckpt.get('best_loss','N/A')}")

    # Data loader
    loader = build_test_loader(args)
    logger.info(f"Test set size: {len(loader.dataset)}")

    # Eval
    result = evaluate(model, loader, device, args, logger)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(result_json, "w") as f:
        json.dump({
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "test_list": str(Path(args.test_list).resolve()),
            "avg_loss": result["avg_loss"],
            "num_batches": len(result["batch_losses"]),
            "num_samples": len(loader.dataset)
        }, f, indent=2)
    logger.info(f"Saved result JSON -> {result_json}")

    if base_args.save_per_batch:  # global switch
        with open(per_batch_json, "w") as f:
            json.dump({
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "test_list": str(Path(args.test_list).resolve()),
                "batch_losses": result["batch_losses"],
                "avg_loss": result["avg_loss"]
            }, f, indent=2)
        logger.info(f"Saved per-batch JSON -> {per_batch_json}")

    # restore base args (for safety if reused)
    for k, v in snap.items():
        setattr(base_args, k, v)

    logger.info(f"Eval finished: {run_name}")


# ---------------- Entry ---------------- #

def main():
    p = argparse.ArgumentParser(description="Batch Model Evaluation Runner")

    # Batch plan + roots used when output_dir is absent in a run
    p.add_argument("--batch_plan", type=str, required=True)
    p.add_argument("--results_root", type=str, default="./eval_results")
    p.add_argument("--log_root", type=str, default="./logs/eval")

    # Data defaults (can be overridden per-run)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--memory_map", action="store_true", default=True)
    p.add_argument("--T_prime", type=int, default=30)
    p.add_argument("--tau_seconds", type=float, default=6.0)

    # Eval knobs
    p.add_argument("--mask_ratio", type=float, default=0.65)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_per_batch", action="store_true", default=False)  # global switch

    # Model hyperparams (defaults; may be overridden per-run)
    p.add_argument("--model_type", type=str, default="mamba", choices=["mamba", "vit"])
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=24)
    p.add_argument("--predictor_depth", type=int, default=6)
    p.add_argument("--drop_path_rate", type=float, default=0.1)
    p.add_argument("--rms_norm", action="store_true", default=False)
    p.add_argument("--fused_add_norm", action="store_true", default=True)
    p.add_argument("--residual_in_fp32", action="store_true", default=True)
    p.add_argument("--bimamba_type", type=str, default="none")
    p.add_argument("--if_bimamba", action="store_true", default=False)
    p.add_argument("--mixer_type", type=str, default="mamba")
    p.add_argument("--if_devide_out", action="store_true", default=True)
    p.add_argument("--predictor_hidden", type=int, default=None)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--momentum", type=float, default=0.992)
    p.add_argument("--norm_target", action="store_true", default=True)
    p.add_argument("--use_res_cond", action="store_true", default=False)

    # Misc
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    plan = json.loads(Path(args.batch_plan).read_text())
    runs = plan.get("runs", [])
    if not runs:
        raise ValueError("JSON plan has no 'runs'.")

    for job in runs:
        # basic required keys per run
        if "test_list" not in job or "checkpoint" not in job:
            raise ValueError(f"Run missing required keys: {job}")
        run_one(args, device, job)


if __name__ == "__main__":
    main()