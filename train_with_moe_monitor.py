"""
训练脚本示例：集成 MoE 梯度监测

演示如何在实际训练循环中使用 MoEGradientMonitor
"""
import torch
import torch.nn as nn
import sys
import os
import logging
from datetime import datetime

sys.path.insert(0, 'mamba_mae/mamba2')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mamba_mae.models_vim_mae import VolumeMambaJEPA
from mamba_mae.moe_gradient_monitor import MoEGradientMonitor


def setup_logger(log_dir='logs'):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'moe_monitor_{timestamp}.log')
    
    # 创建 logger
    logger = logging.getLogger('MoE_Training')
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers
    logger.handlers.clear()
    
    # 文件 handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    return logger


class DummyDataset(torch.utils.data.Dataset):
    """模拟数据集"""
    def __init__(self, num_samples=100, device='cpu'):
        self.num_samples = num_samples
        self.device = device
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 模拟 4D fMRI 数据
        x = torch.randn(1, 20, 32, 32, 32)
        
        meta = {
            'voxel_sizes': torch.tensor([2.0, 2.0, 2.0]),
            'TRs': torch.tensor(2.0),
        }
        
        orig_T = torch.tensor(20)
        
        return x, meta, orig_T


def train_with_monitoring(
    model,
    train_loader,
    optimizer,
    device,
    logger,
    num_epochs=2,
    log_interval=10,
    gradient_log_interval=50,
):
    """
    带 MoE 监测的训练循环
    
    Args:
        model: 模型
        train_loader: 数据加载器
        optimizer: 优化器
        device: 设备
        logger: 日志记录器
        num_epochs: 训练轮数
        log_interval: 日志打印间隔
        gradient_log_interval: 梯度监测间隔
    """
    # 创建 MoE 梯度监测器
    gradient_monitor = MoEGradientMonitor(model.moe, logger=logger)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"开始训练")
    logger.info(f"{'='*80}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    logger.info(f"  Log interval: {log_interval}")
    logger.info(f"  Gradient log interval: {gradient_log_interval}")
    logger.info(f"  Device: {device}")
    logger.info(f"  MoE experts: {model.moe.num_experts}")
    logger.info(f"{'='*80}\n")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*80}\n")
        
        for batch_idx, (data, meta, orig_T) in enumerate(train_loader):
            global_step += 1
            
            # 移动数据到设备
            data = data.to(device)
            meta = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in meta.items()}
            orig_T = orig_T.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            try:
                loss, pred, mask = model(data, mask_ratio=0.6, meta=meta, orig_Ts=orig_T)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（可选）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 定期打印训练信息
                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    logger.info(
                        f"Epoch [{epoch + 1}/{num_epochs}] "
                        f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.6f} (Avg: {avg_loss:.6f})"
                    )
                
                # 定期监测梯度
                if global_step % gradient_log_interval == 0:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔬 Gradient Monitoring at Step {global_step}")
                    logger.info(f"{'='*80}")
                    
                    # 监测梯度统计
                    grad_stats = gradient_monitor.log_gradient_stats(
                        step=global_step,
                        prefix=f"Epoch {epoch + 1} "
                    )
                    
                    # 可选：保存梯度统计到文件
                    # save_gradient_stats(grad_stats, global_step)
                
            except Exception as e:
                logger.error(f"❌ Error at epoch {epoch + 1}, batch {batch_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Epoch 结束
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"\n{'='*80}")
        logger.info(f"Epoch {epoch + 1} Summary")
        logger.info(f"{'='*80}")
        logger.info(f"  Average Loss: {avg_epoch_loss:.6f}")
        logger.info(f"{'='*80}\n")
        
        # 每个 epoch 结束时监测一次梯度
        logger.info(f"\n{'='*80}")
        logger.info(f"🔬 End-of-Epoch Gradient Monitoring")
        logger.info(f"{'='*80}")
        grad_stats = gradient_monitor.log_gradient_stats(
            step=global_step,
            prefix=f"Epoch {epoch + 1} End "
        )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"训练完成")
    logger.info(f"{'='*80}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练 MoE 模型并监测梯度')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--num_samples', type=int, default=20, help='数据集样本数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--log_interval', type=int, default=5, help='日志打印间隔')
    parser.add_argument('--gradient_log_interval', type=int, default=10, help='梯度监测间隔')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"配置")
    logger.info(f"{'='*80}")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info(f"{'='*80}\n")
    
    # 创建模型
    logger.info("📦 创建模型...")
    model = VolumeMambaJEPA(
        embed_dim=256,
        depth=4,
        predictor_depth=2,
        bimamba_type='none',
        mixer_type='mamba',
        device=args.device,
    )
    model = model.to(args.device)
    logger.info(f"  ✓ 模型已创建并移动到 {args.device}")
    
    # 创建数据集和数据加载器
    logger.info("\n📊 创建数据集...")
    dataset = DummyDataset(num_samples=args.num_samples, device=args.device)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    logger.info(f"  ✓ 数据集大小: {len(dataset)}")
    logger.info(f"  ✓ 批次数: {len(train_loader)}")
    
    # 创建优化器
    logger.info("\n⚙️  创建优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger.info(f"  ✓ 优化器: AdamW (lr={args.lr})")
    
    # 开始训练
    train_with_monitoring(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=args.device,
        logger=logger,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval,
        gradient_log_interval=args.gradient_log_interval,
    )


if __name__ == "__main__":
    main()

