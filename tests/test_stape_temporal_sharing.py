"""Regression tests for feature-aligned spatial kernel sharing over time."""
import unittest

import torch
import torch.nn.functional as F

from flexibrain.models.layers.stape import STAPE4D_TimeToSpace


class TemporalSharingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model = STAPE4D_TimeToSpace(
            d_mid=2, d_out=2, kt_base=2,
            kx_base=4, ky_base=4, kz_base=4,
        )
        with torch.no_grad():
            self.model.b_t_first.fill_(0.7)
            self.model.b_xyz_after.fill_(0.3)

    def run_group(self, x, lengths, kt=2):
        return self.model._run_group_time_first(
            x, lengths, kt, 4, 4, 4, list(range(len(lengths))),
            [torch.eye(4) for _ in lengths],
        )

    def reference(self, x, lengths, kt=2):
        """Apply the same spatial kernel separately to each valid time window."""
        m = self.model
        wt = m._get_wt_first(kt, x.device, x.dtype)
        ws = m._get_wxyz_after(4, 4, 4, x.device, x.dtype)
        outputs = []
        for i, length in enumerate(lengths):
            xi = x[i, ..., :length]
            xi = F.pad(xi, (0, (-length) % kt))
            temporal = F.conv1d(xi.reshape(-1, 1, xi.shape[-1]), wt, m.b_t_first, stride=kt)
            spatial = None
            for t in range(temporal.shape[-1]):
                volume = temporal[..., t].reshape(1, 96, 96, 96, m.Dm).permute(0, 4, 1, 2, 3)
                value = F.conv3d(volume, ws, bias=None, stride=4)
                spatial = value if spatial is None else spatial + value
            spatial = spatial + m.b_xyz_after[None, :, None, None, None]
            outputs.append(spatial.permute(0, 2, 3, 4, 1).reshape(-1, m.Do))
        return outputs

    def test_real_forward_matches_independent_windows(self):
        with torch.no_grad():
            for windows in (1, 5, 16, 30):
                with self.subTest(windows=windows):
                    x = torch.randn(1, 96, 96, 96, 2 * windows)
                    tokens, lengths, _, positions = self.run_group(x, [2 * windows])
                    expected = self.reference(x, [2 * windows])[0]
                    self.assertEqual(lengths, [24 ** 3])
                    torch.testing.assert_close(tokens[0] - positions[0], expected, atol=2e-4, rtol=2e-4)

    def test_padding_does_not_change_short_sample(self):
        with torch.no_grad():
            short = torch.randn(1, 96, 96, 96, 3)
            alone = self.run_group(short, [3])[0][0]
            padded = F.pad(short, (0, 3))
            mixed = torch.cat([padded, torch.randn_like(padded)], dim=0)
            together = self.run_group(mixed, [3, 6])[0][0]
            torch.testing.assert_close(alone, together, atol=2e-5, rtol=2e-5)

    def test_parameter_gradients_match_reference(self):
        x = torch.randn(1, 96, 96, 96, 5)
        params = [self.model.w_t_first_base, self.model.b_t_first,
                  self.model.w_xyz_after_base, self.model.b_xyz_after]
        tokens, _, _, positions = self.run_group(x, [5])
        actual = torch.autograd.grad((tokens[0] - positions[0]).square().mean(), params)
        reference = torch.autograd.grad(self.reference(x, [5])[0].square().mean(), params)
        for got, expected in zip(actual, reference):
            self.assertTrue(torch.isfinite(got).all())
            torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)

    def test_public_forward_and_backward(self):
        x = torch.randn(1, 96, 96, 96, 5)
        meta = {0: {"tr": 3.0, "voxel": (3.0, 3.0, 3.0)}}
        tokens, mask, lengths, positions = self.model(x, meta, [5], [torch.eye(4)])
        self.assertEqual(tokens.shape, (1, 24 ** 3, 2))
        self.assertFalse(mask.any())
        self.assertEqual(lengths, [24 ** 3])
        self.assertEqual(positions.shape, tokens.shape)
        tokens.square().mean().backward()
        for p in self.model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_default_channel_count_algebra(self):
        for windows in (5, 16, 30, 32):
            with self.subTest(windows=windows):
                features = torch.randn(1, 16, windows, 2, 2, 2, dtype=torch.float64)
                weights = torch.randn(3, 16, 1, 1, 1, dtype=torch.float64)
                actual = F.conv3d(features.flatten(1, 2), weights.repeat_interleave(windows, dim=1))
                expected = F.conv3d(features.sum(dim=2), weights)
                torch.testing.assert_close(actual, expected)
                old = F.conv3d(features.flatten(1, 2), weights.repeat(1, windows, 1, 1, 1))
                self.assertFalse(torch.allclose(old, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
