import unittest

import torch

from src.model import WavelengthAwareMAE, positional_encoding


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = WavelengthAwareMAE(embed_dim=24, patch_size=2, num_heads=4,
                                        encoder_blocks=1, decoder_blocks=1)
        self.chip = torch.rand(2, 4, 6, 3)
        self.wavelengths = torch.tensor([443., 550., 665.])

    def test_round_trip_and_masked_training(self):
        model, x = self.model, self.chip
        torch.testing.assert_close(model.unpatchify(model.patchify(x), x.shape), x)
        prediction, mask, target = model(x, self.wavelengths)
        self.assertEqual(prediction.shape, (2, 18, 4))
        self.assertTrue((mask.sum(dim=1) == 14).all())
        loss = model.masked_loss(prediction, mask, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (model.patch_embed.weight, model.mask_token, model.pred_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(model.reconstruct(x, self.wavelengths).shape, x.shape)
        changed = torch.rand(1, 6, 4, 5)
        self.assertEqual(model.reconstruct(changed, [410., 443., 500., 550., 665.]).shape,
                         changed.shape)

    def test_physical_wavelengths_and_band_reordering(self):
        a = positional_encoding(1, 1, self.wavelengths[:2], 24)
        b = positional_encoding(1, 1, torch.tensor([400., 443., 550.]), 24)
        torch.testing.assert_close(a, b[1:])
        self.assertFalse(torch.equal(a[0], a[1]))
        self.model.eval()
        with torch.no_grad():
            features = self.model.encode(self.chip, self.wavelengths).reshape(2, 2, 3, 3, 24)
            order = [2, 0, 1]
            reordered = self.model.encode(self.chip[..., order], self.wavelengths[order])
            torch.testing.assert_close(features[..., order, :], reordered.reshape(2, 2, 3, 3, 24))

    def test_loss_uses_only_hidden_tokens(self):
        prediction = torch.tensor([[[100.], [2.]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[False, True]])
        self.assertEqual(self.model.masked_loss(prediction, mask, target).item(), 4.)
        with self.assertRaises(ValueError):
            self.model.masked_loss(prediction, torch.zeros_like(mask), target)

    def test_bad_inputs(self):
        for wavelengths in ([443.], [443., float("nan"), 665.], [-1., 550., 665.]):
            with self.assertRaises(ValueError):
                self.model(self.chip, wavelengths)
        with self.assertRaises(ValueError):
            self.model(self.chip[:, :3], self.wavelengths)
        with self.assertRaises(ValueError):
            self.model(torch.full_like(self.chip, float("nan")), self.wavelengths)
        with self.assertRaises(ValueError):
            self.model(torch.ones(1, 2, 2, 1), [443.])
        for ratio in (0, 1, float("nan")):
            with self.assertRaises(ValueError):
                WavelengthAwareMAE(masking_ratio=ratio)


if __name__ == "__main__":
    unittest.main()
