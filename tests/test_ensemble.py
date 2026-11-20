import numpy as np
import pytest
import torch
import torch.nn as nn

from cxrtrust.ensemble import (
    TTA,
    combine_logits,
    epistemic_uncertainty,
    predict_members,
    summarize_members,
)


def test_combine_logits_modes():
    a, b = np.array([[2.0, -1.0]]), np.array([[4.0, -3.0]])
    np.testing.assert_allclose(combine_logits([a, b]), [[3.0, -2.0]])
    p = combine_logits([a, b], mode="prob")
    expected = 1 / (1 + np.exp(-a)) / 2 + 1 / (1 + np.exp(-b)) / 2
    np.testing.assert_allclose(1 / (1 + np.exp(-p)), expected, atol=1e-9)
    with pytest.raises(ValueError):
        combine_logits([a], mode="x")


def test_epistemic_uncertainty_zero_on_agreement_and_high_on_split():
    agree = [np.full((4, 3), 1.5)] * 3
    assert epistemic_uncertainty(agree).max() == pytest.approx(0.0, abs=1e-9)
    split = [np.full((4, 3), 6.0), np.full((4, 3), -6.0)]
    assert epistemic_uncertainty(split).min() > 0.6  # close to ln 2 = 0.693


def test_ensemble_beats_average_member_on_noisy_members():
    rng = np.random.default_rng(0)
    truth = rng.normal(0, 2, (3000, 4))
    y = (rng.random(truth.shape) < 1 / (1 + np.exp(-truth))).astype(float)
    members = np.stack([truth + rng.normal(0, 2, truth.shape) for _ in range(8)])
    s = summarize_members(members, y)
    assert s["bce_ensemble"] < s["bce_single_mean"] - 0.02
    assert s["epistemic_mean"] > 0


def test_tta_identity_shift_and_zoom():
    x = torch.zeros(1, 1, 16, 16)
    x[..., 6:10, 6:10] = 1.0
    assert TTA.transform(x, 1.0, 0.0, 0.0) is x
    zoomed = TTA.transform(x, 2.0, 0.0, 0.0)
    assert zoomed.sum() > 3.0 * x.sum()  # a 2x zoom covers ~4x the area
    shifted = TTA.transform(x, 1.0, 0.25, 0.0)  # content moves right by 0.25 * width
    cols = torch.nonzero(shifted[0, 0].sum(0) > 0.5).flatten()
    assert cols.min() > 6 and cols.max() > 9
    assert (1.0, 0.0, 0.0) in TTA(views=[(1.05, 0, 0)]).views  # identity is always present


class _Linear(nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x.mean(dim=(2, 3)))


def test_predict_members_shapes_and_tta_mean():
    batches = [{"image": torch.randn(5, 4, 8, 8), "target": torch.zeros(5, 3)} for _ in range(2)]
    models = [_Linear(0), _Linear(1)]
    logits, targets = predict_members(models, batches, torch.device("cpu"))
    assert logits.shape == (2, 10, 3) and targets.shape == (10, 3)
    tta = TTA(views=[(1.0, 0, 0)])
    with_tta, _ = predict_members(models, batches, torch.device("cpu"), tta)
    np.testing.assert_allclose(with_tta, logits, atol=1e-6)  # identity-only TTA changes nothing
