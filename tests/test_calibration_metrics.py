import numpy as np
import pytest
import torch

from cxrtrust.calibration import TemperatureScaler, brier_score, expected_calibration_error, reliability_curve
from cxrtrust.metrics import best_f1_thresholds, multilabel_report


def test_ece_zero_for_perfectly_calibrated_bins():
    probs = np.array([0.2] * 10 + [0.8] * 10)
    targets = np.array([1] * 2 + [0] * 8 + [1] * 8 + [0] * 2)
    assert expected_calibration_error(probs, targets) == pytest.approx(0.0)
    conf, acc, count = reliability_curve(probs, targets)
    assert count.sum() == 20


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, c = 4000, 3
    true_logits = rng.normal(0, 2, (n, c))
    targets = (rng.random((n, c)) < 1 / (1 + np.exp(-true_logits))).astype(np.float32)
    overconfident = torch.tensor(true_logits * 3.0, dtype=torch.float32)
    scaler = TemperatureScaler(c).fit(overconfident, torch.from_numpy(targets))
    np.testing.assert_allclose(scaler.temperature.detach().numpy(), 3.0, rtol=0.15)
    before = expected_calibration_error(torch.sigmoid(overconfident).numpy(), targets)
    after = expected_calibration_error(torch.sigmoid(scaler(overconfident)).detach().numpy(), targets)
    assert after < before


def test_multilabel_report_and_thresholds():
    targets = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.2]])
    rep = multilabel_report(probs, targets, ["a", "b"])
    assert rep["auroc_macro"] == pytest.approx(1.0)
    th = best_f1_thresholds(probs, targets)
    assert ((probs >= th) == targets.astype(bool)).all()
    assert brier_score(probs, targets) < 0.1
