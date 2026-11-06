import numpy as np
import pytest

from cxrtrust.conformal import FNRControl, MondrianConformal, conformal_quantile, false_negative_rate
from cxrtrust.selective import aurc, hamming_loss, image_confidence, risk_coverage, selective_report


def _simulate(n, k=5, seed=0, prevalence=(0.05, 0.1, 0.2, 0.3, 0.5)):
    """Calibrated-ish synthetic probabilities: positives score higher than negatives on average."""
    rng = np.random.default_rng(seed)
    y = (rng.random((n, k)) < np.asarray(prevalence)).astype(float)
    logits = rng.normal(0, 1.2, (n, k)) + 2.0 * (2 * y - 1)
    return 1 / (1 + np.exp(-logits)), y


def test_conformal_quantile_finite_sample_rule():
    s = np.arange(1, 11, dtype=float)  # n = 10
    assert conformal_quantile(s, 0.1) == 10.0  # ceil(11 * 0.9) = 10
    assert conformal_quantile(s, 0.5) == 6.0  # ceil(11 * 0.5) = 6
    assert conformal_quantile(s, 0.01) == float("inf")  # not enough samples for 99 %


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_mondrian_label_conditional_coverage(alpha):
    cal_p, cal_y = _simulate(4000, seed=1)
    test_p, test_y = _simulate(20000, seed=2)
    cp = MondrianConformal(alpha).fit(cal_p, cal_y)
    rep = cp.report(test_p, test_y)
    # coverage >= 1 - alpha for *both* labels of every class, even the 5 %-prevalence one (slack for MC noise)
    assert np.all(rep["coverage_pos"] >= 1 - alpha - 0.02)
    assert np.all(rep["coverage_neg"] >= 1 - alpha - 0.02)
    assert np.all(rep["ambiguity"] < 0.9)  # sets are informative, not trivially {absent, present}


def test_mondrian_requires_fit_and_flags_ambiguity():
    with pytest.raises(RuntimeError):
        MondrianConformal().predict(np.zeros((1, 2)))
    cp = MondrianConformal(0.1)
    cp.q_pos, cp.q_neg = np.array([0.6]), np.array([0.6])
    sets = cp.predict(np.array([[0.5], [0.9], [0.1]]))
    assert sets["ambiguous"][:, 0].tolist() == [True, False, False]
    assert sets["present"][:, 0].tolist() == [True, True, False]


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_fnr_control_guarantee(alpha):
    cal_p, cal_y = _simulate(3000, seed=3)
    test_p, test_y = _simulate(20000, seed=4)
    crc = FNRControl(alpha).fit(cal_p, cal_y)
    rep = crc.report(test_p, test_y)
    assert rep["fnr"] <= alpha + 0.01
    assert rep["fnr"] >= alpha - 0.04  # and it is not needlessly conservative
    # a stricter risk level reports more findings
    assert FNRControl(alpha / 2).fit(cal_p, cal_y).threshold <= crc.threshold


def test_false_negative_rate_ignores_images_without_findings():
    probs = np.array([[0.9, 0.2], [0.1, 0.1]])
    y = np.array([[1, 1], [0, 0]])
    np.testing.assert_allclose(false_negative_rate(probs, y, 0.5), [0.5, 0.0])


def test_risk_coverage_and_aurc():
    loss = np.array([0.0, 0.0, 1.0, 1.0])
    good = np.array([4.0, 3.0, 2.0, 1.0])
    cov, risk = risk_coverage(good, loss)
    np.testing.assert_allclose(cov, [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(risk, [0, 0, 1 / 3, 0.5])
    assert aurc(good, loss) < aurc(-good, loss)
    rep = selective_report(good, loss)
    assert rep["e_aurc"] == pytest.approx(0.0)
    assert rep["risk@100"] == pytest.approx(0.5) and rep["risk@50"] == pytest.approx(0.0)


def test_confidence_scores_rank_errors():
    p, y = _simulate(5000, seed=5)
    loss = hamming_loss(p, y)
    for method in ("entropy", "min_margin"):
        conf = image_confidence(p, method)
        assert conf.min() >= 0 and conf.max() <= 1
        assert aurc(conf, loss) < aurc(np.random.default_rng(0).random(len(loss)), loss)
    with pytest.raises(ValueError):
        image_confidence(p, "bogus")

