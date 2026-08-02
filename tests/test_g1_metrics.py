"""harness/g1_metrics.py -- pure-function checks (ema/fit_slope/loss_fell/
g1a_report) plus a live check of the G1-b student-running probe against
the real model (CPU, same fixture pattern as tests/test_shapes.py)."""
import torch
import pytest

from harness.g1_metrics import ema, fit_slope, g1a_report, g1b_student_running_probe, loss_fell
from photon.config import PhotonConfig
from photon.surgery import load_photon_qwen3


def test_fit_slope_negative_for_falling_series():
    assert fit_slope([10.0, 8.0, 6.0, 4.0, 2.0]) < 0


def test_fit_slope_positive_for_rising_series():
    assert fit_slope([1.0, 2.0, 3.0, 4.0, 5.0]) > 0


def test_fit_slope_zero_for_single_point():
    assert fit_slope([5.0]) == 0.0


def test_loss_fell_true_for_noisy_but_downward_series():
    # noisy but a clear downward trend across the run
    values = [1.0, 1.2, 0.9, 1.1, 0.8, 0.3, 0.4, 0.2, 0.25, 0.15]
    assert loss_fell(values, frac=0.2) is True


def test_loss_fell_false_for_flat_series():
    values = [0.5] * 20
    assert loss_fell(values, frac=0.2) is False


def test_ema_smooths_and_preserves_first_value():
    values = [1.0, 5.0, 1.0, 5.0]
    smoothed = ema(values, alpha=0.5)
    assert smoothed[0] == 1.0
    assert len(smoothed) == len(values)


def test_g1a_report_windowed_initial_final_matches_loss_fell():
    history = {
        "enc1": [1.0, 0.9, 0.8, 0.3, 0.2, 0.1],
        "enc2": [0.01] * 6,
        "dec2": [0.02] * 6,
        "dec1": [10.0, 8.0, 6.0, 4.0, 2.0, 1.0],
    }
    report = g1a_report(history)
    assert report["enc1"]["final"] < report["enc1"]["initial"]
    assert report["enc1"]["fell"] == (report["enc1"]["final"] < report["enc1"]["initial"])
    assert "dec_dominates_enc" in report


def test_g1a_report_dec_dominates_enc_flag():
    history = dict(enc1=[0.01] * 5, enc2=[0.01] * 5, dec2=[1.0] * 5, dec1=[1.0] * 5)
    report = g1a_report(history)
    assert report["dec_dominates_enc"] is True

    history2 = dict(enc1=[0.5] * 5, enc2=[0.5] * 5, dec2=[0.5] * 5, dec1=[0.5] * 5)
    report2 = g1a_report(history2)
    assert report2["dec_dominates_enc"] is False


@pytest.fixture(scope="module")
def photon_real():
    cfg = PhotonConfig(C1=2, C2=2, R1=2, R2=2)
    photon, stock = load_photon_qwen3(cfg)
    del stock
    return photon


def test_g1b_probe_runs_and_returns_bounded_cosines(photon_real):
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (1, 16))
    result = g1b_student_running_probe(photon_real, input_ids)
    assert set(result) == {"cos_h1_hat_h1", "cos_chunk2_h1_hat_h2"}
    for v in result.values():
        assert -1.0 - 1e-4 <= v <= 1.0 + 1e-4


def test_g1b_probe_restores_training_mode(photon_real):
    photon_real.train()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (1, 16))
    g1b_student_running_probe(photon_real, input_ids)
    assert photon_real.training is True
    photon_real.eval()
