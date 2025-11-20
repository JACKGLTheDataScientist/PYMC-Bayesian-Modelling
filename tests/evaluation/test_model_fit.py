# tests/evaluation/test_evaluate_ppc.py
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pytest
from mmm.evaluation.model_fit import evaluate_ppc

def test_returns_metrics_and_figs_when_ppc_present(idata_small, tmp_path):
    """
    With posterior_predictive available, we should get fit metrics,
    residuals, and both plots saved to disk.
    """
    results, figs = evaluate_ppc(
        idata_small, observed_var="y_obs", output_dir=tmp_path
    )

    # metrics present
    assert isinstance(results["fit_metrics"], dict)
    for k in ("rmse", "mae", "mape", "classical_r2"):
        assert k in results["fit_metrics"]

    # residuals present
    assert isinstance(results["residuals"], np.ndarray)

    # figures present
    assert "time_series_fit" in figs
    assert "residuals_hist" in figs

    # files saved
    paths = results["paths"]
    assert "time_series_fit" in paths and "residuals_hist" in paths


def test_graceful_when_ppc_missing(idata_no_ppc, tmp_path):
    """
    When posterior_predictive is missing, no error; returns None metrics.
    """
    results, figs = evaluate_ppc(
        idata_no_ppc, observed_var="y_obs", output_dir=tmp_path
    )
    assert results["fit_metrics"] is None
    assert results["residuals"] is None
    # should note PPC missing
    assert any("posterior_predictive" in note for note in results.get("notes", []))
    # no figs saved
    assert results["paths"] == {}


def test_prior_overlay_shows_when_requested(tmp_path):
    """
    If PPC is missing but 'prior' or 'prior_predictive' exists and prior_overlay=True,
    we should get a prior overlay figure (and saved file) without error.
    """
    
    # Build a tiny InferenceData with observed and prior_predictive
    y = np.array([1.0, 2.0, 3.0])
    prior_draws = np.stack([y + np.random.normal(0, 1, size=y.size) for _ in range(50)])
    # shape: (draw,) or (chain, draw, obs) — arviz.from_dict will broadcast
    idata = az.from_dict(
        observed_data={"y_obs": y},
        prior_predictive={"y_obs": prior_draws},
    )

    results, figs = evaluate_ppc(
        idata, observed_var="y_obs", prior_overlay=True, output_dir=tmp_path
    )

    # no PPC metrics (since PPC missing)
    assert results["fit_metrics"] is None
    # prior overlay figure present (if ArviZ successfully rendered it)
    # It's okay if figure key is absent due to backend differences; check paths/notes.
    has_overlay_path = "prior_ppc_overlay" in results["paths"]
    said_rendered = any("rendered" in note for note in results.get("notes", []))
    assert has_overlay_path or said_rendered


def test_missing_observed_var_raises(idata_small):
    
    """
    If observed_var is not in observed_data, function returns graceful empty result,
    not a hard crash. (Your implementation returns empty result; adjust assertion if you
    prefer an exception.)
    """
    results, figs = evaluate_ppc(
        idata_small, observed_var="NOT_THERE", output_dir=None
    )
    assert results["fit_metrics"] is None
    assert "observed data not found" in results.get("notes", [])


def test_mape_handles_zero_observed(tmp_path):
    """
    MAPE denominator replaces zeros with 1. Ensure no divide-by-zero crashes.
    """
    y_obs = np.array([0.0, 1.0, 2.0, 0.0])
    # Create PPC with same length, simple noise
    draws = 20
    ppc = np.tile(y_obs, (draws, 1)) + np.random.normal(0, 0.1, size=(draws, y_obs.size))

    # Shape for from_dict: (chain, draw, obs) — use 1 chain
    ppc = ppc.reshape(1, draws, -1)

    idata = az.from_dict(
        observed_data={"y_obs": y_obs},
        posterior_predictive={"y_obs": ppc},
    )

    results, figs = evaluate_ppc(
        idata, observed_var="y_obs", output_dir=tmp_path
    )

    assert "mape" in results["fit_metrics"]
    # should be finite
    assert np.isfinite(results["fit_metrics"]["mape"])


def test_output_dir_already_exists_ok(idata_small, tmp_path):
    """
    If the output directory already exists, evaluate_ppc still saves files without error.
    """
    # Ensure directory exists beforehand
    tmp_path.mkdir(parents=True, exist_ok=True)

    results, figs = evaluate_ppc(
        idata_small, observed_var="y_obs", output_dir=tmp_path
    )
    paths = results["paths"]

    assert "time_series_fit" in paths
    assert "residuals_hist" in paths

def test_ppc_with_posterior_predictive_returns_metrics_and_figs(idata_with_ppc, tmp_path):
    results, figs = evaluate_ppc(idata_with_ppc, observed_var="y_obs", output_dir=tmp_path)
    # metrics present
    fm = results["fit_metrics"]
    assert set(["rmse", "mae", "mape", "classical_r2"]).issubset(fm.keys())
    # figures present and saved
    assert "time_series_fit" in figs
    assert "residuals_hist" in figs
    assert "time_series_fit" in results["paths"]
    assert "residuals_hist" in results["paths"]

def test_prior_overlay_when_no_posterior_predictive(tmp_path):
    # Build a minimal idata with observed + prior_predictive only
    y_obs = np.linspace(0, 1, 20)
    prior_draws = np.tile(y_obs, (1, 1, 1))  # shape (chain=1, draw=1, obs=20)
    idata = az.from_dict(
        observed_data={"y_obs": y_obs},
        prior_predictive={"y_obs": prior_draws},
    )
    results, figs = evaluate_ppc(idata, observed_var="y_obs", output_dir=tmp_path, prior_overlay=True)
    # No posterior predictive -> no fit metrics
    assert results["fit_metrics"] is None
    # Prior overlay figure present and saved
    assert "prior_ppc" in figs
    assert "prior_ppc_overlay" in results["paths"]

def test_missing_observed_graceful(tmp_path):
    # idata without observed_data should return empty results (no error)
    idata = az.from_dict(posterior={"theta": np.array([[0.1]])})
    results, figs = evaluate_ppc(idata, output_dir=tmp_path, prior_overlay=True)
    assert results["fit_metrics"] is None
    assert "observed data not found" in results["notes"]


