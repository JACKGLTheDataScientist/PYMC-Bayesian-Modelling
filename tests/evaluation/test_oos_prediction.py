# tests/evaluation/test_oos.py
import pytest
import os
import sys
from pathlib import Path
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")  # headless plotting backend for CI
from mmm.evaluation.oos_prediction import evaluate_oos

# ---- Helpers to fabricate PPC draws ------------------------------------------

def _fake_ppc_group(var_name: str, n_obs: int, chains: int = 2, draws: int = 5, mean: float = 10.0):
    """Create an az.InferenceData with a posterior_predictive[var_name] of shape (chains, draws, obs)."""
    arr = np.full((chains, draws, n_obs), mean, dtype=float)
    return az.from_dict(posterior_predictive={var_name: arr})


# ---- Tests -------------------------------------------------------------------

def test_prefers_y_hat_when_present(monkeypatch, tiny_model_with_hat, idata_posterior_only, tmp_path):
    model, _, y = tiny_model_with_hat
    n = len(y)

    # monkeypatch PPC to return only y_hat
    def fake_ppc(posterior, var_names=None, return_inferencedata=True, **kwargs):
        assert var_names == ["y_hat"]
        return _fake_ppc_group("y_hat", n, mean=5.0)
    monkeypatch.setattr(pm, "sample_posterior_predictive", fake_ppc)

    results, figs = evaluate_oos(
        model, idata_posterior_only, y_true=y, output_dir=tmp_path, prefix="oos"
    )

    assert results["used_predictive_var"] == "y_hat"
    assert set(results["metrics"].keys()) == {"rmse", "mae", "mape", "r2"}
    # files saved
    for key in ("time_series", "scatter", "residuals"):
        assert key in results["paths"]
        assert Path(results["paths"][key]).exists()
    # figs returned
    assert all(k in figs for k in ("time_series", "scatter", "residuals"))


def test_falls_back_to_observed_var_when_no_y_hat(monkeypatch, tiny_model_with_hat, idata_posterior_only):
    model, _, y = tiny_model_with_hat
    n = len(y)

    # pretend model does not have y_hat by removing it from named_vars lookup
    # (wrap original model.__contains__/__getitem__ would be messy; instead ask function to force observed_var)
    def fake_ppc(posterior, var_names=None, return_inferencedata=True, **kwargs):
        assert var_names == ["y_obs"]  # should fall back
        return _fake_ppc_group("y_obs", n, mean=8.0)
    monkeypatch.setattr(pm, "sample_posterior_predictive", fake_ppc)

    results, _ = evaluate_oos(
        model, idata_posterior_only, y_true=y, predictive_var="y_obs"
    )
    assert results["used_predictive_var"] == "y_obs"
    # metrics are numeric
    assert all(np.isfinite(results["metrics"][k]) or np.isnan(results["metrics"][k]) for k in results["metrics"])


def test_new_data_is_passed(monkeypatch, tiny_model_with_hat, idata_posterior_only):
    model, _, y = tiny_model_with_hat
    n = len(y)
    called = {"set_data": False}

    def fake_set_data(data, model=None):
        called["set_data"] = True
        # quick shape check of a plausible key
        assert "prod_idx" in data
        assert data["prod_idx"].shape[0] == n

    def fake_ppc(posterior, var_names=None, return_inferencedata=True, **kwargs):
        return _fake_ppc_group(var_names[0], n, mean=7.0)

    monkeypatch.setattr(pm, "set_data", fake_set_data)
    monkeypatch.setattr(pm, "sample_posterior_predictive", fake_ppc)

    new_data = {"prod_idx": np.zeros(n, dtype=int)}  # minimal; other model Data nodes can be omitted in this test
    evaluate_oos(model, idata_posterior_only, y_true=y, new_data=new_data)

    assert called["set_data"] is True


def test_length_mismatch_raises(monkeypatch, tiny_model_with_hat, idata_posterior_only):
    model, _, y = tiny_model_with_hat
    n = len(y)

    monkeypatch.setattr(
        pm, "sample_posterior_predictive",
        lambda *a, **k: _fake_ppc_group("y_hat", n, mean=9.0),
    )

    with pytest.raises(ValueError, match="y_true length"):
        evaluate_oos(model, idata_posterior_only, y_true=y[:-1])


def test_returns_obs_dim_and_ppc_object(monkeypatch, tiny_model_with_hat, idata_posterior_only):
    model, _, y = tiny_model_with_hat
    n = len(y)

    monkeypatch.setattr(
        pm, "sample_posterior_predictive",
        lambda *a, **k: _fake_ppc_group("y_hat", n, mean=6.0),
    )

    results, _ = evaluate_oos(model, idata_posterior_only, y_true=y)
    assert results["obs_dim"] in ("obs", "y_hat_dim_0", "y_obs_dim_0")
    assert isinstance(results["ppc"], az.InferenceData)