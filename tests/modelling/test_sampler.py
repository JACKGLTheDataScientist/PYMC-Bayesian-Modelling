# tests/modelling/test_sampler.py
import pymc as pm
import arviz as az
import numpy as np
import pytest
from mmm.model.sampler import run_sampler


def test_run_sampler_returns_inferencedata(tiny_model):
    idata = run_sampler(tiny_model, {"draws": 5, "tune": 5, "cores": 1, "chains": 1})
    assert isinstance(idata, az.InferenceData)
    assert "posterior" in idata


def test_run_sampler_uses_custom_cfg(tiny_model, monkeypatch):
    called_args = {}

    def fake_sample(**kwargs):
        nonlocal called_args
        called_args = kwargs
        return az.from_dict(posterior={"mu": np.array([[1.0]])})

    monkeypatch.setattr(pm, "sample", fake_sample)

    _ = run_sampler(
        tiny_model,
        {
            "draws": 20,
            "tune": 10,
            "target_accept": 0.95,
            "chains": 2,
            "cores": 1,
            "seed": 999,
            "init": "jitter+adapt_diag",
            "compute_convergence_checks": False,
        },
    )

    assert called_args["target_accept"] == 0.95
    assert called_args["chains"] == 2
    assert called_args["cores"] == 1
    assert called_args["random_seed"] == 999
    assert called_args["init"] == "jitter+adapt_diag"
    assert called_args["compute_convergence_checks"] is False


def test_prior_and_posterior_predictive_flags(monkeypatch, tiny_model):
    # Counters
    calls = {"prior": 0, "ppc": 0}

    def fake_sample(**kwargs):
        return az.from_dict(posterior={"mu": np.array([[1.0]])})

    def fake_prior(**kwargs):
        calls["prior"] += 1
        return az.from_dict(prior={"mu": np.array([[0.0]])})

    def fake_ppc(posterior, var_names=None, return_inferencedata=True, **kwargs):
        calls["ppc"] += 1
        assert isinstance(posterior, az.InferenceData) or hasattr(posterior, "dims")
        return az.from_dict(posterior_predictive={"y_obs": np.array([[1.0]])})

    monkeypatch.setattr(pm, "sample", fake_sample)
    monkeypatch.setattr(pm, "sample_prior_predictive", fake_prior)
    monkeypatch.setattr(pm, "sample_posterior_predictive", fake_ppc)

    # both on (default)
    _ = run_sampler(tiny_model, {"draws": 5, "tune": 5})
    assert calls["prior"] == 1
    assert calls["ppc"] == 1

    # turn both off
    calls["prior"] = calls["ppc"] = 0
    _ = run_sampler(
        tiny_model,
        {"draws": 5, "tune": 5, "prior_predictive": False, "posterior_predictive": False},
    )
    assert calls["prior"] == 0
    assert calls["ppc"] == 0


def test_ppc_vars_respected(monkeypatch, tiny_model):
    captured = {"var_names": None}

    def fake_sample(**kwargs):
        return az.from_dict(posterior={"mu": np.array([[1.0]])})

    def fake_ppc(posterior, var_names=None, return_inferencedata=True, **kwargs):
        captured["var_names"] = tuple(var_names) if var_names else None
        return az.from_dict(posterior_predictive={"y_obs": np.array([[1.0]])})

    monkeypatch.setattr(pm, "sample", fake_sample)
    monkeypatch.setattr(pm, "sample_prior_predictive", lambda **_: az.from_dict(prior={}))
    monkeypatch.setattr(pm, "sample_posterior_predictive", fake_ppc)

    _ = run_sampler(
        tiny_model,
        {"draws": 5, "tune": 5, "ppc_vars": ("y_obs",)},  # request only in-sample PPC
    )
    assert captured["var_names"] == ("y_obs",)


def test_missing_required_keys_raises(tiny_model):
    with pytest.raises(KeyError):
        run_sampler(tiny_model, {"tune": 10})
    with pytest.raises(KeyError):
        run_sampler(tiny_model, {"draws": 10})
