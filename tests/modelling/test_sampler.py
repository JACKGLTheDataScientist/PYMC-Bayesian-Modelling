import pymc as pm
import arviz as az
import numpy as np
import pytest
from mmm.model.sampler import run_sampler


def test_run_sampler_returns_inferencedata(tiny_model):
    sampler_cfg = {"draws": 5, "tune": 5}  # minimal config
    idata = run_sampler(tiny_model, sampler_cfg)
    assert isinstance(idata, az.InferenceData)
    assert "posterior" in idata
    

def test_run_sampler_uses_custom_cfg(tiny_model, monkeypatch):
    called_args = {}

    def fake_sample(**kwargs):
        nonlocal called_args
        called_args = kwargs
        return az.from_dict(posterior={"mu": np.array([[1]])})

    monkeypatch.setattr(pm, "sample", fake_sample)

    sampler_cfg = {
        "draws": 20,
        "tune": 10,
        "target_accept": 0.95,
        "chains": 2,
        "cores": 1,
        "seed": 999,
    }

    _ = run_sampler(tiny_model, sampler_cfg)

    # Check overrides
    assert called_args["target_accept"] == 0.95
    assert called_args["chains"] == 2
    assert called_args["cores"] == 1
    assert called_args["random_seed"] == 999
