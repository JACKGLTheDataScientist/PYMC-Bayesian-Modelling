import pymc as pm
import numpy as np
import pandas as pd
from mmm.model.model import build_mmm
import pytest

# -------- UNIT TESTS --------------#  
# tests/modelling/test_build_mmm.py
def test_config_vectors_match_channels(model_cfg):
    n = len(model_cfg["channels"])
    for k in ["mu_beta", "sigma_beta", "beta_sigma", "theta_sigma"]:
        assert len(model_cfg[k]) == n, f"{k} must have length {n}"

def test_build_smoke_and_vars(panel_small, model_cfg):
    m = build_mmm(panel_small, model_cfg)
    assert isinstance(m, pm.Model)
    # coords exist
    for cd in ["obs", "channels", "product"]:
        assert cd in m.coords
    # core RVs & deterministics exist
    for name in [
        "mu_beta","beta_sigma","beta","adstock","theta","slope","sigma","nu",
        "mu","contrib_media","contrib_media_total","roi_channel","y_obs",
        "contrib_price","contrib_gdp","contrib_season","contrib_trend","contrib_baseline"
    ]:
        assert name in m.named_vars, f"{name} missing"

def test_noncontiguous_product_ids_ok(panel_small, model_cfg):
    m = build_mmm(panel_small, model_cfg)
    assert len(m.coords["product"]) == 2  # mapping should work
