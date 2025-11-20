import numpy as np
import pymc as pm
import pytest
from mmm.model.model import build_mmm

def test_config_vectors_match_channels(model_cfg):
    n = len(model_cfg["channels"])
    for k in ["mu_beta", "sigma_beta", "beta_sigma", "theta_sigma"]:
        assert len(model_cfg[k]) == n, f"{k} must have length {n}"

def test_build_smoke_and_vars(panel_small_with_feats_scaled, model_cfg):
    panel = panel_small_with_feats_scaled
    m = build_mmm(panel, model_cfg)
    assert isinstance(m, pm.Model)

    # coords
    for cd in ["obs", "channels", "product"]:
        assert cd in m.coords

    must_have = [
        "mu_beta","beta_sigma","beta",
        "adstock","theta","slope",
        "sigma","nu","mu",
        "contrib_media","contrib_media_total",
        "roi_channel","y_obs","contrib_baseline",
        "contrib_controls",  # price + gdp combined (if your model exposes separate, change accordingly)
    ]
    for name in must_have:
        assert name in m.named_vars, f"{name} missing"

    # Because panel has trend + Fourier, these should exist
    for name in ["contrib_trend", "contrib_season"]:
        assert name in m.named_vars, f"{name} missing with engineered features"

    # channel coord length matches config
    assert len(m.coords["channels"]) == len(model_cfg["channels"])

def test_noncontiguous_product_ids_ok(panel_small_with_feats_scaled, model_cfg):
    m = build_mmm(panel_small_with_feats_scaled, model_cfg)
    assert len(m.coords["product"]) == panel_small_with_feats_scaled["product_id"].nunique()

def test_set_data_train_test_swap(panel_small_with_feats_scaled, model_cfg):
    panel = panel_small_with_feats_scaled
    panel_train = panel.iloc[:2].copy()
    panel_test  = panel.iloc[2:].copy()

    m = build_mmm(panel_train, model_cfg)

    channels = list(model_cfg["channels"])
    
    def arrays(df):
        season_cols = [c for c in df.columns if c.startswith(("sin_","cos_"))]
        return {
            "prod_idx": df["product_id"].rank(method="dense").astype(int).to_numpy() - 1,
            "X_spend":  df[[f"spend_{ch}" for ch in channels]].to_numpy(float),
            "price":    df["price"].to_numpy(float),
            "gdp":      df["gdp_index"].to_numpy(float),
            "trend":    df["trend"].to_numpy(float) if "trend" in df.columns else None,
            "X_season": df[season_cols].to_numpy(float) if season_cols else None,
        }

    A_te = arrays(panel_test)

    with m:
        updates = {
            "prod_idx": A_te["prod_idx"],
            "X_spend":  A_te["X_spend"],
            "price":    A_te["price"],
            "gdp":      A_te["gdp"],
        }
        if A_te["trend"] is not None:
            updates["trend"] = A_te["trend"]
        if A_te["X_season"] is not None and A_te["X_season"].size > 0:
            updates["X_season"] = A_te["X_season"]

        pm.set_data(updates)

        # quick prior predictive to force evaluation with new obs length
        ppc = pm.sample_prior_predictive(samples=5, var_names=["y_hat"], random_seed=123)
        yhat = ppc.prior["y_hat"] 

        # obs dimension length should match test length
        obs_len = yhat.sizes.get("obs", None)
        assert obs_len == len(panel_test)
