import pytest
import arviz as az
from mmm.make_synthetic_panel import make_synthetic_panel
from mmm.model import build_mmm

# --- Smoke test ---
def test_build_mmm_runs_and_returns_idata():
    panel = make_synthetic_panel(n_products=2, n_weeks=8, seed=123)
    idata = build_mmm(panel, draws=10, tune=10)

    # Check InferenceData object
    assert isinstance(idata, az.InferenceData)

    # Posterior should contain key parameters
    for var in ["beta", "mu_beta", "beta_baseline", "sigma"]:
        assert var in idata.posterior

    # Log-likelihood present
    assert "y_obs" in idata.log_likelihood


# --- Shape checks ---
def test_beta_shape_matches_products_and_channels():
    panel = make_synthetic_panel(n_products=3, n_weeks=6, seed=42)
    idata = build_mmm(panel, draws=10, tune=10)

    n_products = panel["product_id"].nunique()
    n_channels = 3  # tv, search, social

    beta_shape = idata.posterior["beta"].shape  # (chains, draws, product, channels)

    assert beta_shape[-2] == n_products
    assert beta_shape[-1] == n_channels
