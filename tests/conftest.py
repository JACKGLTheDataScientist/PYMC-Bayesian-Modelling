# tests/conftest.py
import pytest
import numpy as np
import pymc as pm
import arviz as az
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)


@pytest.fixture(scope="function")
def simple_model():
    # minimal example model used by the tests
    x = np.linspace(0, 1, 50)
    y = 1.0 + 2.0 * x + np.random.normal(0, 0.1, size=len(x))

    with pm.Model() as m:
        beta0 = pm.Normal("beta0", 0, 10)
        beta1 = pm.Normal("beta1", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        x_data = pm.Data("x", x)
        pm.Normal("y_obs", beta0 + beta1 * x_data, sigma, observed=y)
    return m

@pytest.fixture(scope="function")
def idata_no_ppc(simple_model) -> az.InferenceData:
    with simple_model:
        idata = pm.sample(
            draws=30, tune=30, chains=2, cores=1, target_accept=0.8, 
            idata_kwargs={"log_likelihood": True}, progressbar=False, return_inferencedata=True)
    return idata


@pytest.fixture(scope="function")
def idata_with_ppc(idata_no_ppc: az.InferenceData, simple_model) -> az.InferenceData:
    """Extend the posterior with PPC; reuse the session posterior."""
    idata = idata_no_ppc.copy()
    with simple_model:
        ppc = pm.sample_posterior_predictive(
            idata.posterior,            
            var_names=["y_obs"],
            random_seed=2025,
            progressbar=False,
            return_inferencedata=True,   
        )
    idata.extend(ppc)                   
    return idata


@pytest.fixture(scope="session")
def idata_small():
    """
    Fit a tiny PyMC model end-to-end and return InferenceData with:
      - posterior
      - sample_stats (energy, diverging)
      - observed_data -> y_obs
      - posterior_predictive -> y_obs
    Kept tiny so tests run quickly and work on Windows (cores=1).
    """
    rng = np.random.default_rng(123)
    n = 40
    x = rng.normal(size=n)
    y = 2.0 + 0.5 * x + rng.normal(scale=0.5, size=n)  # ground truth

    coords = {"obs": np.arange(n)}
    with pm.Model(coords=coords) as model:
        x_data = pm.Data("x", x, dims="obs")
        beta0 = pm.Normal("beta0", 0, 10)
        beta1 = pm.Normal("beta1", 0, 10)
        sigma = pm.HalfNormal("sigma", 1.0)

        mu = beta0 + beta1 * x_data
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs")

        # Small, quick run; cores=1 for Windows friendliness
        idata = pm.sample(
            draws=50,
            tune=50,
            chains=2,
            cores=1,
            target_accept=0.9,
            random_seed=2024,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=False,
        )
        # Posterior predictive for y_obs, keeping chain/draw dims
        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=2025,
            progressbar=False,
        )
        idata.extend(ppc)  # attach posterior_predictive

    return idata


@pytest.fixture
def heldout_small():
    """
    Optional simple holdout 'observed' for OOS tests (same name 'y_obs').
    Your oos function can accept this as a mapping {observed_var: array}.
    """
    rng = np.random.default_rng(456)
    n = 20
    return {"y_obs": 2.0 + 0.5 * rng.normal(size=n) + rng.normal(scale=0.6, size=n)}


@pytest.fixture
def panel_small():
    # 4 rows, 2 products, 3 channels
    return pd.DataFrame({
        "product_id": [0, 0, 101, 101],  # exercise non-contiguous mapping
        "week": [1, 2, 1, 2],
        "kpi_sales": [100., 120., 90., 110.],
        "spend_tv": [10., 20., 5., 15.],
        "spend_search": [5., 10., 2., 8.],
        "spend_social": [3., 5., 1., 4.],
        "price": [50., 50., 55., 55.],
        "gdp_index": [100., 101., 100., 101.],
    })

@pytest.fixture
def panel_small_with_feats(panel_small):
    df = panel_small.copy()
    # trend
    df["trend"] = df["week"].astype(float)

    # simple 1st harmonic Fourier terms (period=52)
    period = 52.0
    w = df["week"].astype(float).to_numpy()
    df["sin_1"] = np.sin(2 * np.pi * 1 * w / period)
    df["cos_1"] = np.cos(2 * np.pi * 1 * w / period)

    # (optional) second harmonic
    # df["sin_2"] = np.sin(2 * np.pi * 2 * w / period)
    # df["cos_2"] = np.cos(2 * np.pi * 2 * w / period)
    return df

@pytest.fixture
def model_cfg():
    return {
        "channels": ["tv", "search", "social"],    # matches spend_* columns
        # per-channel priors
        "mu_beta":     [1000, 500, 300],
        "sigma_beta":  [200,  200, 200],
        "beta_sigma":  [100,  100, 100],
        "theta_sigma": [1,    1,   1  ],
        # hill
        "slope_mu": 1.0,
        "slope_sigma": 0.5,
        # baseline & noise
        "baseline_mu": 20000,
        "baseline_sigma": 5000,
        "noise_sigma": 1000,
        # linear controls (raw)
        "price_mu": 0.0,
        "price_sigma": 10.0,
        "gdp_mu": 0.0,
        "gdp_sigma": 10.0,
        # coefficients for engineered trend/season (on raw features)
        "season_sigma": 0.1,
        "trend_sigma": 0.1,
    }

# A very small dummy model for fast tests
@pytest.fixture
def tiny_model():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=np.random.randn(5))
    return model


@pytest.fixture
def panel_small_scaled():
    df = pd.DataFrame({
        "product_id": [0, 0, 101, 101],   # two products, non-contiguous IDs
        "week":       [1, 2, 1, 2],
        "kpi_sales":  [100., 120., 90., 110.],      # target (raw)
        "price":      [50., 50., 55., 55.],         # control (raw)
        "gdp_index":  [100., 101., 100., 101.],     # control (raw)
        # media (already scaled to [0,1])
        "spend_tv":      [0.20, 0.40, 0.10, 0.30],
        "spend_search":  [0.10, 0.25, 0.05, 0.20],
        "spend_social":  [0.05, 0.10, 0.02, 0.08],
    })
    return df

# --- Same as above, but with engineered trend + Fourier seasonality (still RAW)
@pytest.fixture
def panel_small_with_feats_scaled(panel_small_scaled):
    df = panel_small_scaled.copy()
    # trend (raw, just the time index)
    df["trend"] = df["week"].astype(float)

    # Fourier terms (period=52), raw sine/cosine values
    period = 52.0
    w = df["week"].astype(float).to_numpy()
    df["sin_1"] = np.sin(2 * np.pi * 1 * w / period)
    df["cos_1"] = np.cos(2 * np.pi * 1 * w / period)
    # (optionally) add more harmonics:
    # df["sin_2"] = np.sin(2 * np.pi * 2 * w / period)
    # df["cos_2"] = np.cos(2 * np.pi * 2 * w / period)
    return df

@pytest.fixture(scope="function")
def idata_small_diag():
    """Tiny model to keep tests fast but realistic."""
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    y = 1.5 + 0.7 * x + rng.normal(scale=0.5, size=n)

    coords = {"obs": np.arange(n)}
    with pm.Model(coords=coords) as model:
        beta0 = pm.Normal("beta0", 0, 5)
        beta1 = pm.Normal("beta1", 0, 5)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal("y_obs", mu=beta0 + beta1 * x, sigma=sigma, observed=y, dims="obs")

        idata = pm.sample(
            draws=200, tune=200, chains=2, cores=1,
            target_accept=0.9, random_seed=123,
            return_inferencedata=True, idata_kwargs={"log_likelihood": True},
            progressbar=False,
        )
    return idata


@pytest.fixture(scope="function")
def idata_no_stats():
    """
    Minimal InferenceData without sample_stats to test graceful fallbacks
    for divergences and BFMI.
    """
    return az.from_dict(posterior={"theta": np.array([[0.0, 1.0]])})



@pytest.fixture
def tiny_model_with_hat():
    """Tiny model that defines both y_obs (likelihood) and y_hat (predictive head)."""
    rng = np.random.default_rng(0)
    n = 8
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(0, 0.2, size=n)

    coords = {"obs": np.arange(n)}
    with pm.Model(coords=coords) as m:
        beta0 = pm.Normal("beta0", 0, 5)
        beta1 = pm.Normal("beta1", 0, 5)
        sigma = pm.HalfNormal("sigma", 1.0)
        mu = beta0 + beta1 * x
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="obs")
        # clean predictive head
        pm.Normal("y_hat", mu=mu, sigma=sigma, dims="obs")
    return m, x, y


@pytest.fixture
def idata_posterior_only():
    """Minimal posterior group (no sample_stats etc.) to feed evaluate_oos_simple."""
    # 2 chains, 5 draws, 1 scalar param
    return az.from_dict(posterior={"beta0": np.zeros((2, 5, 1))})


