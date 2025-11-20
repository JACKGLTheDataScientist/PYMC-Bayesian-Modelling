######################################
# Scenario Simulation - Global and Hierarichal
######################################
import numpy as np
import pandas as pd
import arviz as az
from typing import Dict

# -------------------------------
# Helper transformations
# -------------------------------

def adstock_geometric(spend: np.ndarray, decay: float) -> np.ndarray:
    """Apply geometric adstock transformation."""
    result = np.zeros_like(spend)
    for t in range(len(spend)):
        result[t] = spend[t] + decay * (result[t - 1] if t > 0 else 0)
    return result


def hill_saturation(adstock_spend: np.ndarray, theta: float, slope: float) -> np.ndarray:
    """Apply Hill saturation transformation (diminishing returns)."""
    return np.power(adstock_spend, slope) / (np.power(theta, slope) + np.power(adstock_spend, slope))


def media_response(spend_array: np.ndarray, beta: float, decay: float, theta: float, slope: float) -> np.ndarray:
    """Compute weekly media contribution."""
    ad = adstock_geometric(spend_array, decay)
    sat = hill_saturation(ad, theta, slope)
    return beta * sat


# -------------------------------
# Main Scenario Simulation
# -------------------------------
from mmm.transformation.modelling_transformations import adstock_geometric, hill_saturation

# --- Define scenario changes per channel (in percentages)
scenario_changes = {
    "tv": 1.10,        # +10%
    "search": 0.90,    # -10%
    "social": 1.20     # +20%
}

# --- Extract posterior samples
posterior = idata.posterior

# Flatten chains + draws
posterior_flat = posterior.stack(sample=("chain", "draw"))

# --- Extract model parameters
betas   = posterior_flat["beta_obs"].values   # shape (N, C, S)
adstock = posterior_flat["adstock"].values    # (C, S)
theta   = posterior_flat["theta"].values      # (C, S)
slope   = posterior_flat["slope"].values      # (C, S)
mu_base = posterior_flat["mu"].values         # (N, S)

# --- Extract channel list
channels = list(model_cfg["channels"])

# --- Adjust spends according to the scenario
X_spend = panel[[f"spend_{ch}" for ch in channels]].to_numpy()
X_scenario = X_spend.copy()

for i, ch in enumerate(channels):
    X_scenario[:, i] *= scenario_changes.get(ch, 1.0)

# --- Apply transformations using posterior parameters
def apply_transform(X, ad, th, sl):
    """Adstock + Hill saturation transform for a single channel."""
    aded = adstock_geometric(X, ad)
    return hill_saturation(aded, th, sl)

# --- Compute scenario predictions across samples
n_samples = betas.shape[-1]
n_obs = X_spend.shape[0]
n_channels = len(channels)

media_contrib_new = np.zeros((n_obs, n_channels, n_samples))

for s in range(n_samples):
    for j in range(n_channels):
        transformed = apply_transform(X_scenario[:, j], adstock[j, s], theta[j, s], slope[j, s])
        media_contrib_new[:, j, s] = betas[:, j, s] * transformed

# --- Sum across channels to get total media contribution
media_total_new = media_contrib_new.sum(axis=1)

# --- Add baseline and controls (already in μ_base)
#    So compute new μ as baseline part + new media contribution - old media contribution
media_total_old = (posterior_flat["contrib_media_total"].values).T  # (N, S)
mu_new = mu_base - media_total_old + media_total_new

# --- Wrap into xarray for convenience
mu_new_da = xr.DataArray(mu_new, dims=["obs", "sample"], name="mu_scenario")

# --- Compare distributions (posterior means)
mu_base_mean = mu_base.mean(axis=1)
mu_new_mean = mu_new.mean(axis=1)

delta = mu_new_mean - mu_base_mean
uplift_pct = (mu_new_mean / mu_base_mean - 1) * 100

