# src/mmm/evaluation/oos.py
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr

# Reuse your helpers to avoid duplication
from mmm.evaluation.sampling_diagnostics import _save, _fig_from_axes  # already in your repo

def evaluate_oos_predictions(
    model,
    idata,
    new_data: dict | None,
    y_true,
    observed_var: str = "y_obs",
    output_dir: str | Path | None = None,
    prefix: str = "oos",
    progressbar: bool = False,
    random_seed: int | None = 2025,
):
    """
    Evaluate OUT-OF-SAMPLE predictive accuracy for a fitted PyMC model.

    Parameters
    ----------
    model : pm.Model
        The compiled PyMC model used for training. Must contain pm.Data nodes
        so we can swap to holdout (via pm.set_data).
    idata : az.InferenceData
        Result of pm.sample(..., return_inferencedata=True). We will use idata.posterior.
    new_data : dict or None
        Mapping of pm.Data node names -> holdout arrays (e.g., {"X_spend": X_oos, "week": week_oos, ...}).
        If None, we assume the model already has the holdout data set.
    y_true : array-like
        Ground-truth holdout values (1-D) for the same target RV as `observed_var`.
    observed_var : str
        Name of the likelihood RV to draw in PPC (e.g., "y_obs").
    output_dir : str | Path | None
        If provided, write PNGs to this directory (like other evaluators).
    prefix : str
        Filename prefix for saved figures (e.g., 'oos_time_series.png').
    progressbar : bool
        Show PyMC PPC progress.
    random_seed : int | None
        Seed for reproducibility of PPC draws.

    Returns
    -------
    results : dict
        {
          "metrics": {...},       # RMSE/MAE/MAPE/sMAPE/R²/coverage/width/bias
          "paths": {...},         # saved figure paths (if output_dir)
          "obs_dim": str,         # detected observation dimension in PPC
          "ppc": az.InferenceData # the OOS PPC draws (posterior_predictive group)
        }
    figs : dict[str, matplotlib.figure.Figure]
        Figures (closed after creation): "time_series", "scatter", "residuals_hist", "pit".
    """
    # ---------- 1) Generate OOS PPC ----------
    with model:
        if new_data:
            pm.set_data(new_data, model=model)

        # Use posterior group only (version-safe), not full idata
        ppc_idata = pm.sample_posterior_predictive(
            idata.posterior,
            var_names=[observed_var],
            return_inferencedata=True,
            random_seed=random_seed,
            progressbar=progressbar,
        )

    # Extract PPC draws for the target variable
    y_pred_ppc = ppc_idata.posterior_predictive[observed_var]  # dims: ('chain','draw', obs_dim)
    # Detect the observation dimension (anything not chain/draw)
    obs_dim = next(d for d in y_pred_ppc.dims if d not in ("chain", "draw"))

    # Mean prediction per observation for scalar metrics
    y_pred_mean = y_pred_ppc.mean(dim=("chain", "draw")).to_numpy()

    # Compute 50% and 95% intervals via quantiles
    q = y_pred_ppc.quantile([0.025, 0.25, 0.75, 0.975], dim=("chain", "draw"))  # dims: ('quantile', obs_dim)
    q025 = q.sel(quantile=0.025).to_numpy().squeeze()
    q250 = q.sel(quantile=0.25).to_numpy().squeeze()
    q750 = q.sel(quantile=0.75).to_numpy().squeeze()
    q975 = q.sel(quantile=0.975).to_numpy().squeeze()

    # ---------- 2) Metrics ----------
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.shape[0] != y_pred_mean.shape[0]:
        raise ValueError(f"y_true length {y_true.shape[0]} != predictions length {y_pred_mean.shape[0]}")

    resid = y_true - y_pred_mean
    denom = np.where(y_true == 0, 1.0, y_true)

    rmse   = float(np.sqrt(np.mean((resid) ** 2)))
    mae    = float(np.mean(np.abs(resid)))
    mape   = float(np.mean(np.abs(resid) / denom))
    smape  = float(np.mean(2.0 * np.abs(resid) / (np.abs(y_true) + np.abs(y_pred_mean) + 1e-12)))

    # Classical R² guarded against constant targets
    sse = float(np.sum((y_true - y_pred_mean) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2  = float(1.0 - sse / sst) if sst > 0 else np.nan

    coverage_95 = float(np.mean((y_true >= q025) & (y_true <= q975)))
    coverage_50 = float(np.mean((y_true >= q250) & (y_true <= q750)))
    width_95    = float(np.mean(q975 - q025))
    bias        = float(np.mean(y_pred_mean - y_true))

    # PIT calibration (probability integral transform)
    # stack draws -> ("draws", obs_dim); then PIT_i = P(ŷ <= y_i)
    y_flat = y_pred_ppc.stack(draws=("chain", "draw"))  # dims: ('draws', obs_dim)
    # broadcast compare and average across draws
    pit = (y_flat <= xr.DataArray(y_true, dims=(obs_dim,))).mean(dim="draws").to_numpy()
    pit_mean = float(np.mean(pit))
    pit_std  = float(np.std(pit))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "coverage_50": coverage_50,
        "coverage_95": coverage_95,
        "pi_width_95_mean": width_95,
        "bias_mean": bias,
        "pit_mean": pit_mean,     # ~0.5 if well calibrated
        "pit_std": pit_std,       # ~sqrt(1/12)=0.2887 if uniform
    }

    # ---------- 3) Figures ----------
    figs, paths = {}, {}

    # Time series overlay with 95% band (x-axis = 0..N-1 unless provided by caller later)
    x = np.arange(y_pred_mean.shape[0])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_true, label="Observed")
    ax.plot(x, y_pred_mean, label="Predicted mean")
    ax.fill_between(x, q025, q975, alpha=0.3, label="95% interval")
    ax.set_title("OOS Observed vs Predicted (95% interval)")
    ax.legend()
    figs["time_series"] = fig
    p = _save(fig, output_dir, f"{prefix}_time_series")
    if p: paths["time_series"] = p
    plt.close(fig)

    # Scatter y vs ŷ with 45° line
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred_mean, s=16, alpha=0.8)
    lo = min(np.min(y_true), np.min(y_pred_mean))
    hi = max(np.max(y_true), np.max(y_pred_mean))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted mean")
    ax.set_title("OOS Observed vs Predicted")
    figs["scatter"] = fig
    p = _save(fig, output_dir, f"{prefix}_scatter")
    if p: paths["scatter"] = p
    plt.close(fig)

    # Residuals histogram
    fig, ax = plt.subplots()
    ax.hist(resid, bins=20, edgecolor="k")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("OOS Residuals Histogram")
    figs["residuals_hist"] = fig
    p = _save(fig, output_dir, f"{prefix}_residuals_hist")
    if p: paths["residuals_hist"] = p
    plt.close(fig)

    results = {
        "metrics": metrics,
        "paths": paths,
        "obs_dim": obs_dim,
        "ppc": ppc_idata,  
    }
    return results, figs
