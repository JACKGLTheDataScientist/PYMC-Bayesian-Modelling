# src/mmm/evaluation/oos.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt


def _extract_var_from_idata(idata: az.InferenceData, name: str):
    """Return DataArray for `name` from posterior_predictive or predictions."""
    if hasattr(idata, "posterior_predictive") and name in idata.posterior_predictive:
        return idata.posterior_predictive[name]
    if hasattr(idata, "predictions") and name in idata.predictions:
        return idata.predictions[name]
    raise KeyError(f"'{name}' not found in posterior_predictive or predictions.")


def evaluate_oos(
    model: pm.Model,
    idata: az.InferenceData,
    y_true: np.ndarray,
    new_data: Optional[dict] = None,
    observed_var: str = "y_obs",      # Likelihood name (used for forecast evaluation)
    include_y_hat: bool = False,      # Also recompute noise-free mean path if present
    output_dir: Optional[str | Path] = None,
    prefix: str = "oos",
) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
    """
    OOS evaluation with posterior predictive forecasting + in-sample LOO & Pareto-k.

    Steps:
      1) (Optional) pm.set_data(new_data) to point model at holdout features.
      2) Posterior predictive for observed_var (default 'y_obs') — includes observation noise.
         Optionally also recompute 'y_hat' (noise-free mean) if present in the model.
      3) Compute RMSE/MAE/MAPE/R² on predictive mean vs y_true.
      4) Save three plots (time-series, scatter, residuals) if output_dir is provided.
      5) Compute in-sample LOO and PSIS Pareto-k diagnostics (requires log_likelihood in idata).

    Returns
    -------
    results: dict with keys
      - "metrics": RMSE/MAE/MAPE/R²
      - "loo": aggregate LOO summary
      - "loo_pareto_k": { mean, max, n_warn, n_bad, warn_idx, bad_idx, khat_plot }
      - "paths": saved figures
      - "obs_dim": observation dim label
      - "ppc": posterior predictive InferenceData
      - "y_hat_mean": optional noise-free mean (if requested and available)
    figs: dict of matplotlib Figures {"time_series","scatter","residuals"}
    """
    
    # --- Validate vars in graph ---
    if observed_var not in model.named_vars:
        raise ValueError(f"'{observed_var}' not found in model.named_vars.")
    want_names = [observed_var]
    if include_y_hat and ("y_hat" in model.named_vars):
        want_names.append("y_hat")

    # --- Output folder (optional) ---
    out = None
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

    # --- Point model at OOS features & sample PPC ---
    with model:
        if new_data is not None:
            pm.set_data(new_data, model=model)

        ppc = pm.sample_posterior_predictive(
            idata.posterior,
            var_names=want_names,
            return_inferencedata=True,
            progressbar=False,
        )

    # --- Pull y_obs PPC for evaluation ---
    y_ppc = _extract_var_from_idata(ppc, observed_var)
    # detect observation axis (anything that's not chain/draw)
    obs_dim = next(d for d in y_ppc.dims if d not in ("chain", "draw"))

    y_mean = y_ppc.mean(("chain", "draw")).to_numpy().reshape(-1)
    q025   = y_ppc.quantile(0.025, ("chain", "draw")).to_numpy().reshape(-1)
    q975   = y_ppc.quantile(0.975, ("chain", "draw")).to_numpy().reshape(-1)

    y_true = np.asarray(y_true).reshape(-1)
    if y_true.shape[0] != y_mean.shape[0]:
        raise ValueError(f"y_true length {y_true.shape[0]} != predictions length {y_mean.shape[0]}")

    resid = y_true - y_mean
    rmse  = float(np.sqrt(np.mean(resid**2)))
    mae   = float(np.mean(np.abs(resid)))
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    mape  = float(np.mean(np.abs(resid) / denom))
    sse   = float(np.sum((y_true - y_mean) ** 2))
    sst   = float(np.sum((y_true - y_true.mean()) ** 2))
    r2    = float(1.0 - sse / sst) if sst > 0 else np.nan
    metrics = {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

    # --- Optional noise-free path (diagnostic) ---
    y_hat_mean = None
    if include_y_hat and ("y_hat" in model.named_vars):
        try:
            y_hat_da = _extract_var_from_idata(ppc, "y_hat")
            y_hat_mean = y_hat_da.mean(("chain", "draw")).to_numpy().reshape(-1)
        except KeyError:
            y_hat_mean = None

    # --- In-sample LOO (aggregate) ---
    try:
        loo_res = az.loo(idata, pointwise=False)
        loo = {
            "elpd_loo": float(loo_res.elpd_loo),
            "elpd_loo_se": float(loo_res.elpd_loo_se),
            "p_loo": float(loo_res.p_loo),
            "looic": float(getattr(loo_res, "looic", -2 * loo_res.elpd_loo)),
            "scale": str(getattr(loo_res, "scale", "log")),
        }
    except Exception as e:
        loo = {"error": f"LOO failed: {type(e).__name__}: {e}"}

    # --- PSIS-LOO Pareto-k (pointwise, influential observations) ---
    loo_k = None
    khat_plot_path = None
    try:
        loo_pw = az.loo(idata, pointwise=True)  # requires idata.log_likelihood
        # Pareto-k DataArray -> numpy
        k = np.asarray(loo_pw.pareto_k)
        n_warn = int((k > 0.7).sum())
        n_bad  = int((k > 1.0).sum())
        warn_idx = np.where(k > 0.7)[0].tolist()
        bad_idx  = np.where(k > 1.0)[0].tolist()
        loo_k = {
            "mean": float(np.mean(k)),
            "max": float(np.max(k)),
            "n_warn_k_gt_0.7": n_warn,
            "n_bad_k_gt_1.0": n_bad,
            "warn_idx": warn_idx,
            "bad_idx": bad_idx,
        }
        # Plot khat and save
        kh_ax = az.plot_khat(loo_pw)
        # robustly get figure
        fig = getattr(kh_ax, "figure", plt.gcf())
        if out is not None:
            khat_plot_path = out / f"{prefix}_khat.png"
            fig.savefig(khat_plot_path, dpi=150, bbox_inches="tight")
            khat_plot_path = str(khat_plot_path)
        plt.close(fig)
    except Exception as e:
        loo_k = {"error": f"Pareto-k failed: {type(e).__name__}: {e}"}

    # --- Plots ---
    figs, paths = {}, {}
    x = np.arange(len(y_true))

    # 1) time series with 95% PI
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_true, label="Observed")
    ax.plot(x, y_mean, label="Predicted mean")
    ax.fill_between(x, q025, q975, alpha=0.3, label="95% PI")
    if y_hat_mean is not None:
        ax.plot(x, y_hat_mean, lw=1.0, alpha=0.8, label="y_hat (mean, noise-free)")
    ax.set_title(f"OOS: observed vs predicted ({observed_var})")
    ax.legend()
    figs["time_series"] = fig
    if out:
        p = out / f"{prefix}_time_series.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths["time_series"] = str(p)
    plt.close(fig)

    # 2) scatter observed vs predicted
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_mean, s=16, alpha=0.85)
    lo, hi = float(min(y_true.min(), y_mean.min())), float(max(y_true.max(), y_mean.max()))
    ax.plot([lo, hi], [lo, hi], "--")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted mean")
    ax.set_title("OOS: observed vs predicted")
    figs["scatter"] = fig
    if out:
        p = out / f"{prefix}_scatter.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths["scatter"] = str(p)
    plt.close(fig)

    # 3) residuals histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(resid, bins=20, edgecolor="k")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_title("OOS residuals")
    figs["residuals"] = fig
    if out:
        p = out / f"{prefix}_residuals.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        paths["residuals"] = str(p)
    plt.close(fig)

    # --- Aggregate outputs ---
    results = {
        "metrics": metrics,
        "loo": loo,                              # in-sample LOO summary
        "loo_pareto_k": {
            **(loo_k or {}),
            "khat_plot": khat_plot_path,
        },
        "paths": paths,
        "obs_dim": obs_dim,
        "ppc": ppc,
        "y_hat_mean": y_hat_mean,
    }
    return results, figs

