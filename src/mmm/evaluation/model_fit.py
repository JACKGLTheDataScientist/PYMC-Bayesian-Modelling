import matplotlib
matplotlib.use("Agg", force=True)
import os
from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def evaluate_posteriors_and_ppc(
    idata,
    var_names=None,
    observed_var="y_obs",
    output_dir=None,
    max_overlay_vars=9,       # cap overlay grid for readability
):
    """
    Return posterior summaries, PPC tables, residuals, and plots.
    Also creates prior predictive and prior-vs-posterior comparison figs
    when those groups exist in `idata`.

    Returns: (results_dict, figs_dict)
      results_dict:
        - posterior_summary: pd.DataFrame
        - ppc_summary: pd.DataFrame | None
        - residuals: np.ndarray | None
        - fit_metrics: dict | None
      figs_dict: dict[str, matplotlib.figure.Figure]
        Possible keys: 'posterior', 'forest', 'ppc', 'residuals_hist',
                       'time_series_fit', 'residuals_time',
                       'prior_ppc', 'prior_vs_posterior'
    """
    # Make sure output dir exists if requested
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = None

    # -------- Posterior parameter summary --------
    posterior_summary = az.summary(idata, var_names=var_names)

    # -------- Posterior Predictive Checks --------
    ppc_summary = None
    residuals = None
    fit_metrics = None
    figs = {}

    if "posterior_predictive" in idata:
        # observed (xarray -> numpy for metrics)
        y_obs_da = idata.observed_data[observed_var]           # dims: (obs,)
        y_obs = y_obs_da.values                                # np (obs,)

        # posterior predictive draws: dims typically ('chain','draw','obs' or 'y_obs_dim_0')
        y_pred_ppc = idata.posterior_predictive[observed_var]

        # Mean prediction across draws -> (obs,)
        y_pred_mean = y_pred_ppc.mean(dim=("chain", "draw")).values

        residuals = y_obs - y_pred_mean

        # Robust MAPE denominator
        denom = np.where(y_obs == 0, 1.0, y_obs)

        fit_metrics = {
            "rmse": float(np.sqrt(((y_obs - y_pred_mean) ** 2).mean())),
            "mae": float(np.abs(y_obs - y_pred_mean).mean()),
            "mape": float((np.abs((y_obs - y_pred_mean) / denom)).mean()),
            "classical r2": float(
                1 - ((y_obs - y_pred_mean) ** 2).sum()
                / ((y_obs - y_obs.mean()) ** 2).sum()
            ),
        }

        # PPC summaries (per-variable stats over posterior_predictive group)
        ppc_summary = az.summary(idata, group="posterior_predictive")

        # Residuals histogram
        fig, ax = plt.subplots()
        ax.hist(residuals, bins=20, edgecolor="k")
        ax.set_title("Residuals Histogram")
        figs["residuals_hist"] = fig
        if out: fig.savefig(out / "residuals_hist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Observed vs Predicted mean with 95% interval (quantiles across draws)
        # detect obs dimension (anything not chain/draw)
        obs_dim = next(d for d in y_pred_ppc.dims if d not in ("chain", "draw"))
        # 95% interval via quantiles (robust across xarray versions)
        q = y_pred_ppc.quantile([0.025, 0.975], dim=("chain", "draw"))  # dims ('quantile', obs_dim)
        lower = q.sel(quantile=0.025).to_numpy().squeeze()
        upper = q.sel(quantile=0.975).to_numpy().squeeze()

        x = np.arange(y_pred_mean.shape[0])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y_obs, label="Observed")
        ax.plot(x, y_pred_mean, label="Predicted mean")
        ax.fill_between(x, lower, upper, alpha=0.3, label="95% interval")
        ax.set_title("Observed vs Predicted with 95% interval")
        ax.legend()
        figs["time_series_fit"] = fig
        if out: fig.savefig(out / "time_series_fit.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Residuals over time
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(residuals, marker="o", linestyle="-")
        ax.axhline(0, color="red", linestyle="--")
        ax.set_title("Residuals over Time")
        figs["residuals_time"] = fig
        if out: fig.savefig(out / "residuals_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    else:
        print("no posterior predictive in idata")

    # -------- Prior Predictive Check (if present) --------
    if "prior_predictive" in idata:
        # This overlays observed against draws from the prior predictive
        az.plot_ppc(idata, data_pairs={observed_var: observed_var}, group="prior")
        figs["prior_ppc"] = plt.gcf()
        if out: figs["prior_ppc"].savefig(out / "prior_ppc.png", dpi=150, bbox_inches="tight")
        plt.close(figs["prior_ppc"])

    # -------- Prior vs Posterior overlay for parameters (if present) --------
    if "prior" in idata and "posterior" in idata:
        # Choose variables to overlay:
        if var_names is not None:
            overlay_vars = [v for v in var_names
                            if v in idata.posterior and v in idata.prior]
        else:
            # intersection of variables available in both groups
            overlay_vars = list(set(idata.posterior.data_vars).intersection(set(idata.prior.data_vars)))
            # typically you don’t want to overlay huge tensors—cap and keep scalars/1D first
            # prioritize scalars by filtering for <=1 dims (excluding chain/draw)
            def _is_scalar_or_1d(da):
                other_dims = [d for d in da.dims if d not in ("chain", "draw")]
                return len(other_dims) <= 1
            overlay_vars = [v for v in overlay_vars if _is_scalar_or_1d(idata.posterior[v])]
            overlay_vars = overlay_vars[:max_overlay_vars]

        if overlay_vars:
            cols = min(3, len(overlay_vars))
            rows = int(np.ceil(len(overlay_vars) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)
            for ax, v in zip(axes.flat, overlay_vars):
                post = idata.posterior[v].stack(sample=("chain", "draw")).values.ravel()
                prior = idata.prior[v].stack(sample=("chain", "draw")).values.ravel()
                # simple, robust overlay (no seaborn dependency)
                ax.hist(prior, bins=50, density=True, alpha=0.35, label="prior")
                ax.hist(post,  bins=50, density=True, alpha=0.35, label="posterior")
                ax.set_title(v)
                ax.legend()
            # If grid is larger than vars, clear extra axes
            for ax in axes.flat[len(overlay_vars):]:
                ax.axis("off")
            figs["prior_vs_posterior"] = fig
            if out: fig.savefig(out / "prior_vs_posterior.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # -------- Standard posterior diagnostic plots --------
    az.plot_posterior(idata, var_names=var_names)
    figs["posterior"] = plt.gcf()
    if out: figs["posterior"].savefig(out / "posterior_params.png", dpi=150, bbox_inches="tight")
    plt.close(figs["posterior"])

    az.plot_forest(idata, var_names=var_names, combined=True)
    figs["forest"] = plt.gcf()
    if out: figs["forest"].savefig(out / "posterior_forest.png", dpi=150, bbox_inches="tight")
    plt.close(figs["forest"])

    if "posterior_predictive" in idata:
        az.plot_ppc(idata, data_pairs={observed_var: observed_var})
        figs["ppc"] = plt.gcf()
        if out: figs["ppc"].savefig(out / "ppc.png", dpi=150, bbox_inches="tight")
        plt.close(figs["ppc"])

    return {
        "posterior_summary": posterior_summary,
        "ppc_summary": ppc_summary,
        "residuals": residuals,
        "fit_metrics": fit_metrics,
    }, figs
