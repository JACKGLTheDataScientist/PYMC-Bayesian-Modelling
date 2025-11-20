# src/mmm/evaluation/simple_fit.py
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def _ensure_dir(path):
    """Create and return Path (or None)."""
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stack_samples(da):
    """
    Return a (samples, obs) ndarray regardless of (chain, draw) presence.
    Expects exactly one non-sample axis (the observation axis).
    """
    if ("chain" in da.dims) and ("draw" in da.dims):
        da = da.stack(sample=("chain", "draw"))
    elif "sample" not in da.dims:
        da = da.expand_dims(sample=[0])  # single-sample fallback

    obs_dims = [d for d in da.dims if d != "sample"]
    if len(obs_dims) != 1:
        raise ValueError(f"_stack_samples expects exactly 1 obs dim, got {da.dims}")
    return da.transpose("sample", obs_dims[0]).values


def _get_draws(idata, observed_var: str):
    """Prefer posterior_predictive[observed_var]; otherwise fallback to posterior['mu']."""
    if hasattr(idata, "posterior_predictive") and observed_var in idata.posterior_predictive:
        return _stack_samples(idata.posterior_predictive[observed_var])
    return _stack_samples(idata.posterior["mu"])


def _interval(draws, hdi_prob: float):
    """
    True Highest Density Interval per observation from draws (samples, obs).
    Returns lo, hi arrays of shape (obs,).
    """
    # az.hdi expects sample axis first here; returns (obs, 2)
    lo_hi = az.hdi(draws, hdi_prob=hdi_prob)
    return lo_hi[:, 0], lo_hi[:, 1]


def _metrics(y, pred, lo, hi):
    """Standard fit metrics (pooled)."""
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae  = float(np.mean(np.abs(resid)))
    mape = float(np.mean(np.abs(resid) / np.where(y == 0, 1.0, y)))
    r2   = float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    return rmse, mae, mape, r2, coverage


def _get_coord(idata, name):
    """Return coord values `name` from the first group that has it."""
    for g in ("posterior", "prior", "posterior_predictive", "prior_predictive",
              "observed_data", "constant_data"):
        grp = getattr(idata, g, None)
        if getattr(grp, "coords", None) is not None and name in grp.coords:
            return grp.coords[name].values
    raise KeyError(f"Coord '{name}' not found in any InferenceData group.")

def _ppc_stat(draws, y, stat="mean", q=None):
    """
    draws: (samples, n_obs) posterior predictive
    y:     (n_obs,) observed
    Returns (sim_stats, obs_stat, title)
    """
    stat = stat.lower()
    if stat == "mean":
        sim = draws.mean(axis=1)
        obs = y.mean()
        title = "PPC: distribution of mean(ỹ)"
    elif stat == "sum":
        sim = draws.sum(axis=1)
        obs = y.sum()
        title = "PPC: distribution of sum(ỹ)"
    elif stat in ("sd", "std"):
        sim = draws.std(axis=1, ddof=1)
        obs = y.std(ddof=1)
        title = "PPC: distribution of sd(ỹ)"
    elif stat in ("median", "med"):
        sim = np.median(draws, axis=1)
        obs = np.median(y)
        title = "PPC: distribution of median(ỹ)"
    elif stat in ("quantile", "quant", "q"):
        if q is None:
            raise ValueError("q must be provided for quantile PPC")
        sim = np.quantile(draws, q, axis=1)
        obs = np.quantile(y, q)
        title = f"PPC: distribution of Q{int(q*100)}(ỹ)"
    else:
        raise ValueError(f"Unknown stat='{stat}'")
    return sim, obs, title


# ----------------------------
# Global pooled fit (obs-level) with product sections
# ----------------------------
def evaluate_global_fit(
    idata,
    observed_var: str = "y_obs",
    output_dir: str | Path | None = None,
    hdi_prob: float = 0.90,
    prefix: str = "",
    product_idx_name: str = "prod_idx",
    product_coord: str = "product",
    show_product_sections: bool = True,
    sort_products_for_plot: bool = False,
    label_sections: bool = True,
):
    """
    Global (pooled) fit across all observations (time x product flattened).
    - Draws product section boundaries (and optional labels) on the pooled plot.
    - If `sort_products_for_plot=True`, reorders a *copy* of the data so sections
      appear as big contiguous blocks in product index order (doesn't affect metrics).

    Saves under ./eval/fit/global by default.
    """
    out = _ensure_dir(output_dir or (Path.cwd() / "eval" / "fit" / "global"))

    # Observed + predictive
    y = idata.observed_data[observed_var].values            # (n_obs,)
    draws = _get_draws(idata, observed_var)                 # (samples, n_obs)
    if draws.shape[1] != len(y):
        raise ValueError(f"n_obs mismatch: draws has {draws.shape[1]} obs, y has {len(y)}")

    # Optional product indices (for sections)
    prod_idx = None
    if hasattr(idata, "constant_data") and product_idx_name in idata.constant_data:
        prod_idx = idata.constant_data[product_idx_name].values.astype(int)
        if len(prod_idx) != len(y):
            raise ValueError(f"len(prod_idx)={len(prod_idx)} != len(y)={len(y)}")
    else:
        if show_product_sections or sort_products_for_plot or label_sections:
            print(f"[warn] '{product_idx_name}' not found in idata.constant_data; "
                  "section boundaries/labels will be skipped.")
            show_product_sections = False
            sort_products_for_plot = False
            label_sections = False

    # Summaries per observation
    pred = draws.mean(axis=0)
    lo, hi = _interval_hdi(draws, hdi_prob)
    rmse, mae, mape, r2, coverage = _metrics(y, pred, lo, hi)

    # For plotting, optionally reorder by product so sections are clean blocks
    if sort_products_for_plot and (prod_idx is not None):
        order = np.argsort(prod_idx)
        y_plot, pred_plot, lo_plot, hi_plot = y[order], pred[order], lo[order], hi[order]
        prod_idx_plot = prod_idx[order]
    else:
        y_plot, pred_plot, lo_plot, hi_plot = y, pred, lo, hi
        prod_idx_plot = prod_idx

    x = np.arange(len(y_plot))

    # 1) Observed vs Predicted with interval
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y_plot,   label="Observed")
    ax.plot(x, pred_plot, label="Predicted mean")
    ax.fill_between(x, lo_plot, hi_plot, alpha=0.3, label=f"{int(hdi_prob*100)}% HDI")
    ax.set_title("Observed vs Predicted (global pooled)")
    ax.legend()

    # Product section boundaries + labels
    if show_product_sections and (prod_idx_plot is not None):
        change_pts = np.where(np.diff(prod_idx_plot) != 0)[0]
        for cp in change_pts:
            ax.axvline(cp + 0.5, color="gray", alpha=0.25, lw=1)

        if label_sections:
            try:
                products = _get_coord(idata, product_coord)
            except KeyError:
                products = np.arange(prod_idx_plot.max() + 1)
            starts = np.r_[0, change_pts + 1]
            ends   = np.r_[change_pts, len(prod_idx_plot) - 1]
            ymax = ax.get_ylim()[1]
            for s, e in zip(starts, ends):
                mid = (s + e) // 2
                label = str(products[prod_idx_plot[mid]]) if prod_idx_plot is not None else ""
                ax.text(mid, ymax, label, ha="center", va="bottom", fontsize=8, alpha=0.6)

    fig.savefig(out / f"{prefix}global_obs_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) PPC histogram (pooled)
    sim, obs, title = _ppc_stat(draws, y, stat="mean")  # or "sum"/"sd"/"median"/("quantile", q=0.9)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sim, bins=40, alpha=0.8, edgecolor="k")
    ax.axvline(obs, color="k", ls="--", lw=2, label="Observed")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out / f"{prefix}global_ppc_stat_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) Residuals over pooled index
    resid_plot = y_plot - pred_plot
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.scatter(x, resid_plot, s=16)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title("Residuals over time (global pooled)")
    ax.set_xlabel("Row index (panel order{} )".format(
        ", sorted by product" if sort_products_for_plot and prod_idx is not None else ""
    ))
    ax.set_ylabel("Obs - Pred")
    fig.savefig(out / f"{prefix}global_residuals_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "coverage": coverage,
        "n_obs": int(len(y)), "output_dir": str(out),
    }


# ----------------------------
# Per-product fit
# ----------------------------
def evaluate_product_fit(
    idata,
    observed_var: str = "y_obs",
    product_coord: str = "product",
    prod_idx_name: str = "prod_idx",
    output_dir: str | Path | None = None,
    hdi_prob: float = 0.90,
    plot_each: bool = True,
    prefix: str = "",
):
    out = _ensure_dir(output_dir or (Path.cwd() / "eval" / "fit" / "per_product"))

    # Observed vector and posterior predictive draws
    y = idata.observed_data[observed_var].values                   # (N,)
    draws = _get_draws(idata, observed_var)                        # (S, N)

    # Product index per observation (required)
    if "constant_data" not in idata or prod_idx_name not in idata.constant_data:
        raise KeyError(f"'{prod_idx_name}' not found in idata.constant_data — "
                       f"add pm.Data('{prod_idx_name}', prod_idx_arr, dims='obs') before sampling.")
    prod_idx = idata.constant_data[prod_idx_name].values           # (N,)

    # Sanity check: lengths must match
    if len(prod_idx) != draws.shape[1] or len(y) != draws.shape[1]:
        raise ValueError(
            f"Length mismatch: len(prod_idx)={len(prod_idx)}, len(y)={len(y)}, "
            f"n_obs_from_draws={draws.shape[1]}. Ensure obs ordering matches."
        )

    # Product labels (fallback to 0..P-1 if coord missing)
    try:
        products = _get_coord(idata, product_coord)
    except KeyError:
        products = np.arange(prod_idx.max() + 1)
        print(f"[warn] Coord '{product_coord}' not found; using numeric labels 0..{products[-1]}.")

    rows = []
    for p_i in range(int(prod_idx.max()) + 1):
        mask = (prod_idx == p_i)
        if not mask.any():
            continue
        y_p = y[mask]                     # (N_p,)
        d_p = draws[:, mask]              # (S, N_p)
        pred_p = d_p.mean(axis=0)
        lo_p, hi_p = _interval(d_p, hdi_prob)

        rmse, mae, mape, r2, coverage = _metrics(y_p, pred_p, lo_p, hi_p)
        label = str(products[p_i]) if p_i < len(products) else str(p_i)
        rows.append({"product": label, "RMSE": rmse, "MAE": mae, "MAPE": mape,
                     "R2": r2, "Coverage": coverage, "n_obs": int(mask.sum())})

        if plot_each:
            t = np.arange(mask.sum())
            fig, ax = plt.subplots(figsize=(9, 3.2))
            ax.plot(t, y_p, label="Observed")
            ax.plot(t, pred_p, label="Pred. mean")
            ax.fill_between(t, lo_p, hi_p, alpha=0.25, label=f"{int(hdi_prob*100)}% interval")
            ax.set_title(f"PPC fit — {label}")
            ax.legend()
            fig.savefig(out / f"{prefix}product_{label}_obs_vs_pred.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 2.6))
            ax.scatter(t, y_p - pred_p, s=14)
            ax.axhline(0, color="k", ls="--", lw=1)
            ax.set_title(f"Residuals — {label}")
            ax.set_xlabel("Time index (within product)")
            ax.set_ylabel("Obs - Pred")
            fig.savefig(out / f"{prefix}product_{label}_residuals.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    table = pd.DataFrame(rows).sort_values("R2", ascending=False)
    table.to_csv(out / f"{prefix}per_product_metrics.csv", index=False)
    return {"table": table, "output_dir": str(out)}


# ----------------------------
# Aggregated (sum over products per time)
# ----------------------------
def evaluate_aggregated_fit(
    idata,
    observed_var: str = "y_obs",
    time_idx_name: str = "obs_idx",
    time_coord: str | None = "time",
    output_dir: str | Path | None = None,
    hdi_prob: float = 0.90,
    prefix: str = "",
):
    """
    Aggregated fit over time:
      - Requires idata.constant_data[time_idx_name] giving an integer time index per obs (0..T-1).
      - Sums observed and predictive across products for each time period.
    Saves under ./eval/fit/aggregated by default.
    """
    out = _ensure_dir(output_dir or (Path.cwd() / "eval" / "fit" / "aggregated"))

    # Inputs
    y = idata.observed_data[observed_var].values          # (n_obs,)
    if "constant_data" not in idata or time_idx_name not in idata.constant_data:
        raise KeyError(f"'{time_idx_name}' not found in idata.constant_data")
    t_idx = idata.constant_data[time_idx_name].values.astype(int)  # (n_obs,)
    T = int(t_idx.max() + 1)
    t_labels = None
    if time_coord is not None:
        try:
            t_labels = _get_coord(idata, time_coord)
        except KeyError:
            t_labels = None

    # Posterior predictive draws
    draws = _get_draws(idata, observed_var)               # (samples, n_obs)

    # Aggregate by time: use bincount per sample
    agg_y = np.bincount(t_idx, weights=y, minlength=T)    # (T,)
    agg_draws = np.vstack([
        np.bincount(t_idx, weights=draws[s], minlength=T)
        for s in range(draws.shape[0])
    ])                                                    # (samples, T)

    pred = agg_draws.mean(axis=0)
    lo, hi = _interval(agg_draws, hdi_prob)

    rmse, mae, mape, r2, coverage = _metrics(agg_y, pred, lo, hi)

    x = np.arange(T) if t_labels is None else np.arange(len(t_labels))
    xticks = t_labels if t_labels is not None else x

    # 1) Aggregated observed vs predicted over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, agg_y, label="Observed (sum over products)")
    ax.plot(x, pred, label="Predicted mean (sum over products)")
    ax.fill_between(x, lo, hi, alpha=0.3, label=f"{int(hdi_prob*100)}% interval")
    ax.set_title("Observed vs Predicted (aggregated by time)")
    if t_labels is not None and len(xticks) <= 24:
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.legend()
    fig.savefig(out / f"{prefix}agg_obs_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) Residuals over time (aggregated)
    resid = agg_y - pred
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.scatter(x, resid, s=16)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title("Residuals over time (aggregated)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Obs - Pred")
    if t_labels is not None and len(xticks) <= 24:
        ax.set_xticks(x)
        ax.set_xticklabels(xticks, rotation=45, ha="right")
    fig.savefig(out / f"{prefix}agg_residuals_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) PPC histogram (aggregated draws pooled)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(agg_draws.ravel(), bins=40, alpha=0.75, edgecolor="k")
    ax.axvline(agg_y.mean(), color="k", ls="--", label="Observed mean")
    ax.set_title("PPC Histogram (aggregated)")
    ax.legend()
    fig.savefig(out / f"{prefix}agg_ppc_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "coverage": coverage,
        "n_periods": int(T), "output_dir": str(out),
    }


