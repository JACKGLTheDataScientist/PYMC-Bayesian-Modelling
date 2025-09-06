# sampling_diagnostics.py
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def _fig_from_axes(axes):
    """Normalize ArviZ returns (Axes or array of Axes) to a single Figure."""
    try:
        # az.plot_* may return an Axes or ndarray of Axes
        ax0 = axes.ravel()[0] if hasattr(axes, "ravel") else axes
        return getattr(ax0, "figure", plt.gcf())
    except Exception:
        return plt.gcf()

def _save(fig, output_dir, name, dpi=150):
    """Save a figure to output_dir/name.png if output_dir is provided."""
    if output_dir is None:
        return None
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)

def evaluate_sampling_diagnostics(
    idata,
    var_names=None,
    output_dir=None,
    prefix="diag",
):
    """
    Diagnose PyMC NUTS sampling health (no LOO/WAIC, no model-fit plots).

    Parameters
    ----------
    idata : az.InferenceData
        From pm.sample(..., return_inferencedata=True).
    var_names : list[str] | None
        Optional subset for summaries/plots.
    output_dir : str | Path | None
        If provided, save PNGs to this folder.
    prefix : str
        Filename prefix for saved figures.

    Returns
    -------
    results : dict
        {
          "summary": pandas.DataFrame,
          "diagnostics": {
            "chains", "draws_per_chain", "total_draws",
            "divergences", "bfmi_by_chain", "bfmi_min",
            "rhat_max", "ess_bulk_min", "ess_tail_min",
            "accept_rate_mean", "step_size_mean", "step_size_sd",
            "tree_depth_max",
          },
          "paths": {figure_name: file_path, ...}  # only if output_dir is set
        }
    figs : dict[str, matplotlib.figure.Figure]
        {"trace","rank","energy","autocorr"} figures (already closed).
    """
    # ---------- Tabular summary ----------
    summary = az.summary(idata, var_names=var_names)

    # ---------- Pull sampler stats safely ----------
    ss = idata.sample_stats  # xarray.Dataset
    chains = ss.sizes.get("chain")
    draws = ss.sizes.get("draw")
    total_draws = (chains * draws) if (chains and draws) else None

    divergences = None
    if "diverging" in ss:
        divergences = int(ss["diverging"].sum().item())

    # BFMI (per chain)
    bfmi_da = az.bfmi(idata)  # DataArray over chain
    bfmi_vals = np.ravel(bfmi_da.values) if getattr(bfmi_da, "values", None) is not None else []
    bfmi_by_chain = [float(x) for x in bfmi_vals]
    bfmi_min = float(np.min(bfmi_by_chain)) if len(bfmi_by_chain) else None

    def _safe_col(df, col, reducer):
        return float(reducer(df[col])) if (col in df.columns and len(df)) else None

    rhat_max     = _safe_col(summary, "r_hat", np.nanmax)
    ess_bulk_min = _safe_col(summary, "ess_bulk", np.nanmin)
    ess_tail_min = _safe_col(summary, "ess_tail", np.nanmin)

    # Acceptance rate differs across ArviZ/PyMC versions
    accept_rate_mean = None
    for k in ("acceptance_rate", "mean_tree_accept"):
        if k in ss:
            accept_rate_mean = float(ss[k].mean().item())
            break

    step_size_mean = step_size_sd = None
    if "step_size" in ss:
        step_size_mean = float(ss["step_size"].mean().item())
        step_size_sd   = float(ss["step_size"].std().item())

    tree_depth_max = int(ss["tree_depth"].max().item()) if "tree_depth" in ss else None

    # ---------- Figures (core sampler-health only) ----------
    figs, paths = {}, {}

    ax = az.plot_trace(idata, var_names=var_names, figsize=(10, 6))
    fig = _fig_from_axes(ax)
    figs["trace"] = fig
    p = _save(fig, output_dir, f"{prefix}_trace")
    if p: paths["trace"] = p
    plt.close(fig)

    ax = az.plot_rank(idata, var_names=var_names)
    fig = _fig_from_axes(ax)
    figs["rank"] = fig
    p = _save(fig, output_dir, f"{prefix}_rank")
    if p: paths["rank"] = p
    plt.close(fig)

    ax = az.plot_energy(idata)  # checks E-BFMI shape/overlap
    fig = _fig_from_axes(ax)
    figs["energy"] = fig
    p = _save(fig, output_dir, f"{prefix}_energy")
    if p: paths["energy"] = p
    plt.close(fig)

    ax = az.plot_autocorr(idata, var_names=var_names)
    fig = _fig_from_axes(ax)
    figs["autocorr"] = fig
    p = _save(fig, output_dir, f"{prefix}_autocorr")
    if p: paths["autocorr"] = p
    plt.close(fig)

    # (Optional) R-hat and ESS visual summaries—wrapped to be version-safe
    try:
        ax = az.plot_rhat(idata, var_names=var_names)
        fig = _fig_from_axes(ax)
        figs["rhat"] = fig
        p = _save(fig, output_dir, f"{prefix}_rhat")
        if p: paths["rhat"] = p
        plt.close(fig)
    except Exception:
        pass

    try:
        ax = az.plot_ess(idata, var_names=var_names)
        fig = _fig_from_axes(ax)
        figs["ess"] = fig
        p = _save(fig, output_dir, f"{prefix}_ess")
        if p: paths["ess"] = p
        plt.close(fig)
    except Exception:
        pass

    diagnostics = {
        "chains": chains,
        "draws_per_chain": draws,
        "total_draws": total_draws,
        "divergences": divergences,
        "bfmi_by_chain": bfmi_by_chain,
        "bfmi_min": bfmi_min,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_bulk_min,
        "ess_tail_min": ess_tail_min,
        "accept_rate_mean": accept_rate_mean,
        "step_size_mean": step_size_mean,
        "step_size_sd": step_size_sd,
        "tree_depth_max": tree_depth_max,
    }

    return {"summary": summary, "diagnostics": diagnostics, "paths": paths}, figs

