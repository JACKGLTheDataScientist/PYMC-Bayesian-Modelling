from pathlib import Path
import json
import arviz as az
import matplotlib.pyplot as plt

def _fig(obj):
    ax0 = obj.ravel()[0] if hasattr(obj, "ravel") else obj
    return getattr(ax0, "figure", plt.gcf())

def _save(fig, output_dir, name, dpi=150):
    if not output_dir:
        return None
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return str(path)

def sampling_diagnostics(idata, var_names=None, output_dir=None, prefix="diag", coords=None):
    """
    Sampling diagnostics:
      - summary (R-hat, ESS)   [subset to var_names if provided]
      - trace plot             [subset to var_names if provided]
      - energy plot (BFMI)     
      - divergences count
      - BFMI (min across chains)
      - tree depth stats
      - step size & acceptance
      - autocorr & rank plots  [subset to var_names if provided]
    """
    # ---------------------------
    # Resolve the variable list
    # ---------------------------
    posterior_vars = set(idata.posterior.data_vars) if hasattr(idata, "posterior") else set()
    requested = list(var_names) if var_names else None
    if requested:
        kept = [v for v in requested if v in posterior_vars]
        missing = [v for v in requested if v not in posterior_vars]
    else:
        kept, missing = None, []

    # Build an InferenceData for plotting (only chosen vars in posterior)
    idata_plot = idata
    if kept:
        idata_plot = idata.copy()
        idata_plot.posterior = idata.posterior[kept]

    # ---------------------------
    # Summary (subset-aware)
    # ---------------------------
    summary = az.summary(
        idata if not kept else idata_plot,
        var_names=kept,  # None -> all
        coords=coords
    )

    # ---------------------------
    # sample_stats-based metrics
    # ---------------------------
    ss = getattr(idata, "sample_stats", None)

    divergences = int(ss["diverging"].sum().item()) if (ss is not None and "diverging" in ss) else None

    bfmi_da = az.bfmi(idata)
    bfmi_min = float(bfmi_da.min().item()) if getattr(bfmi_da, "size", 0) else None

    tree_depth_stats = None
    if ss is not None and "tree_depth" in ss:
        td = ss["tree_depth"]
        td_max = int(td.max().item())
        td_hits = int((td == td_max).sum().item())
        td_total = int(td.size)
        td_ratio = float(td_hits / td_total) if td_total else None
        tree_depth_stats = {
            "max_observed": td_max,
            "hits_at_max": td_hits,
            "hits_ratio": td_ratio,
            "total_draws": td_total,
        }

    step_size_stats = None
    if ss is not None and "step_size" in ss:
        step_size_stats = {
            "mean": float(ss["step_size"].mean().item()),
            "sd": float(ss["step_size"].std().item()),
        }

    acceptance_stats = None
    if ss is not None and "acceptance_rate" in ss:
        acceptance_stats = {
            "mean": float(ss["acceptance_rate"].mean().item()),
            "sd": float(ss["acceptance_rate"].std().item()),
        }
    elif ss is not None and "accept_rate" in ss:
        acceptance_stats = {
            "mean": float(ss["accept_rate"].mean().item()),
            "sd": float(ss["accept_rate"].std().item()),
        }

    # ---------------------------
    # Plots (subset-aware)
    # ---------------------------
    figs, paths, warnings = {}, {}, []

    # Trace (subset to kept)
    tr = az.plot_trace(
        idata_plot if kept else idata,  # if no kept, plot all
        var_names=kept,                 # None -> all
        coords=coords,
        figsize=(10, 6)
    )
    fig = _fig(tr); figs["trace"] = fig
    p = _save(fig, output_dir, f"{prefix}_trace")
    if p: paths["trace"] = p
    plt.close(fig)

    # Energy (global/BFMI)
    en = az.plot_energy(idata)
    fig = _fig(en); figs["energy"] = fig
    p = _save(fig, output_dir, f"{prefix}_energy")
    if p: paths["energy"] = p
    plt.close(fig)

    # Autocorrelation (subset)
    if kept:
        ac = az.plot_autocorr(idata_plot, var_names=kept)
        fig = _fig(ac); figs["autocorr"] = fig
        p = _save(fig, output_dir, f"{prefix}_autocorr")
        if p: paths["autocorr"] = p
        plt.close(fig)

        # Rank (subset)
        rp = az.plot_rank(idata_plot, var_names=kept)
        fig = _fig(rp); figs["rank"] = fig
        p = _save(fig, output_dir, f"{prefix}_rank")
        if p: paths["rank"] = p
        plt.close(fig)

    # ---------------------------
    # Results payload
    # ---------------------------
    if requested and missing:
        warnings.append(f"Ignored missing variables: {missing}")

    results = {
        "requested_vars": requested,
        "plotted_vars": kept if kept else "ALL",
        "missing_vars": missing,
        "summary": summary,
        "divergences": divergences,
        "bfmi_min": bfmi_min,
        "tree_depth": tree_depth_stats,
        "step_size": step_size_stats,
        "acceptance": acceptance_stats,
        "paths": paths,
        "warnings": warnings,
    }

    # Save tables if output_dir provided
    if output_dir:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)

        summary_path = out / f"{prefix}_summary.csv"
        summary.to_csv(summary_path)

        stats_path = out / f"{prefix}_stats.json"
        stats_path.write_text(json.dumps(results, indent=2, default=str))

        results["summary_path"] = str(summary_path)
        results["stats_path"] = str(stats_path)

    return results, figs

