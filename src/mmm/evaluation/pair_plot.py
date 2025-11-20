# src/mmm/evaluation/pair_plot.py
from pathlib import Path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import arviz as az

def pair_plot(
    idata,
    out_path,
    var_names,
    coords=None,
    kind="kde",           # "kde" or "scatter"
    divergences=True,     # overlay red X for divergent transitions
    max_vars=4,           # cap to keep the grid reasonable
):
    """
    Create a single ArviZ pair plot for up to `max_vars` variables from the posterior.
    Saves a PNG to `out_path` and returns the path.
    """
    if not hasattr(idata, "posterior") or idata.posterior is None:
        raise ValueError("idata has no posterior group.")

    # Keep only variables that are actually in the posterior, and cap the count
    present = [v for v in var_names if v in idata.posterior.data_vars]
    if len(present) < 2:
        raise ValueError("Need at least two variables that exist in the posterior to generate pair plots.")
    present = present[:max_vars]

    ax = az.plot_pair(
        idata,
        var_names=present,
        coords=coords or None,
        kind=kind,
        marginals=True,
        point_estimate="median",
        divergences=divergences)

    # Get a figure handle robustly
    try:
        fig = ax[0, 0].figure
    except Exception:
        fig = getattr(ax, "figure", plt.gcf())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)

