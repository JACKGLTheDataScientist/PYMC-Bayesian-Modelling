from pathlib import Path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import arviz as az
import xarray as xr


def plot_prior_posterior_forest(
    idata,
    out_dir,
    var_names,          
    hdi_prob=0.9,
    chunk=10,
    coords=None,
    prefix = ""
):
    """
    Compare PRIOR vs POSTERIOR for variables in `var_names`.

    - Random variables' priors are read from `idata.prior`.
    - Deterministics (e.g., 'beta') are read from `idata.prior_predictive`
    - Posterior values come from `idata.posterior`.

    Returns a list of saved PNG paths showing prior distributions vs posteriors.
    """
    has_post = hasattr(idata, "posterior") and (idata.posterior is not None)
    has_prior = hasattr(idata, "prior") and (idata.prior is not None)
    has_prior_pred = hasattr(idata, "prior_predictive") and (idata.prior_predictive is not None)

    if not has_post:
        raise ValueError("No 'posterior' group in idata.")
        return []
    if not var_names:
        raise ValueError("var_names is empty; set evaluation.forest_plot.variables in YAML.")
        return []

    prior_vars = set(idata.prior.data_vars) if has_prior else set()
    prior_pred_vars = set(idata.prior_predictive.data_vars) if has_prior_pred else set()
    post_vars = set(idata.posterior.data_vars)

    # keep vars that are in POSTERIOR and in (PRIOR or PRIOR_PREDICTIVE)
    candidates = [v for v in var_names if (v in post_vars and (v in prior_vars or v in prior_pred_vars))]
    if not candidates:
            raise ValueError("None of the requested variables are in both posterior and prior/prior_predictive.")
            return []

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Build a merged "prior" dataset from prior+prior_predictive
    merged_prior = {}
    for v in candidates:
        if v in prior_vars:
            merged_prior[v] = idata.prior[v]
        elif v in prior_pred_vars:
            merged_prior[v] = idata.prior_predictive[v]

    prior_ds = xr.Dataset(data_vars=merged_prior)
    post_ds  = idata.posterior[candidates]

    # Feed both into az.plot_forest as separate "models"
    prior_id = az.InferenceData(posterior=prior_ds)
    post_id  = az.InferenceData(posterior=post_ds)

    paths = []
    for i in range(0, len(candidates), chunk):
        chunk_vars = candidates[i:i+chunk]
        az.plot_forest(
            [prior_id, post_id],
            model_names=["prior", "posterior"],
            var_names=chunk_vars,
            combined=True,
            hdi_prob=hdi_prob,
            coords=coords,
        )
        
        fig = plt.gcf()
        fig.tight_layout()
        fname = f"{prefix}forest_prior_vs_posterior_{i//chunk+1}.png" \
                if prefix else f"forest_prior_vs_posterior_{i//chunk+1}.png"
        png = out_dir / fname
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(png))
    return paths
