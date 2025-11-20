# src/mmm/inference/prior_predictive.py
import pymc as pm
import arviz as az

def run_posterior_predictive(model: pm.Model, idata: az.InferenceData, cfg: dict) -> az.InferenceData:
    """
    Run posterior predictive sampling for specified variables in a Bayesian MMM model.

    Parameters
    ----------
    model : pm.Model
        The PyMC model object (already built via build_mmm()).
    idata : az.InferenceData
        The posterior InferenceData returned from sampling.
    cfg : dict
        Configuration dictionary containing:
            - posterior_samples (int): number of posterior predictive samples.
            - observed_var (str): observed variable name (usually 'y_obs').
            - ppc_vars (list/tuple): additional deterministics or nodes to sample.
    Returns
    -------
    az.InferenceData
        Updated InferenceData including posterior predictive group.
    """
    n_samples = int(cfg.get("posterior_samples", 1000)) # How many KPI prediction iterations using posterior distributions
    observed_var = cfg.get("observed_var", "y_obs") # Name of observed KPI node 
    user_vars = tuple(cfg.get("ppc_vars", ())) # Deterministic variables required to be predicted using posteriors

    with model:
        # All free RVs (posterior nodes) - collects all stochastic posterior nodes 
        param_names = [rv.name for rv in model.free_RVs]
        keep = list(dict.fromkeys(param_names + [observed_var]))

        # Include user-specified deterministics (if they exist in graph)
        if user_vars:
            available = set(model.named_vars.keys())
            keep += [v for v in user_vars if v in available]
            keep = list(dict.fromkeys(keep))

        posterior_ppc = pm.sample_posterior_predictive(
            trace=idata,
            var_names=keep,
            extend_inferencedata=True,
            random_seed=cfg.get("seed", 123),
            return_inferencedata=True,
        )

    return posterior_ppc


# By default, all posteriors are generated when the model is sampled
# By default, posterior predictive check will sample the y_obs and generate predictive draws that a PPC can be generated from
# Variables included in var_names argument will also be generated/predicted and saved in idata object  