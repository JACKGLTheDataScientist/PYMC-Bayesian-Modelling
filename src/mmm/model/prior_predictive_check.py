# src/mmm/inference/prior_predictive.py
import pymc as pm
import arviz as az

def run_prior_predictive(model: pm.Model, cfg: dict) -> az.InferenceData:
    """
    Run prior predictive sampling for specified variables.
    """
    n_samples = int(cfg.get("prior_samples", 500)) # Number of KPI prediction iterations using only priors
    observed_var = cfg.get("observed_var", "y_obs") # Observed variable name
    user_vars = tuple(cfg.get("prior_vars", ())) # Additional deterministic variables to obtain prior distributions for (i.e. beta_mpc for media effects)
    
    with model:
        param_names = [rv.name for rv in model.free_RVs]
        keep = list(dict.fromkeys(param_names + [observed_var]))

        # add custom deterministics if present
        if user_vars:
            available = set(model.named_vars.keys())
            keep += [v for v in user_vars if v in available]
            keep = list(dict.fromkeys(keep))
        
        prior_idata = pm.sample_prior_predictive(
            samples=n_samples,
            var_names=keep,
            return_inferencedata=True,
        )
    
    return prior_idata

# Simulates draws from all random variables in the model. Generates priors for all stochastic/random nodes and samples using those priors to generate a prior predictive group for the KPI. 
# Code above has option to include deterministic prior draws (like might want to observe priors for deterministic beta_mpc so can compare posteriors across different media channels)
# pm.sample_prior_predictive does not include deterministic variables by default unless specify in var_names - this argument tells pymc which variables to predict using priors and include in the resulting inference data object