import pymc as pm
import arviz as az

def run_sampler(model, sampler_cfg):
    """
    Sample from a PyMC model using NUTS, then (optionally) attach
    prior- and posterior-predictive draws to the same InferenceData.
    """
    # Required (fail fast if missing)
    draws = sampler_cfg["draws"]
    tune  = sampler_cfg["tune"]

    # Optionals (safe defaults)
    obs_var  = sampler_cfg.get("observed_var", "y_obs")  # set to "y_data" if your model uses that name
    do_prior = sampler_cfg.get("prior_predictive", True)
    do_ppc   = sampler_cfg.get("posterior_predictive", True)
    prior_n  = sampler_cfg.get("prior_samples", 1000)

    with model:
        # --- Posterior
        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=sampler_cfg.get("target_accept", 0.8),
            chains=sampler_cfg.get("chains", 4),
            cores=sampler_cfg.get("cores", 4),
            random_seed=sampler_cfg.get("seed", 123),
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            progressbar=sampler_cfg.get("progressbar", True),
        )

        # --- Prior predictive (optional)
        if do_prior:
            prior_idata = pm.sample_prior_predictive(
                samples=prior_n,
                return_inferencedata=True,
            )
            idata.extend(prior_idata)

        # --- Posterior predictive (optional)
        if do_ppc:
            ppc_idata = pm.sample_posterior_predictive(
                idata=idata,
                var_names=[obs_var],      # must match your likelihood RV name
                return_inferencedata=True,
            )
            idata.extend(ppc_idata)

    return idata

