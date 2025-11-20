from typing import Sequence
import pymc as pm
import arviz as az

def run_sampler(model: pm.Model, cfg: dict) -> az.InferenceData:
    """
    Sample a PyMC model using the standard PyTensor backend (no JAX/Numpyro).

    Supports:
      - Prior predictive sampling
      - Posterior sampling
      - Posterior predictive sampling
    """

    # --- Config parameters ---
    draws = int(cfg.get("draws", 2000))  
    tune = int(cfg.get("tune", 1000))    
    
    chains = int(cfg.get("chains", 2))            
    cores = int(cfg.get("cores", 1))              
    target_accept = float(cfg.get("target_accept", 0.8))
    seed = int(cfg.get("seed", 123))
    do_prior = bool(cfg.get("prior_predictive", True))
    do_ppc = bool(cfg.get("posterior_predictive", True))
    prior_n = int(cfg.get("prior_samples", 500))
    ppc_vars = tuple(cfg.get("ppc_vars", "y_obs"))
    observed_var = cfg.get("observed_var", "y_obs")
    progressbar = bool(cfg.get("progressbar", True))
    user_prior_vars = tuple(cfg.get("prior_vars", ()))

    with model:
        # -------------------------
        # Posterior sampling
        # -------------------------
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            init=cfg.get("init", "auto"),
            compute_convergence_checks=cfg.get("compute_convergence_checks", True),
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},  # useful for LOO/WAIC
            progressbar=progressbar,
        )

        # -------------------------
        # Prior predictive sampling
        # -------------------------
        if do_prior:
            param_names = [rv.name for rv in model.free_RVs]
            keep = list(dict.fromkeys(param_names + [observed_var]))

            # Include requested extra variables (e.g. deterministics)
            if user_prior_vars:
                available = set(model.named_vars.keys())
                keep += [v for v in user_prior_vars if v in available]
                keep = list(dict.fromkeys(keep))

            prior_idata = pm.sample_prior_predictive(
                samples=prior_n,
                var_names=keep,
                return_inferencedata=True,
            )
            idata.extend(prior_idata)

        # -------------------------
        # Posterior predictive sampling
        # -------------------------
        if do_ppc and ppc_vars:
            available = set(model.named_vars.keys())
            vnames = [v for v in ppc_vars if v in available]
            if vnames:
                ppc_idata = pm.sample_posterior_predictive(
                    idata.posterior,
                    var_names=vnames,
                    return_inferencedata=True,
                )
                idata.extend(ppc_idata)

    return idata


