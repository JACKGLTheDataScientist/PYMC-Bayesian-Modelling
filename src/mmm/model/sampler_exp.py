from typing import Sequence
import pymc as pm
import arviz as az
import jax
from pymc.sampling.jax import sample_numpyro_nuts


def run_sampler(model: pm.Model, cfg: dict) -> az.InferenceData:
    """
    Run posterior sampling using either PyTensor (default) or JAX (NumPyro) backend.

    cfg options:
        backend: "pytensor" (default) or "jax"
        draws, tune, chains, cores, target_accept, seed, progressbar as before
        chain_method: "parallel" (default) or "vectorized" for JAX backend
        jax_enable_x64: bool, default True

    Returns:
        az.InferenceData with posterior and log_likelihood
    """
    draws = int(cfg.get("draws", 2000))
    tune = int(cfg.get("tune", 2000))
    chains = int(cfg.get("chains", 2))
    cores = int(cfg.get("cores", 1))
    target_accept = float(cfg.get("target_accept", 0.9))
    seed = int(cfg.get("seed", 123))
    progressbar = bool(cfg.get("progressbar", True))
    backend = cfg.get("backend", "pytensor")  # "pytensor" or "jax"
    chain_method = cfg.get("chain_method", "parallel")  # "vectorized" - run chains simultaneously inside one process, "parallel" - run chains independently on seperate CPU cores
    jax_enable_x64 = bool(cfg.get("jax_enable_x64", True))

    if backend == "jax" and jax_enable_x64:
        # enable double precision for stability
        jax.config.update("jax_enable_x64", True)

    with model:
        if backend == "jax":
            print("🔹 Using JAX (NumPyro) backend for sampling...")
            idata = pm.sampling.jax.sample_numpyro_nuts(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=seed,
                progressbar=progressbar,
                chain_method=chain_method,  # "parallel" for CPU, "vectorized" for GPU
                idata_kwargs={"log_likelihood": True},
            )
        else:
            print("🔹 Using default PyTensor (C backend) sampler...")
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                target_accept=target_accept,
                random_seed=seed,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
                progressbar=progressbar,
            )

    return idata
