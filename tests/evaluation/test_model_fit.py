import numpy as np
import pandas as pd
import arviz as az
from mmm.evaluation.model_fit import evaluate_posteriors_and_ppc

def _has_figlike(obj) -> bool:
    return hasattr(obj, "savefig") or hasattr(obj, "figure")

def test_with_ppc_returns_metrics_and_figs(idata_with_ppc: az.InferenceData):
    results, figs = evaluate_posteriors_and_ppc(idata_with_ppc, observed_var="y_obs")

    # Metrics present and finite
    assert isinstance(results["posterior_summary"], pd.DataFrame)
    assert not results["posterior_summary"].empty
    assert results["ppc_summary"] is not None
    assert results["residuals"] is not None
    fm = results["fit_metrics"]
    for k in ["rmse", "mae", "mape", "classical r2"]:
        assert k in fm and np.isfinite(fm[k])

    # Residuals shape matches observed length (regardless of obs dim name)
    n_obs = idata_with_ppc.observed_data["y_obs"].sizes.get(
        "obs", idata_with_ppc.observed_data["y_obs"].sizes.get("y_obs_dim_0")
    )
    assert results["residuals"].shape[0] == n_obs

    # Required figures
    for name in ["posterior", "forest", "ppc", "residuals_hist", "time_series_fit", "residuals_time"]:
        assert name in figs and _has_figlike(figs[name])

    # Optional figures (only if groups exist)
    if "prior_predictive" in idata_with_ppc:
        assert "prior_ppc" in figs and _has_figlike(figs["prior_ppc"])
    if "prior" in idata_with_ppc:
        assert "prior_vs_posterior" in figs and _has_figlike(figs["prior_vs_posterior"])


def test_var_names_filter_limits_posterior_summary(idata_with_ppc: az.InferenceData):
    results, _ = evaluate_posteriors_and_ppc(idata_with_ppc, var_names=["beta0", "beta1"], observed_var="y_obs")
    idx = results["posterior_summary"].index.astype(str).tolist()
    assert any("beta0" in s for s in idx)
    assert any("beta1" in s for s in idx)
    assert not any("sigma" in s for s in idx)


def test_output_dir_saves_expected_files(idata_with_ppc: az.InferenceData, tmp_path):
    outdir = tmp_path / "exp"
    results, figs = evaluate_posteriors_and_ppc(
        idata_with_ppc, observed_var="y_obs", output_dir=str(outdir)
    )
    # Always expected:
    expect = ["posterior_params.png", "posterior_forest.png", "ppc.png",
              "residuals_hist.png", "time_series_fit.png", "residuals_time.png"]
    for fname in expect:
        f = outdir / fname
        assert f.exists(), f"{fname} should be saved"

    # Optional, if groups exist:
    if "prior_predictive" in idata_with_ppc:
        assert (outdir / "prior_ppc.png").exists()
    if "prior" in idata_with_ppc:
        assert (outdir / "prior_vs_posterior.png").exists()


