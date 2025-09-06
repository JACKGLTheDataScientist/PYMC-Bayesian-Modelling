# tests/evaluation/test_sampling_diagnostics.py
import os
import numpy as np
import pandas as pd
import arviz as az
from mmm.evaluation.sampling_diagnostics import evaluate_sampling_diagnostics


def _has_figlike(obj) -> bool:
    # Works for both Matplotlib Figure and Axes-return styles normalized by your code
    return hasattr(obj, "savefig") or hasattr(obj, "figure")


def _finite_or_none(x):
    return (x is None) or np.isfinite(x)


def test_diagnostics_basic_types_and_keys(idata_no_ppc: az.InferenceData):
    """
    Smoke test: runs on a small posterior-only idata (no PPC),
    checks returned structures and core keys.
    """
    results, figs = evaluate_sampling_diagnostics(
        idata_no_ppc, var_names=None, output_dir=None
    )

    # Summary table
    assert isinstance(results["summary"], pd.DataFrame)
    assert not results["summary"].empty

    # Diagnostics dict and required keys
    diag = results["diagnostics"]
    expected_keys = {
        "chains",
        "draws_per_chain",
        "total_draws",
        "divergences",
        "bfmi_by_chain",
        "bfmi_min",
        "rhat_max",
        "ess_bulk_min",
        "ess_tail_min",
        "accept_rate_mean",
        "step_size_mean",
        "step_size_sd",
        "tree_depth_max",
    }
    assert expected_keys.issubset(set(diag.keys()))

    # Chains/draws should be positive ints for this fixture (2 chains, 30 draws)
    assert isinstance(diag["chains"], int) and diag["chains"] >= 1
    assert isinstance(diag["draws_per_chain"], int) and diag["draws_per_chain"] >= 1
    assert isinstance(diag["total_draws"], int) and diag["total_draws"] >= 1

    # Divergences should be an int >= 0
    assert isinstance(diag["divergences"], int) and diag["divergences"] >= 0

    # BFMI list per chain; values should be finite numbers
    # BFMI may be per-chain or unavailable depending on ArviZ/PyMC version and stats
    bfmi_list = diag["bfmi_by_chain"]
    assert isinstance(bfmi_list, list)
    if bfmi_list:  # if per-chain values are present
        assert len(bfmi_list) == diag["chains"]
        assert np.all(np.isfinite(bfmi_list))
    else:
        # fall back to bfmi_min being None or finite
        assert (diag["bfmi_min"] is None) or np.isfinite(diag["bfmi_min"])


    # Optional numeric diagnostics can be None or finite
    for k in ["bfmi_min", "rhat_max", "ess_bulk_min", "ess_tail_min",
              "accept_rate_mean", "step_size_mean", "step_size_sd"]:
        assert _finite_or_none(diag[k])

    # tree_depth_max can be None or an int >= 0
    tdm = diag["tree_depth_max"]
    assert (tdm is None) or (isinstance(tdm, (int, np.integer)) and tdm >= 0)

    # No files were requested
    assert isinstance(results["paths"], dict) and len(results["paths"]) == 0

    # Figures returned: trace/rank/energy/autocorr at minimum
    for name in ["trace", "rank", "energy", "autocorr"]:
        assert name in figs and _has_figlike(figs[name])


def test_var_names_filter_limits_summary(idata_no_ppc: az.InferenceData):
    """
    If we pass var_names, summary should focus on those parameters.
    """
    results, _ = evaluate_sampling_diagnostics(
        idata_no_ppc, var_names=["beta0", "beta1"], output_dir=None
    )
    idx = results["summary"].index.astype(str).tolist()
    # at least those two should be present, and others should not appear if filtered
    assert any("beta0" in s for s in idx)
    assert any("beta1" in s for s in idx)
    assert not any("sigma" in s for s in idx)  # excluded by var_names


def test_paths_written_when_output_dir_provided(tmp_path, idata_small: az.InferenceData):
    """
    When output_dir is provided, PNGs should be written and paths returned.
    """
    results, figs = evaluate_sampling_diagnostics(
        idata_small, var_names=None, output_dir=tmp_path, prefix="tstdiag"
    )

    # Core figures must be saved
    expected_figs = ["trace", "rank", "energy", "autocorr"]
    for name in expected_figs:
        assert name in figs
        assert name in results["paths"]
        p = results["paths"][name]
        assert os.path.isfile(p)
        assert p.endswith(".png")

    # Optional figures (version dependent) may or may not be present
    for optional in ["rhat", "ess"]:
        if optional in figs:
            assert optional in results["paths"]
            assert os.path.isfile(results["paths"][optional])


def test_handles_small_runs_without_crashing(idata_with_ppc: az.InferenceData):
    """
    Works even when the sampling run is tiny (few draws, possible NaNs).
    """
    results, figs = evaluate_sampling_diagnostics(idata_with_ppc)

    # Diagnostics exist; values may be None or finite due to tiny runs
    diag = results["diagnostics"]
    assert "divergences" in diag and isinstance(diag["divergences"], int)

    # Figures exist
    for name in ["trace", "rank", "energy", "autocorr"]:
        assert name in figs and _has_figlike(figs[name])
