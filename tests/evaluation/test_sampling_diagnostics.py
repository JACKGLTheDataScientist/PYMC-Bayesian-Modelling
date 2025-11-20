# tests/diagnostics/test_diagnostics_minimal.py
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg", force=True)  # avoid GUI backends in CI
import pytest

from mmm.evaluation.sampling_diagnostics import sampling_diagnostics

# --------- Tests ---------
def test_returns_expected_keys_and_types(idata_small_diag, tmp_path):
    results, figs = sampling_diagnostics(idata_small_diag, output_dir=tmp_path, prefix="t")

    # results shape
    assert isinstance(results, dict)
    assert "summary" in results and "paths" in results and "bfmi_min" in results
    assert hasattr(results["summary"], "index")  # pandas DataFrame-like

    # figs shape
    assert isinstance(figs, dict)
    assert "trace" in figs and "energy" in figs
    # matplotlib Figure has a "savefig" attr
    assert hasattr(figs["trace"], "savefig")
    assert hasattr(figs["energy"], "savefig")


def test_summary_includes_convergence_columns(idata_small_diag):
    results, _ = sampling_diagnostics(idata_small_diag)
    summary = results["summary"]
    for col in ("r_hat", "ess_bulk", "ess_tail"):
        assert col in summary.columns, f"{col} missing from az.summary output"


def test_saves_trace_and_energy_plots(idata_small_diag, tmp_path):
    results, _ = sampling_diagnostics(idata_small_diag, output_dir=tmp_path, prefix="diag")
    paths = results["paths"]
    assert "trace" in paths and "energy" in paths
    assert (tmp_path / "diag_trace.png").exists()
    assert (tmp_path / "diag_energy.png").exists()


def test_reports_divergences_and_bfmi(idata_small_diag):
    results, _ = sampling_diagnostics(idata_small_diag)
    # divergences may be zero, but should be an int
    assert results["divergences"] is None or isinstance(results["divergences"], int)
    # bfmi_min should be a float (or None if not computable)
    assert results["bfmi_min"] is None or isinstance(results["bfmi_min"], float)


def test_var_names_filters_summary_and_plots(idata_small_diag, tmp_path):
    # keep only beta0 in summary/trace
    results, figs = sampling_diagnostics(
        idata_small_diag, var_names=["beta0"], output_dir=tmp_path, prefix="sub"
    )
    summary = results["summary"]
    assert "beta0" in summary.index
    assert "beta1" not in summary.index  # filtered out

def test_bfmi_fallback_without_sample_stats(idata_no_stats, tmp_path):
    # idata_no_stats is in your conftest and has only a posterior group
    results, figs = sampling_diagnostics(idata_no_stats, output_dir=tmp_path, prefix="nostats2")
    assert "summary" in results
    assert results["bfmi_min"] is None  # fallback path
    # still produced plots & saved them
    assert "trace" in figs and "energy" in figs
    assert "trace" in results["paths"] and "energy" in results["paths"]
