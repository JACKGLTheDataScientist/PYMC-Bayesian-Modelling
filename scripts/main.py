# scripts/main.py
import argparse, os, sys, time, json, yaml, logging, random, platform, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import arviz as az
import multiprocessing as mp
import pandas as pd

# Importing MMM package - self built 
from mmm.data.make_synthetic_panel import make_synthetic_panel
from mmm.feature_engineering.engineered_features import add_trend, add_fourier_terms
from mmm.model.model import build_mmm
from mmm.model.sampler import run_sampler
from mmm.evaluation.model_fit import evaluate_ppc
from mmm.evaluation.forest_plot import plot_prior_posterior_forest
from mmm.evaluation.pair_plot import pair_plot

# ---- Project root anchored ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def setup_logging(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(run_dir / "run.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )

def save_env(run_dir: Path):
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        meta["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        meta["git_commit"] = None
    (run_dir / "env.json").write_text(json.dumps(meta, indent=2))

def get_run_dir(output_root: Path, cfg_path: str, run_override: str | None) -> Path:
    # Prefer explicit --run, then RUN_ID env, then timestamped default
    run_id = run_override or os.environ.get("RUN_ID") or \
             f"{Path(cfg_path).stem}_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    os.environ["RUN_ID"] = run_id
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_dataset(df: pd.DataFrame, run_dir: Path, name: str = "panel_engineered"):
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        fp = data_dir / f"{name}.parquet"
        df.to_parquet(fp, index=False)
    except Exception:
        fp = data_dir / f"{name}.csv"
        df.to_csv(fp, index=False)
    manifest = {
        "file": fp.name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info(f"Saved engineered dataset -> {fp}")


# Defining Main to run workflow
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_baseline.yaml")
    ap.add_argument("--run", default=None, help="Optional run name (overrides timestamp)")
    args = ap.parse_args()

    # Resolve config path relative to project root (robust to CWD)
    cfg_path = (PROJECT_ROOT / args.config).resolve()
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed        = cfg.get("seed", 123)
    model_cfg   = cfg["model_cfg"]
    sampler_cfg = cfg["sampler"]

    np.random.seed(seed); random.seed(seed)

    # Always write under <project-root>/<output_dir>
    output_root = (PROJECT_ROOT / cfg.get("output_dir", "output")).resolve()
    run_dir = get_run_dir(output_root, cfg_path=str(cfg_path), run_override=args.run)

    setup_logging(run_dir)
    logging.info(f"PROJECT_ROOT={PROJECT_ROOT}")
    logging.info(f"Config path = {cfg_path}")
    logging.info(f"Output root = {output_root}")
    logging.info(f"PID={os.getpid()} | RUN_ID={os.environ['RUN_ID']} | run_dir={run_dir}")

    save_env(run_dir)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))

    # ----------- Modelling Process Started --------------

    # ---- Creating Synthetic Data ---- 
    panel = make_synthetic_panel(n_products=3, n_weeks=156, seed=seed)

    # ---- Feature Engineering ---- 
    panel = add_fourier_terms(panel, K=2, period=52, time_col="week")  # sin_*, cos_*
    panel = add_trend(panel, time_col="week")
    save_dataset(panel, run_dir, name="panel_engineered")

    # ---- Building Model ---- 
    mmm = build_mmm(panel, model_cfg)
    backend = sampler_cfg.get("backend", "pytensor").lower()
    logging.info(f"Using backend: {backend.upper()}")

    # ----- MCMC NUTS Sampler ----- 
    t0 = time.time()
    idata = run_sampler(mmm, sampler_cfg)
    logging.info(f"Sampling finished in {time.time()-t0:.1f}s")

    out_path = run_dir / "posterior.nc" # Outputting nc idata object
    az.to_netcdf(idata, out_path)
    logging.info(f"Saved InferenceData -> {out_path}")

    # ---- Posterior & prior PPC / model fit ----
    ppc_dir = run_dir / "eval" / "ppc"
    ppc_dir.mkdir(parents=True, exist_ok=True)
    
    results, _ = evaluate_ppc(
        idata=idata,
        observed_var="y_obs",
        output_dir=ppc_dir,
        prior_overlay=True,     # set False if you want to skip prior overlay
    )
    
    # Save metrics JSON
    (ppc_dir / "metrics.json").write_text(
        json.dumps(results.get("fit_metrics", {}), indent=2)
    )
    logging.info(f"PPC metrics -> {(ppc_dir / 'metrics.json')}")
    
    # Save residuals & ACF info 
    resid = results.get("residuals")
    if resid is not None:
        # simple one-column CSV showing residuals over time
        (ppc_dir / "residuals.csv").write_text(
            "residual\n" + "\n".join(str(float(r)) for r in np.asarray(resid).ravel())
        )
        logging.info(f"Residuals CSV -> {(ppc_dir / 'residuals.csv')}")
    
    acf_info = results.get("residual_acf_sm")
    if acf_info:
        (ppc_dir / "residual_acf.json").write_text(json.dumps(acf_info, indent=2))
        logging.info(f"Residual ACF JSON -> {(ppc_dir / 'residual_acf.json')}")
    
    logging.info(f"PPC figures -> {ppc_dir} (paths: {results.get('paths', {})})")


    # ---- Forest plot (prior vs posterior) ----
    evaluation  = cfg.get("evaluation") or {}
    forest_cfg  = evaluation.get("forest_plot") or {}
    vars_from_yaml = forest_cfg.get("variables") or []

    if forest_cfg.get("enabled", True) and vars_from_yaml:
        forest_dir = run_dir / "eval" / "forest"
        try:
            paths = plot_prior_posterior_forest(
                idata=idata,
                out_dir=forest_dir,
                var_names=vars_from_yaml,                
                hdi_prob=float(forest_cfg.get("hdi_prob", 0.9)),
                chunk=int(forest_cfg.get("chunk", 10)),
                coords=forest_cfg.get("coords"),
                strict=False,
            )
            logging.info(f"Forest plots saved: {len(paths)} -> {forest_dir}")
        except Exception as e:
            logging.warning(f"Forest plotting skipped: {e}")
    else:
        logging.info("Forest plotting disabled or no variables specified in YAML.")

    # This test can be ran after sampling using the saved idata object and tester script

    # ---- Pair Plots (observe divergence causes and multicollinearity) ----
    pair_cfg = (cfg.get("evaluation") or {}).get("pair_plot") or {}
    if pair_cfg.get("enabled", True):
        pair_dir = run_dir / "eval" / "pairs"
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Variables to show (must exist in idata.posterior)
        try: 
            var_names = pair_cfg.get("variables") 
            logging.info(f"variables included in pair plot {var_names}")
        except: 
            logging.warning(f"Variables not found in config file")

        # Which channels/products to slice
        chan_list = pair_cfg.get("channels") or list(model_cfg["channels"])
        prod_list = pair_cfg.get("products") or [0]

        # ArviZ settings
        kind = pair_cfg.get("kind", "kde")
        show_div = bool(pair_cfg.get("divergences", True))

        saved = []
        for ch in chan_list:
            for pid in prod_list:
                # coords slice: theta/slope have dims=("channels",); beta has dims=("product","channels")
                coords = {"channels": [ch], "product": [pid]}
                out_path = pair_dir / f"pair_{ch}_prod{pid}.png"
                try:
                    path = pair_plot(
                        idata=idata,
                        out_path=out_path,
                        var_names=var_names,
                        coords=coords,
                        kind=kind,
                        divergences=show_div,
                        max_vars=4,
                    )
                    saved.append(path)
                except Exception as e:
                    logging.warning(f"Pair plot skipped for {ch}/prod{pid}: {e}")

        logging.info(f"Pair plots saved: {len(saved)} -> {pair_dir}")
    else:
        logging.info("Pair plotting disabled in config.")

    # This test can be ran after sampling using the saved idata object and tester script
    # This is recommended to observe certain variables as not all can be included

    # ---- Sampling Diagnostics ---- 
    


    
    # ---- OOS Prediction ----
    


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()

