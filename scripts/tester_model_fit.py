# scripts/tester_model_fit.py
import argparse
from pathlib import Path
import json
import sys
import arviz as az

# Import your evaluators (update the import path if your module name differs)
from mmm.evaluation.model_fit import (
    evaluate_global_fit,
    evaluate_product_fit,
    evaluate_aggregated_fit,  # make sure this exists in your evaluation module
)

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate MMM fit from a saved posterior.nc (global, per-product, optional aggregated)."
    )
    ap.add_argument("--nc", required=True, help="Path to posterior.nc")
    ap.add_argument("--out", default=None, help="Base output directory (default: <nc_parent>/eval/fit)")
    ap.add_argument("--obs", default="y_obs", help="Observed variable name in the model (default: y_obs)")
    ap.add_argument("--product-coord", default="product", help="Product coord name in idata (default: product)")
    ap.add_argument("--prod-idx-name", default="prod_idx", help="Name of product index in constant_data (default: prod_idx)")
    ap.add_argument("--time-idx-name", default="time_idx", help="Name of time index in constant_data for aggregation (default: time_idx)")
    ap.add_argument("--time-coord", default="time", help="Optional time coord for prettier x-axis labels (default: time)")
    ap.add_argument("--hdi", type=float, default=0.90, help="Credible interval probability (default: 0.90)")
    ap.add_argument("--prefix", default="", help="Filename prefix for saved artifacts (default: '')")
    return ap.parse_args()

def main():
    args = parse_args()

    nc_path = Path(args.nc).resolve()
    if not nc_path.exists():
        sys.exit(f"[ERROR] File not found: {nc_path}")

    print(f"[info] Loading InferenceData from: {nc_path}")
    idata = az.from_netcdf(nc_path)

    # Default output base: <run-folder>/eval/fit
    base_out = Path(args.out) if args.out else (nc_path.parent / "eval" / "fit")
    (base_out).mkdir(parents=True, exist_ok=True)
    print(f"[info] Outputs will be saved under: {base_out}")

    # Small heads-up for PPC availability
    if hasattr(idata, "posterior_predictive") and args.obs in idata.posterior_predictive:
        print(f"[info] posterior_predictive[{args.obs}] found with dims: {idata.posterior_predictive[args.obs].dims}")
    else:
        print(f"[warn] posterior_predictive[{args.obs}] not found; falling back to posterior['mu'] for predictions.")

    # ---------- Global pooled fit ----------
    print("\n== Global (pooled) fit ==")
    g = evaluate_global_fit(
        idata,
        observed_var="y_obs",
        output_dir=base_out / "global",
        hdi_prob=0.90,
        prefix=args.prefix,
        product_idx_name="prod_idx",
        product_coord="product",
        show_product_sections=True,
        sort_products_for_plot=True,   # set True to force contiguous blocks per product
        label_sections=True,
    )
    print(g)
    with open(base_out / "global" / f"{args.prefix}metrics.json", "w") as f:
        json.dump(g, f, indent=2)

    # ---------- Per-product fit ----------
    print("\n== Per-product fit ==")
    try:
        p = evaluate_product_fit(
            idata,
            observed_var=args.obs,
            product_coord=args.product_coord,
            prod_idx_name=args.prod_idx_name,
            output_dir=base_out / "per_product",
            hdi_prob=args.hdi,
            plot_each=True,
            prefix=args.prefix,
        )
        print(p["table"].head())
    except Exception as e:
        print(f"[error] Per-product fit failed: {type(e).__name__}: {e}")


    # ---------- Aggregated (optional) ----------
    print("\n== Aggregated (sum over products per time) fit ==")
    try:
        a = evaluate_aggregated_fit(
            idata,
            observed_var=args.obs,
            time_idx_name=args.time_idx_name,
            time_coord=args.time_coord,
            output_dir=base_out / "aggregated",
            hdi_prob=args.hdi,
            prefix=args.prefix,
        )
        print(a)
        with open(base_out / "aggregated" / f"{args.prefix}metrics.json", "w") as f:
            json.dump(a, f, indent=2)
    except KeyError as e:
        print(f"[warn] Skipping aggregated fit: {e}")

    print("\n[done] Evaluation artifacts written to:")
    print(f"  - {base_out / 'global'}")
    print(f"  - {base_out / 'per_product'}")

if __name__ == "__main__":
    main()
