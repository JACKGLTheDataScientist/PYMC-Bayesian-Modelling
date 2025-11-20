# eval_ppc_standalone.py
import argparse
import arviz as az
from pathlib import Path
from mmm.evaluation.model_fit import evaluate_ppc  

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="Path to posterior.nc")
    ap.add_argument("--out", default=None, help="Output dir for figures (default: alongside nc)")
    ap.add_argument("--observed", default="y_obs")
    ap.add_argument("--prior", action="store_true", help="Also draw prior PPC overlay")
    args = ap.parse_args()

    nc_path = Path(args.nc)
    if not nc_path.exists():
        raise FileNotFoundError(f"Not found: {nc_path.resolve()}")

    idata = az.from_netcdf(nc_path)
    out_dir = Path(args.out) if args.out else nc_path.parent / "eval"

    results, _ = evaluate_ppc(
        idata,
        observed_var=args.observed,
        output_dir=out_dir,
        prior_overlay=args.prior,
    )
    print("Notes:", results['notes'])
    print("Fit metrics:", results['fit_metrics'])
    print(f"Figures saved to: {out_dir}")
    print(idata.groups())

if __name__ == "__main__":
    main()
