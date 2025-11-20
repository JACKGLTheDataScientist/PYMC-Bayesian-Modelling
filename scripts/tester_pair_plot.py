# scripts/tester_sampling_diagnostics.py
import argparse
from pathlib import Path
import json
import arviz as az
from mmm.evaluation.sampling_diagnostics import sampling_diagnostics

def main():
    ap = argparse.ArgumentParser(description="Quick sampling diagnostics tester")
    ap.add_argument("--nc", required=True, help="Path to posterior.nc")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: <nc_dir>/eval/diag)")
    ap.add_argument("--vars", default=None,
                    help="Comma-separated var names (optional; omit for all)")
    ap.add_argument("--prefix", default="diag", help="Filename prefix for figs")
    args = ap.parse_args()

    idata = az.from_netcdf(args.nc)

    # Default output dir
    out_dir = Path(args.out) if args.out else (Path(args.nc).parent / "eval" / "diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    var_names = args.vars.split(",") if args.vars else None

    results, _ = sampling_diagnostics(
        idata=idata,
        var_names=var_names,          # None => all posterior vars
        output_dir=str(out_dir),      # saves diag_trace.png, diag_energy.png here
        prefix=args.prefix,
    )

    # Persist the summary and scalar stats
    results["summary"].to_csv(out_dir / f"{args.prefix}_summary.csv")
    (out_dir / f"{args.prefix}_stats.json").write_text(json.dumps({
        "divergences": results["divergences"],
        "bfmi_min": results["bfmi_min"],
        "paths": results["paths"],
    }, indent=2))

    print("Saved:", results["paths"])
    print("Summary CSV:", out_dir / f"{args.prefix}_summary.csv")
    print("Stats JSON:", out_dir / f"{args.prefix}_stats.json")

if __name__ == "__main__":
    main()

