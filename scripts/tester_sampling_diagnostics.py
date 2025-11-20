#!/usr/bin/env python3
import argparse
from pathlib import Path
import arviz as az
import json
from mmm.evaluation.sampling_diagnostics import sampling_diagnostics

def main():
    ap = argparse.ArgumentParser(description="Quick sampling diagnostics tester")
    ap.add_argument("--nc", required=True, help="Path to posterior.nc")
    ap.add_argument("--out", default=None, help="Output dir (default: <nc_dir>/eval/diag)")
    ap.add_argument("--vars", default=None, help="Comma-separated var names (optional)")
    ap.add_argument("--channels", nargs="*", default=None, help="Channel names/indices (e.g. tv search)")
    ap.add_argument("--prod", nargs="*", type=int, default=None, help="Product indices (e.g. 0 1)")
    ap.add_argument("--prefix", default="diag", help="Filename prefix for figs")
    args = ap.parse_args()

    # Loading in idata object from folder
    idata = az.from_netcdf(args.nc)

    # Choosing folder to save outputs
    out_dir = Path(args.out) if args.out else (Path(args.nc).parent / "eval" / "diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    # build kwargs conditionally
    kwargs = dict(
        idata=idata,
        output_dir=str(out_dir),
        prefix=args.prefix,
    )

    # Pass var_names if user supplied --vars
    if args.vars:
        # 
        tmp = []
        for tok in args.vars.split():
            tmp.extend([t.strip() for t in tok.split(",") if t.strip()])
        kwargs["var_names"] = tmp

    # pass coords if user supplied either channels or prod
    coords = {}
    if args.channels is not None:
        coords["channels"] = args.channels
    if args.prod is not None:
        coords["product"] = args.prod
    if coords:
        kwargs["coords"] = coords
    
    results, _ = sampling_diagnostics(**kwargs)

if __name__ == "__main__":
    main()
