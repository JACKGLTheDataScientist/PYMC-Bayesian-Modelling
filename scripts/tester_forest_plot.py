import argparse
from pathlib import Path
import arviz as az
from mmm.evaluation.forest_plot import plot_prior_posterior_forest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True, help="Path to posterior.nc")
    ap.add_argument("--out", default=None, help="Output dir (default: <nc_dir>/eval/forest_test)")
    ap.add_argument("--vars", nargs="*", default=[
        "mu_beta", "beta_sigma", "adstock", "theta", "slope",
        "beta_baseline", "sigma", "nu_raw", "beta"
    ], help="Variables to compare (must exist in BOTH prior & posterior)")
    ap.add_argument("--hdi", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--prod", nargs="*", type=int, default=None, help="Optional product indices (for 'beta')")
    ap.add_argument("--ch", nargs="*", default=None, help="Optional channel names/indices (for 'beta')")
    ap.add_argument("--prefix", default="", help="Optional prefix for saved PNG filenames")
    args = ap.parse_args()

    # Loading in posterior object from directory
    nc_path = Path(args.nc)
    idata = az.from_netcdf(nc_path)

    # Showing idata groups
    print("groups:", idata.groups())
    if hasattr(idata, "prior") and idata.prior is not None:
        print("prior vars:", list(idata.prior.data_vars))
    if hasattr(idata, "posterior") and idata.posterior is not None:
        print("posterior vars:", list(idata.posterior.data_vars))

    # Setting product and channel coords
    coords = {}
    if args.prod is not None:
        coords["product"] = args.prod
    if args.ch is not None:
        coords["channels"] = args.ch

    # Setting output directory
    out_dir = Path(args.out) if args.out else nc_path.parent / "eval" / "forest_test"

    # Running the function to generate posterior and priors
    paths = plot_prior_posterior_forest(
        idata=idata,
        out_dir=out_dir,
        var_names=args.vars,
        hdi_prob=args.hdi,
        chunk=args.chunk,
        coords=(coords or None),
        prefix = args.prefix
    )
    print("Saved:")
    for p in paths:
        print(" -", p)

if __name__ == "__main__":
    main()
