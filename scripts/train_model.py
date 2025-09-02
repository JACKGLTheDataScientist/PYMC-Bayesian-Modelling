# train.py
import argparse
import pandas as pd
import arviz as az

from mmm.synthetic_data import make_synthetic_panel
from mmm.trend import add_trend, add_fourier_terms
from mmm.model import build_mmm

def main(args):
    # 1. Load or generate data
    if args.synthetic:
        panel = make_synthetic_panel(
            n_products=args.n_products,
            n_weeks=args.n_weeks,
            seed=args.seed
        )
    else:
        # In production, replace this with a real data load
        panel = pd.read_csv(args.data_path)

    # 2. Feature engineering
    panel = add_trend(panel, time_col="week")
    panel = add_fourier_terms(panel, time_col="week", period=52, K=2)

    # 3. Train model
    idata = build_mmm(
        panel,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept
    )

    # 4. Save results
    az.to_netcdf(idata, args.output)
    print(f"Saved model inference data to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic panel data")
    parser.add_argument("--n_products", type=int, default=5)
    parser.add_argument("--n_weeks", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, help="Path to real data CSV")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--target_accept", type=float, default=0.9)
    parser.add_argument("--output", type=str, default="results/inference_data.nc")
    args = parser.parse_args()

    main(args)
