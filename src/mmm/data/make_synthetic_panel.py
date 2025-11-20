import numpy as np
import pandas as pd
from mmm.data.transformations_datagen import adstock_datagen, hill_datagen

def _bounded_positive(x, lo=1e-3, hi=None):
    return float(np.clip(x, lo, hi if hi is not None else np.inf))

def make_synthetic_panel(
    n_markets=3, n_products_per_market=3, n_weeks=156, seed=42,
    start_date="2020-01-06"  # first Monday of 2020
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Create a weekly datetime index ---
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")  # weekly Mondays
    weeks = np.arange(n_weeks)
    panel_data = []

    # ---- Base media effect scalars ----
    BETA_TV, BETA_SEARCH, BETA_SOCIAL = 10_000.0, 7_000.0, 5_000.0
    GAMMA_TV, GAMMA_SEARCH, GAMMA_SOCIAL = 2.5, 1.8, 1.5
    THETA_PCTL = 0.65  # half-saturation quantile

    # ---- Create market and product structure ----
    market_ids = [f"MKT_{i+1}" for i in range(n_markets)]

    for mkt in market_ids:
        for p in range(n_products_per_market):
            # adstock decays (TV high, Search low, Social mid)
            lam_tv     = float(np.clip(rng.normal(0.5, 0.2), 0.001, 0.99))
            lam_search = float(np.clip(rng.normal(0.2, 0.2), 0.001, 0.99))
            lam_social = float(np.clip(rng.normal(0.3, 0.1), 0.001, 0.99))

            # Spend patterns
            spend_tv     = rng.gamma(5, 1000, n_weeks) * (1 + 0.20 * p)
            spend_search = rng.gamma(4,  800, n_weeks) * (1 + 0.10 * p)
            spend_social = rng.gamma(3,  500, n_weeks) * (1 + 0.05 * p)

            # Adjust early/late weeks + bursts for realism
            spend_tv[: n_weeks // 3] *= 0.6
            spend_tv[-n_weeks // 3 :] *= 1.8
            mask_zero  = rng.random(n_weeks) < 0.05
            mask_burst = rng.random(n_weeks) < 0.08
            spend_tv[mask_zero]  = 0.0
            spend_tv[mask_burst] *= rng.uniform(1.5, 2.5, mask_burst.sum())

            # --- Apply adstock + saturation ---
            ad_tv     = adstock_datagen(spend_tv,     lam_tv)
            ad_search = adstock_datagen(spend_search, lam_search)
            ad_social = adstock_datagen(spend_social, lam_social)

            theta_tv     = _bounded_positive(np.quantile(ad_tv,     THETA_PCTL))
            theta_search = _bounded_positive(np.quantile(ad_search, THETA_PCTL))
            theta_social = _bounded_positive(np.quantile(ad_social, THETA_PCTL))

            gamma_tv     = _bounded_positive(rng.normal(GAMMA_TV,     0.3), lo=0.5, hi=5.0)
            gamma_search = _bounded_positive(rng.normal(GAMMA_SEARCH, 0.3), lo=0.5, hi=5.0)
            gamma_social = _bounded_positive(rng.normal(GAMMA_SOCIAL, 0.3), lo=0.5, hi=5.0)

            eff_tv     = hill_datagen(ad_tv,     theta=theta_tv,     gamma=gamma_tv)
            eff_search = hill_datagen(ad_search, theta=theta_search, gamma=gamma_search)
            eff_social = hill_datagen(ad_social, theta=theta_social, gamma=gamma_social)

            contrib_tv     = BETA_TV     * eff_tv
            contrib_search = BETA_SEARCH * eff_search
            contrib_social = BETA_SOCIAL * eff_social

            # ---------------------------
            # Control Variables (Haleon-style)
            # ---------------------------

            price = rng.normal(100, 5, n_weeks) * (1 + 0.03 * p)

            feature_display = np.clip(
                0.1 + 0.05 * np.sin(2 * np.pi * weeks / 26) + rng.normal(0, 0.02, n_weeks),
                0, 0.25
            )

            numeric_distribution = np.clip(
                0.8 + 0.05 * np.sin(2 * np.pi * weeks / 52) + rng.normal(0, 0.02, n_weeks),
                0.7, 1.0
            )

            competitor_spend = rng.gamma(3, 700, n_weeks) * (1 + 0.05 * np.sin(2 * np.pi * weeks / 26))

            weather_index = 1 + 0.4 * np.sin(2 * np.pi * (weeks - 13) / 52) + rng.normal(0, 0.05, n_weeks)
            temperature = 10 + 10 * np.sin(2 * np.pi * (weeks - 13) / 52) + rng.normal(0, 1, n_weeks)

            # ---------------------------
            # Structural effects
            # ---------------------------
            seasonality = 20000 * np.sin(2 * np.pi * weeks / 52) + 10000 * np.cos(2 * np.pi * weeks / 52)
            baseline = 50000 / (1 + np.exp(-0.05 * (weeks - 50)))

            # Peaks: first 2 weeks of Dec, last 2 weeks of Jan
            weeks_iso = dates.isocalendar().week.to_numpy()
            dec_peaks = np.isin(weeks_iso, [48, 49]) * 25000
            jan_peaks = np.isin(weeks_iso, [1, 2]) * 20000

            # ---------------------------
            # KPI (Final)
            # ---------------------------
            kpi = (
                baseline
                + seasonality
                + contrib_tv + contrib_search + contrib_social
                - 200 * price
                + 100 * numeric_distribution
                + 150 * feature_display
                - 50 * competitor_spend / 1000
                + 100 * weather_index
                + 30 * temperature
                + dec_peaks + jan_peaks
                + rng.normal(0, 5000, n_weeks)
            )
            kpi = np.maximum(kpi, 0)

            # ---------------------------
            # Combine
            # ---------------------------
            df = pd.DataFrame({
                "market": mkt,
                "product_id": f"{mkt}_P{p+1}",
                "date": dates,
                "week": weeks,
                "spend_tv": spend_tv,
                "spend_search": spend_search,
                "spend_social": spend_social,
                "price": price,
                "feature_display": feature_display,
                "numeric_distribution": numeric_distribution,
                "competitor_spend": competitor_spend,
                "weather_index": weather_index,
                "temperature": temperature,
                "gdp_index": 100 + 0.1 * weeks + rng.normal(0, 2, n_weeks),
                "kpi_sales": kpi,
            })
            panel_data.append(df)

    return pd.concat(panel_data, ignore_index=True)


