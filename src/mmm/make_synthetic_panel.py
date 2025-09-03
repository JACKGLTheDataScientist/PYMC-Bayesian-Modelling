import numpy as np
import pandas as pd
from .transformations_datagen import adstock, hill_saturation

def make_synthetic_panel(n_products=5, n_weeks=104, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)
    panel_data = []

    for p in range(n_products):
        
        # product-specific params
        lam_tv = float(np.clip(rng.normal(0.5, 0.2), 0.001, 0.99))
        lam_search = float(np.clip(rng.normal(0.2, 0.2), 0.001, 0.99))
        lam_social = float(np.clip(rng.normal(0.3, 0.1), 0.001, 0.99))

        theta_tv, gamma_tv = rng.normal(5000, 500), rng.normal(1.5, 0.5)
        theta_search, gamma_search = rng.normal(10000, 500), rng.normal(2, 1)
        theta_social, gamma_social = rng.normal(5000, 500), rng.normal(2, 1)

        # spends
        spend_tv = rng.gamma(5, 1000, n_weeks) * (1 + 0.2 * p)
        spend_search = rng.gamma(4, 800, n_weeks) * (1 + 0.1 * p)
        spend_social = rng.gamma(3, 500, n_weeks) * (1 + 0.05 * p)

        # transformations
        ad_tv = adstock(spend_tv, lam_tv)
        ad_search = adstock(spend_search, lam_search)
        ad_social = adstock(spend_social, lam_social)

        eff_tv = hill_saturation(ad_tv, theta = theta_tv, gamma = gamma_tv)
        eff_search = hill_saturation(ad_search, theta = theta_search, gamma = gamma_search)
        eff_social = hill_saturation(ad_social, theta = theta_social, gamma = gamma_social)

        # controls
        price = rng.normal(100, 5, n_weeks) * (1 + 0.05 * p)
        gdp_index = 100 + 0.1 * weeks + rng.normal(0, 2, n_weeks)
        seasonality = 20000 * np.sin(2*np.pi*weeks/52) + 10000 * np.cos(2*np.pi*weeks/52)
        baseline = 50000 / (1 + np.exp(-0.05 * (weeks - 50)))

        # KPI
        kpi = (
            baseline + eff_tv + eff_search + eff_social
            - 200 * (price - 100)
            + 100 * gdp_index
            + seasonality
            + rng.normal(0, 5000, n_weeks)
        )

        kpi = np.maximum(kpi, 0)

        df = pd.DataFrame({
            "product_id": p, "week": weeks,
            "spend_tv": spend_tv, "spend_search": spend_search, "spend_social": spend_social,
            "price": price, "gdp_index": gdp_index, "kpi_sales": kpi
        })
        panel_data.append(df)

    return pd.concat(panel_data, ignore_index=True)
