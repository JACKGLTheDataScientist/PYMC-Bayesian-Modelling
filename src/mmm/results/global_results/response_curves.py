#####################################
# Annual Response Curves
#####################################
import numpy as np
import arviz as az

def response_curve(
    idata,
    panel,
    channel,
    market,
    product,
    weeks=52,
    spend_prefix="spend_",
    beta_var="beta_mpc",       # your hierarchical β(market, product, channel)
    slope_var="slope",
    theta_var="theta",
    decay_var="adstock",
):
    """
    Compute Bayesian response curves for a single channel and a specific (market, product).

    Steps:
    1) Define annual spend levels → convert to weekly spend
    2) Extract posterior draws of adstock, slope, theta
    3) Extract coefficient for selected market × product × channel
    4) Apply transformations → weekly → annual contributions
    5) Aggregate to annual contribution
    6) Return spend grid + posterior samples for plotting credible intervals
    """

    # -----------------------------------------
    # 1. Extract market/product indices
    # -----------------------------------------
    market_idx = list(idata.posterior.coords["market"].values).index(market)
    product_idx = list(idata.posterior.coords["product"].values).index(product)

    # -----------------------------------------
    # 2. Extract posterior samples
    # -----------------------------------------
    slope = (
        idata.posterior[slope_var]
        .sel(channel=channel)
        .stack(sample=("chain", "draw"))
        .values
    )
    theta = (
        idata.posterior[theta_var]
        .sel(channel=channel)
        .stack(sample=("chain", "draw"))
        .values
    )
    decay = (
        idata.posterior[decay_var]
        .isel(channel=channel)
        .stack(sample=("chain", "draw"))
        .values
    )

    # hierarchical β(market, product, channel)
    beta = (
        idata.posterior[beta_var]
        .sel(market=market, product=product, channel=channel)
        .stack(sample=("chain", "draw"))
        .values
    )

    n_samples = len(beta)

    # -----------------------------------------
    # 3. Construct annual spend levels
    # -----------------------------------------
    panel["year"] = pd.to_datetime(panel["week"]).dt.year
    current_annual = panel.groupby("year")[f"{spend_prefix}{channel}"].sum().mean()

    annual_spend_levels = np.linspace(0, current_annual * 3, 30)

    # storage: (samples × spend levels)
    annual_contrib = np.zeros((n_samples, len(annual_spend_levels))) # Each row a posterior draw, each column annual spend level

    # -----------------------------------------
    # 4. Loop over posterior draws - simulating one believable world
    # -----------------------------------------
    for s in range(n_samples):

        b  = beta[s]
        sl = slope[s]
        th = theta[s]
        de = decay[s]

        for j, annual_spend in enumerate(annual_spend_levels):

            weekly_spend = annual_spend / weeks
            spend_ts = np.repeat(weekly_spend, weeks)

            # --- Adstock
            ad = np.zeros_like(spend_ts)
            for t in range(weeks):
                ad[t] = spend_ts[t] + de * (ad[t-1] if t > 0 else 0)

            # --- Saturation (Hill)
            sat = (ad**sl) / (th**sl + ad**sl)

            # --- Weekly → Annual
            annual_contrib[s, j] = (b * sat).sum()

    # -----------------------------------------
    # 5. Summary
    # -----------------------------------------
    return {
        "annual_spend_levels": annual_spend_levels,
        "annual_contrib_samples": annual_contrib,
        "annual_contrib_mean": annual_contrib.mean(axis=0),
    }

