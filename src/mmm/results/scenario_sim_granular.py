# Running scenario simulation for different market x product combinations
def scenario_analysis_by_group(base_spends_df, delta_pct, beta, decay, theta, slope, channels):
    """
    Run scenario simulation for each market x product combination.

    Parameters
    ----------
    base_spends_df : pandas.DataFrame
        Must have columns:
          - 'market'
          - 'product'
          - one spend column per channel in `channels`
        Each row is a single (market, product) combo with current/base spend levels.

    delta_pct : dict
        Requested % changes per channel. Example:
        {"tv": +0.10, "search": -0.20}
        means: increase TV spend by 10%, cut Search spend by 20%.
        If a channel isn't in the dict, we leave it unchanged.

    beta : something indexable like beta[market][product][channel]
        Effect size per channel for that specific (market, product).
        This is basically your `beta_mpc` posterior mean from the model.

    decay, theta, slope : arrays shaped (n_channels,)
        Adstock decay, Hill theta, Hill slope per channel.

    channels : list of str
        Channel order, e.g. ["tv", "search", "social"].
        Must match the spend column ordering and parameter ordering.
    """

    results = []

    # Loop through each unique (market, product) pair in the spend data
    for (mkt, prod), rows in base_spends_df.groupby(["market", "product"]):

        # 1. Get the base spend vector for THIS market x product
        base_spend = rows[channels].values.squeeze()
        # -> base_spend is now like [tv_spend_for_this_pair,
        #                            search_spend_for_this_pair,
        #                            social_spend_for_this_pair]

        # 2. Get the model parameters for THIS market x product
        beta_mp  = beta[mkt][prod]   # shape (n_channels,)
        decay_mp = decay             # (n_channels,)
        theta_mp = theta             # (n_channels,)
        slope_mp = slope             # (n_channels,)

        # 3. Run scenario for this one market x product
        res = scenario_analysis(
            base_spend=base_spend,
            delta_pct=delta_pct,
            beta=beta_mp,
            decay=decay_mp,
            theta=theta_mp,
            slope=slope_mp,
            channels=channels
        )

        # Add identifiers so we know which row this result belongs to
        res.update({"market": mkt, "product": prod})

        # Store result
        results.append(res)

    # Turn list of dicts into a nice table for analysis
    return pd.DataFrame(results)            