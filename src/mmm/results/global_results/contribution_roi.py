###########################################
# Media Channel Coefficients, ROI, Contribution Summary
###########################################
import pandas as pd
import numpy as np
import arviz as az

# Summary Channel Statistics
def summarize_channel_metrics(idata, panel, spend_prefix="spend_", contrib_var="contrib_media_ch"):
    """
    Summarize PyMC MMM posterior results at the channel level.
    
    Parameters
    ----------
    idata : arviz.InferenceData
        InferenceData object containing posterior draws.
    panel : pd.DataFrame
        DataFrame with channel spend columns matching the pattern f"{spend_prefix}{channel}".
    spend_prefix : str, optional
        Prefix used for spend columns in panel (default 'spend_').
    contrib_var : str, optional
        Variable name in posterior representing channel contributions (default 'contrib_media_ch') - Deterministic
    
    Returns
    -------
    pd.DataFrame
        DataFrame summarizing coefficient, ROI, and total contribution statistics per channel.
    """
    channels = idata.posterior.coords["channel"].values

    # --- Coefficients ---
    coef_rows = []
    for channel in channels:
        vals = (
            idata.posterior["mu_global"]
            .sel(channel=channel)
            .stack(sample=("chain", "draw"))
            .values
        )
        coef_rows.append({
            "Channel": str(channel),
            "Coefficient_mean": np.mean(vals),
            "Coefficient_5th": np.percentile(vals, 5),
            "Coefficient_95th": np.percentile(vals, 95),
        })
    df_coef = pd.DataFrame(coef_rows)

    # --- Contributions & ROI ---
    contrib_rows = []
    total_contrib_means = []

    for channel in channels:
        contrib = (
            idata.posterior[contrib_var]
            .sel(channel=channel)
            .stack(sample=("chain", "draw"))
            .values
        )
        sum_contrib = contrib.sum(axis=1)
        total_mean = np.mean(sum_contrib)
        total_contrib_means.append(total_mean)

        total_spend = panel[f"{spend_prefix}{channel}"].sum()
        ROI = sum_contrib / total_spend

        contrib_rows.append({
            "Channel": str(channel),
            "Total_Contribution_Mean": total_mean,
            "Total_Contribution_5th": np.percentile(sum_contrib, 5),
            "Total_Contribution_95th": np.percentile(sum_contrib, 95),
            "ROI_Mean": np.mean(ROI),
            "ROI_5th": np.percentile(ROI, 5),
            "ROI_95th": np.percentile(ROI, 95),
        })

    df_contrib = pd.DataFrame(contrib_rows)
    total_all = np.sum(total_contrib_means)
    df_contrib["Pct_Total_Contribution"] = (
        df_contrib["Total_Contribution_Mean"] / total_all * 100
    )

    # --- Merge ---
    df_summary = pd.merge(df_coef, df_contrib, on="Channel")
    df_summary = df_summary.sort_values(by="Total_Contribution_Mean", ascending=False).reset_index(drop=True)

    return df_summary



# Contribution Over time per channel
def channel_contribution_over_time(
    idata,
    dataset,
    date_col="week",
    contrib_var="contrib_media_ch",
):
    """
    Extracts posterior contribution samples over time per channel
    and returns time series data for plotting contribution trajectories.

    Parameters
    ----------
    idata : arviz.InferenceData
        PyMC InferenceData object containing posterior samples.
    dataset : pd.DataFrame
        Panel data with date_col corresponding to observation order in idata.
    date_col : str, optional
        Column name for the temporal ordering (e.g. 'week').
    contrib_var : str, optional
        Variable name in posterior representing channel contributions.

    Returns
    -------
    dict
        Dictionary keyed by channel name with:
            - 'weeks' : sorted date series
            - 'samples' : np.ndarray of shape (n_samples, n_obs)
            - 'mean' : posterior mean over samples (n_obs,)
    """

    # Ensure chronological order
    dataset_sorted = dataset.sort_values(date_col).reset_index(drop=True)
    weeks = dataset_sorted[date_col].values

    # Determine the order of observations to align with idata
    order = np.argsort(dataset[date_col].values)

    channels = idata.posterior.coords["channel"].values
    output = {}

    for channel in channels:
        # Extract posterior contributions for this channel
        contrib = (
            idata.posterior[contrib_var]
            .sel(channel=channel)
            .stack(sample=("chain", "draw"))  # (n_samples, n_obs)
        )

        # Reorder obs to match sorted weeks
        contrib_sorted = contrib.isel(obs=order)

        # Posterior mean over samples
        mean_contrib = contrib_sorted.mean(dim="sample")

        # Computing ROI mean of channel over time
        roi_samples = idata.posterior["roi_channel_time"].sel(channel = channel).stack(sample = ("chain", "draw")) # (n_samples, n_obs) 

        roi_mean_overtime = roi_samples.mean(axis = 0)
        roi_mean = roi_samples.mean(axis = 1)

        output[str(channel)] = {
            "weeks": weeks,
            "samples": contrib_sorted.values,  # shape: (n_samples, n_obs)
            "mean": mean_contrib.values,       # shape: (n_obs,)
            "roi_samples": roi_samples.values,
            "roi_overtime": roi_mean_overtime, 
            "roi_mean":roi_mean
        }

    return output


        









    