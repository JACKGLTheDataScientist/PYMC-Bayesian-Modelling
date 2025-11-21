# src/mmm/model/model.py
import numpy as np
import pymc as pm
import pytensor.tensor as at
from mmm.transformation.modelling_transformations import adstock_geometric, hill_saturation

def build_mmm(panel, model_cfg):
    """
    Bayesian MMM where:
      - ONLY media spends are transformed inside the graph (adstock + saturation with priors)
      - Trend and seasonality (Fourier) are PRE-ENGINEERED in `panel`
    """
    # ----------------
    # Coordinates, Indices & Arrays
    # ----------------
    channels = list(model_cfg['channels']) # Making channel list for coords
    markets = np.sort(panel['market'].unique()) # Extracting unique markets for coords
    products = np.sort(panel['product_id'].unique()) # Extracting unique products for coords
    obs_idx = np.arange(len(panel), dtype="int64") # Creates an integer index for each observation 

    # Setting coords dictionary
    coords = {"channel": channels, "market": markets, "product": products, "obs": obs_idx}
    
    # --- Market and product row indices ---
    mar_map = {pid: i for i, pid in enumerate(markets)} # Mapping of market to interger {1: UK, 2: US} etc...
    mar_idx_arr = panel['market'].map(mar_map).to_numpy(dtype="int64") # Maps market index to market column (obs) - market index per row
    
    prod_map = {pid: i for i, pid in enumerate(products)} # Mapping of product to interger {1: pr1, 2: pr2} etc...
    prod_idx_arr = panel["product_id"].map(prod_map).to_numpy(dtype="int64") # Maps product index per row
    
    # --- Media spend array ---
    X_spend_arr = panel[[f"spend_{ch}" for ch in channels]].to_numpy(dtype=np.float64)
    
    # --- KPI ---
    y_obs_arr = panel["kpi_sales"].to_numpy(dtype=np.float64)
    
    # --- Controls ---
    price_arr = panel["price"].to_numpy(dtype=np.float64)
    feature_arr = panel["feature_display"].to_numpy(dtype=np.float64)
    dist_arr = panel["numeric_distribution"].to_numpy(dtype=np.float64)
    comp_arr = panel["competitor_spend"].to_numpy(dtype=np.float64)
    weather_arr = panel["weather_index"].to_numpy(dtype=np.float64)
    temp_arr = panel["temperature"].to_numpy(dtype=np.float64)
    gdp_arr = panel["gdp_index"].to_numpy(dtype=np.float64)
    
    # --- Trend variable ---
    trend_arr = panel["week"].to_numpy(dtype=np.float64)
    
    # --- Fourier Seasonality ---
    season_cols = [c for c in panel.columns if c.startswith("sin_") or c.startswith("cos_")]
    X_season_arr = panel[season_cols].to_numpy(dtype=np.float64) if season_cols else None
    if season_cols:
        coords["season_term"] = season_cols


    with pm.Model(coords=coords) as mmm:
        # ----------------
        # Data containers: Mutable Pytensor variables
        # ----------------
        market_idx = pm.Data("mar_idx", mar_idx_arr, dims="obs") 
        prod_idx = pm.Data("prod_idx", prod_idx_arr, dims="obs")
        X_spend = pm.Data("X_spend", X_spend_arr, dims=("obs", "channel"))
        y_data = pm.Data("y_data", y_obs_arr, dims="obs")
        
        # Controls
        price = pm.Data("price", price_arr, dims="obs")
        feature = pm.Data("feature_display", feature_arr, dims="obs")
        distribution = pm.Data("numeric_distribution", dist_arr, dims="obs")
        competitor = pm.Data("competitor_spend", comp_arr, dims="obs")
        weather = pm.Data("weather_index", weather_arr, dims="obs")
        temperature = pm.Data("temperature", temp_arr, dims="obs")
        gdp = pm.Data("gdp", gdp_arr, dims="obs")
        
        # Trend and Fourier seasonality
        trend = pm.Data("trend", trend_arr, dims="obs")
        if X_season_arr is not None:
            X_season = pm.Data("X_season", X_season_arr, dims=("obs", "season_term"))


        # ----------------
        # Media effects (crossed hierarchy) + in-graph transforms
        # ----------------
        
        # ====================================
        # Example 1: Cross Hierarichal structure: Market x Product Multiplicative deviation from the global mean
        # ====================================
        # ---------------------------------------
        # # Define the parameters in the deviation priors (sigma_market, sigma_product etc...) parameters on log scale (i.e. np.log)
        # # When deviations are combined with z for non-centered hierarichy and then exponentiated and combined with lognormal global prior (has been exp to natural scale already), this introduces multiplicative deviation (%deviations)
        # # This structure enforces positive media coefficients for channels are product x market levels given the exp elements in beta_mpc. Z can be <0 but once exp, it becomes positive. 
        # # Need to be careful as can cause unidentifitability issues if deviations are large (deviation goes large quickly across market product deviations) - given % deviations rather than additive (+)
        # ---------------------------------------

        # Start
        # # Defining config values defined in config file
        # mu_vals        = np.asarray(model_cfg["mu_global"],    dtype=np.float64)  # per-channel central tendency (>0)
        # sigma_vals     = np.asarray(model_cfg["sigma_global"], dtype=np.float64)  # per-channel multiplicative spread (>1)
        # vals_mar_sigma = np.asarray(model_cfg["sigma_market"], dtype=np.float64)  # per-channel market spread (>1)
        # vals_prod_sigma = np.asarray(model_cfg["sigma_product"],dtype=np.float64)  # per-channel product spread (>1)
        # vals_prod_mar_sigma = np.asarray(model_cfg["sigma_product_market"],dtype=np.float64)  # per-channel product spread (>1)

        # # ---- Global (positive, per channel) ----
        # mu_global = pm.LogNormal("mu_global", mu=np.log(mu_vals), sigma=np.log(sigma_vals), dims="channel")  # (C,)

        # # ---- Market deviations (M, C) ----
        # sigma_market = pm.HalfNormal("sigma_market", sigma=np.log(vals_mar_sigma), dims="channel")  # (C,) 
        # z_market     = pm.Normal("z_market", 0.0, 1.0, dims=("market", "channel"))                  # (M, C)
        # mu_market = pm.Deterministic("mu_market",
        #                              mu_global * at.exp(sigma_market * z_market), # Multiplicative on the natural scale
        #                              dims=("market", "channel"))                                    # (M, C)

        # # ---- Product deviations (P, C) ----
        # sigma_product = pm.HalfNormal("sigma_product", sigma=np.log(vals_prod_sigma), dims="channel")  # (C,)
        # z_product     = pm.Normal("z_product", 0.0, 1.0, dims=("product", "channel"))                  # (P, C)
        # mu_product = pm.Deterministic("mu_product",
        #                               mu_global * at.exp(sigma_product * z_product),
        #                               dims=("product", "channel"))                                     # (P, C)

        # # ---- Crossed interaction (M, P, C) ----
        # # Fine grained differences not explained by either market nor product deviations
        # sigma_market_product = pm.HalfNormal("sigma_market_product", sigma=np.log(vals_prod_mar_sigma), dims="channel")        # (C,)
        # z_market_product     = pm.Normal("z_market_product", 0.0, 1.0, dims=("market", "product", "channel"))  # (M,P,C)


        # # Media product x market specific coefficients
        # # ---- The beta_mpc (M,P,C) parameter flipped onto log scale - Additive ----
        # # log(beta_mpc) = log(mu_global)
        # #                + sigma_market * z_market[m,c]
        # #                + sigma_product * z_product[p,c]
        # #                + sigma_mp * z_market_product[m,p,c]


        # # ---- The beta_mpc (M,P,C) parameter flipped onto natural scale - multiplicative ----
        # beta_mpc = pm.Deterministic(
        #     "beta_mpc",
        #     mu_global[None, None, :] *
        #     at.exp(sigma_market[None, None, :]* z_market[:,  None,  :]) *
        #     at.exp(sigma_product[None, None, :]* z_product[None, :,   :]) *
        #     at.exp(sigma_market_product[None, None, :]* z_market_product),      
        #     dims=("market", "product", "channel")
        # )  


        # ==================================
        # Example 2: Crossed Hierarichy with additive deviation
        # ==================================
        # ----------------------------------
        # Still keeping global effect strictly positive by using lognormal prior - global effect has multiplicative deviation around mean
        # Difference from above is additive structure in beta_mpc - additive deviation from mean across markets, products and market x product (intracies) from global mean
        # This helps limit blowup in deviations coming from multiplicative version above
        # This structure can in rare occasions generate negative media coefficients for channels are product x market levels if sigma is large enough and z is negative. Setting sigmas small overcomes this 
        # ----------------------------------
        # Defining from config file the global effect and variation, along with market and product deviations
        mu_vals = np.asarray(model_cfg["mu_global"], dtype = np.float64) # Global effect 
        sigma_vals = np.asarray(model_cfg['sigma_global'], dtype = np.float64) # Variation in global effect - multiplicative 
        vals_mar_sigma = np.asarray(model_cfg['sigma_market'], dtype = np.float64) # Market variation from global 
        vals_prod_sigma = np.asarray(model_cfg['sigma_product'], dtype = np.float64) # Product variation from global
        vals_prod_mar_sigma = np.asarray(model_cfg['sigma_product_market'], dtype = np.float64) # Deviations from market x product combinations

        # Defining prior for global effect - strictly positive
        mu_global = pm.LogNormal("mu_global", mu = np.log(mu_vals), sigma = np.log(sigma_vals), dims = "channel")  # Parameters on log scale (multiplicative variation)

        # Market Deviation from global effects
        sigma_market = pm.HalfNormal("sigma_market", sigma = vals_mar_sigma, dims = "channel") # Additive natural scale deviation from global mean (no logging parameters)
        z_market = pm.Normal("z_market", mu = 0, sigma = 1, dims = ("market", "channel")) # Non-centered hierarichy - prevents strong correlation coupled relationship with global (slow sampling). Defines directional variation from global mean( (+/-) for each market per channel (Allows positive and negative deviation given z can be +-). 

        # Product Deviation from global effects 
        sigma_product = pm.HalfNormal("sigma_product", sigma = vals_prod_sigma, dims = "channel") # Additive natural scale deviation from global mean (no logging parameters)
        z_product = pm.Normal("z_product", mu = 0, sigma = 1, dims = ("product", "channel")) # Non-centered hierarichy - prevents strong correlation coupled relationship with global (slow sampling). Defines directional variation from global mean( (+/-)

        # Market x product specific deviations from global - unique nuances
        sigma_market_product = pm.HalfNormal("sigma_market_product", sigma = vals_prod_mar_sigma, dims = "channel")
        z_market_product = pm.Normal("z_market_product", mu = 0, sigma = 1, dims = ("market", "product", "channel")) # Non-centered hierarichy - prevents strong correlation coupled relationship with global (slow sampling). Defines directional variation from global mean( (+/-) for each product per channel (Allows positive and negative deviation given z can be +-). 

        # Deterministic variable to define channel product x market specific coefficients - additive deviation
        beta_mpc = pm.Deterministic(
            "beta_mpc",
            mu_global[None, None, :] +
            sigma_market[None, None, :]* z_market[:,  None,  :] +   
            sigma_product[None, None, :]* z_product[None, :,   :] +
            sigma_market_product[None, None, :]* z_market_product,      
            dims=("market", "product", "channel")
        )  

        # The above generates a 3D tensor of coefficients - one coefficient per market x product x channel combination: 
        # Important to look at the shapes of the components in beta_mpc by looking at dims in each prior set above 
        # **mu_global** has original shape (C,) - one global coefficient per channel. By adding [None, None, :] make it (1, 1, C). Mu global then broadcasted across all markets and products - gives every market×product cell same base channel coefficient.
        # **sigma_market** has original shape (C,) - same shape as mu_global so broadcasted the same way
        # **z_market** has shape (M,C) so by adding [:, None, :] broadcasts into (M, 1, C) 

                              
    
        # # ================================
        # # Example 3: Truncated Crossed Hierarchy with additive deviation (using different priors)
        # # ================================

        # # ----------------------------------
        # # Enforcing positivity in global effect using truncated normal - global effect has additive deviation from mean with upper and lower bounds
        # # Difference from above is enforcing positivity using truncated normal - still additive variation between market, product and market x product deviations from the global mean
        # # Using gamma to enforce positive deviations that are usually small but allow occasionally large deviations
        # # ----------------------------------
        # # Defining from config file the global effect and variation, along with market and product deviations
        # mu_vals = np.asarray(model_cfg["mu_global"], dtype = np.float64) # Global effect 
        # sigma_vals = np.asarray(model_cfg['sigma_global'], dtype = np.float64) # Variation in global effect
        # vals_mar_sigma = np.asarray(model_cfg['sigma_market'], dtype = np.float64) # Market variation from global 
        # vals_prod_sigma = np.asarray(model_cfg['sigma_product'], dtype = np.float64) # Product variation from global
        # vals_prod_mar_sigma = np.asarray(model_cfg['sigma_product_market'], dtype = np.float64) # Deviations from market x product combinations

        # # Defining Global Effect 
        # mu_global = pm.TruncatedNormal("mu_global", mu = mu_vals, sigma = sigma_vals, lower = 0, dims = "channel")

        # # Defining market deviation from global 
        # sigma_market = pm.Gamma("sigma_market", mu = vals_mar_sigma, sigma = vals_mar_sigma/2, dims = "channel")
        # z_market = pm.Normal("z_market", mu = 0, sigma = 1, dims = ("market", "channel")) # Non-centered hierarichy - prevents strong correlation coupled relationship with global (slow sampling). Defines directional variation from global mean( (+/-)

        # # Defining product deviation from global
        # sigma_product = pm.Gamma("sigma_product", mu = vals_prod_sigma, sigma = vals_prod_sigma/2, dims = ("channel"))
        # z_prodoct = pm.Normal("z_product", mu = 0, sigma = 1, dims = ("product", "channel")) # Non-centered hierarichy - prevents strong correlation coupled relationship with global (slow sampling). Defines directional variation from global mean( (+/-)

        # # Defining product x market specific deviations 
        # sigma_market_product = pm.Gamma("sigma_market_product", mu = vals_prod_mar_sigma, sigma = vals_prod_mar_sigma/2, dims = ("channel")) # Defines magnitude of deviation 
        # z_market_product = pm.Normal("z_market_product", mu = 0, sigma = 1, dims = ("market", "product", "channel")) # Non-centered hierarichy - prevents strong correlation coupled relationship with global (slow sampling). Defines directional variation from global mean( (+/-)

        # # Deterministic variable to define channel product x market specific coefficients - additive deviation
        # beta_mpc = pm.Deterministic(
        #     "beta_mpc",
        #     mu_global[None, None, :] +
        #     sigma_market[None, None, :]* z_market[:,  None,  :] +   
        #     sigma_product[None, None, :]* z_product[None, :,   :] +
        #     sigma_market_product[None, None, :]* z_market_product,      
        #     dims=("market", "product", "channel")
        # )  

        

        # ----------------
        # Transformation priors (Geometric Adstock + Hill Saturation)
        # ----------------

        # ===========================
        # Beta prior for adstock decay
        # ===========================
        # The adstock parameter models *memory* — how long the impact of a media impression/spend persists over time.        
        # The Beta distribution is ideal for adstock decay because:
        # - It is bounded between 0 and 1 - cannot be negative or exceed 100% (i.e 1)        
        
        # Mathematically:
        #     adstock ~ Beta(alpha, beta)
        #     mean = alpha / (alpha + beta) - center value of distribution
        #     concentration (confidence/K) = alpha + beta = higher summed value means more confidence in distribution - narrower 
        #   → k = 6  → fairly loose, weakly-informative
        #   → k = 10 → moderately confident
        #   → k > 20 → very confident and tight around the mean
        # Example:
        # - If you believe media has moderate persistence with decay ≈ 0.4 (and range ~0.2–0.8):
        #     mean = 0.4
        #     alpha = mean * k = 0.4 * 10 = 4
        #     beta  = (1 - mean) * k = 6
        #   → adstock ~ Beta(4, 6)
        #   → centered near 0.4 with mild uncertainty - fairly narrow       
        
        # Converting halflife (50% effect from media spend 8 weeks ago) to decay geometric factor includes:
        # For 8 week half life: then e ** (log(0.5) / 8)  = 0.917 is the decay factor meaning 91.7% of last weeks media effect remains 
        
        # Advantages:
        # - Naturally enforces bounds (0,1) with smooth tails — no truncation causing sampling issues
        # - Easy to interpret and calibrate via mean and distribution concentration
        # - Encodes intuitive beliefs directly about persistence
        # Common disadvantages 
        # - Hard to tune intuitively — (alpha, beta) are not obvious parameters to interpret.
        # - Beta distributions can become numerically unstable near 0 or 1 if parameters are too tight (e.g., alpha,beta < 1).
        # - If decay priors are too vague (alpha + beta < 4), the sampler may explore extreme regions (e.g., adstock ≈ 0 or ≈ 1) → leads to identifiability issues between adstock and saturation (theta).
        # - If too narrow, the model over-constrains media lag behaviour → poor fit and divergences.
        # - Because adstock affects all lagged spend, misspecification propagates through the model, amplifying sampling instability.
        
        # Multimodality - multiple different parameter combinations that fit data equally well due to:
        # - Poorly scaled priors on adstock can interact with HalfNormal noise priors to produce multimodal posteriors. 
        # If adstock prior too tight or vague then short and long memory can correspond to large and small media effects if noise prior too loose to absorb residual error. Causes adstock posterior to be bimodal - chains poorly mix
        # If geometric decay too high: Over smoothing (flatten peaks), over-attribution to media effects, multicollinearity with trend
        # If geometric decay too low: Most media effect happens immediately, sharp peaks and trough remain, under attribute the effect of media spends
        
        # Diagnostic tip:
        # - Always inspect the posterior histogram of adstock — if it piles up at 0 or 1, the prior is too weak or too tight.
        # - Run prior predictive checks to visualize carryover shapes implied by your Beta prior before fitting the full model.
        
        adstock = pm.Beta(
            "adstock",
            alpha=np.asarray(model_cfg["geometric_alpha"], dtype=np.float64),
            beta=np.asarray(model_cfg["geometric_beta"],  dtype=np.float64),
            dims="channel",
        )



        # Other prior distributions that can be used for adstock decay include: 
        # 1) Truncated Normal - although hard boundaries with sampler not good, doesn't capture skewness in distribution unlike beta


        # =======================
        # Hill Function - Saturation Diminishing returns 
        # =======================

        # --------------------------
        # LogNormal prior for THETA
        # --------------------------
        # Theta is half max saturation point: where response from media spend reaches 50% of maximum effect - how quickly saturation occurs
        # The LogNormal distribution is used and defines a variable whose logarithm follows a Normal distribution:
        #     log(theta) ~ Normal(mu, sigma)
        # This means:
        # - On the log scale: uncertainty is symmetric and additive (+)
        # - On the natural scale: theta is strictly positive, right-skewed, and deviation is multiplicative around exp(mu).
        
        # Intuitively:
        # - mu = np.log(x) ensures the prior is specified in log space (so mu and sigma act additively in log space and multiplicative when exponetiated in natural space).
        # - Exponentiation back to natural space enforces positivity and yields multiplicative deviation. The result is a right-skewed prior that is strictly positive 
        
        # Advantages of lognormal include:
        # - The sampler is free to explore in log-space (which is unconstrained) before exponentiate back to natural scale - no instance of sampler hitting a hard wall at zero like truncation (halfnormal or truncated normal)
        # - Introduces multiplicative deviation rather than fixed differences (like normal) - unrealistic
        # - Less divergence risk as HMC/NUTS sampler performs best when parameters are unconstrained - take any real number, which lognormal does then exp back to natural scale 
        # Beware: 
        # - Non-identifiabilty; Theta and slope parameters for media spends are often non-identifbale when both weakly informative (high theta and steep slope explain data as well as low theta and low slope) 
        # - Also can trade off with betas and adstock so need to ensure weakly informative priors (wide priors allow the sampler to explore unrealistic regions and create multimodality) 
        # - Pair plots and no/low mixing of chains can indicate non-identifiabilty 
 
        # i.e. mu = np.log(10000), sigma = np.log(1.2) - indicates natural scale centered at 10000 with approx. 20% deviation at the 68% range
        theta = pm.LogNormal(
            "theta",
            mu=np.log(np.asarray(model_cfg["theta_mu"], dtype=np.float64)),
            sigma=np.log(np.asarray(model_cfg["theta_sigma"], dtype=np.float64)),
            dims="channel",
        )

        # Other prior distributions used for max saturation:
        # 1) 



        # --------------------------
        # Truncated Normal prior for slope parameter
        # ---------------------------
        # Slope controls the steepness of the saturation curve in the hill function - how sharply the response accelerates before flattening 
        # Normal ranges from 1 - 3 with 1 indicating gradual response soft saturation (S-curve), and 3 > sharp steepness (3 is rarely used)
        # A TruncatedNormal prior is used to:
        # - Center the distribution around a realistic mean (mu).
        # - Constrain values to a plausible physiological/economic range (lower, upper).
        # - Enforce positivity and prevent extreme slope shapes that break interpretability.
        # Mathematically:
        #     slope ~ Normal(mu, sigma), truncated to [lower, upper]
        # This means:
        # - The underlying slope follows a Normal distribution, but values below 'lower' or above 'upper' are cut off and renormalized to keep total probability = 1.
        # - Within those bounds, uncertainty is symmetric and additive around 'mu'.
        # - Truncation enforces plausible ranges, preventing the sampler from exploring unrealistic regions (e.g., negative slopes or overly steep curves).
        # Intuitively:
        # - mu: expected slope center (e.g., 1.5 means moderate steep response).
        # - sigma: uncertainty (spread) around the mean in linear space (additive)
        # - lower/upper: physical or domain-informed limits (e.g., 0.5–3) ensuring realism.
        # Example:
        #     mu=1.5, sigma=0.5, lower=0.5, upper=3
        #     → allows 68% of prior mass roughly between [1.0, 2.0], 95% between 0.5 - 2.5. Prior distribution bounded hard at 0.5 and 3.
        # Advantages of TruncatedNormal include:
        # - Enforces physically meaningful parameter bounds without hard rejections during sampling.
        # - Keeps sampling smooth and efficient — HMC/NUTS can explore freely within [lower, upper] - although causes hard bounds that can cause divergences
        # - Encodes domain knowledge directly: slope must be positive, moderate, and not extreme.   
        # Interpretation tip:
        # - Posterior values near the bounds may indicate overly tight limits or data pushing outside prior expectations.
        # - A broad sigma relative to (upper - lower) gives a weakly-informative prior.

        # Beware: 
        # - Non-identifiabilty issues - set weakly informative priors 
        # slope = pm.TruncatedNormal(
        #     "slope",
        #     mu=np.asarray(model_cfg["slope_mu"],    dtype=np.float64),
        #     sigma=np.asarray(model_cfg["slope_sigma"], dtype=np.float64),
        #     lower=np.asarray(model_cfg["slope_lower"], dtype=np.float64),
        #     upper=np.asarray(model_cfg["slope_upper"], dtype=np.float64),
        #     dims="channel",
        # )

        # Other distributions used for slope: 
        # 1) Lognormal - Smoother Sampling and no harsh truncation
        slope = pm.LogNormal("slope", 
                             mu = np.log(model_cfg["slope_mu"]), 
                             sigma = np.log(model_cfg["slope_sigma"]), 
                             dims = "channel")


        # =================================
        # Transforming media spends and getting per observation contributions from media to input into mu
        # =================================
        
        # ---- Per-row coefficients used in contributions: (N, C) ----
        beta_obs = pm.Deterministic(
            "beta_obs",
            beta_mpc[market_idx, prod_idx, :],
            dims=("obs", "channel")
        )  # (N, C)

        # What the above does:
        # Used pytensor integer indexing to select the correct corresponding coefficient from the beta_mpc (m, p, c) array and assign it to the correct row using the mark_idx and prod_idx created which have shapes (n,)
        # The resulting array is shape (n,c) containing the correct coefficients per row for each media channel 


        # Transform media channel spends
        transformed = []
        for j in range(len(channels)):
            ad  = adstock_geometric(X_spend[:, j], adstock[j])
            sat = hill_saturation(ad, theta[j], slope[j])
            transformed.append(sat)
        X_media = at.stack(transformed, axis=1)  # (N, C)

        # - Applies the corresponding geometric decay factor values created by the beta prior - indexes them given the channel dims in the beta prior
        # - Does the same process for hill function transforming the spends
        # - Combines the transformed spends into a 2D tensor (N, C) - each column is the adstocked + saturated transformed spend for a channel

        # ---- Media contributions at the row level ----
        media_contrib_obs_ch = beta_obs * X_media               # (N, C)
        # Multiplying both arrays of shape (N, C) elementwise gives the per-observation, per-channel contributions (in KPI units), i.e. how much each channel adds to the expected outcome in that week.
        
        media_contrib_obs    = media_contrib_obs_ch.sum(axis=1) # (N,) - input into the mu for model
        # Summing across channels collapses the per observational contribution into a single total media contribution per observation (media_contrib_obs), which is then added into the model’s mean structure μ 


        # ------------------------
        # Other media transformations that could be used along with corresponding priors
        # ------------------------
        
        # -------------------------
        # How to know correct transformations have been made: 
        # -------------------------
        # Stages include: 

        # 1) Pre-Model Sanity 
        # - Build adstocked media using different decay factors and cross with KPI - high correlation indicates potentially correct transformations 
        # - Check lagged correlation between media spends and KPI - gives idea of what transformation to use - look at time series plot between KPI and media spends
        # - Plot KPI vs adstocked spends - inspect saturation and if a non-linear relationship exists and where it reaches half saturation 
        # - Align half life adstocks with domain knowledge (i.e. not applying 12-week half life to paid search as unrealistic)
        # - Ensure priors are not too wide as can create unidentifiabilty (explaining the data equally well with different transformations)

        # 2) In Model Behaviour
        # - Prior predictive checks for transformed channel generate realistic contributions - check in prior predictive checks (using var_names arg in prior predictive check function)
        # - If prior + posterior predictive check too smooth or spiky or lagged then could be incorrect transformations - could also be omitted variable bias.  
        # - Check posteriors of transformation parameters - aren't flat, multimodal and chains not mixing (r-hat > 1.01)
        # - Posteriors align with domain knowledge and not too wide
        # - Posteriors not slamed against bounds of priors - prior too restrictive and model trying to escape 
        # - Too wide posterior can inidcate multimodal behaviour 
        # - Good model WAIC or LOO - compare models with different media transformations and prior settings

        # 3) Predictive Validity
        # - PPC fits the observed data well - if way too noisy or smooth this can indicate off transformations 
        # - R**2 is good and model predicts well insample and oos
        # - Response curves (predicted KPI from transformed spends x coefficient) are accurate and realistic (monotonic concave curves)

        # 4) PPC indicators of incorrect transformations
        # - Model predictions typically above or below real KPI
        # - PPC captures trend and magnitude but peaks misaligned in time - model appears delayed or too reactive
        # - Model predictions much smoother than observed KPI - spikes/seasonal peaks muted - too high or low theta in hill function dampens sensitivity
        # - PPC too spikey could indicate too lower adstock decay

        # 5) Residual Analysis
        # - Residuals look random and peaks captured is a good sign that transformations are captured
        # - Model predicts peaks too early or late is indicating adstock too low or high
        # - Residuals mostly positive or negative can indicate saturation transformation wrong (too high - media is underattributed, too low - media is overestimated) - and/or omitted variable bias or incorrect prior settings
        # - Residuals spike after spend - adstock too high


        # ===========================
        # Control Variables
        # ===========================
        # Global controls (shared across all products & markets) - no hierarichal partial pooling
        weather_beta = pm.Normal("beta_weather", mu=model_cfg['weather_mu'], sigma=model_cfg['weather_sigma']) # Scale (0 - 1)
        temp_beta    = pm.Normal("beta_temperature", mu=model_cfg['temp_mu'], sigma=model_cfg['temp_sigma']) # Scale (-1 - 1)
        gdp_beta     = pm.Normal("beta_gdp", mu=model_cfg["gdp_mu"], sigma=model_cfg["gdp_sigma"]) # Scaled using x - mean / SD so average GDP corresponds to 0. Prior mu indicates increase in KPI per 1SD increase in gdp_index 
        

        # -------------------------------
        # Price
        # -------------------------------
        # Price has been logged and scaled using x - x.mean / std meaning priors need to correspond to change in KPI from a 1 SD increase in price (% increase in price as been logged)
        # This is a crossed hierarichal setup: market × product - effect of price per product x market changes
       
        # Global prior: average price elasticity across all brands and markets
        price_mu_global = pm.TruncatedNormal(
            "price_mu_global",
            mu=model_cfg["price_mu"],
            sigma=model_cfg["price_sigma"],
            upper=0.0,
        ) 
        
        # Deviation scales - variation in price effect across markets and products 
        sigma_price_market = pm.HalfNormal("sigma_price_market", sigma=model_cfg['price_sigma_mar']) # Deviation in price effect across markets from global 
        sigma_price_product = pm.HalfNormal("sigma_price_product", sigma=model_cfg['price_sigma_prod']) # Deviation in price effect across products from global 
        sigma_price_mp = pm.HalfNormal("sigma_price_market_product", sigma=model_cfg['price_sigma_mar_prod']) # Deviation in individual product x market pairs from global
        
        # Latent z's
        z_price_market = pm.Normal("z_price_market", 0, 1, dims="market")
        z_price_product = pm.Normal("z_price_product", 0, 1, dims="product")
        z_price_mp = pm.Normal("z_price_mp", 0, 1, dims=("market", "product"))
        
        # Build full crossed coefficient (market × product)
        beta_price_mp = pm.Deterministic(
            "beta_price_mp",
            price_mu_global
            + sigma_price_market * z_price_market[:, None]
            + sigma_price_product * z_price_product[None, :]
            + sigma_price_mp * z_price_mp,
            dims=("market", "product")
        )
        
        # Indexing by row’s market/product to assign coefficients per obersveration in the dataset
        price_beta_obs = pm.Deterministic(
            "beta_price_obs",
            beta_price_mp[market_idx, prod_idx],
            dims="obs"
        )

        # ----------------------------------
        # Feature Display - indicates proportion of share of products displayed in shops - promotional shelves 
        # ----------------------------------
        # Data is scaled to between 0-1 - indicates proportion
        # This is a crossed hierarichy setup: market × product - effect per product x market combination - Feature display might have different effect for different products in different markets
        
        # Global prior for Feature & Display effect 
        feature_mu_global = pm.Normal("feature_mu_global", mu=model_cfg['feature_mu'], sigma=model_cfg['feature_sigma'])
        
        # Devations in feature & display effect across markets and products
        sigma_feature_market = pm.HalfNormal("sigma_feature_market", sigma=model_cfg['feature_sigma_mar'])
        sigma_feature_product = pm.HalfNormal("sigma_feature_product", sigma=model_cfg['feature_sigma_prod'])
        sigma_feature_mp = pm.HalfNormal("sigma_feature_market_product", sigma=model_cfg['feature_sigma_mar_prod'])

        # Non-centered deviations
        z_feature_market = pm.Normal("z_feature_market", 0, 1, dims="market")
        z_feature_product = pm.Normal("z_feature_product", 0, 1, dims="product")
        z_feature_mp = pm.Normal("z_feature_mp", 0, 1, dims=("market", "product"))

        # Creater pytensor array to set product x market specific coefficients
        beta_feature_mp = pm.Deterministic(
            "beta_feature_mp",
            feature_mu_global
            + sigma_feature_market * z_feature_market[:, None]
            + sigma_feature_product * z_feature_product[None, :]
            + sigma_feature_mp * z_feature_mp,
            dims=("market", "product")
        )

        # Indexing coefficients across observations
        feature_beta_obs = pm.Deterministic(
            "beta_feature_obs",
            beta_feature_mp[market_idx, prod_idx],
            dims="obs"
        )

        # ---------------------------------
        # Competitor Spend
        # ---------------------------------
        # Data has been divided by 1000 so priors indicate the increase in KPI per £1k increase in competitor spend. 
        # Competitor spend global effect - across all markets
        comp_mu = pm.Normal("comp_mu_global", mu=model_cfg['comp_mu'], sigma=model_cfg['comp_sigma'])

        # Deviation in competitor spend (/£1k) effect
        sigma_comp = pm.HalfNormal("sigma_comp", sigma=model_cfg['comp_sigma_mar'])
        
        # Non-centered hierarhcy and directional element
        z_comp = pm.Normal("z_comp", 0, 1, dims="market")

        # Competitor spend /£1k effect betas per market
        competitor_beta = pm.Deterministic(
            "beta_competitor",
            comp_mu + sigma_comp * z_comp,
            dims="market")

        # Per observation beta - this gets multiplied by competitor spend below
        competitor_obs = pm.Deterministic("Competitor_obs", 
                                          competitor_beta[market_idx], 
                                          dims = "obs")

        
        # Control contributions
        ctrl_term = (
            price_beta_obs * price              # product x market -level
            + feature_beta_obs * feature        # product x market -level
            + competitor_obs * competitor       # market-level
            + weather_beta * weather            # global
            + temp_beta * temperature           # global
            + gdp_beta * gdp                    # global
        )

        
        # ===================
        # Baseline
        # ===================
        # Baseline is the long-term natural level of the KPI that exists without media or external influences on KPI
        # Represents organic demand or brand equity floor in each market x product combination

        # --------------------------
        # 1) No pooling baseline 
        # --------------------------
        # Individual baseline per market x product are learnt individually. 
        # Each baseline term represents product x market natural KPI level - baseline can vary dramtically across different products in different markets
        # Only use if adequate rich data to learn individual baseline levels accurately as no pooling - if noisy weak signal then overfitting and uncertain baselines generated. 

        # --- Setting baseline priors ---
        # baseline = pm.Normal("beta_baseline",
        #              mu=model_cfg["baseline_mu"],
        #              sigma=model_cfg["baseline_sigma"],
        #              dims=("market","product"))

        # --- Setting correct baseline for corresponding rows in the dataset - input into model ---
        # baseline_obs = pm.Deterministic("baseline_obs", baseline[market_idx, product_idx], dims = "obs") 
        

        # -----------------------------
        # 2) Partial Pooling Hierarichal baseline
        # -----------------------------
        # Market x Products share global baseline information - baselines allowed to vary across products x markets from global - crossed hierarichy
        # Improves stabilty and shrinkage when data is weak and uninformative - baselines gets pooled

        # --- Extracting prior settings from config file ---
        baseline_cfg = model_cfg["baseline"]
        mu_baseline_vals       = np.asarray(baseline_cfg["baseline_mu_global"])
        mu_baseline_sigma_vals = np.asarray(baseline_cfg["baseline_sigma"])
        sigma_baseline_mar_val = np.asarray(baseline_cfg["baseline_mar_sigma"])
        sigma_baseline_prod_val = np.asarray(baseline_cfg["baseline_prod_sigma"])
        sigma_baseline_mar_prod_val = np.asarray(baseline_cfg["baseline_prod_mar_sigma"])

        # --- Global baseline (positive scale with multiplicative deviation) ---
        mu_baseline = pm.LogNormal(
            "mu_baseline",
            mu=np.log(mu_baseline_vals),
            sigma=np.log(mu_baseline_sigma_vals)
        )
        
        # --- Market baseline deviations from global ---
        sigma_baseline_mar = pm.HalfNormal("sigma_baseline_mar", sigma=sigma_baseline_mar_val)
        z_baseline_mar = pm.Normal("z_baseline_mar", 0, 1, dims="market")
        
        # --- Product baseline deviations from global ---
        sigma_baseline_prod = pm.HalfNormal("sigma_baseline_prod", sigma=sigma_baseline_prod_val)
        z_baseline_prod = pm.Normal("z_baseline_prod", 0, 1, dims="product")
        
        # --- Market × product deviations from global ---
        sigma_baseline_mar_prod = pm.HalfNormal("sigma_baseline_mar_prod", sigma=sigma_baseline_mar_prod_val)
        z_baseline_mar_prod = pm.Normal("z_baseline_mar_prod", 0, 1, dims=("market", "product"))
        
        # --- Combined baseline for each market × product pair ---
        baseline_beta_mpc = pm.Deterministic(
            "baseline_beta_mpc",
            mu_baseline
            + sigma_baseline_mar * z_baseline_mar[:, None]         # market effect broadcast over products
            + sigma_baseline_prod * z_baseline_prod[None, :]       # product effect broadcast over markets
            + sigma_baseline_mar_prod * z_baseline_mar_prod,       # market×product interaction deviation from baseline
            dims=("market", "product")
        )

        # --- Per-row baseline assignment - input into the model ---
        baseline_obs = pm.Deterministic("baseline_obs", baseline_beta_mpc[market_idx, prod_idx])


        # -------------------
        # 3) Baseline Structural Change
        # -------------------
        # A permanent, known change in the KPI’s natural level or trend slope — typically from major business or external events: Regulatory or global events (COVID), rebranding, Competitor Exit or Entry
        # Beware: May not be needed if using polynominal trends or time-varying baseline as can absorb changes 
        # Can introduce a hierarichal variant allowing different structural break effects across markets 

        # -------- 1) Dummy variable for known structural break ----------
        # Adds constant (structural_shift_term) to original baseline after breakpoint to make new baseline
        if baseline_cfg.get("structural_change", {}).get("enabled", False):
            scfg = baseline_cfg["structural_change"]
            break_at = scfg["break_at"]
            level_shift_arr = (panel["trend"].to_numpy() >= break_at).astype(float)
            level_shift = pm.Data("level_shift", level_shift_arr, dims="obs")

            # Setting structual change prior - how much natural KPI level changes every week 
            baseline_shift = pm.Normal(
                "baseline_shift",
                mu=scfg["mu"],
                sigma=scfg["sigma"]
            )
            structural_shift_term = baseline_shift * level_shift
        else:
            structural_shift_term = at.constant(0.0)
        
        # -------- 2) Piecewise Baseline Slope ------------------------
        # Change in baseline slope (not static) - rate of organic/decay changes after known point
        # Continuous with a bend at a breakpoint - no step change unlike dummy variable method above
        # Two linear segemnts with different slopes (B_pre, B_post) joined smoothly

        # This piecewise structural change in baseline is global - effects products x markets the same - added to baseline obs which is product x market specific and generated above in hierarichal setup
        # This structural change is normally in line with global economic slowdown - could extend to partial pooling over markets/products to get different changes in baseline. 
        if baseline_cfg.get("piecewise", {}).get("enabled", False):
            pcfg = baseline_cfg["piecewise"]
            break_at = pcfg["break_at"]
            trend_arr = panel["trend"].to_numpy().astype(float)
        
            pre_mask = pm.Data("pre_mask", (trend_arr < break_at).astype(float), dims="obs") # Subset pre structural break point data
            post_mask = pm.Data("post_mask", (trend_arr >= break_at).astype(float), dims="obs") # Subset post structural break point data
            
            trend_centered_post = trend_arr - break_at # Center the breakpoint at 0
        
            beta_pre = pm.Normal("beta_pre", mu=pcfg["pre_mu"], sigma=pcfg["pre_sigma"]) # Pre structural break baseline trend
            beta_post = pm.Normal("beta_post", mu=pcfg["post_mu"], sigma=pcfg["post_sigma"]) # Post structural break baseline trend

            # Ensures post break line continues from the same values as the pre-break line but follows different slope (beta_pre * break_at) - controls change in baseline growth
            baseline_piecewise_term = (
                beta_pre * (trend_arr * pre_mask)
                + (beta_pre * break_at + beta_post * trend_centered_post) * post_mask
            )
        else:
            baseline_piecewise_term = at.constant(0.0)


        # -------------------
        # 3) Time-Varying Baseline (stochastic)
        # -------------------
        # Adds a latent evolving process that lets baseline drift over time - capture flexible/organic fluctuations that deterministic trend terms cannot

        tv_cfg = baseline_cfg.get("time_varying_baseline", {}) 
        tv_type = tv_cfg.get("type", "none").lower()
        
        if tv_type == "gaussian_rw":
            # --- Gaussian Random Walk (GRW) ---
            # A Gaussian Random Walk is a stochastic process where each period’s value equals the previous value plus a random Gaussian shock.
            # Used to model residual baseline fluctuations that are not explained by deterministic terms (baseline_obs + structural_shift_term + baseline_piecewise_term)
            # Use when baseline wanders slowly upwards/downwards from unmodelled shifts in baseline, mild local flucutations, temporary drifts due to untracked effects (competitor promotions etc...) - suspect nonlinear drift and haven't imposed polynominal
            # It’s useful when the KPI’s underlying level shows slow, unpredictable movements upward or downward that linear or polynomial trends can’t capture, such as gradual shifts in organic demand, brand equity, or unmodelled macro effects.
            # Avoid overlapping with strong polynominal trends - these already capture slow drift so might overfit 
            # If GRW sigma too high allows baseline to wiggle too much and may abosrb signal of media and/or trend     
            # Cumulative change = sigma * sqrt(no. of modelling weeks). So sigma = 800 for 154 modelled weeks = 10000 units attributed to gaussian RW baseline.  

            # Defining the wiggle around the KPI (week-to-week noise)
            sigma_rw_val = float(tv_cfg.get("sigma", 800.0)) # If expect 10000 wiggle around baseline over 3 years (154 weeks) then 10000/sqrt(154) = 806 - this is the sigma. 
            sigma_rw = pm.HalfNormal("sigma_baseline_rw", sigma_rw_val)
        
            # Latent state evolving week to week
            baseline_rw = pm.GaussianRandomWalk(
                "baseline_rw",
                sigma=sigma_rw,
                dims="obs"
            )
        
            tv_baseline_term = baseline_rw  # added to baseline_total downstream
        
        elif tv_type == "ar1":
            # --- AR(1) process ---
            # Use when KPI baseline fluctuates around mean level with short-term persistence.
            # Captures temporary deviations that revert back to baseline over time (not cumulative drift).
            # If rho = 1 then behaves like GRW, if rho < 1 temporary deviations not slow drift 
            # Use when KPI shows short-lived deviations (short economic shocks or stock issues)
            # Beware using with seasonality or dummy variables as these might already capture these deviations
            rho_val = float(tv_cfg.get("rho", 0.9))         # persistence: 1 → slow drift; <1 → mean-reverting
            sigma_ar_val = float(tv_cfg.get("sigma", 800.0)) # volatility of shocks
            sigma_ar = pm.HalfNormal("sigma_baseline_ar1", sigma_ar_val)
        
            baseline_ar1 = pm.AR(
                "baseline_ar1",
                rho=rho_val,
                sigma=sigma_ar,
                dims="obs"
            )
        
            tv_baseline_term = baseline_ar1  # added to baseline_total downstream
        
        elif tv_type == "none":
            tv_baseline_term = at.constant(0.0)  # no stochastic time-varying component
        
        else:
            raise ValueError(f"Unknown time_varying_baseline type: {tv_type}")


        # Combining baseline components to input into mu
        baseline_total = (
            baseline_obs                  # Static hierarchical baseline (market x product specific baseline)
            + structural_shift_term       # Permanent level shift (dummy structural change)
            + baseline_piecewise_term     # Slope change (piecewise baseline drift)
            + tv_baseline_term            # Stochastic baseline variation (RW or AR(1))
        )

        # --------------------
        # Trend 
        # --------------------
        
        # Deterministic trend (steady/smooth predictable drift/growth/decay) 
        trend_det_term = 0.0 # A)

        trend_cfg = model_cfg.get("trend", {})
        det_cfg   = (trend_cfg.get("deterministic") or {"type": "linear"}) # default to linear trend term (unidirectional monotonic trend over time)

        # ---- (A) Deterministic Trend ----
        # Uses the pre-engineered `trend` column if present in data (e.g., week index or z-scored time)
        if trend_arr is not None:
            trend = trend_arr
            det_type = det_cfg.get("type", "linear").lower() # extracting trend type (linear, polynomial, piecewise) 

            # Handle scalar or list mu/sigma gracefully
            det_mu_raw = det_cfg.get("mu", 0.0) # extracting mu for implementing trend prior
            det_sigma_raw = det_cfg.get("sigma", 1.0) # extracting SD for trend prior
            
            # Convert to float only if scalar
            if isinstance(det_mu_raw, (int, float)):
                det_mu = float(det_mu_raw)
            else:
                det_mu = np.asarray(det_mu_raw, dtype=np.float64)
            
            if isinstance(det_sigma_raw, (int, float)):
                det_sigma = float(det_sigma_raw)
            else:
                det_sigma = np.asarray(det_sigma_raw, dtype=np.float64)

            if det_type == "linear":
                # Linear deterministic trend - monotonic increase or decrease in KPI overtime with same slope
                beta_trend = pm.Normal("beta_trend", det_mu, det_sigma)
                trend_det_term = beta_trend * trend 

            elif det_type == "polynomial":  
                # Used when the trend accelerates or deccelerates over time - isn't linear 
                # Polynom allows curved deterministic trend and different slope in trend overtime
                # If polynominal set to 2 then have a linear and curvature terms  
                # T**2 controls curvature and gets very large so careful setting prior 
                degree = int(det_cfg.get("degree", 2))
                mmm.add_coord("trend_degree", list(range(1, degree + 1)))  # coordinate labels for [1, 2, ...]
                trend_powers = [trend ** d for d in range(1, degree + 1)]      # Build polynomial trend terms using time index/trend array            
                det_mu = np.asarray(det_mu, dtype=np.float64)          # [expected weekly increase from linear trend, increase in curvature] - remember t**2 gets very large so set priors carefully
                det_sigma = np.asarray(det_sigma, dtype=np.float64)    # uncertainty for each coefficient for linear trend and polynominal
            
                beta_trend_poly = pm.Normal(
                    "beta_trend_poly",
                    mu=det_mu,
                    sigma=det_sigma,
                    dims="trend_degree"
                ) # Creates vector of prior draws for each polynominal coefficient. Each controls how strongly each power of time affects KPI trend (t & t**2) 

                # --- Construct full trend term ---
                trend_det_term = at.stack(trend_powers, axis=1) @ beta_trend_poly # at.stack creates 2D matrix with each column per power. The @ performs matrix multiplication between matrix of trend powers and vector of corresponding coefficients
                
            elif det_type == "piecewise":
                # Piecewise: different trend before/after a breakpoint in the time index
                # Different slopes for different time intervals but still forms one continuous curve - no sudden jumps but a bend at a known time point that introduces a new rate in trend 
                # Would use when underlying structural changes in trend of KPI. Can be caused by: Covid effecting demand levels, New pricing model causes sustained increased demand, baseline awareness rises from campaign 
                break_at = det_cfg.get("break_at", None) # time interger specified in config
                if break_at is None:
                    raise ValueError("trend.deterministic.break_at must be set for piecewise.")
                    
                # Creating data containers for before and after breakpoint
                pre_mask_arr  = (panel["trend"].to_numpy() <  break_at)
                post_mask_arr = (panel["trend"].to_numpy() >= break_at)
                pre_mask  = pm.Data("trend_pre_mask",  pre_mask_arr,  dims="obs")
                post_mask = pm.Data("trend_post_mask", post_mask_arr, dims="obs")

                # Two seperate slope coefficients - one pre-break and one post break
                beta_trend_pre  = pm.Normal("beta_trend_pre",  0.0, det_sigma)
                beta_trend_post = pm.Normal("beta_trend_post", 0.0, det_sigma)

                trend_centered_post = trend - break_at # Trend continuous across break as trend_centered_post = 0 at break_point, then + 1, +2 etc... thereafter so post-slope starts from 0 at break and increases afterwards with continuous join
                trend_det_term = beta_trend_pre * (trend * pre_mask) + beta_trend_post * (trend_centered_post * post_mask)

            elif det_type == "none":
                trend_det_term = at.constant(0.0)

            else:
                raise ValueError(f"Unknown deterministic trend type: {det_type}")


        # -----------------
        # Seasonality - partial pooling using market
        # -----------------
        # Fourier Term Priors
        # Used to capture repeating patterns in KPI over a fixed period - weekly pattern within a year, holiday spikes, cold/flu season
        # Implement smooth periodic basis functions that repeat over a known period - capture smooth patterns/trends
        # Higher harmonics capture more frequent patterns within the time specified (i.e. 52 weeks for annual seasonality) 
        # Better than monthly dummies which overfit and cannot capture smooth periodic structure (also increase dimensionality) 
        
        # Fourier terms approximate repeating annual seasonality using sin/cosine waves.
        # 2 harmonics = 4 columns: [sin1, cos1, sin2, cos2].
        # k=1 captures smooth yearly pattern; k=2 adds sub-annual patterns (2 peaks/troughs).
        
        # Priors:
        # - mu_season ~ Normal(0, base_sigma)  → assume no seasonality effect on average.
        # - sigma_season decays with harmonic index (5000, 2500) → shrink higher harmonics.
        # - beta_season(product, term) → each product has its own seasonal curve
        #   while partially pooled around the global mean/scale.

        season_cfg = model_cfg.get("seasonality", {})
        season_enabled = season_cfg.get("enabled", True)
        n_harmonics = int(season_cfg.get("n_harmonics", 1))
        expected_amp = float(season_cfg.get("expected_amplitude", 10000.0))  # Expected single peak amplitude from seasonality in KPI units
        
        base_sigma = expected_amp / np.sqrt(2) 
        
        # Shrink effect for higher harmonic terms
        harmonic_ids = np.repeat(np.arange(1, n_harmonics + 1), 2)
        sigma_vec = base_sigma / harmonic_ids 

        # Setting seasonality variation for across markets and products
        market_sigma = float(season_cfg.get("market_sigma", base_sigma / 2))
        product_sigma = float(season_cfg.get("product_sigma", base_sigma / 3))
        market_product_sigma = float(season_cfg.get("market_product_sigma", base_sigma/4))
        
        season_term = 0.0
        if season_enabled and X_season_arr is not None and X_season_arr.shape[1] > 0:
        
           # --- Global mean per Fourier term ---
            mu_season = pm.Normal(
                "mu_season",
                mu=0,
                sigma=sigma_vec,
                dims="season_term"
            ) # Global seasonal term effects (6,)
        
            # --- Market-level deviations ---
            sigma_market = pm.HalfNormal("sigma_season_market", sigma=market_sigma, dims = "season_term") # Market deviations from global for seasonal terms 
            z_market = pm.Normal("z_season_market", mu=0, sigma=1, dims=("market", "season_term")) # Non-centered directional prior
        
            # --- Product-level deviations (within market) ---
            sigma_product = pm.HalfNormal("sigma_season_product", sigma=product_sigma, dims = "season_term") # Product deviation from global for seasonal terms
            z_product = pm.Normal("z_season_product", mu=0, sigma=1, dims=("product", "season_term")) # Non-centered directional prior

            # --- Product x Market specific deviations (products seasonality differing per market) --- 
            sigma_product_market = pm.HalfNormal('sigma_season_market_product', sigma = market_product_sigma, dims = "season_term") # Market x product specific deviations from global
            z_product_market = pm.Normal("z_season_product_market", mu=0, sigma=1, dims=("market", "product", "season_term")) # Non-centered directional prior

            # Defining market x product specific coefficients
            beta_season = pm.Deterministic(
                "beta_season",
                mu_season[None, None, :] + 
                sigma_market * z_market[:,None,:] + 
                sigma_product * z_product[None, :, :] + 
                sigma_product_market * z_product_market,
                dims=("market", "product", "season_term")
            )
        
            # --- Combine Fourier basis with coefficients ---
            season_term = at.sum(
                X_season * beta_season[market_idx, prod_idx, :],
                axis=1
            )
        
        else:
            season_term = at.zeros_like(trend)  # or at.constant(0.0)


        # ------------------
        # Dummy / Special-Event Variables (if configured)
        # ------------------
        dummy_cfg = model_cfg.get("dummies", {})
        dummy_terms = 0.0  # will accumulate all dummy contributions
        
        for name, spec in dummy_cfg.items():
            if not spec.get("enabled", False):
                continue  # skip disabled dummies
        
            # Ensure dummy exists in panel
            if name not in panel.columns:
                print(f"[WARNING] Dummy variable '{name}' not found in panel; skipping.")
                continue
        
            # Load as data container
            dummy_arr = panel[name].to_numpy(dtype=np.float64)
            dummy_data = pm.Data(name, dummy_arr, dims="obs")
        
            # Priors from config
            mu = float(spec.get("mu", 0.0))
            sigma = float(spec.get("sigma", 1000.0))
            lower = spec.get("lower", None)
            upper = spec.get("upper", None)
        
            # Build prior dynamically
            if lower is not None or upper is not None:
                beta = pm.TruncatedNormal(
                    f"beta_{name}",
                    mu=mu,
                    sigma=sigma,
                    lower=lower,
                    upper=upper,
                    dims="product",
                )
            else:
                beta = pm.Normal(
                    f"beta_{name}",
                    mu=mu,
                    sigma=sigma,
                    dims="product",
                )
        
            # Add contribution term
            dummy_terms += beta[prod_idx] * dummy_data

        # -------------------
        # Noise/sigma
        # -------------------
        # Residual scale - measures how much of the KPI's variation is unexplained by the model (media, controls, seasonality) 
        # Captures random deviations between mu and actual KPI
        # Need to ensure matches with likleihood: if normal or student T then additive (constant variance), if gamma or lognormal then multiplicative (variance scales with mean)
        # Check in EDA how variable data is (using SD). This can be used to set sigma. 


        # Why halfnormal: 
        # - Ensures positive noise - noise cannot be negative
        # - Allows values close to 0/small noise but allows larger
        # - interpretable: most prior mass below 2 x scale

        sigma = pm.HalfNormal("sigma", sigma=model_cfg["noise_sigma"])

        # Signs noise prior incorrectly specified:
        # Too tight: 
        # - Posterior hugs prior edges 
        # - Model acts overconfident (very narrow credible intervals in PPC) 
        # - Divergences/poor sampling
        # - PPC not capturing variability of observed data
        # - Model coefficients and other variables inflated - picking up attribution that is noise
        # - Posteriors unrealistically narrow

        # Too Wide: 
        # - Posteior of noise extremely wide after sampling
        # - PPC overdispersed - credible intervals too wide in PPC (histogram and time-series)
        # - Credible intervals on posteriors too wide - signal drowned by unbounded residual variance


        # ----------------
        # Mean structure 
        # ----------------
        mu = (
            baseline_total
            + trend_det_term 
            + media_contrib_obs
            + ctrl_term
            + season_term
            + dummy_terms
        )

        ## NOTE: If the model's mean structure dominates the likelihood shape (captures majority of variation in KPI) then only small residual part come from likelihood. 
        ## This means model overfits, and the properities like generating predicted skewed KPI disappears - PPC may look more gaussian. 
        
        # =================
        # Deterministic Nodes 
        # =================
        # Deterministics are computed variables - not random themselves but a function of other random variables (from priors)
        # Allow derived quantities to be tracked like media contributions, ROIs etc... - They can be observed in the posterior group of the idata object
        # Prevent manual calculation after building the MMM model and sampling
        # Allow combined effect across hierarchies to be captured - like beta_mpc
        # Doesn't introduce extra sampling - computed from existing draws 

        # ------------------ 
        # Media contributions
        # ------------------
        # Per observational contribution per media channel
        pm.Deterministic(
            "contrib_media_ch",
            media_contrib_obs_ch, # Comes from multiplying beta_mpc * X_spends
            dims=("obs", "channel"),
        )

        # Per Obs Contribution per channel
        for i, channel in enumerate(channels):
            pm.Deterministic(
                f"media_contri_{channel}",
                media_contrib_obs_ch[:, i],
                dims="obs"
            )

        # Total Media Contribution (across channels) over time 
        pm.Deterministic(
            "contrib_media_total",
            media_contrib_obs_ch.sum(axis=1), # Sum all channels together
            dims="obs",
        )

        # Total Channel Contribution 
        pm.Deterministic("total_contrib_channel", 
                         media_contrib_obs_ch.sum(axis = 0), # Total media contribution per channel
                        dims = "channel", )

        # Total ROI per channel
        pm.Deterministic(
            "roi_channel",
            media_contrib_obs_ch.sum(axis=0) / (X_spend.sum(axis=0) + 1e-8),
            dims="channel",
        )

    
        # Total ROI per channel over time
        pm.Deterministic("roi_channel_time", 
                         media_contrib_obs_ch / (X_spend + 1e-8), 
                         dims = ("obs", "channel"))

        # Share of total media impact each channel explains
        pm.Deterministic(
            "share_contrib_channel",
            media_contrib_obs_ch.sum(axis=0) / media_contrib_obs_ch.sum(),
            dims="channel",
        )


        # ------------------ 
        # Baseline  
        # ------------------

        # Static baseline contribution
        pm.Deterministic("contrib_baseline", baseline_obs, dims="obs")

        
        # # Check if any term has a dimensionality attribute (i.e. not scalar) or is nonzero symbolic
        # pm.Deterministic(
        #         "contrib_baseline_drift",
        #         structural_shift_term + baseline_piecewise_term + tv_baseline_term,
        #         dims="obs",
        # ) # Structural and drift baseline contributions


        # -------------------
        # Trend
        # -------------------

        # Total trend contribution
        pm.Deterministic("contrib_trend", trend_det_term, dims="obs")

        # --------------------
        # Seasonality 
        # --------------------
        # Seasonality contribution
        pm.Deterministic("season_amplitude", season_term, dims="obs")

        # -----------------
        # Controls 
        # -----------------
        pm.Deterministic("contrib_controls_total", ctrl_term, dims="obs")
        pm.Deterministic("contrib_price", price_beta_obs * price, dims="obs")        
        pm.Deterministic("contrib_feature", feature_beta_obs * feature, dims="obs")
        pm.Deterministic("contrib_competition", competitor_beta[market_idx] * competitor, dims="obs")
        pm.Deterministic("contrib_weather", weather_beta * weather, dims="obs")
        pm.Deterministic("contrib_temperature", temp_beta * temperature, dims="obs")
        pm.Deterministic("contrib_gdp", gdp_beta * gdp, dims="obs")
       
        # ------------------
        # Dummies / Special events
        # ------------------
        pm.Deterministic("contrib_dummies", dummy_terms, dims="obs")
        
        # -----------------
        # Total mean prediction 
        # -----------------
        pm.Deterministic("mu", mu, dims="obs")
        
        # ------------------
        # Prediction for OOS / PPC
        # ------------------
        # pm.Deterministic("y_hat", mu, dims="obs")

        
        # ----------------
        # Likelihood 
        # ----------------
        # -------------------------------
        # Normal Likelihood 
        # -------------------------------
        ## Use when the KPI is continuous, roughly symmetric around a central mean, 
        ## Deviations (residuals) can be both positive and negative with approximately constant variance
        ## The Normal likelihood assumes *additive, homoscedastic noise*: meaning magnitude of residual variation is roughly the same across all predicted KPI levels.
        ## This is the most common baseline likelihood for continuous outcomes in MMMs 
        ## Use when residuals are well-behaved and not strongly skewed or heavy-tailed.
        
        ## Diagnostic checks before use:
        ## - Histogram of KPI looks symmetric (bell-shaped) around its mean.
        ## - Residuals from a linear model look roughly Gaussian (QQ plot follows 45° line).
        ## - Variance of residuals is roughly constant across fitted KPI values (mostly homoscedasticity).
        ## - No systematic skewness/long tails or fat tails in residuals.
        
        ## Parameters:
        ## - mu     → expected KPI mean per observation (from your model structure).
        ## - sigma  → standard deviation of the residuals (noise around the mean).
        
        # ## Normal Likelihood:
        # y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_data, dims="obs")

        ## Interpretation:
        ## - Posterior mean of coefficients and contributions are in *additive raw KPI units* 
        ##   (e.g., a coefficient of 500 means a 1-unit increase in a predictor adds 500 KPI units on average).
        ## - Sigma (σ) represents the typical absolute deviation between observed and predicted KPI.
        
        ## Use this when:
        ## - KPI is continuous and roughly symmetric.
        ## - Residual variance is stable across fitted values (no heteroscedasticity).
        ## - No major outliers or long-tailed residuals.
        
        ## Don’t use when:
        ## - KPI is strictly positive and right-skewed (consider Gamma or LogNormal).
        ## - KPI residuals show long tails or large outliers (consider Student-T).
        ## - KPI variance grows with the mean (heteroscedastic — consider Gamma).
        ## - KPI has many zeros or non-Gaussian structure (use zero-inflated or alternative likelihood).
        
        ## Consequences of misuse:
        ## - If residuals are non-Gaussian (skewed or heavy-tailed), the model can:
        ##    • underestimate uncertainty (too narrow credible intervals), or  
        ##    • inflate noise σ and distort other parameter posteriors.  
        ## - Leads to biased predictions and misleading attribution of effects.

        
        
        # -------------------------------
        # Student T Likelihood 
        # -------------------------------
        ## Use when KPI is continuous, roughly symmetric around a mean (can take positive or negative deviations), 
        ## but shows *occasional outliers* or heavy tails that violate the Normal assumption.
        
        ## Conceptually similar to a Normal distribution but with heavier tails, 
        ## meaning extreme deviations have higher probability and therefore less influence on the posterior mean.  
        ## This makes the model more robust to noise spikes, data quality issues, or outlier weeks.
        
        ## Diagnostic checks before use:
        ## - Histogram of KPI looks roughly symmetric but with occasional extreme highs/lows.
        ## - Residuals from a Normal model show long tails or deviate sharply at ends in a QQ plot.
        ## - Posterior predictive checks under Normal likelihood underpredict extremes (tails too thin).
        ## - Noise parameter (σ) inflates unrealistically large under Normal likelihood.
        
        ## Parameters:
        ## - mu     → expected KPI mean per observation (same interpretation as Normal).
        ## - sigma  → residual standard deviation (scale of noise).
        ## - nu     → degrees of freedom (tail thickness; controls how heavy the tails are).
        ##       • nu → ∞ : becomes equivalent to Normal
        ##       • nu ≈ 10 : moderate tails, handles moderate outliers
        ##       • nu ≈ 3–5 : strong tails, robust to extreme outliers
        
        ## Implementation:
        ## - Common to use Exponential(1/10) prior for nu_raw and shift by +2 to keep nu > 2 
        ##   (ensures finite variance but allows heavy tails when needed).
        ## - Smaller nu → heavier tails but slower convergence; too small (≈2–3) can increase divergences.
        
        # ## Student T Likelihood:
        # nu_raw = pm.Exponential("nu_raw", 1.0 / 10.0)  # mean = 10
        # nu = pm.Deterministic("nu", nu_raw + 2.0)      # ensures nu > 2 (finite variance)
        # y_obs = pm.StudentT("y_obs", mu=mu, sigma=sigma, nu=nu, observed=y_data)


        ## Interpretation:
        ## - Contributions and coefficients remain *additive in raw KPI units*, same as with Normal.
        ## - Posterior credible intervals are wider in the tails but more stable overall.
        ## - Outlier observations have less pull on the posterior mean (more robust inference).
        
        ## Use this over Normal when:
        ## - KPI residuals show occasional extreme deviations or noise bursts.
        ## - Normal PPC underestimates tail behaviour.
        ## - You want robustness without transforming KPI scale.
        
        ## Don't use when:
        ## - KPI is strictly positive and right-skewed (use Gamma or LogNormal instead).
        ## - KPI residuals are symmetric and light-tailed (Normal is simpler and faster).
        ## - KPI has structural zeros (use zero-inflated models instead).


        # -------------------------------
        # Gamma Likelihood 
        # -------------------------------
        ## Use when KPI is strictly positive, continuous, and right-skewed 
        ## Appropriate when variance increases proportionally with the mean (heteroscedasticity) as the Gamma distribution models relative—not absolute—noise. 
        ## Diagnostic checks before use:
        ## - Histogram of KPI shows strong right skew (long positive tail).
        ## - Scatterplot of mean vs variance across groups shows - as average KPI value (mean) increases across groups (e.g., markets, products, or time), the spread or variability (variance) of those values also increases.
        ## - Residual plots from a Normal model show increasing spread at higher fitted values.    
        ## Interpretation:
        ## - Effects and contributions remain in *additive raw KPI units* (same interpretation as Normal).
        ## - Only the likelihood (noise model) changes to multiplicative — the mean structure is identical.   
        ## Technical notes:
        ## - `mu` is the expected KPI mean per observation.
        ## - `sigma` controls the *coefficient of variation* (relative dispersion): 
        ##     e.g. sigma = 0.2 ⇒ typical variability of ±20% around the mean value.
        ##     IF sigma is small, implied skew is small
        
        ## Implementation details:
        ## - Clip mu to enforce positivity (Gamma support y>0): 
        ##     ensures model predictions never fall below 0, avoiding numerical errors.
        ## - Must also ensure observed y values are > 0:
        ##     for rare zeros, clip slightly upward (e.g. 1e-3);
        ##     for frequent zeros, use a zero-inflated Gamma mixture instead.

        ## BEWARE: The gamma distribution is right skewed only when values are small. As KPI mean increases the distribution becomes symmetric. 

        
        # ## Gamma distribution
        y_data = np.clip(y_obs_arr, 1e-3, None) # Making KPI data non-zero
        y_obs = pm.Gamma(
            "y_obs",
            mu=at.clip(mu, 1e-3, np.inf),
            sigma=sigma,  # sigma ≈ relative SD (0.1–0.5 typical)
            observed=y_data,
        )
        
        ## Don't use when:
        ## - KPI can be zero or negative frequently (consider Zero-Inflated Gamma or LogNormal).
        ## - KPI is symmetric around a mean (use Normal or Student-T).
        ## - KPI residuals have extreme outliers (use Student-T instead).


        # -------------------------------
        # Negative Binomial Likelihood 
        # -------------------------------
        
        ## Use when KPI represents *count data* (non-negative integers: 0, 1, 2, …), 
        ## and the data are **overdispersed** — that is, the variance is larger than the mean.  
        ## Common for event counts such as conversions, signups, donations, or clicks.
        
        ## Unlike the Poisson distribution (which assumes mean = variance), the Negative Binomial introduces an overdispersion parameter (α) 
        ## to allow more flexible modeling of noisy or variable count data.
        
        ## Diagnostic checks before use:
        ## - KPI values are discrete (integer-valued) and non-negative.
        ## - Histogram shows a long right tail (many small counts, few large ones).
        ## - Empirical variance of KPI > mean (overdispersion).
        ## - Poisson model residuals underpredict extreme counts.
        
        ## Parameters:
        ## - mu     → expected count per observation (from your model structure).
        ## - alpha  → overdispersion parameter (controls variance); smaller α = higher variance.  
        ##            Var(Y) = μ + μ² / α  
        ##            As α → ∞, the distribution approaches Poisson.
        
        # ## Negative Binomial:
        # alpha = pm.Exponential("alpha", 1)  # weakly-informative prior for dispersion
        # y_obs = pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y_data)
        
        ## Interpretation:
        ## - Effects are in *additive count units*: a coefficient of 50 means the predictor increases expected count by ~50.
        ## - mu represents the expected count rate (average number of events per observation).
        ## - Posterior predictive checks should align on both mean and dispersion (variance).
        
        ## Use this when:
        ## - KPI is count-based (e.g., donations, conversions, purchases per week).
        ## - Variance grows faster than the mean (overdispersed counts).
        ## - Data include zeros naturally (not structural zeros).
        
        ## Don’t use when:
        ## - KPI is continuous (use Normal, Gamma, or LogNormal instead).
        ## - KPI variance ≈ mean (Poisson may suffice).
        ## - KPI has excessive zeros (use Zero-Inflated Negative Binomial).
        ## - KPI is strictly positive continuous (Gamma or LogNormal better suited).
        
        ## Consequences of misuse:
        ## - Using Normal for counts can yield non-integer or negative predictions.
        ## - Using Poisson when data are overdispersed underestimates uncertainty 
        ##   and leads to overconfident posterior intervals.
        ## - Setting α too tight (e.g., too large) removes dispersion flexibility, 
        ##   producing underfit residuals.


        # # ----- Lognormal ------
        # ## Use when KPI is strictly positive and multiplicative in nature (% change not additive raw changes)
        # ## Enforces positivty and suitable when the log(KPI) is normal
        # ## Use when variance is proportionate to the mean 
        # sigma = pm.HalfNormal("sigma", 0.3)
        # y_obs = pm.LogNormal("y_obs", mu=pm.math.log(y_hat + 1e-6), sigma=sigma, observed=y)


    return mmm
