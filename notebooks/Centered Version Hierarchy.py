# --- Global effect ---
mu_global = pm.LogNormal(
    "mu_global",
    mu=np.log([10000, 20000, 15000]),
    sigma=np.log([1.5, 1.2, 1.3]),
    dims="channel"
)

# --- Market-level deviation (centered) ---
sigma_market = pm.HalfNormal("sigma_market", sigma=[1000, 2000, 3000], dims="channel")
beta_market = pm.Normal(
    "beta_market",
    mu=mu_global,
    sigma=sigma_market,
    dims=("market", "channel")
)

# --- Product-level deviation (centered) ---
sigma_product = pm.HalfNormal("sigma_product", sigma=[2000, 3000, 4000], dims="channel")
beta_product = pm.Normal(
    "beta_product",
    mu=mu_global,
    sigma=sigma_product,
    dims=("product", "channel")
)

# --- Market × Product residual deviation ---
sigma_prod_mar = pm.HalfNormal("sigma_prod_mar", sigma=[200, 300, 300], dims="channel")
beta_prod_mar = pm.Normal(
    "beta_prod_mar",
    mu=0,
    sigma=sigma_prod_mar,
    dims=("market", "product", "channel")
)

# --- Combine into final coefficients ---
beta_mpc = pm.Deterministic(
    "beta_mpc",
    mu_global[None, None, :] +
    beta_market[:, None, :] +
    beta_product[None, :, :] +
    beta_prod_mar,
    dims=("market", "product", "channel")
)

                               
