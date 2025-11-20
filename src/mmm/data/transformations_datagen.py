import numpy as np

# Adstock function 
## Geometric Adstock that implements lingering effect of media over different time periods 
def adstock_datagen(x: np.array, lam: float): 
    """ 
    Apply geometric adstock transformation to a media time series.

    Adstock models the lingering effect of media exposure over time:
    today's effect is a weighted sum of today's spend plus a decayed
    carryover of past spend. This captures memory effects in consumer
    response to advertising.

    The geometric adstock function is defined as:
        out[t] = x[t] + lam * out[t-1]
    where lam ∈ [0,1) is the decay parameter.

    Parameters
    ----------
    x : np.ndarray
        1D array of non-negative media spend or exposures over time.
    lam : float
        Decay parameter in [0,1). Higher values produce longer-lasting
        carryover effects, while values close to 0 imply very short memory.

    Returns
    -------
    out : np.ndarray
        Transformed array of the same shape as x, representing the
        adstocked media variable.

    Raises
    ------
    TypeError
        If `x` is not a numpy.ndarray or `lam` is not a float.
    ValueError
        If `lam` is not in [0,1), or if `x` contains NaNs.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([100, 0, 0, 0, 0])
    >>> adstock(x, 0.5)
    array([100. ,  50. ,  25. ,  12.5,   6.25])
    """
    
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be of type numpy.ndarrray, got {type(x)}")
    if not isinstance(lam, float):
        raise TypeError(f"lam must be float, got {type(lam)}")
    if not 0 <= lam <= 1: 
        raise ValueError(f"lam must be between 0-1")
    if np.isnan(x).any():
        raise ValueError("x contains NaNs, clean your data before applying adstock.")
    
    # Core logic
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + (out[t-1] * lam if t > 0 else 0.0)
    return out

## Hill function saturation - models saturation with an inflexion point. import numpy as np
def hill_datagen(x: np.ndarray, theta: float, gamma: float) -> np.ndarray:
    """
    Apply Hill saturation transformation to media spend or exposures.

    The Hill function models diminishing incremental effects of spend:
    as spend increases, incremental impact slows down and approaches
    a saturation level. It is a flexible S-shaped curve often used in
    marketing mix modeling.

    The function is defined as:
        f(x) = x^gamma / (theta^gamma + x^gamma)

    - theta controls the half-saturation point (spend level where f(x) = 0.5).
    - gamma controls steepness of the curve (higher gamma = steeper).

    Parameters
    ----------
    x : np.ndarray
        1D array of non-negative media spend or exposures.
    theta : float
        Half-saturation parameter (>0). Value of x where the transformed
        response equals 0.5.
    gamma : float
        Slope parameter (>0). Higher values make the curve steeper around
        the half-saturation point.

    Returns
    -------
    out : np.ndarray
        Transformed array of the same shape as x, values in the interval (0,1).

    Raises
    ------
    TypeError
        If x is not a numpy.ndarray, or if theta/gamma are not numeric.
    ValueError
        If x contains negative values, or if theta/gamma are not strictly >0.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 10, 50, 100, 200])
    >>> hill_transform(x, theta=50, gamma=2)
    array([0.        , 0.03846154, 0.5       , 0.8       , 0.94117647])
    """

    # Type checks
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be numpy.ndarray, got {type(x)}")
    if not isinstance(theta, (float, int)):
        raise TypeError(f"theta must be float, got {type(theta)}")
    if not isinstance(gamma, (float, int)):
        raise TypeError(f"gamma must be float, got {type(gamma)}")

    # Value checks
    if np.any(x < 0):
        raise ValueError("x must be non-negative (spend/exposures).")
    if theta <= 0:
        raise ValueError("theta must be > 0.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")

    x = x.astype(float)
    theta, gamma = float(theta), float(gamma)

    return (x ** gamma) / (theta ** gamma + x ** gamma + 1e-8)
