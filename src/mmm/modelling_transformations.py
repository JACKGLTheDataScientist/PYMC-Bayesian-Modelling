import numpy as np
import pytensor.tensor as pt

import pytensor.tensor as pt

def adstock_geometric(x, alpha, L=21):
    """
    Geometric adstock transformation with consistent padding.
    
    Parameters
    ----------
    x : 1D PyTensor vector
        Input spend time series
    alpha : scalar
        Decay parameter between 0 and 1
    L : int
        Maximum lag length
    
    Returns
    -------
    1D PyTensor vector of same length as x
    """
    n = x.shape[0]
    x_lags = []
    for l in range(L):
        if l == 0:
            lagged = x
        else:
            # drop last l values
            truncated = x[:-l]
            # pad with l zeros at the front
            pad = pt.zeros((l,), dtype=x.dtype)
            lagged = pt.concatenate([pad, truncated])
        # force length n by slicing
        lagged = lagged[:n]
        x_lags.append(lagged * (alpha**l))
    return pt.sum(pt.stack(x_lags, axis=0), axis=0)
    

## Hill function saturation - models saturation with an inflexion point. import numpy as np
def hill_saturation(x: np.ndarray, theta: float, gamma: float) -> np.ndarray:
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
