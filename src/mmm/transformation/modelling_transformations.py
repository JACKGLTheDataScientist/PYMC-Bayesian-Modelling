import numpy as np
import pytensor.tensor as pt
from pytensor import scan 

def adstock_geometric(x, alpha):
    """
    Geometric adstock.
    Recursive implementation: y[t] = x[t] + alpha * y[t-1]
    """
    def step(x_t, y_tm1, alpha):
        return x_t + alpha * y_tm1

    outputs, _ = scan(
        fn=step,
        sequences=x,
        outputs_info=pt.zeros(()),  # initial y[0] = 0
        non_sequences=alpha,
    )
    return outputs


    

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
    return (x ** gamma) / (theta ** gamma + x ** gamma + 1e-8)
