import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt

from mmm.modelling_transformations import adstock_geometric, hill_saturation

# ----------------------------
# Tests for adstock_geometric
# ----------------------------
def test_adstock_geometric_basic():
    """Check that adstock works with simple inputs."""
    x = pt.as_tensor_variable(np.array([1.0, 0.0, 0.0, 0.0]))
    lam = 0.5
    L = 4

    out = adstock_geometric(x, lam, L)
    f = pytensor.function([], out)
    result = f()

    # Manually compute expected
    expected = np.array([1.0, 0.5, 0.25, 0.125])

    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_adstock_geometric_lag_limit():
    """Ensure lag length L truncates effect properly."""
    x = pt.as_tensor_variable(np.array([1.0] + [0.0] * 9))
    lam = 0.9
    L = 3

    out = adstock_geometric(x, lam, L)
    f = pytensor.function([], out)
    result = f()

    # Same length as input
    assert result.shape[0] == 10

    # Within lag length → memory exists
    assert result[1] > 0  

    # After L=3, effects should be truncated (zero)
    assert np.all(result[L+1:] == 0.0)



# ----------------------------
# Tests for hill_saturation
# ----------------------------
def test_hill_saturation_basic():
    """Check Hill transform outputs expected values."""
    x = np.array([0, 10, 50, 100])
    theta, gamma = 50, 2

    result = hill_saturation(x, theta, gamma)

    # At theta=50 and gamma=2 → f(50)=0.5
    assert np.isclose(result[2], 0.5, atol=1e-6)

    # Values should be bounded in [0,1]
    assert np.all(result >= 0) and np.all(result <= 1)


def test_hill_saturation_invalid_inputs():
    """Ensure Hill transform raises errors for invalid inputs."""
    with pytest.raises(TypeError):
        hill_saturation([1, 2, 3], 50, 2)  # not np.ndarray

    with pytest.raises(ValueError):
        hill_saturation(np.array([-1, 2, 3]), 50, 2)  # negative values

    with pytest.raises(ValueError):
        hill_saturation(np.array([1, 2, 3]), -10, 2)  # invalid theta

    with pytest.raises(ValueError):
        hill_saturation(np.array([1, 2, 3]), 10, 0)  # invalid gamma
