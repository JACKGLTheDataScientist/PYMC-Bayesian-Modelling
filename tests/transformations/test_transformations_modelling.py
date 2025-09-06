import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt

from mmm.transformation.modelling_transformations import adstock_geometric, hill_saturation

# ----------------------------
# Tests for adstock_geometric
# ----------------------------
def test_adstock_geometric_decay():
    """Ensure recursive adstock decays but never hard-truncates."""
    x = pt.as_tensor_variable(np.array([1.0] + [0.0] * 9))
    lam = 0.9

    out = adstock_geometric(x, lam)
    f = pytensor.function([], out)
    result = f()

    # Same length as input
    assert result.shape[0] == 10

    # Each step should be smaller than the previous
    assert np.all(result[1:] < result[:-1])

    # Decay should get close to zero
    assert result[-1] < 0.5

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
