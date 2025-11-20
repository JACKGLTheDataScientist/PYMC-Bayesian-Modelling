import numpy as np
import pytest
from mmm.data.transformations_datagen import adstock_datagen, hill_datagen

## Adstock Unit Tests
def test_adstock_basic_case():
    x = np.array([100, 0, 0, 0, 0])
    out = adstock_datagen(x, 0.5)
    expected = np.array([100, 50, 25, 12.5, 6.25])
    np.testing.assert_allclose(out, expected)

def test_adstock_zero_decay():
    x = np.array([10, 20, 30])
    out = adstock_datagen(x, 0.0)
    np.testing.assert_array_equal(out, x)  # no carryover effect

def test_adstock_high_decay():
    x = np.array([10, 0, 0])
    out = adstock_datagen(x, 0.9)
    assert out[1] > 0 and out[2] > 0  # lingering effect

def test_adstock_invalid_lam():
    x = np.array([10, 20])
    with pytest.raises(ValueError):
        adstock_datagen(x, 1.5)  # invalid decay

def test_adstock_non_numpy_input():
    with pytest.raises(TypeError):
        adstock_datagen([10, 20], 0.5)  # list instead of np.ndarray

def test_adstock_with_nan_values():
    x = np.array([10, np.nan, 20])
    with pytest.raises(ValueError):
        adstock_datagen(x, 0.5)


## Hill Function Unit Tests
def test_hill_basic_behavior():
    x = np.array([0, 50, 100, 200])
    out = hill_datagen(x, theta=50, gamma=2)
    assert np.all((0 <= out) & (out <= 1))  # values should be between 0 and 1
    assert np.isclose(out[2], 0.8, atol=0.05)  # sanity check near expected value

def test_hill_half_saturation():
    x = np.array([50.0])
    out = hill_datagen(x, theta=50, gamma=2)
    np.testing.assert_allclose(out, np.array([0.5]), rtol=1e-2)

def test_hill_large_values_saturate():
    x = np.array([1e6])
    out = hill_datagen(x, theta=50, gamma=2)
    assert np.isclose(out, 1.0, atol=1e-6)

def test_hill_invalid_inputs():
    x = np.array([10, 20])
    with pytest.raises(ValueError):
        hill_datagen(x, theta=-10, gamma=2)  # invalid theta
    with pytest.raises(ValueError):
        hill_datagen(x, theta=10, gamma=0)   # invalid gamma

def test_hill_negative_x():
    x = np.array([-5, 10])
    with pytest.raises(ValueError):
        hill_datagen(x, theta=50, gamma=2)

def test_hill_non_numpy_input():
    with pytest.raises(TypeError):
        hill_datagen([10, 20], theta=50, gamma=2)

