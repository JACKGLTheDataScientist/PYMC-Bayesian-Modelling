# tests/test_runtime_versions.py
def test_runtime_versions():
    import sys, inspect
    import pymc as pm, arviz as az
    from pymc.sampling import forward

    print("\nPYTHON:", sys.executable)
    print("PYMC:", pm.__version__)
    print("ARVIZ:", az.__version__)
    print("PPC signature:", inspect.signature(forward.sample_posterior_predictive))

    assert True
