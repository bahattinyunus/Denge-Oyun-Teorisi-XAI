"""
Cost Optimizer Tests
"""

import numpy as np
import pytest
from explainers.cost_optimizer import FastLinearSHAP, CoalitionSampler


def test_fast_linear_shap():
    # Linear model: y = 2.0*x0 - 3.0*x1 + 5.0
    coef = np.array([2.0, -3.0])
    intercept = 5.0
    means = np.array([1.0, 2.0])

    explainer = FastLinearSHAP(coef, intercept, feature_means=means)
    
    # Instance x = [3.0, 0.0]
    # phi_0 = 2.0 * (3.0 - 1.0) = 4.0
    # phi_1 = -3.0 * (0.0 - 2.0) = 6.0
    # Base value = 5.0 + 2*1 - 3*2 = 1.0
    # f(x) = 5.0 + 2*3 - 3*0 = 11.0
    # Sum phi = 4 + 6 = 10, Base + Sum = 1 + 10 = 11.0 (Efficiency verified!)
    res = explainer.explain(np.array([3.0, 0.0]))
    assert np.isclose(res["values"][0, 0], 4.0)
    assert np.isclose(res["values"][0, 1], 6.0)
    assert np.isclose(res["base_values"][0], 1.0)
    assert np.isclose(np.sum(res["values"][0]) + res["base_values"][0], 11.0)


def test_coalition_sampler():
    sampler = CoalitionSampler(n_features=4, n_samples=20)
    coalitions = sampler.sample_coalitions()
    assert coalitions.shape[1] == 4
    # Antithetic property: first half and second half should be complementary
    half = len(coalitions) // 2
    assert np.all(coalitions[:half] == ~coalitions[half:])
