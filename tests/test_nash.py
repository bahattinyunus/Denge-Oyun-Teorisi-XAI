"""
Nash Equilibrium Unit Tests
"""

import numpy as np
import pytest
from core_theory.nash_equilibrium import (
    FeaturePayoffMatrix,
    compute_nash_equilibrium,
    detect_feature_suppression,
    compute_feature_synergy,
    NashFeatureRanker,
)


def test_nash_equilibrium_basic():
    # 2x2 symmetric game: Prisoner's dilemma payoff structure
    matrix = np.array([
        [3.0, 0.0],
        [5.0, 1.0],
    ])
    result = compute_nash_equilibrium(matrix)
    assert "mixed_equilibrium" in result
    assert "expected_payoff" in result
    assert np.isclose(np.sum(result["mixed_equilibrium"]), 1.0)


def test_feature_payoff_from_correlations():
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(100, 3))
    y = 2.0 * X[:, 0] + 0.5 * X[:, 1]
    
    payoff = FeaturePayoffMatrix.from_correlations(X, y, feature_names=["F0", "F1", "F2"])
    assert payoff.matrix.shape == (3, 3)
    # F0 should have highest isolated payoff
    assert payoff.matrix[0, 0] > payoff.matrix[2, 2]


def test_suppression_and_synergy_detection():
    # Construct a matrix with clear suppression and synergy
    matrix = np.array([
        [10.0, 2.0, 10.0],   # F0 suppressed by F1 (drops from 10 to 2)
        [2.0, 5.0, 8.0],
        [10.0, 8.0, 4.0],    # F2 + F1 has synergy
    ])
    suppressions = detect_feature_suppression(matrix, threshold=0.3)
    assert len(suppressions) > 0
    assert suppressions[0]["suppressed_index"] == 0
    assert suppressions[0]["suppressor_index"] == 1


def test_nash_ranker():
    payoff = FeaturePayoffMatrix(np.array([
        [5.0, 2.0],
        [1.0, 0.5],
    ]), feature_names=["A", "B"])
    ranker = NashFeatureRanker(payoff)
    ranked = ranker.rank()
    assert len(ranked) == 2
    assert ranked[0]["feature"] == "A"
