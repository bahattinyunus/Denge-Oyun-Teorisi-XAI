"""
Maynard Smith ESS and Evolutionary Game Theory Tests
"""

import numpy as np
import pytest
from core_theory.maynard_ess_filter import (
    EvolutionaryGame,
    verify_ess_condition,
    simulate_replicator_dynamics,
    MaynardESSFilter,
)


def test_evolutionary_game_fitness():
    payoff = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])
    game = EvolutionaryGame(payoff, feature_names=["Hawk", "Dove"])
    pop = np.array([0.5, 0.5])
    fit = game.fitness(pop)
    assert len(fit) == 2
    avg_fit = game.average_fitness(pop)
    assert avg_fit > 0


def test_verify_ess_condition():
    # Hawk-Dove game payoff
    # V=4, C=6
    # Hawk vs Hawk: (V-C)/2 = -1
    # Hawk vs Dove: V = 4
    # Dove vs Hawk: 0
    # Dove vs Dove: V/2 = 2
    payoff = np.array([
        [-1.0, 4.0],
        [0.0, 2.0],
    ])
    # Interior ESS strategy: p = V / C = 4 / 6 = 2/3 Hawk, 1/3 Dove
    p_ess = np.array([2.0 / 3.0, 1.0 / 3.0])
    res = verify_ess_condition(payoff, p_ess)
    assert res["is_ess"] is True


def test_replicator_dynamics_simulation():
    payoff = np.array([
        [3.0, 0.0],
        [5.0, 1.0],
    ])
    traj, times = simulate_replicator_dynamics(payoff, generations=50, dt=0.05)
    assert traj.shape == (50, 2)
    # Total shares should always sum to 1.0
    for row in traj:
        assert np.isclose(np.sum(row), 1.0, atol=1e-3)


def test_maynard_ess_filter():
    rng = np.random.RandomState(42)
    X = rng.normal(0, 1, size=(200, 4))
    # Feature 0 is strong and robust
    # Feature 3 is pure weak noise
    y = 3.0 * X[:, 0] + 1.5 * X[:, 1] + 0.0 * X[:, 3]

    ess_filter = MaynardESSFilter(n_generations=50, mutation_rate=0.1, random_state=42)
    result = ess_filter.fit_filter(X, y, feature_names=["Strong", "Medium", "Weak", "Dummy"])
    
    assert "selected_features" in result
    assert "Strong" in result["selected_features"]
    assert len(result["feature_scores"]) == 4
