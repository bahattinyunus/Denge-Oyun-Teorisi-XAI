"""
Lloyd Shapley Cooperative Game and Axioms Tests
"""

import numpy as np
import pytest
from core_theory.shapley_axioms import (
    ShapleyGame,
    compute_exact_shapley,
    verify_efficiency_axiom,
    verify_symmetry_axiom,
    verify_dummy_axiom,
    verify_additivity_axiom,
    verify_all_shapley_axioms,
)
from data_ops.synthetic_coalitions import create_axiomatic_game


def test_exact_shapley_simple():
    # 2-player game:
    # v({}) = 0, v({0}) = 2, v({1}) = 4, v({0, 1}) = 10
    # phi_0 = 1/2 * (2 - 0) + 1/2 * (10 - 4) = 1 + 3 = 4
    # phi_1 = 1/2 * (4 - 0) + 1/2 * (10 - 2) = 2 + 4 = 6
    def char_fn(S):
        if not S:
            return 0.0
        if S == (0,):
            return 2.0
        if S == (1,):
            return 4.0
        if set(S) == {0, 1}:
            return 10.0
        return 0.0

    game = ShapleyGame(2, characteristic_fn=char_fn)
    phi = compute_exact_shapley(game)
    assert np.isclose(phi[0], 4.0)
    assert np.isclose(phi[1], 6.0)
    # Check efficiency: 4 + 6 = 10 = v({0, 1})
    assert np.isclose(np.sum(phi), 10.0)


def test_all_four_axioms():
    game = create_axiomatic_game(n_players=4)
    report = verify_all_shapley_axioms(game)
    
    assert report["efficiency"]["passed"] is True
    assert report["symmetry"]["passed"] is True
    assert report["dummy_player"]["passed"] is True
    assert report["additivity"]["passed"] is True
    assert report["all_passed"] is True
