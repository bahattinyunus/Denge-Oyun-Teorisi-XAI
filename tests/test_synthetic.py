"""
Synthetic Coalitions Dataset Tests
"""

import numpy as np
import pytest
from data_ops.synthetic_coalitions import generate_coalition_dataset, create_axiomatic_game


def test_generate_coalition_dataset():
    dataset = generate_coalition_dataset(n_samples=300, random_state=42)
    assert dataset.X.shape == (300, 6)
    assert dataset.y.shape == (300,)
    assert len(dataset.feature_names) == 6
    assert "Dummy_Noise" in dataset.feature_names
    assert dataset.roles["Dummy_Noise"] == "dummy"


def test_create_axiomatic_game():
    game = create_axiomatic_game(n_players=4)
    assert game.n_players == 4
    # Dummy player (index 3) should have zero marginal contribution
    val_with_dummy = game.v((0, 3))
    val_without_dummy = game.v((0,))
    assert val_with_dummy == val_without_dummy
