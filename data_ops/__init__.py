"""
Denge-Oyun-Teorisi-XAI Data Operations Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- synthetic_coalitions: Synthetic game generator for game-theoretic benchmarking.
"""

from .synthetic_coalitions import (
    CoalitionDataset,
    generate_coalition_dataset,
    create_axiomatic_game,
)

__all__ = [
    "CoalitionDataset",
    "generate_coalition_dataset",
    "create_axiomatic_game",
]
