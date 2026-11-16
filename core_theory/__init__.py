"""
Denge-Oyun-Teorisi-XAI Core Game Theory Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- nash_equilibrium: Non-cooperative game theory, feature competition, and equilibrium analysis.
- maynard_ess_filter: Evolutionary game theory (Maynard Smith) and Evolutionarily Stable Strategy (ESS) feature filtering.
- shapley_axioms: Cooperative game theory, characteristic functions, and the 4 fundamental Shapley axioms.
"""

from .nash_equilibrium import (
    FeaturePayoffMatrix,
    compute_nash_equilibrium,
    detect_feature_suppression,
    compute_feature_synergy,
    NashFeatureRanker,
)
from .maynard_ess_filter import (
    EvolutionaryGame,
    MaynardESSFilter,
    verify_ess_condition,
    simulate_replicator_dynamics,
)
from .shapley_axioms import (
    ShapleyGame,
    compute_exact_shapley,
    verify_efficiency_axiom,
    verify_symmetry_axiom,
    verify_dummy_axiom,
    verify_additivity_axiom,
    verify_all_shapley_axioms,
)

__all__ = [
    "FeaturePayoffMatrix",
    "compute_nash_equilibrium",
    "detect_feature_suppression",
    "compute_feature_synergy",
    "NashFeatureRanker",
    "EvolutionaryGame",
    "MaynardESSFilter",
    "verify_ess_condition",
    "simulate_replicator_dynamics",
    "ShapleyGame",
    "compute_exact_shapley",
    "verify_efficiency_axiom",
    "verify_symmetry_axiom",
    "verify_dummy_axiom",
    "verify_additivity_axiom",
    "verify_all_shapley_axioms",
]
