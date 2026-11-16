"""
Denge-Oyun-Teorisi-XAI Explainers Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- waterfall_local: Local micro-simulation waterfall explainer.
- beeswarm_global: Global summary and beeswarm macro-strategy importance maps.
- cost_optimizer: High-performance TreeSHAP, LinearSHAP, and coalition sampling accelerators.
"""

from .waterfall_local import render_waterfall, WaterfallExplainer
from .beeswarm_global import render_summary, BeeswarmExplainer
from .cost_optimizer import (
    FastTreeSHAP,
    FastLinearSHAP,
    CoalitionSampler,
    benchmark_shap_latency,
)

__all__ = [
    "render_waterfall",
    "WaterfallExplainer",
    "render_summary",
    "BeeswarmExplainer",
    "FastTreeSHAP",
    "FastLinearSHAP",
    "CoalitionSampler",
    "benchmark_shap_latency",
]
