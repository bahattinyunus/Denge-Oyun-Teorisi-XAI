"""
Explainers (Waterfall and Beeswarm) Tests
"""

import numpy as np
import pytest
from explainers.waterfall_local import WaterfallExplainer, render_waterfall
from explainers.beeswarm_global import BeeswarmExplainer, render_summary


def test_waterfall_explainer():
    vals = np.array([1.5, -0.8, 0.4])
    base = 10.0
    data = np.array([3.2, 1.1, 4.5])
    names = ["A", "B", "C"]

    we = WaterfallExplainer(base_value=base, values=vals, data=data, feature_names=names)
    summary = we.text_summary()
    assert "E[f(x)]" in summary
    assert "f(x)" in summary

    fig = we.plot(show=False)
    assert fig is not None


def test_beeswarm_explainer():
    vals = np.array([
        [1.0, -0.5, 0.2],
        [0.8, -0.3, 0.1],
        [1.2, -0.7, 0.3],
    ])
    names = ["Feat1", "Feat2", "Feat3"]
    be = BeeswarmExplainer(values=vals, feature_names=names)
    df = be.compute_global_importance()
    assert len(df) == 3
    assert df.iloc[0]["Feature"] == "Feat1"

    fig = be.plot(show=False)
    assert fig is not None
