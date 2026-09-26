import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal.uplift import UpliftSegmentationError, evaluate_all_uplift, evaluate_uplift


def _data():
    return pd.DataFrame(
        {
            "customer_id": list("abcdef"),
            "treatment": [10, 0, 20, 0, 10, 20],
            "purchase": [1, 0, 1, 0, 0, 1],
            "ITE_10": [0.9, 0.8, 0.7, 0.2, 0.1, 0.0],
            "ITE_20": [0.8, 0.7, 0.6, 0.3, 0.2, 0.1],
        }
    )


def test_uplift_evaluation_ranks_by_effect_and_returns_plot_data():
    result = evaluate_uplift(_data(), 10)

    assert result.x.tolist() == [1, 2, 3, 4]
    assert result.y[0] == pytest.approx(1.0)
    assert result.random_baseline[0] == pytest.approx(result.final_incremental_gain / 4)
    assert result.plot_data["x"] == result.x.tolist()
    assert np.isfinite(result.area_under_curve)
    assert result.treated_count == result.control_count == 2


def test_both_contrasts_and_validation():
    data = _data()
    results = evaluate_all_uplift(data)
    assert set(results) == {10, 20}

    with pytest.raises(UpliftSegmentationError, match="only 0, 10, or 20"):
        evaluate_uplift(data.assign(treatment=[99, 0, 10, 0, 10, 0]), 10)
    with pytest.raises(UpliftSegmentationError, match="metric"):
        evaluate_uplift(data, 10, metric="unknown")
