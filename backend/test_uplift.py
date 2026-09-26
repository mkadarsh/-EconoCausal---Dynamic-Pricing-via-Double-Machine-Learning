import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal.uplift import UpliftSegmentationError, segment_customers
from app.causal.uplift_config import SegmentationConfig


def _row(customer_id, baseline, ite_10, ite_20):
    return {
        "customer_id": customer_id,
        "baseline_purchase_probability": baseline,
        "purchase_probability_10": baseline + ite_10,
        "purchase_probability_20": baseline + ite_20,
        "ITE_10": ite_10,
        "ITE_20": ite_20,
    }


def test_classification_boundaries_and_output_columns():
    data = pd.DataFrame(
        [
            _row("negative", 0.5, -0.01, 0.01),
            _row("sure", 0.70, 0.02, 0.01),
            _row("persuadable", 0.40, 0.05, 0.01),
            _row("unlikely", 0.30, 0.02, 0.01),
            _row("unclassified", 0.50, 0.02, 0.01),
        ]
    )

    result = segment_customers(data)

    assert result.customers.columns.tolist() == [
        "customer_id",
        "segment",
        "baseline_probability",
        "probability_10",
        "probability_20",
        "ITE_10",
        "ITE_20",
    ]
    assert result.customers.set_index("customer_id")["segment"].to_dict() == {
        "negative": "Potential Negative Response",
        "sure": "Sure Buyer",
        "persuadable": "Persuadable",
        "unlikely": "Unlikely Buyer",
        "unclassified": "Unclassified Response",
    }


def test_custom_thresholds_and_statistics():
    data = pd.DataFrame([_row("a", 0.6, 0.1, 0.0), _row("b", 0.1, 0.0, 0.0)])
    result = segment_customers(
        data,
        SegmentationConfig(
            high_baseline_probability=0.6,
            low_baseline_probability=0.2,
            meaningful_positive_ite=0.08,
            small_absolute_ite=0.03,
        ),
    )

    assert result.customers.loc[0, "segment"] == "Persuadable"
    assert result.customers.loc[1, "segment"] == "Unlikely Buyer"
    assert result.statistics["customer_count"].sum() == 2
    assert result.statistics["customer_fraction"].sum() == pytest.approx(1.0)


def test_invalid_input_is_reported():
    data = pd.DataFrame([_row("a", 0.5, 0.1, 0.0)]).drop(columns=["ITE_20"])

    with pytest.raises(UpliftSegmentationError, match="required"):
        segment_customers(data)
