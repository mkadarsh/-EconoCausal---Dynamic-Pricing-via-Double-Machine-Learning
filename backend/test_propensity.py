import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal.propensity import PropensityError, analyze_propensity, estimate_propensity


def _campaign() -> pd.DataFrame:
    rows = []
    for index in range(40):
        treatment = 10 if index % 4 == 0 else 20 if index % 7 == 0 else 0
        rows.append(
            {
                "customer_id": f"c{index}",
                "age": 20 + index,
                "income": 20_000 + index * 500,
                "previous_purchases": index % 5,
                "website_visits": index % 9,
                "customer_tenure": 1 + index % 12,
                "previous_spend": 10 + index * 2,
                "engagement_score": 10 + index * 1.5,
                "treatment": treatment,
                "discount_amount": treatment,
                "purchase": int(index % 3 != 0),
                "revenue": float(index * 2 if index % 3 else 0),
            }
        )
    return pd.DataFrame(rows)


def test_propensity_calculation_and_visualization_data():
    result = estimate_propensity(_campaign(), 10)

    assert result.treated_count == 10
    # The $10 contrast excludes the separate $20 treatment arm.
    assert result.control_count == 26
    assert result.propensity_scores.shape[0] == 36
    assert result.propensity_scores["propensity_score"].between(0, 1).all()
    assert result.mean_propensity_treated > result.mean_propensity_control
    assert result.overlap_statistics["has_common_support"]


def test_both_discount_comparisons_and_matching():
    results = analyze_propensity(_campaign())

    assert set(results) == {10, 20}
    for result in results.values():
        stats = result.matching_statistics
        assert stats.matched_pair_count == len(result.matched_treatment_rows)
        assert stats.matched_pair_count == len(result.matched_control_rows)
        assert result.matched_treatment_rows["treatment"].eq(1).all()
        assert result.matched_control_rows["treatment"].eq(0).all()
        assert result.matched_control_rows["customer_id"].is_unique


def test_caliper_and_invalid_discount_are_reported():
    with pytest.raises(PropensityError, match="10 or 20"):
        estimate_propensity(_campaign(), 15)
    with pytest.raises(PropensityError, match="caliper"):
        estimate_propensity(_campaign(), 10, caliper=-0.1)


def test_missing_required_feature_is_rejected():
    campaign = _campaign().drop(columns=["income"])

    with pytest.raises(ValueError, match="missing required"):
        estimate_propensity(campaign, 10)
