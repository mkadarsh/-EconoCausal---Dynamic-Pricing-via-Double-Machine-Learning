import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.optimization.budget_optimizer import (
    BudgetOptimizationError,
    optimize_campaign,
    validate_budget,
)


def _campaign(count: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(count)],
            "baseline_purchase_probability": [0.2] * count,
            "purchase_probability_10": [0.5, 0.3, 0.35, 0.35][:count],
            "purchase_probability_20": [0.7, 0.6, 0.3, 0.45][:count],
            "ITE_10": [0.3, 0.1, 0.15, 0.15][:count],
            "ITE_20": [0.5, 0.4, 0.1, 0.25][:count],
            "expected_order_value": [100.0] * count,
            "discount_cost": [10.0] * count,
        }
    )


def test_zero_budget_assigns_no_discount():
    result = optimize_campaign(_campaign(), 0)
    assert result.recommendations["recommended_discount"].eq(0).all()
    assert result.summary.budget_used == 0
    assert result.summary.number_no_discount == 4


def test_small_budget_never_exceeds_budget_and_selects_best_value():
    result = optimize_campaign(_campaign(), 10)
    assert result.summary.budget_used <= 10
    assert result.summary.number_targeted == 1
    assert result.recommendations.loc[0, "recommended_discount"] == 10


def test_large_budget_targets_profitable_customers_only():
    result = optimize_campaign(_campaign(), 1_000)
    assert result.summary.budget_used <= 1_000
    assert result.summary.number_targeted == 4
    assert result.summary.number_no_discount == 0


def test_insufficient_customers_and_all_customer_outputs():
    result = optimize_campaign(_campaign(1), 10)
    assert len(result.recommendations) == 1
    assert result.recommendations.columns.tolist() == [
        "customer_id",
        "recommended_discount",
        "expected_purchase_probability",
        "expected_incremental_revenue",
        "discount_cost",
        "estimated_net_value",
    ]


def test_invalid_negative_budget():
    with pytest.raises(BudgetOptimizationError, match="non-negative"):
        validate_budget(-1)
    with pytest.raises(BudgetOptimizationError, match="non-negative"):
        optimize_campaign(_campaign(), -1)
