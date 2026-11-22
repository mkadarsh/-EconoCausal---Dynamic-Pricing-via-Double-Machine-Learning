import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal.refutation import (
    RefutationError,
    RefutationResult,
    _refutation_passed,
    audit_all_refutations,
    audit_refutations,
)


def _campaign() -> pd.DataFrame:
    rows = []
    for index in range(30):
        treatment = 10 if index % 3 == 0 else 20 if index % 5 == 0 else 0
        rows.append(
            {
                "customer_id": f"c{index}",
                "age": 20 + index,
                "income": 30_000 + index * 100,
                "previous_purchases": index % 4,
                "website_visits": index % 6,
                "customer_tenure": 1 + index,
                "previous_spend": 10 + index,
                "engagement_score": 20 + index,
                "treatment": treatment,
                "discount_amount": treatment,
                "purchase": int(index % 2 == 0),
                "revenue": float(index),
            }
        )
    return pd.DataFrame(rows)


def test_refutation_result_is_ui_serializable():
    result = RefutationResult(
        test_name="Placebo treatment",
        estimated_effect=0.2,
        refuted_effect=0.01,
        passed=True,
        interpretation="Diagnostic result.",
        limitations="Not proof.",
    )
    assert result.to_dict()["test_name"] == "Placebo treatment"
    assert result.to_dict()["passed"] is True


def test_invalid_refutation_configuration_is_reported():
    with pytest.raises(RefutationError, match="num_simulations"):
        audit_refutations(_campaign(), 10, num_simulations=0)
    with pytest.raises(RefutationError, match="subset_fraction"):
        audit_refutations(_campaign(), 10, subset_fraction=1)
    with pytest.raises(RefutationError, match="10 or 20"):
        audit_refutations(_campaign(), 15)


def test_dowhy_numpy_significance_maps_to_pass_status():
    result = SimpleNamespace(
        refutation_result={
            "p_value": 0.5,
            "is_statistically_significant": np.False_,
        }
    )

    assert _refutation_passed(result) is True


def test_missing_or_incompatible_dowhy_is_explicit():
    result = audit_refutations(_campaign(), 10, num_simulations=1)

    assert result.discount_amount == 10
    assert len(result.results) == 3
    assert all(item.test_name for item in result.results)
    if any(item.status == "unavailable" for item in result.results):
        assert all(item.passed is None for item in result.results)


def test_audit_supports_both_discount_comparisons():
    audits = audit_all_refutations(_campaign(), num_simulations=1)

    assert set(audits) == {10, 20}
    assert all(len(audit.results) == 3 for audit in audits.values())
