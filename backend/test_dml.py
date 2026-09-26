import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal.dml import (
    DMLError,
    estimate_ate,
    estimate_ite,
    get_model_summary,
    train_dml_model,
    train_dml_models,
)

econml = pytest.importorskip("econml")


def _campaign(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    engagement = rng.normal(50, 15, rows)
    treatment = rng.choice([0, 10, 20], rows, p=[0.5, 0.25, 0.25])
    baseline = 0.05 + 0.003 * engagement
    probability = np.clip(
        baseline
        + np.where(treatment == 10, 0.12, 0)
        + np.where(treatment == 20, 0.2, 0),
        0.02,
        0.95,
    )
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(rows)],
            "age": rng.integers(20, 70, rows),
            "income": rng.integers(20_000, 100_000, rows),
            "previous_purchases": rng.integers(0, 8, rows),
            "website_visits": rng.integers(0, 20, rows),
            "customer_tenure": rng.integers(1, 60, rows),
            "previous_spend": rng.uniform(0, 1_000, rows),
            "engagement_score": engagement,
            "treatment": treatment,
            "discount_amount": treatment,
            "purchase": (rng.random(rows) < probability).astype(int),
            "revenue": rng.uniform(0, 100, rows),
        }
    )


def test_dml_fit_returns_ite_ate_intervals_and_metadata():
    result = train_dml_model(_campaign(), 10, n_estimators=20)

    assert result.ite.size == 120
    assert result.customer_id.iloc[0] == "c0"
    assert np.isfinite(result.ate)
    assert result.treatment_count > 0
    assert result.control_count > 0
    assert result.metadata["estimator"] == "LinearDML"
    assert estimate_ate(result) == result.ate
    assert estimate_ite(result).equals(result.ite)
    assert get_model_summary(result)["ite_column"] == "ITE_10"


def test_both_models_preserve_customer_mapping():
    result_10, result_20, ite_table = train_dml_models(_campaign(), n_estimators=20)

    assert result_10.ite.size == result_20.ite.size == 120
    assert ite_table.columns.tolist() == ["customer_id", "ITE_10", "ITE_20"]
    assert ite_table["customer_id"].is_unique


def test_dml_rejects_invalid_discount_and_insufficient_data():
    with pytest.raises(DMLError, match="10 or 20"):
        train_dml_model(_campaign(), 15)

    small = _campaign(6)
    small["treatment"] = 0
    small["discount_amount"] = 0
    with pytest.raises(DMLError, match="both treated and control"):
        train_dml_model(small, 10)
