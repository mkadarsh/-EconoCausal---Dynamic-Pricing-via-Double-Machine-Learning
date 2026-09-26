import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal.preprocessing import PreprocessingError, preprocess_campaign


def _campaign() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["a", "b", "c", "d", "e", "f"],
            "age": [20, 30, 40, 50, 60, 70],
            "income": [20_000, 30_000, 40_000, 50_000, 60_000, 70_000],
            "previous_purchases": [0, 1, 2, 3, 4, 5],
            "website_visits": [1, 2, 3, 4, 5, 6],
            "customer_tenure": [1, 2, 3, 4, 5, 6],
            "previous_spend": [10, 20, 30, 40, 50, 60],
            "engagement_score": [10, 20, 30, 40, 50, 60],
            "treatment": [0, 10, 20, 0, 10, 20],
            "discount_amount": [0, 10, 20, 0, 10, 20],
            "purchase": [0, 1, 0, 1, 1, 1],
            "revenue": [0, 20, 0, 40, 30, 50],
        }
    )


def test_preprocessing_returns_standardized_features_and_binary_contrasts():
    prepared = preprocess_campaign(_campaign())

    assert prepared.customer_id.tolist() == ["a", "b", "c", "d", "e", "f"]
    assert prepared.X.columns.tolist() == list(prepared.feature_columns)
    assert "customer_id" not in prepared.X.columns
    assert prepared.X.isna().sum().sum() == 0
    assert prepared.T10.tolist() == [False, True, False, False, True, False]
    assert prepared.T20.tolist() == [False, False, True, False, False, True]
    assert prepared.binary_datasets[10].T.tolist() == [0, 1, 0, 1]
    assert prepared.binary_datasets[20].T.tolist() == [0, 1, 0, 1]
    assert prepared.binary_datasets[10].customer_id.tolist() == ["a", "b", "d", "e"]


def test_preprocessing_median_imputes_missing_numeric_features():
    campaign = _campaign()
    campaign.loc[1, "income"] = None

    prepared = preprocess_campaign(campaign)

    assert prepared.X.isna().sum().sum() == 0
    assert prepared.imputer.statistics_[1] == 50_000


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("treatment", 15, "Invalid treatment"),
        ("purchase", 2, "binary"),
        ("age", "not numeric", "non-numeric"),
    ],
)
def test_preprocessing_rejects_invalid_values(column, value, message):
    campaign = _campaign()
    if column == "age":
        campaign["age"] = campaign["age"].astype(object)
    campaign.loc[0, column] = value

    with pytest.raises(PreprocessingError, match=message):
        preprocess_campaign(campaign)


def test_preprocessing_rejects_duplicate_customer_ids():
    campaign = _campaign()
    campaign.loc[1, "customer_id"] = campaign.loc[0, "customer_id"]

    with pytest.raises(PreprocessingError, match="unique"):
        preprocess_campaign(campaign)
