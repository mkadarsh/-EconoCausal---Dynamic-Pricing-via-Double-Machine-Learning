import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.causal.dag import (
    CausalGraphError,
    create_causal_graph,
    get_dowhy_model,
    identify_estimand,
)


def test_graph_contains_observed_confounders_and_campaign_paths():
    graph = create_causal_graph("T10", "purchase")

    assert '"T10" -> "purchase";' in graph
    assert '"purchase" -> "revenue";' in graph
    for confounder in (
        "age",
        "income",
        "previous_purchases",
        "website_visits",
        "customer_tenure",
        "previous_spend",
        "engagement_score",
    ):
        assert f'"{confounder}" -> "T10";' in graph
        assert f'"{confounder}" -> "purchase";' in graph


def test_graph_rejects_blank_node_names():
    with pytest.raises(CausalGraphError, match="non-empty"):
        create_causal_graph("", "purchase")


def test_model_and_estimand_for_ten_dollar_contrast():
    dowhy = pytest.importorskip("dowhy")
    data = pd.DataFrame(
        {
            "T10": [0, 1, 0, 1, 0, 1, 0, 1],
            "purchase": [0, 1, 0, 1, 1, 1, 0, 0],
            "age": [20, 21, 30, 31, 40, 41, 50, 51],
            "income": [20, 21, 30, 31, 40, 41, 50, 51],
            "previous_purchases": [0, 1, 0, 1, 0, 1, 0, 1],
            "website_visits": [1, 2, 1, 2, 1, 2, 1, 2],
            "customer_tenure": [1, 2, 1, 2, 1, 2, 1, 2],
            "previous_spend": [10, 20, 10, 20, 10, 20, 10, 20],
            "engagement_score": [10, 20, 10, 20, 10, 20, 10, 20],
        }
    )
    model = get_dowhy_model(data, "T10", "purchase")
    estimand = identify_estimand(model)

    assert estimand is not None


def test_model_requires_binary_preprocessed_treatment():
    data = pd.DataFrame({"treatment": [0, 10, 20], "purchase": [0, 1, 1]})
    for column in (
        "age",
        "income",
        "previous_purchases",
        "website_visits",
        "customer_tenure",
        "previous_spend",
        "engagement_score",
    ):
        data[column] = [1, 2, 3]

    with pytest.raises(CausalGraphError, match="binary"):
        get_dowhy_model(data, "treatment", "purchase")
