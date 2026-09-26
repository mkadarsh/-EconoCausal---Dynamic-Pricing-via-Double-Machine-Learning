"""Customer response segmentation from causal uplift estimates.

The segment labels describe estimated response patterns; they do not assert
guaranteed individual behavior. In particular, an ITE is an estimated
individual-level causal effect, not an observed or certain outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.causal.uplift_config import SegmentationConfig

REQUIRED_COLUMNS = (
    "customer_id",
    "baseline_purchase_probability",
    "purchase_probability_10",
    "purchase_probability_20",
    "ITE_10",
    "ITE_20",
)
OUTPUT_COLUMNS = REQUIRED_COLUMNS
SEGMENTS = (
    "Potential Negative Response",
    "Sure Buyer",
    "Persuadable",
    "Unlikely Buyer",
    "Unclassified Response",
)


class UpliftSegmentationError(ValueError):
    """Raised when counterfactual probability data cannot be segmented."""


@dataclass(frozen=True)
class SegmentationResult:
    """Customer-level segment labels and aggregate segment statistics."""

    customers: pd.DataFrame
    statistics: pd.DataFrame
    config: SegmentationConfig


def _validate_config(config: SegmentationConfig) -> None:
    values = (
        config.high_baseline_probability,
        config.low_baseline_probability,
        config.meaningful_positive_ite,
        config.small_absolute_ite,
        config.negative_ite,
    )
    if not all(np.isfinite(value) for value in values):
        raise UpliftSegmentationError("Segmentation thresholds must be finite numbers")
    if not 0 <= config.low_baseline_probability < config.high_baseline_probability <= 1:
        raise UpliftSegmentationError(
            "Baseline thresholds must satisfy 0 <= low < high <= 1"
        )
    if config.meaningful_positive_ite <= 0 or config.small_absolute_ite < 0:
        raise UpliftSegmentationError(
            "ITE thresholds must be positive meaningful and non-negative small-effect values"
        )
    if config.negative_ite >= 0:
        raise UpliftSegmentationError("negative_ite must be below zero")


def _validate_input(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise UpliftSegmentationError(f"Data is missing required columns: {missing}")
    if data["customer_id"].isna().any() or data["customer_id"].duplicated().any():
        raise UpliftSegmentationError("customer_id must be present and unique")
    numeric = data[list(REQUIRED_COLUMNS[1:])].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        invalid = numeric.columns[numeric.isna().any()].tolist()
        raise UpliftSegmentationError(f"Numeric response columns contain invalid values: {invalid}")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise UpliftSegmentationError("Numeric response columns must contain finite values")
    probabilities = numeric[
        ["baseline_purchase_probability", "purchase_probability_10", "purchase_probability_20"]
    ]
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise UpliftSegmentationError("Purchase probabilities must lie between 0 and 1")


def _segment(row: pd.Series, config: SegmentationConfig) -> str:
    effects = np.array([row["ITE_10"], row["ITE_20"]], dtype=float)
    max_effect = float(effects.max())
    max_absolute_effect = float(np.abs(effects).max())
    baseline = float(row["baseline_probability"])

    # Precedence makes overlapping boundary cases deterministic.
    if (effects <= config.negative_ite).any():
        return "Potential Negative Response"
    if baseline >= config.high_baseline_probability and max_absolute_effect < config.small_absolute_ite:
        return "Sure Buyer"
    if max_effect >= config.meaningful_positive_ite:
        return "Persuadable"
    if baseline <= config.low_baseline_probability and max_absolute_effect < config.small_absolute_ite:
        return "Unlikely Buyer"
    return "Unclassified Response"


def _statistics(customers: pd.DataFrame) -> pd.DataFrame:
    grouped = customers.groupby("segment", sort=False)
    stats = grouped.agg(
        customer_count=("customer_id", "size"),
        mean_baseline_probability=("baseline_probability", "mean"),
        mean_probability_10=("probability_10", "mean"),
        mean_probability_20=("probability_20", "mean"),
        mean_ITE_10=("ITE_10", "mean"),
        mean_ITE_20=("ITE_20", "mean"),
    ).reset_index()
    stats["customer_fraction"] = stats["customer_count"] / len(customers)
    return stats


def segment_customers(
    data: pd.DataFrame, config: SegmentationConfig | None = None
) -> SegmentationResult:
    """Classify customers and calculate segment-level descriptive statistics."""

    selected_config = config or SegmentationConfig()
    _validate_config(selected_config)
    _validate_input(data)
    customers = data.loc[:, REQUIRED_COLUMNS].copy()
    customers = customers.rename(
        columns={
            "baseline_purchase_probability": "baseline_probability",
            "purchase_probability_10": "probability_10",
            "purchase_probability_20": "probability_20",
        }
    )
    customers["segment"] = customers.apply(
        lambda row: _segment(row, selected_config), axis=1
    )
    customers = customers[
        [
            "customer_id",
            "segment",
            "baseline_probability",
            "probability_10",
            "probability_20",
            "ITE_10",
            "ITE_20",
        ]
    ]
    return SegmentationResult(
        customers=customers,
        statistics=_statistics(customers),
        config=selected_config,
    )
