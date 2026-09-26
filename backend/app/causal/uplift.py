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
EVALUATION_COLUMNS = ("customer_id", "treatment", "purchase")


class UpliftSegmentationError(ValueError):
    """Raised when counterfactual probability data cannot be segmented."""


@dataclass(frozen=True)
class SegmentationResult:
    """Customer-level segment labels and aggregate segment statistics."""

    customers: pd.DataFrame
    statistics: pd.DataFrame
    config: SegmentationConfig


@dataclass(frozen=True)
class UpliftEvaluation:
    """Plotly-ready ranked uplift/Qini evaluation for one discount contrast."""

    discount_amount: int
    x: np.ndarray
    y: np.ndarray
    random_baseline: np.ndarray
    metric: str
    area_under_curve: float
    final_incremental_gain: float
    treated_count: int
    control_count: int

    @property
    def plot_data(self) -> dict[str, list[float]]:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "random_baseline": self.random_baseline.tolist(),
        }


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


def _evaluation_frame(data: pd.DataFrame, discount_amount: int) -> pd.DataFrame:
    if discount_amount not in (10, 20):
        raise UpliftSegmentationError("discount_amount must be 10 or 20")
    required = set(EVALUATION_COLUMNS) | {f"ITE_{discount_amount}"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise UpliftSegmentationError(
            f"Evaluation data is missing required columns: {missing}"
        )
    frame = data.loc[:, sorted(required)].copy()
    frame["treatment"] = pd.to_numeric(frame["treatment"], errors="coerce")
    frame["purchase"] = pd.to_numeric(frame["purchase"], errors="coerce")
    frame[f"ITE_{discount_amount}"] = pd.to_numeric(
        frame[f"ITE_{discount_amount}"], errors="coerce"
    )
    if frame.isna().any().any():
        raise UpliftSegmentationError("Evaluation data contains missing or non-numeric values")
    allowed_campaign_treatments = {0, 10, 20}
    if not frame["treatment"].isin(allowed_campaign_treatments).all():
        raise UpliftSegmentationError(
            "Evaluation treatment must contain only 0, 10, or 20"
        )
    if not frame["purchase"].isin({0, 1}).all():
        raise UpliftSegmentationError("Evaluation purchase must be binary")
    if frame["customer_id"].duplicated().any():
        raise UpliftSegmentationError("Evaluation customer_id values must be unique")
    frame = frame.loc[frame["treatment"].isin({0, discount_amount})].reset_index(drop=True)
    if not (frame["treatment"] == discount_amount).any() or not (frame["treatment"] == 0).any():
        raise UpliftSegmentationError(
            f"Evaluation requires both treated and control rows for ${discount_amount}"
        )
    return frame


def evaluate_uplift(
    data: pd.DataFrame,
    discount_amount: int,
    *,
    metric: str = "uplift",
) -> UpliftEvaluation:
    """Evaluate ranked causal effects against observed outcomes.

    Customers are ranked by estimated ITE, then cumulative incremental gain is
    estimated as treated outcomes minus the control outcome rate scaled by the
    treated count. This is the standard observational Qini-style construction;
    it is descriptive and relies on exchangeability, overlap, and stable
    treatment/outcome definitions. It does not prove model superiority.
    """

    if metric not in {"uplift", "qini", "cumulative_incremental_gain"}:
        raise UpliftSegmentationError(
            "metric must be 'uplift', 'qini', or 'cumulative_incremental_gain'"
        )
    frame = _evaluation_frame(data, discount_amount)
    effect_column = f"ITE_{discount_amount}"
    frame = frame.sort_values(effect_column, ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    treated = frame["treatment"].eq(discount_amount)
    control = frame["treatment"].eq(0)
    control_rate = frame.loc[control, "purchase"].mean()
    treated_outcomes = (frame["purchase"] * treated).cumsum().to_numpy(dtype=float)
    treated_seen = treated.cumsum().to_numpy(dtype=float)
    gain = treated_outcomes - treated_seen * float(control_rate)
    x = np.arange(1, len(frame) + 1, dtype=float)
    final_gain = float(gain[-1])
    random_baseline = final_gain * x / len(frame)
    area = float(np.trapezoid(gain - random_baseline, x))
    return UpliftEvaluation(
        discount_amount=discount_amount,
        x=x,
        y=gain,
        random_baseline=random_baseline,
        metric=metric,
        area_under_curve=area,
        final_incremental_gain=final_gain,
        treated_count=int(treated.sum()),
        control_count=int(control.sum()),
    )


def evaluate_all_uplift(data: pd.DataFrame) -> dict[int, UpliftEvaluation]:
    """Evaluate both supported discount contrasts from one result table."""

    return {
        discount: evaluate_uplift(
            data,
            discount,
            metric="cumulative_incremental_gain",
        )
        for discount in (10, 20)
    }


def evaluate_uplift_curve(data: pd.DataFrame, discount_amount: int) -> UpliftEvaluation:
    """Return the ranked cumulative uplift curve for one discount contrast."""

    return evaluate_uplift(data, discount_amount, metric="uplift")


def evaluate_qini_curve(data: pd.DataFrame, discount_amount: int) -> UpliftEvaluation:
    """Return the ranked Qini-style cumulative incremental gain curve."""

    return evaluate_uplift(data, discount_amount, metric="qini")


def evaluate_cumulative_incremental_gain(
    data: pd.DataFrame, discount_amount: int
) -> UpliftEvaluation:
    """Return cumulative incremental gain and its random-targeting baseline."""

    return evaluate_uplift(
        data,
        discount_amount,
        metric="cumulative_incremental_gain",
    )
