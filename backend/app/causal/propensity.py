"""Propensity scores and simple matching for the campaign contrasts.

This module estimates treatment assignment from the observed covariates only.
Matching therefore balances measured covariates through their propensity score;
it does not prove that unobserved confounding has been eliminated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.causal.preprocessing import (
    BinaryTreatmentDataset,
    PreprocessedCampaign,
    preprocess_campaign,
)

SUPPORTED_DISCOUNTS: Final[frozenset[int]] = frozenset({10, 20})


class PropensityError(ValueError):
    """Raised when propensity estimation or matching cannot be performed."""


@dataclass(frozen=True)
class MatchingStatistics:
    """Summary of one-to-one nearest-neighbor matching."""

    candidate_treated_count: int
    candidate_control_count: int
    matched_pair_count: int
    unmatched_treated_count: int
    unmatched_control_count: int
    mean_absolute_score_distance: float | None
    max_absolute_score_distance: float | None


@dataclass(frozen=True)
class PropensityAnalysis:
    """Propensity model output for one discount versus no discount."""

    discount_amount: int
    model: LogisticRegression
    propensity_scores: pd.DataFrame
    treated_count: int
    control_count: int
    mean_propensity_treated: float
    mean_propensity_control: float
    overlap_statistics: dict[str, float | int | bool | None]
    matched_treatment_rows: pd.DataFrame
    matched_control_rows: pd.DataFrame
    matching_statistics: MatchingStatistics


def _get_dataset(
    campaign: pd.DataFrame | PreprocessedCampaign, discount_amount: int
) -> BinaryTreatmentDataset:
    if discount_amount not in SUPPORTED_DISCOUNTS:
        raise PropensityError(
            f"discount_amount must be 10 or 20; received {discount_amount}"
        )
    prepared = (
        campaign if isinstance(campaign, PreprocessedCampaign) else preprocess_campaign(campaign)
    )
    dataset = prepared.binary_datasets[discount_amount]
    if dataset.T.nunique() != 2:
        raise PropensityError(
            f"${discount_amount} contrast must contain both treated and control rows"
        )
    return dataset


def _overlap_statistics(scores: pd.DataFrame) -> dict[str, float | int | bool | None]:
    treated = scores.loc[scores["treatment"] == 1, "propensity_score"]
    control = scores.loc[scores["treatment"] == 0, "propensity_score"]
    lower = max(float(treated.min()), float(control.min()))
    upper = min(float(treated.max()), float(control.max()))
    in_support = scores["propensity_score"].between(lower, upper)
    return {
        "treated_min": float(treated.min()),
        "treated_max": float(treated.max()),
        "control_min": float(control.min()),
        "control_max": float(control.max()),
        "common_support_lower": lower,
        "common_support_upper": upper,
        "common_support_width": max(0.0, upper - lower),
        "in_common_support_count": int(in_support.sum()),
        "in_common_support_fraction": float(in_support.mean()),
        "has_common_support": bool(lower <= upper),
    }


def _match_nearest(
    scores: pd.DataFrame,
    caliper: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, MatchingStatistics]:
    if caliper is not None and (not np.isfinite(caliper) or caliper < 0):
        raise PropensityError("caliper must be a finite non-negative number or None")

    treated = scores.loc[scores["treatment"] == 1].copy()
    control = scores.loc[scores["treatment"] == 0].copy()
    available_controls = set(control.index)
    pairs: list[tuple[int, int, float]] = []
    for treated_index, treated_row in treated.sort_values("propensity_score").iterrows():
        if not available_controls:
            break
        candidates = control.loc[list(available_controls)]
        distances = (candidates["propensity_score"] - treated_row["propensity_score"]).abs()
        control_index = int(distances.idxmin())
        distance = float(distances.loc[control_index])
        if caliper is not None and distance > caliper:
            continue
        pairs.append((int(treated_index), control_index, distance))
        available_controls.remove(control_index)

    matched_treatment = treated.loc[[pair[0] for pair in pairs]].copy()
    matched_control = control.loc[[pair[1] for pair in pairs]].copy()
    for frame in (matched_treatment, matched_control):
        frame.reset_index(drop=True, inplace=True)
    distances = [pair[2] for pair in pairs]
    statistics = MatchingStatistics(
        candidate_treated_count=len(treated),
        candidate_control_count=len(control),
        matched_pair_count=len(pairs),
        unmatched_treated_count=len(treated) - len(pairs),
        unmatched_control_count=len(control) - len(pairs),
        mean_absolute_score_distance=float(np.mean(distances)) if distances else None,
        max_absolute_score_distance=float(np.max(distances)) if distances else None,
    )
    return matched_treatment, matched_control, statistics


def estimate_propensity(
    campaign: pd.DataFrame | PreprocessedCampaign,
    discount_amount: int,
    *,
    caliper: float | None = None,
    random_state: int = 42,
) -> PropensityAnalysis:
    """Fit logistic assignment scores and perform one-to-one score matching.

    ``campaign`` may be raw campaign data or an existing
    :class:`PreprocessedCampaign`. The returned ``propensity_scores`` table is
    ready for plotting and retains customer IDs.
    """

    dataset = _get_dataset(campaign, discount_amount)
    if not dataset.X.select_dtypes(include=[np.number]).shape[1] == dataset.X.shape[1]:
        raise PropensityError("Propensity features must be numeric")
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    try:
        model.fit(dataset.X, dataset.T)
    except (ValueError, TypeError) as exc:
        raise PropensityError(
            f"Could not fit propensity model for ${discount_amount} vs $0: {exc}"
        ) from exc

    scores = model.predict_proba(dataset.X)[:, 1]
    score_table = pd.DataFrame(
        {
            "customer_id": dataset.customer_id.to_numpy(),
            "treatment": dataset.T.to_numpy(),
            "treatment_group": np.where(dataset.T.to_numpy() == 1, "treated", "control"),
            "propensity_score": scores,
            "discount_amount": discount_amount,
        }
    )
    matched_treatment, matched_control, matching_statistics = _match_nearest(
        score_table, caliper
    )
    overlap = _overlap_statistics(score_table)
    return PropensityAnalysis(
        discount_amount=discount_amount,
        model=model,
        propensity_scores=score_table,
        treated_count=int(dataset.T.sum()),
        control_count=int((dataset.T == 0).sum()),
        mean_propensity_treated=float(scores[dataset.T.to_numpy() == 1].mean()),
        mean_propensity_control=float(scores[dataset.T.to_numpy() == 0].mean()),
        overlap_statistics=overlap,
        matched_treatment_rows=matched_treatment,
        matched_control_rows=matched_control,
        matching_statistics=matching_statistics,
    )


def analyze_propensity(
    campaign: pd.DataFrame | PreprocessedCampaign,
    *,
    caliper: float | None = None,
) -> dict[int, PropensityAnalysis]:
    """Run propensity analysis for both supported discount contrasts."""

    return {
        discount: estimate_propensity(campaign, discount, caliper=caliper)
        for discount in sorted(SUPPORTED_DISCOUNTS)
    }
