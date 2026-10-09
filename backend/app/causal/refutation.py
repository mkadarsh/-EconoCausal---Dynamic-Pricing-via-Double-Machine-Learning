"""DoWhy refutation audit for the two observational discount contrasts.

Refutation tests challenge an identified estimate under alternative data
perturbations. They are diagnostic checks, not mathematical proofs that the
DML or DoWhy causal assumptions are true. This module uses DoWhy's own
backdoor estimate as the refutation target because a fitted EconML estimator
is not universally accepted by DoWhy's ``refute_estimate`` API across library
versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd

from app.causal.dag import get_dowhy_model
from app.causal.preprocessing import FEATURE_COLUMNS, preprocess_campaign

SUPPORTED_DISCOUNTS: Final[frozenset[int]] = frozenset({10, 20})
REFUTER_METHODS: Final[tuple[tuple[str, str], ...]] = (
    ("placebo_treatment_refuter", "Placebo treatment"),
    ("random_common_cause", "Random common cause"),
    ("data_subset_refuter", "Data subset"),
)


class RefutationError(ValueError):
    """Raised when refutation input or configuration is invalid."""


@dataclass(frozen=True)
class RefutationResult:
    """UI-ready result for one DoWhy refutation attempt."""

    test_name: str
    estimated_effect: float | None
    refuted_effect: float | None
    passed: bool | None
    interpretation: str
    limitations: str
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "estimated_effect": self.estimated_effect,
            "refuted_effect": self.refuted_effect,
            "passed": self.passed,
            "interpretation": self.interpretation,
            "limitations": self.limitations,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class RefutationAudit:
    """All refutation results for one discount comparison."""

    discount_amount: int
    results: tuple[RefutationResult, ...]
    estimation_method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "discount_amount": self.discount_amount,
            "estimation_method": self.estimation_method,
            "results": [result.to_dict() for result in self.results],
        }


def _require_dowhy() -> Any:
    try:
        from dowhy import CausalModel
    except ImportError as exc:
        raise ImportError(
            "DoWhy is required for refutation audits. "
            "Install backend requirements with: pip install -r backend\\requirements.txt"
        ) from exc
    return CausalModel


def _binary_data(data: pd.DataFrame, discount_amount: int) -> pd.DataFrame:
    if discount_amount not in SUPPORTED_DISCOUNTS:
        raise RefutationError(
            f"discount_amount must be 10 or 20; received {discount_amount}"
        )
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    preprocess_campaign(data)
    source = data.loc[
        data["treatment"].isin({0, discount_amount}),
        [*FEATURE_COLUMNS, "treatment", "purchase"],
    ].copy()
    source["binary_treatment"] = (
        source["treatment"].astype(int) == discount_amount
    ).astype(int)
    source.drop(columns=["treatment"], inplace=True)
    if source["binary_treatment"].nunique() != 2:
        raise RefutationError(
            f"${discount_amount} vs $0 requires both treated and control observations"
        )
    if source["purchase"].nunique() != 2:
        raise RefutationError(
            f"${discount_amount} vs $0 requires both purchase outcomes"
        )
    return source.reset_index(drop=True)


def _number(value: Any) -> float | None:
    try:
        number = float(np.asarray(value).squeeze())
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _refuted_effect(refutation: Any) -> float | None:
    for name in ("new_effect", "refuted_effect"):
        if hasattr(refutation, name):
            value = _number(getattr(refutation, name))
            if value is not None:
                return value
    return None


def _estimate_effect(model: Any, estimand: Any) -> Any:
    return model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        test_significance=True,
    )


def _run_refuter(
    estimate: Any,
    method: str,
    *,
    num_simulations: int,
    subset_fraction: float,
) -> Any:
    kwargs: dict[str, Any] = {"refutation_method": method, "num_simulations": num_simulations}
    if method == "placebo_treatment_refuter":
        kwargs["placebo_type"] = "permute"
    elif method == "data_subset_refuter":
        kwargs["subset_fraction"] = subset_fraction
    return estimate.refute_estimate(**kwargs)


def audit_refutations(
    data: pd.DataFrame,
    discount_amount: int,
    *,
    num_simulations: int = 10,
    subset_fraction: float = 0.8,
) -> RefutationAudit:
    """Run DoWhy placebo, random-common-cause, and subset refuters.

    A completed test is marked passed only when DoWhy's refuter reports a
    non-refutation. Unsupported/failing version integrations are returned as
    ``status='unavailable'`` with ``passed=None`` so a UI cannot mistake them
    for successful evidence.
    """

    if num_simulations < 1:
        raise RefutationError("num_simulations must be at least 1")
    if not 0 < subset_fraction < 1:
        raise RefutationError("subset_fraction must be between 0 and 1")
    binary = _binary_data(data, discount_amount)
    try:
        _require_dowhy()
    except ImportError as exc:
        limitations = (
            "DoWhy is not installed; no refutation was run. Refutation tests "
            "are diagnostics, not proof of causal validity."
        )
        return RefutationAudit(
            discount_amount=discount_amount,
            estimation_method="DoWhy backdoor.linear_regression",
            results=tuple(
                RefutationResult(
                    test_name=test_name,
                    estimated_effect=None,
                    refuted_effect=None,
                    passed=None,
                    interpretation="Refutation unavailable in the current environment.",
                    limitations=limitations,
                    status="unavailable",
                    error=str(exc),
                )
                for _, test_name in REFUTER_METHODS
            ),
        )
    model = get_dowhy_model(
        binary,
        treatment="binary_treatment",
        outcome="purchase",
    )
    try:
        from app.causal.dag import identify_estimand

        estimand = identify_estimand(model)
        estimate = _estimate_effect(model, estimand)
    except Exception as exc:
        limitations = (
            "The DoWhy identification/estimation step failed, so no refutation "
            "was run. This is not evidence for or against the causal model."
        )
        return RefutationAudit(
            discount_amount=discount_amount,
            estimation_method="DoWhy backdoor.linear_regression",
            results=tuple(
                RefutationResult(
                    test_name=test_name,
                    estimated_effect=None,
                    refuted_effect=None,
                    passed=None,
                    interpretation="The DoWhy audit could not construct its reference estimate.",
                    limitations=limitations,
                    status="unavailable",
                    error=str(exc),
                )
                for _, test_name in REFUTER_METHODS
            ),
        )

    estimated_effect = _number(getattr(estimate, "value", estimate))
    results: list[RefutationResult] = []
    limitations = (
        "Diagnostic only: this perturbation test cannot prove causal validity, "
        "rule out unmeasured confounding, or establish DML superiority."
    )
    for method, test_name in REFUTER_METHODS:
        try:
            refutation = _run_refuter(
                estimate,
                method,
                num_simulations=num_simulations,
                subset_fraction=subset_fraction,
            )
            refuted_effect = _refuted_effect(refutation)
            passed_value = getattr(refutation, "refutation_result", None)
            passed = passed_value if isinstance(passed_value, bool) else None
            results.append(
                RefutationResult(
                    test_name=test_name,
                    estimated_effect=estimated_effect,
                    refuted_effect=refuted_effect,
                    passed=passed,
                    interpretation=(
                        "DoWhy completed the refuter; a pass means the estimate was "
                        "not materially contradicted by this perturbation."
                        if passed is not False
                        else "DoWhy reported evidence against the original estimate under this perturbation."
                    ),
                    limitations=limitations,
                )
            )
        except Exception as exc:
            results.append(
                RefutationResult(
                    test_name=test_name,
                    estimated_effect=estimated_effect,
                    refuted_effect=None,
                    passed=None,
                    interpretation="This refuter could not be executed with the installed DoWhy workflow.",
                    limitations=limitations,
                    status="unavailable",
                    error=str(exc),
                )
            )
    return RefutationAudit(
        discount_amount=discount_amount,
        results=tuple(results),
        estimation_method="DoWhy backdoor.linear_regression",
    )


def audit_all_refutations(
    data: pd.DataFrame,
    *,
    num_simulations: int = 10,
    subset_fraction: float = 0.8,
) -> dict[int, RefutationAudit]:
    """Run the audit for both supported binary discount comparisons."""

    return {
        discount: audit_refutations(
            data,
            discount,
            num_simulations=num_simulations,
            subset_fraction=subset_fraction,
        )
        for discount in sorted(SUPPORTED_DISCOUNTS)
    }
