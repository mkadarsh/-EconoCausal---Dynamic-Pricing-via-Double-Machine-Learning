"""Exact discrete budget optimization for promotional discount assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, Bounds, milp
from scipy.sparse import lil_matrix

ACTION_COLUMNS = {
    0: ("baseline_purchase_probability", "ITE_0"),
    10: ("purchase_probability_10", "ITE_10"),
    20: ("purchase_probability_20", "ITE_20"),
}
REQUIRED_COLUMNS = (
    "customer_id",
    "baseline_purchase_probability",
    "purchase_probability_10",
    "purchase_probability_20",
    "ITE_10",
    "ITE_20",
    "expected_order_value",
    "discount_cost",
)
OUTPUT_COLUMNS = (
    "customer_id",
    "recommended_discount",
    "expected_purchase_probability",
    "expected_incremental_revenue",
    "discount_cost",
    "estimated_net_value",
)


class BudgetOptimizationError(ValueError):
    """Raised when an allocation cannot be validated or solved."""


@dataclass(frozen=True)
class CampaignSummary:
    total_budget: float
    budget_used: float
    remaining_budget: float
    expected_incremental_revenue: float
    expected_discount_cost: float
    number_targeted: int
    number_no_discount: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget": self.total_budget,
            "budget_used": self.budget_used,
            "remaining_budget": self.remaining_budget,
            "expected_incremental_revenue": self.expected_incremental_revenue,
            "expected_discount_cost": self.expected_discount_cost,
            "number_targeted": self.number_targeted,
            "number_no_discount": self.number_no_discount,
        }


@dataclass(frozen=True)
class OptimizationResult:
    recommendations: pd.DataFrame
    summary: CampaignSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendations": self.recommendations.to_dict(orient="records"),
            "summary": self.summary.to_dict(),
        }


def validate_budget(budget: float) -> float:
    """Validate and normalize a non-negative finite marketing budget."""

    try:
        value = float(budget)
    except (TypeError, ValueError) as exc:
        raise BudgetOptimizationError("budget must be a finite non-negative number") from exc
    if not np.isfinite(value) or value < 0:
        raise BudgetOptimizationError("budget must be a finite non-negative number")
    return value


def _validate_input(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise BudgetOptimizationError("campaign allocation data must contain customers")
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise BudgetOptimizationError(f"Allocation data is missing required columns: {missing}")
    if data["customer_id"].isna().any() or data["customer_id"].duplicated().any():
        raise BudgetOptimizationError("customer_id must be present and unique")
    numeric = data.loc[:, REQUIRED_COLUMNS[1:]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        bad = numeric.columns[numeric.isna().any()].tolist()
        raise BudgetOptimizationError(
            f"Allocation numeric columns contain missing, non-numeric, or non-finite values: {bad}"
        )
    if (numeric["expected_order_value"] < 0).any() or (numeric["discount_cost"] < 0).any():
        raise BudgetOptimizationError(
            "expected_order_value and discount_cost must be non-negative"
        )
    for column in ("discount_cost_10", "discount_cost_20"):
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all() or (values < 0).any():
                raise BudgetOptimizationError(f"{column} must be finite and non-negative")
    probabilities = numeric[
        [
            "baseline_purchase_probability",
            "purchase_probability_10",
            "purchase_probability_20",
        ]
    ]
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise BudgetOptimizationError("purchase probabilities must lie between 0 and 1")
    return data.copy().assign(**{column: numeric[column] for column in numeric.columns})


def _action_costs(row: pd.Series) -> dict[int, float]:
    """Support one common cost column and optional action-specific columns."""

    if {"discount_cost_10", "discount_cost_20"}.issubset(row.index):
        return {
            0: 0.0,
            10: float(row["discount_cost_10"]),
            20: float(row["discount_cost_20"]),
        }
    base_cost = float(row["discount_cost"])
    return {0: 0.0, 10: base_cost, 20: base_cost * 2.0}


def _build_options(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return costs, incremental revenue, and net value matrices (rows x actions)."""

    expected_value = data["expected_order_value"].to_numpy(dtype=float)
    baseline = data["baseline_purchase_probability"].to_numpy(dtype=float)
    probabilities = np.column_stack(
        [
            baseline,
            data["purchase_probability_10"].to_numpy(dtype=float),
            data["purchase_probability_20"].to_numpy(dtype=float),
        ]
    )
    effects = np.column_stack(
        [
            np.zeros(len(data)),
            data["ITE_10"].to_numpy(dtype=float),
            data["ITE_20"].to_numpy(dtype=float),
        ]
    )
    costs = np.array(
        [[_action_costs(row)[action] for action in (0, 10, 20)] for _, row in data.iterrows()]
    )
    incremental_revenue = effects * expected_value[:, None]
    net_value = incremental_revenue - costs
    return costs, incremental_revenue, net_value


def calculate_campaign_summary(
    recommendations: pd.DataFrame, total_budget: float
) -> CampaignSummary:
    """Calculate budget and value totals from a validated allocation table."""

    budget = validate_budget(total_budget)
    missing = sorted(set(OUTPUT_COLUMNS).difference(recommendations.columns))
    if missing:
        raise BudgetOptimizationError(f"Recommendations are missing columns: {missing}")
    costs = pd.to_numeric(recommendations["discount_cost"], errors="coerce")
    incremental = pd.to_numeric(
        recommendations["expected_incremental_revenue"], errors="coerce"
    )
    if costs.isna().any() or incremental.isna().any():
        raise BudgetOptimizationError("Recommendation cost and value columns must be numeric")
    used = float(costs.sum())
    if used > budget + 1e-8:
        raise BudgetOptimizationError("recommendation allocation exceeds the specified budget")
    targeted = recommendations["recommended_discount"].ne(0)
    return CampaignSummary(
        total_budget=budget,
        budget_used=used,
        remaining_budget=max(0.0, budget - used),
        expected_incremental_revenue=float(incremental.sum()),
        expected_discount_cost=used,
        number_targeted=int(targeted.sum()),
        number_no_discount=int((~targeted).sum()),
    )


def _greedy_allocate(
    campaign: pd.DataFrame,
    budget: float,
    max_customers: int | None,
) -> OptimizationResult:
    """Fast allocation fallback for large campaigns with no MILP stall."""

    costs, incremental_revenue, net_value = _build_options(campaign)
    probabilities = np.column_stack(
        [
            campaign["baseline_purchase_probability"].to_numpy(dtype=float),
            campaign["purchase_probability_10"].to_numpy(dtype=float),
            campaign["purchase_probability_20"].to_numpy(dtype=float),
        ]
    )
    target_limit = len(campaign) if max_customers is None else min(int(max_customers), len(campaign))
    actions = np.zeros(len(campaign), dtype=int)
    chosen_action_index = np.zeros(len(campaign), dtype=int)
    remaining = budget
    candidates: list[tuple[float, int, int]] = []
    for customer in range(len(campaign)):
        for action_index, action in ((1, 10), (2, 20)):
            action_cost = float(costs[customer, action_index])
            action_net = float(net_value[customer, action_index])
            if action_cost > 0 and action_net > 0:
                candidates.append((action_net / action_cost, customer, action_index))
    candidates.sort(key=lambda item: item[0], reverse=True)
    targeted = 0
    for _, customer, action_index in candidates:
        if targeted >= target_limit:
            break
        if actions[customer] != 0:
            continue
        action_cost = float(costs[customer, action_index])
        if action_cost <= remaining:
            chosen_action_index[customer] = action_index
            actions[customer] = (0, 10, 20)[action_index]
            remaining -= action_cost
            targeted += 1

    selected_costs = costs[np.arange(len(campaign)), chosen_action_index]
    recommendations = pd.DataFrame(
        {
            "customer_id": campaign["customer_id"].to_numpy(),
            "recommended_discount": actions,
            "expected_purchase_probability": probabilities[
                np.arange(len(campaign)), chosen_action_index
            ],
            "expected_incremental_revenue": incremental_revenue[
                np.arange(len(campaign)), chosen_action_index
            ],
            "discount_cost": selected_costs,
            "estimated_net_value": net_value[
                np.arange(len(campaign)), chosen_action_index
            ],
        }
    )
    return OptimizationResult(
        recommendations=recommendations,
        summary=calculate_campaign_summary(recommendations, budget),
    )


def optimize_campaign(
    data: pd.DataFrame, budget: float, max_customers: int | None = None
) -> OptimizationResult:
    """Allocate at most one discrete action per customer under ``budget``.

    The MILP has one binary variable for each customer/action pair. Its
    objective maximizes estimated net value, where incremental revenue is
    ``ITE * expected_order_value`` and net value subtracts the discount cost.
    The zero-discount action has zero cost and zero incremental value.
    """

    total_budget = validate_budget(budget)
    if max_customers is not None and (max_customers < 0 or int(max_customers) != max_customers):
        raise BudgetOptimizationError("max_customers must be a non-negative integer")
    campaign = _validate_input(data)
    costs, incremental_revenue, net_value = _build_options(campaign)
    customers, action_count = len(campaign), 3
    if customers > 1_000:
        return _greedy_allocate(campaign, total_budget, max_customers)
    variable_count = customers * action_count
    objective = -net_value.ravel()
    target_limit = customers if max_customers is None else min(int(max_customers), customers)
    if total_budget == 0:
        target_limit = 0
    if target_limit == 0:
        return OptimizationResult(
            recommendations=pd.DataFrame(
                {
                    "customer_id": campaign["customer_id"].to_numpy(),
                    "recommended_discount": np.zeros(customers, dtype=int),
                    "expected_purchase_probability": campaign[
                        "baseline_purchase_probability"
                    ].to_numpy(dtype=float),
                    "expected_incremental_revenue": np.zeros(customers, dtype=float),
                    "discount_cost": np.zeros(customers, dtype=float),
                    "estimated_net_value": np.zeros(customers, dtype=float),
                }
            ),
            summary=calculate_campaign_summary(
                pd.DataFrame(
                    {
                        "customer_id": campaign["customer_id"].to_numpy(),
                        "recommended_discount": np.zeros(customers, dtype=int),
                        "expected_purchase_probability": campaign[
                            "baseline_purchase_probability"
                        ].to_numpy(dtype=float),
                        "expected_incremental_revenue": np.zeros(customers, dtype=float),
                        "discount_cost": np.zeros(customers, dtype=float),
                        "estimated_net_value": np.zeros(customers, dtype=float),
                    }
                ),
                total_budget,
            ),
        )
    constraint_matrix = lil_matrix((customers + 2, variable_count), dtype=float)
    for customer in range(customers):
        constraint_matrix[customer, customer * action_count : (customer + 1) * action_count] = 1
    constraint_matrix[customers, :] = costs.ravel()
    constraint_matrix[customers + 1, 1::action_count] = 1
    constraint_matrix[customers + 1, 2::action_count] = 1
    lower = np.r_[np.ones(customers), -np.inf, -np.inf]
    upper = np.r_[np.ones(customers), total_budget, target_limit]
    try:
        solution = milp(
            c=objective,
            integrality=np.ones(variable_count),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(constraint_matrix.tocsr(), lower, upper),
            options={
                "disp": False,
                "presolve": True,
                "time_limit": 60.0,
                "mip_rel_gap": 0.01,
            },
        )
    except Exception as exc:
        raise BudgetOptimizationError(f"Discrete budget optimization failed: {exc}") from exc
    if solution.x is None:
        raise BudgetOptimizationError(
            f"Discrete budget optimization failed: {solution.message}"
        )

    selected = np.rint(solution.x).astype(int).reshape(customers, action_count)
    chosen_actions = np.array([0, 10, 20])[selected.argmax(axis=1)]
    chosen_index = selected.argmax(axis=1)
    recommendations = pd.DataFrame(
        {
            "customer_id": campaign["customer_id"].to_numpy(),
            "recommended_discount": chosen_actions,
            "expected_purchase_probability": np.take_along_axis(
                np.column_stack(
                    [
                        campaign["baseline_purchase_probability"],
                        campaign["purchase_probability_10"],
                        campaign["purchase_probability_20"],
                    ]
                ),
                chosen_index[:, None],
                axis=1,
            ).ravel(),
            "expected_incremental_revenue": np.take_along_axis(
                incremental_revenue, chosen_index[:, None], axis=1
            ).ravel(),
            "discount_cost": np.take_along_axis(costs, chosen_index[:, None], axis=1).ravel(),
            "estimated_net_value": np.take_along_axis(
                net_value, chosen_index[:, None], axis=1
            ).ravel(),
        },
        columns=OUTPUT_COLUMNS,
    )
    summary = calculate_campaign_summary(recommendations, total_budget)
    return OptimizationResult(recommendations=recommendations, summary=summary)
