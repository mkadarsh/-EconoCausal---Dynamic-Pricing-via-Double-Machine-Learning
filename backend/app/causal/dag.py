"""DoWhy causal graph and estimand helpers for campaign analysis.

The campaign is observational. Customer attributes are treated as observed
pre-treatment confounders because they can influence both promotion assignment
and purchase propensity. The graph does not claim that these attributes are
the only causes of either variable.

Treatment:
    A binary contrast produced by preprocessing: ``T10`` (10 dollars versus
    no discount) or ``T20`` (20 dollars versus no discount).
Outcome:
    ``purchase`` for the MVP causal analysis. ``revenue`` can be requested for
    downstream revenue analysis when it is present in the supplied data.
Observed confounders:
    age, income, previous_purchases, website_visits, customer_tenure,
    previous_spend, and engagement_score.

The assumptions are limited: these confounders are measured before treatment,
their relevant common causes are observed, treatment is well-defined, and
there is sufficient overlap between treated and untreated customers. The
observational synthetic data cannot establish that no unmeasured confounding
exists, and its generated relationships are not evidence about a real market.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from app.causal.preprocessing import FEATURE_COLUMNS

try:
    from dowhy import CausalModel
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    CausalModel = None

DEFAULT_TREATMENT: Final[str] = "treatment"
DEFAULT_OUTCOME: Final[str] = "purchase"
REVENUE_COLUMN: Final[str] = "revenue"
SUPPORTED_BINARY_TREATMENTS: Final[frozenset[str]] = frozenset({"T10", "T20"})


class CausalGraphError(ValueError):
    """Raised when a graph, model, or estimand cannot be constructed."""


def create_causal_graph(
    treatment: str = DEFAULT_TREATMENT,
    outcome: str = DEFAULT_OUTCOME,
) -> str:
    """Return a DOT graph for one binary discount contrast.

    ``treatment`` is normally ``T10`` or ``T20`` after preprocessing. The
    default name ``treatment`` is useful when constructing a generic graph.
    """

    if not isinstance(treatment, str) or not treatment.strip():
        raise CausalGraphError("treatment must be a non-empty column name")
    if not isinstance(outcome, str) or not outcome.strip():
        raise CausalGraphError("outcome must be a non-empty column name")

    confounder_edges = "\n".join(
        f'    "{confounder}" -> "{treatment}";\n'
        f'    "{confounder}" -> "purchase";'
        for confounder in FEATURE_COLUMNS
    )
    outcome_edges = (
        f'    "{treatment}" -> "purchase";\n'
        f'    "purchase" -> "revenue";'
        if outcome == "purchase"
        else f'    "{treatment}" -> "purchase";\n'
        f'    "purchase" -> "{outcome}";'
    )
    return (
        "digraph EconoCausal {\n"
        "    graph [bb=\"0,0,1,1\"];\n"
        "    rankdir=LR;\n"
        f"{confounder_edges}\n"
        f"{outcome_edges}\n"
        "}\n"
    )


def _require_dowhy() -> type:
    if CausalModel is None:
        raise ImportError(
            "DoWhy is required for causal graph modeling. "
            "Install backend requirements with: pip install -r backend\\requirements.txt"
        )
    return CausalModel


def _validate_inputs(data: pd.DataFrame, treatment: str, outcome: str) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise CausalGraphError("data must contain at least one row")
    required_columns = (*FEATURE_COLUMNS, treatment, outcome)
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise CausalGraphError(
            f"Data is missing required causal columns or observed confounders: {missing}"
        )
    values = set(data[treatment].dropna().unique())
    if values != {0, 1}:
        raise CausalGraphError(
            f"Treatment column '{treatment}' must be binary with both 0 and 1 present; "
            f"found {sorted(values)}. Use preprocessing T10 or T20 for a discount contrast."
        )
    if data[treatment].isna().any() or data[outcome].isna().any():
        raise CausalGraphError(
            f"Treatment '{treatment}' and outcome '{outcome}' cannot contain missing values"
        )


def get_dowhy_model(
    data: pd.DataFrame,
    treatment: str = DEFAULT_TREATMENT,
    outcome: str = DEFAULT_OUTCOME,
):
    """Construct a DoWhy ``CausalModel`` for a validated binary contrast."""

    _validate_inputs(data, treatment, outcome)
    model_class = _require_dowhy()
    graph = create_causal_graph(treatment=treatment, outcome=outcome)
    try:
        return model_class(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )
    except Exception as exc:
        raise CausalGraphError(
            f"DoWhy could not construct the causal model for "
            f"'{treatment}' -> '{outcome}': {exc}"
        ) from exc


def identify_estimand(model, *, proceed_when_unidentifiable: bool = False):
    """Identify the causal estimand and provide a useful failure message."""

    if model is None or not hasattr(model, "identify_effect"):
        raise TypeError("model must be a DoWhy CausalModel")
    try:
        estimand = model.identify_effect(
            proceed_when_unidentifiable=proceed_when_unidentifiable
        )
    except Exception as exc:
        raise CausalGraphError(f"DoWhy could not identify the requested estimand: {exc}") from exc
    if estimand is None:
        raise CausalGraphError("DoWhy returned no estimand for the requested treatment and outcome")
    return estimand
