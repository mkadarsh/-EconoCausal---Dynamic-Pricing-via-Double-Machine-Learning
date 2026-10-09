"""Double Machine Learning treatment-effect estimation with EconML.

Two separate binary DML models are fit:

* ``$10`` versus ``$0``
* ``$20`` versus ``$0``

The models adjust for the observed campaign covariates prepared by
``preprocess_campaign``. This supports conditional treatment-effect estimates,
but does not remove bias from unmeasured confounding, treatment interference,
poor overlap, or misspecified nuisance models. The estimates are not simple
treated/control averages or correlations: EconML's orthogonal DML estimator
residualizes treatment and outcome using cross-fitted nuisance models.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from app.causal.preprocessing import (
    BinaryTreatmentDataset,
    PreprocessedCampaign,
    preprocess_campaign,
)

SUPPORTED_DISCOUNTS: Final[frozenset[int]] = frozenset({10, 20})

try:
    from econml.dml import LinearDML
except ImportError:  # pragma: no cover - depends on optional environment dependency
    LinearDML = None


class DMLError(ValueError):
    """Raised when a DML model cannot be trained or queried."""


@dataclass(frozen=True)
class DMLResult:
    """A fitted DML model and customer-level effect outputs."""

    discount_amount: int
    model: Any
    customer_id: pd.Series
    ite: pd.Series
    ate: float
    ite_lower: pd.Series | None
    ite_upper: pd.Series | None
    baseline_purchase_probability: pd.Series
    treatment_purchase_probability: pd.Series
    ate_lower: float | None
    ate_upper: float | None
    treatment_count: int
    control_count: int
    feature_columns: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def ite_column(self) -> str:
        return f"ITE_{self.discount_amount}"

    @property
    def treatment_effect_distribution(self) -> dict[str, float]:
        values = self.ite.to_numpy(dtype=float)
        return {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        }

    @property
    def counterfactual_probabilities(self) -> pd.DataFrame:
        """Return customer-level predictive/counterfactual probability data."""

        return pd.DataFrame(
            {
                "customer_id": self.customer_id.to_numpy(),
                "baseline_purchase_probability": self.baseline_purchase_probability.to_numpy(),
                f"purchase_probability_{self.discount_amount}": (
                    self.treatment_purchase_probability.to_numpy()
                ),
                self.ite_column: self.ite.to_numpy(),
            }
        )


def _require_econml() -> Any:
    if LinearDML is None:
        raise ImportError(
            "EconML is required for DML estimation. "
            "Install backend requirements with: pip install -r backend\\requirements.txt"
        )
    return LinearDML


def _dataset(
    campaign: pd.DataFrame | PreprocessedCampaign, discount_amount: int
) -> tuple[PreprocessedCampaign, BinaryTreatmentDataset]:
    if discount_amount not in SUPPORTED_DISCOUNTS:
        raise DMLError(f"discount_amount must be 10 or 20; received {discount_amount}")
    prepared = (
        campaign if isinstance(campaign, PreprocessedCampaign) else preprocess_campaign(campaign)
    )
    dataset = prepared.binary_datasets[discount_amount]
    treatment_values = dataset.T.to_numpy()
    if dataset.X.empty or not np.isfinite(dataset.X.to_numpy()).all():
        raise DMLError(f"Features for ${discount_amount} contrast are empty or non-finite")
    if np.unique(treatment_values).size != 2:
        raise DMLError(
            f"${discount_amount} contrast requires both treated and control observations"
        )
    return prepared, dataset


def _intervals(model: Any, X: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not hasattr(model, "effect_interval"):
        return None, None
    try:
        lower, upper = model.effect_interval(X)
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None, None


def _ate_interval(model: Any) -> tuple[float | None, float | None]:
    if not hasattr(model, "ate_interval"):
        return None, None
    try:
        lower, upper = model.ate_interval()
        return float(np.asarray(lower).squeeze()), float(np.asarray(upper).squeeze())
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None, None


def _probability_pair(
    raw_baseline: np.ndarray, ite_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep baseline and baseline-plus-effect inside [0, 1] exactly."""

    lower = np.maximum(0.0, -ite_values)
    upper = np.minimum(1.0, 1.0 - ite_values)
    if np.any(lower > upper):
        raise DMLError(
            "DML effects cannot be combined with a valid probability baseline; "
            "at least one individual effect lies outside [-1, 1]"
        )
    baseline = np.clip(raw_baseline, lower, upper)
    return baseline, baseline + ite_values


def train_dml_model(
    campaign: pd.DataFrame | PreprocessedCampaign,
    discount_amount: int,
    *,
    random_state: int = 42,
    cv: int = 3,
    n_estimators: int = 200,
) -> DMLResult:
    """Fit EconML ``LinearDML`` and estimate ITEs for every campaign customer."""

    if cv < 2:
        raise DMLError("cv must be at least 2")
    prepared, dataset = _dataset(campaign, discount_amount)
    if min(int(dataset.T.sum()), int((dataset.T == 0).sum())) < cv:
        raise DMLError(
            f"${discount_amount} contrast has too few observations per arm for cv={cv}"
        )
    if n_estimators < 10:
        raise DMLError("n_estimators must be at least 10")

    estimator_class = _require_econml()
    model_y = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model_t = LogisticRegression(max_iter=1000, random_state=random_state)
    baseline_model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    model = estimator_class(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=cv,
        random_state=random_state,
    )
    try:
        model.fit(
            dataset.Y.to_numpy(dtype=float),
            dataset.T.to_numpy(dtype=int),
            X=dataset.X,
        )
        raw_ite_values = np.asarray(model.effect(prepared.X), dtype=float).reshape(-1)
        ite_values = np.clip(raw_ite_values, -1.0, 1.0)
        ate = float(np.mean(ite_values))
        control_mask = dataset.T.to_numpy(dtype=int) == 0
        if np.unique(dataset.Y.to_numpy()[control_mask]).size < 2:
            raise DMLError(
                f"${discount_amount} control outcomes must contain both purchase classes "
                "to estimate baseline probabilities"
            )
        baseline_model.fit(
            dataset.X.loc[control_mask],
            dataset.Y.to_numpy(dtype=int)[control_mask],
        )
        raw_baseline_values = baseline_model.predict_proba(prepared.X)[:, 1]
        baseline_values, treatment_probability_values = _probability_pair(
            raw_baseline_values, ite_values
        )
    except (ValueError, TypeError, RuntimeError) as exc:
        raise DMLError(
            f"EconML could not fit the ${discount_amount} vs $0 DML model: {exc}"
        ) from exc

    lower, upper = _intervals(model, prepared.X)
    ate_lower, ate_upper = _ate_interval(model)
    metadata = {
        "estimator": estimator_class.__name__,
        "outcome": "purchase",
        "treatment": f"${discount_amount} discount vs $0",
        "nuisance_model_y": model_y.__class__.__name__,
        "nuisance_model_t": model_t.__class__.__name__,
        "baseline_probability_model": baseline_model.__class__.__name__,
        "probability_approximation": (
            "baseline is a control-arm predictive probability; treatment probability "
            "is baseline plus EconML ITE; baseline is constrained so both remain in [0, 1]"
        ),
        "crossfit_folds": cv,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "features": list(prepared.feature_columns),
        "ite_probability_clipped_count": int(np.sum(np.not_equal(raw_ite_values, ite_values))),
        "observed_confounding_adjustment": True,
    }
    return DMLResult(
        discount_amount=discount_amount,
        model=model,
        customer_id=prepared.customer_id.copy(),
        ite=pd.Series(ite_values, name=f"ITE_{discount_amount}"),
        ate=ate,
        ite_lower=(
            pd.Series(lower, name=f"ITE_{discount_amount}_lower") if lower is not None else None
        ),
        ite_upper=(
            pd.Series(upper, name=f"ITE_{discount_amount}_upper") if upper is not None else None
        ),
        baseline_purchase_probability=pd.Series(
            baseline_values, name="baseline_purchase_probability"
        ),
        treatment_purchase_probability=pd.Series(
            treatment_probability_values,
            name=f"purchase_probability_{discount_amount}",
        ),
        ate_lower=ate_lower,
        ate_upper=ate_upper,
        treatment_count=int(dataset.T.sum()),
        control_count=int((dataset.T == 0).sum()),
        feature_columns=prepared.feature_columns,
        metadata=metadata,
    )


def estimate_ite(
    model_result: DMLResult, X: pd.DataFrame | None = None
) -> pd.Series:
    """Estimate ITEs from a fitted result, optionally for a new feature matrix."""

    if not isinstance(model_result, DMLResult):
        raise TypeError("model_result must be a DMLResult")
    features = X
    if features is None:
        return model_result.ite.copy()
    missing = sorted(set(model_result.feature_columns) - set(features.columns))
    if missing:
        raise DMLError(f"X is missing required feature columns: {missing}")
    try:
        values = np.asarray(model_result.model.effect(features[list(model_result.feature_columns)]))
    except (ValueError, TypeError, RuntimeError) as exc:
        raise DMLError(f"Could not estimate individual treatment effects: {exc}") from exc
    return pd.Series(values.reshape(-1), name=model_result.ite_column, index=features.index)


def estimate_ate(model_result: DMLResult) -> float:
    """Return the fitted model's average treatment effect."""

    if not isinstance(model_result, DMLResult):
        raise TypeError("model_result must be a DMLResult")
    return model_result.ate


def get_model_summary(model_result: DMLResult) -> dict[str, Any]:
    """Return serializable metadata and effect summaries for API use."""

    if not isinstance(model_result, DMLResult):
        raise TypeError("model_result must be a DMLResult")
    return {
        **model_result.metadata,
        "discount_amount": model_result.discount_amount,
        "treatment_count": model_result.treatment_count,
        "control_count": model_result.control_count,
        "ate": model_result.ate,
        "ate_lower": model_result.ate_lower,
        "ate_upper": model_result.ate_upper,
        "ite_column": model_result.ite_column,
        "treatment_effect_distribution": model_result.treatment_effect_distribution,
        "probability_columns": [
            "baseline_purchase_probability",
            f"purchase_probability_{model_result.discount_amount}",
        ],
    }


def train_dml_models(
    campaign: pd.DataFrame | PreprocessedCampaign,
    *,
    random_state: int = 42,
    cv: int = 3,
    n_estimators: int = 200,
) -> tuple[DMLResult, DMLResult, pd.DataFrame]:
    """Fit both contrasts and return a customer-mappable ITE table."""

    result_10 = train_dml_model(
        campaign, 10, random_state=random_state, cv=cv, n_estimators=n_estimators
    )
    result_20 = train_dml_model(
        campaign, 20, random_state=random_state, cv=cv, n_estimators=n_estimators
    )
    raw_baseline = result_10.baseline_purchase_probability.to_numpy()
    ite_10 = result_10.ite.to_numpy()
    ite_20 = result_20.ite.to_numpy()
    lower = np.maximum.reduce([np.zeros_like(raw_baseline), -ite_10, -ite_20])
    upper = np.minimum.reduce([np.ones_like(raw_baseline), 1.0 - ite_10, 1.0 - ite_20])
    if np.any(lower > upper):
        raise DMLError(
            "DML effects cannot be combined with a shared probability baseline; "
            "at least one customer has an effect outside [-1, 1]"
        )
    shared_baseline = np.clip(raw_baseline, lower, upper)
    result_10 = replace(
        result_10,
        baseline_purchase_probability=pd.Series(
            shared_baseline, name="baseline_purchase_probability"
        ),
        treatment_purchase_probability=pd.Series(
            shared_baseline + result_10.ite.to_numpy(), name="purchase_probability_10"
        ),
    )
    result_20 = replace(
        result_20,
        baseline_purchase_probability=pd.Series(
            shared_baseline, name="baseline_purchase_probability"
        ),
        treatment_purchase_probability=pd.Series(
            shared_baseline + result_20.ite.to_numpy(), name="purchase_probability_20"
        ),
    )
    ite_table = pd.DataFrame(
        {
            "customer_id": result_10.customer_id,
            "baseline_purchase_probability": result_10.baseline_purchase_probability.to_numpy(),
            "purchase_probability_10": result_10.treatment_purchase_probability.to_numpy(),
            "purchase_probability_20": result_20.treatment_purchase_probability.to_numpy(),
            "ITE_10": result_10.ite.to_numpy(),
            "ITE_20": result_20.ite.to_numpy(),
        }
    )
    return result_10, result_20, ite_table
