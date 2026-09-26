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

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
        ite_values = np.asarray(model.effect(prepared.X), dtype=float).reshape(-1)
        ate = float(np.mean(ite_values))
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
        "crossfit_folds": cv,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "features": list(prepared.feature_columns),
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
    ite_table = pd.DataFrame(
        {
            "customer_id": result_10.customer_id,
            "ITE_10": result_10.ite.to_numpy(),
            "ITE_20": result_20.ite.to_numpy(),
        }
    )
    return result_10, result_20, ite_table
