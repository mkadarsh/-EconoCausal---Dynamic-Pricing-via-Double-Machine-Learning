"""Prepare campaign observations for binary-treatment causal estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

OUTCOME_COLUMN: Final[str] = "purchase"
TREATMENT_COLUMN: Final[str] = "treatment"
IDENTIFIER_COLUMN: Final[str] = "customer_id"
ALLOWED_TREATMENTS: Final[frozenset[int]] = frozenset({0, 10, 20})
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "age",
    "income",
    "previous_purchases",
    "website_visits",
    "customer_tenure",
    "previous_spend",
    "engagement_score",
)


class PreprocessingError(ValueError):
    """Raised when campaign data cannot be prepared for causal analysis."""


@dataclass(frozen=True)
class BinaryTreatmentDataset:
    """One binary contrast and the customer rows used for that contrast."""

    discount_amount: int
    customer_id: pd.Series
    X: pd.DataFrame
    T: pd.Series
    Y: pd.Series

    @property
    def treatment_name(self) -> str:
        return f"${self.discount_amount} vs $0"


@dataclass(frozen=True)
class PreprocessedCampaign:
    """Prepared campaign data shared by the two binary causal analyses.

    ``X``, ``T10``, ``T20``, and ``Y`` retain every validated campaign row.
    ``binary_datasets`` contains the correctly filtered rows for fitting each
    contrast, excluding the other non-control discount.
    """

    customer_id: pd.Series
    X: pd.DataFrame
    T10: pd.Series
    T20: pd.Series
    Y: pd.Series
    feature_columns: tuple[str, ...]
    binary_datasets: dict[int, BinaryTreatmentDataset]
    imputer: SimpleImputer
    scaler: StandardScaler


def _validate_columns(campaign: pd.DataFrame) -> None:
    required = {IDENTIFIER_COLUMN, OUTCOME_COLUMN, TREATMENT_COLUMN, *FEATURE_COLUMNS}
    missing = sorted(required.difference(campaign.columns))
    if missing:
        raise PreprocessingError(f"Campaign data is missing required columns: {missing}")


def _numeric_series(campaign: pd.DataFrame, column: str) -> pd.Series:
    converted = pd.to_numeric(campaign[column], errors="coerce")
    invalid = converted.isna() & campaign[column].notna()
    if invalid.any():
        rows = converted.index[invalid].tolist()[:5]
        raise PreprocessingError(
            f"Column '{column}' contains non-numeric values at rows: {rows}"
        )
    if not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
        raise PreprocessingError(f"Column '{column}' contains non-finite values")
    return converted


def preprocess_campaign(campaign: pd.DataFrame) -> PreprocessedCampaign:
    """Validate and transform a campaign DataFrame for causal modeling.

    Numeric feature missing values are median-imputed and then standardized.
    Customer identifiers are kept separately and never included in ``X``.
    Treatment indicators are represented on all rows by ``T10`` and ``T20``;
    use ``binary_datasets[10]`` and ``binary_datasets[20]`` for estimator
    training so the competing discount is not treated as a control.
    """

    if not isinstance(campaign, pd.DataFrame):
        raise TypeError("campaign must be a pandas DataFrame")
    if campaign.empty:
        raise PreprocessingError("Campaign data must contain at least one row")
    _validate_columns(campaign)

    customer_id = campaign[IDENTIFIER_COLUMN].copy()
    if customer_id.isna().any() or customer_id.astype(str).str.strip().eq("").any():
        raise PreprocessingError("Column 'customer_id' contains missing or blank identifiers")
    if customer_id.duplicated().any():
        raise PreprocessingError("Column 'customer_id' must contain unique identifiers")

    treatment = _numeric_series(campaign, TREATMENT_COLUMN)
    if treatment.isna().any():
        raise PreprocessingError("Column 'treatment' contains missing values")
    if not treatment.isin(ALLOWED_TREATMENTS).all():
        invalid = sorted(set(treatment.dropna().astype(int)) - ALLOWED_TREATMENTS)
        raise PreprocessingError(
            f"Invalid treatment values found: {invalid}; expected only 0, 10, or 20"
        )
    treatment = treatment.astype(int)

    outcome = _numeric_series(campaign, OUTCOME_COLUMN)
    if outcome.isna().any():
        raise PreprocessingError("Column 'purchase' contains missing values")
    if not outcome.isin({0, 1}).all():
        raise PreprocessingError("Column 'purchase' must contain only binary values 0 or 1")
    outcome = outcome.astype(int)

    numeric_features = pd.DataFrame(
        {column: _numeric_series(campaign, column) for column in FEATURE_COLUMNS},
        index=campaign.index,
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(numeric_features))
    X = pd.DataFrame(transformed, columns=FEATURE_COLUMNS, index=campaign.index)

    T10 = (treatment == 10).rename("T10")
    T20 = (treatment == 20).rename("T20")
    binary_datasets: dict[int, BinaryTreatmentDataset] = {}
    for discount in (10, 20):
        mask = treatment.isin({0, discount})
        binary_datasets[discount] = BinaryTreatmentDataset(
            discount_amount=discount,
            customer_id=customer_id.loc[mask].reset_index(drop=True),
            X=X.loc[mask].reset_index(drop=True),
            T=(treatment.loc[mask] == discount).astype(int).rename("treatment").reset_index(
                drop=True
            ),
            Y=outcome.loc[mask].rename("purchase").reset_index(drop=True),
        )

    return PreprocessedCampaign(
        customer_id=customer_id.reset_index(drop=True),
        X=X.reset_index(drop=True),
        T10=T10.reset_index(drop=True),
        T20=T20.reset_index(drop=True),
        Y=outcome.rename("purchase").reset_index(drop=True),
        feature_columns=FEATURE_COLUMNS,
        binary_datasets=binary_datasets,
        imputer=imputer,
        scaler=scaler,
    )
