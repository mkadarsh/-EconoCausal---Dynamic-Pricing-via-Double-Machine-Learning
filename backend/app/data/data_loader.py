"""Data loader and validator for campaign CSV files.

load_csv(path: str) -> pandas.DataFrame
- Validates required columns
- Detects missing values
- Validates treatment values are in {0,10,20}
- Validates numeric columns are numeric (attempts conversion)

Raises informative errors when validations fail.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "income",
    "previous_purchases",
    "website_visits",
    "customer_tenure",
    "previous_spend",
    "engagement_score",
    "treatment",
    "discount_amount",
    "purchase",
    "revenue",
]


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV at path and return a validated pandas DataFrame.

    Raises ValueError for validation problems.
    """
    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Campaign data must contain at least one row")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Detect missing values
    if df[REQUIRED_COLUMNS].isnull().any(axis=None):
        null_counts = df[REQUIRED_COLUMNS].isnull().sum()
        raise ValueError(f"Missing values detected in columns: {null_counts[null_counts>0].to_dict()}")
    if df["customer_id"].astype(str).str.strip().eq("").any() or df["customer_id"].duplicated().any():
        raise ValueError("customer_id must contain unique, non-blank identifiers")

    # Validate numeric columns can be converted
    numeric_cols = [
        "treatment",
        "age",
        "income",
        "previous_purchases",
        "website_visits",
        "customer_tenure",
        "previous_spend",
        "engagement_score",
        "discount_amount",
        "purchase",
        "revenue",
    ]
    for col in numeric_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            invalid_rows = converted[converted.isna()].index.tolist()
            raise ValueError(f"Column {col} contains non-numeric values at rows: {invalid_rows[:5]}")
        df[col] = converted

    if not np.isfinite(df[numeric_cols].to_numpy(dtype=float)).all():
        raise ValueError("Numeric columns must contain finite values")

    # Validate treatment values after numeric coercion so CSV values such as
    # "10" are treated consistently with numeric input.
    allowed = {0, 10, 20}
    unique_t = set(df["treatment"].unique())
    if not unique_t.issubset(allowed):
        raise ValueError(f"Invalid treatment values found: {unique_t - allowed}")
    if not df["purchase"].isin({0, 1}).all():
        raise ValueError("purchase must contain only 0 or 1")
    if not (df["discount_amount"] == df["treatment"]).all():
        raise ValueError("discount_amount must match treatment")

    return df
