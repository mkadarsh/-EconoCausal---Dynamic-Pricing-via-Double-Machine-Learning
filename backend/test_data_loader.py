"""Tests for the data loader.

These tests require pandas; if pandas is not installed the test is skipped.
"""
import sys
from pathlib import Path
import pytest

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.data.synthetic_data import generate_sample_campaign


def test_load_csv_validates_and_returns_df(tmp_path):
    out = tmp_path / "sample_campaign.csv"
    generate_sample_campaign(str(out), n_customers=200, random_seed=42)

    # Lazy import pandas and data loader
    import pandas as pd
    from app.data.data_loader import load_csv

    df = load_csv(str(out))
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"customer_id", "treatment", "purchase", "revenue"}
    assert df.shape[0] == 200


def test_load_csv_rejects_invalid_treatment(tmp_path):
    out = tmp_path / "invalid.csv"
    generate_sample_campaign(str(out), n_customers=10, random_seed=1)
    lines = out.read_text(encoding="utf-8").splitlines()
    first_row = lines[1].split(",")
    first_row[8] = "99"
    first_row[9] = "99.0"
    lines[1] = ",".join(first_row)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    from app.data.data_loader import load_csv

    with pytest.raises(ValueError, match="Invalid treatment"):
        load_csv(out)


def test_load_csv_rejects_empty_and_non_finite_data(tmp_path):
    from app.data.data_loader import load_csv

    empty = tmp_path / "empty.csv"
    empty.write_text(
        "customer_id,age,income,previous_purchases,website_visits,customer_tenure,"
        "previous_spend,engagement_score,treatment,discount_amount,purchase,revenue\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one row"):
        load_csv(empty)

    out = tmp_path / "infinite.csv"
    generate_sample_campaign(str(out), n_customers=10, random_seed=1)
    frame = out.read_text(encoding="utf-8").replace(",0.0,", ",inf,", 1)
    out.write_text(frame, encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_csv(out)
