"""Tests for the synthetic data generator.

These tests are minimal and avoid requiring pandas so they can run in
lightweight environments. They check that the generator creates a CSV with
expected columns and row count.
"""
import sys
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.data.synthetic_data import REQUIRED_COLUMNS, generate_sample_campaign


def test_generator_creates_csv(tmp_path):
    out = tmp_path / "sample_campaign.csv"
    path = generate_sample_campaign(str(out), n_customers=5000, random_seed=123)
    assert Path(path).exists(), "CSV file was not created"

    # Quick header and row count check without loading the full dataset.
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        assert header == REQUIRED_COLUMNS
        rows = sum(1 for _ in fh)
        assert rows == 5000
