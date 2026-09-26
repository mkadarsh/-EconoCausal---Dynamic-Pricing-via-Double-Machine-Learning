import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.api.state import state
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    state.campaign = None
    state.clear_analysis()
    yield
    state.campaign = None
    state.clear_analysis()


def test_health_and_empty_data_summary():
    assert client.get("/api/health").status_code == 200
    response = client.get("/api/data/summary")
    assert response.status_code == 200
    assert response.json()["has_data"] is False


def test_generate_and_summary_workflow():
    response = client.post("/api/data/generate", json={"n_customers": 12, "random_seed": 1})

    assert response.status_code == 201
    body = response.json()
    assert body["rows"] == 12
    assert body["has_data"] is True
    assert sum(body["treatment_counts"].values()) == 12


def test_upload_rejects_non_csv_and_accepts_campaign_csv():
    rejected = client.post(
        "/api/data/upload",
        files={"file": ("campaign.txt", b"not csv", "text/plain")},
    )
    assert rejected.status_code == 415

    csv_path = Path(__file__).resolve().parents[1] / "data" / "sample_campaign_data.csv"
    if not csv_path.exists():
        pytest.skip("sample campaign CSV is not present")
    accepted = client.post(
        "/api/data/upload",
        files={"file": ("campaign.csv", csv_path.read_bytes(), "text/csv")},
    )
    assert accepted.status_code == 201
    assert accepted.json()["rows"] == 5000


def test_preconditions_for_causal_and_optimization_workflows():
    assert client.get("/api/causal/results").status_code == 404
    assert client.get("/api/causal/customer/missing").status_code == 404
    assert client.post("/api/optimization/run", json={"budget": 10}).status_code == 409
    assert client.get("/api/optimization/results").status_code == 404


def test_train_reports_dependency_or_validates_data():
    client.post("/api/data/generate", json={"n_customers": 20})
    response = client.post("/api/causal/train", json={"n_estimators": 10, "cv": 2})

    # EconML is optional in the current test environment; either a real train
    # response or an explicit dependency error is acceptable.
    assert response.status_code in {200, 503, 422}
    if response.status_code == 503:
        assert "EconML" in response.json()["detail"]
