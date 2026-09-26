"""Dataset lifecycle endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.schemas.api import DataSummary, GenerateDataRequest
from app.api.state import state
from app.data.data_loader import load_csv
from app.data.synthetic_data import generate_sample_campaign

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


def _summary() -> DataSummary:
    if state.campaign is None:
        return DataSummary(rows=0, columns=[], treatment_counts={}, purchase_rate=0, has_data=False)
    campaign = state.campaign
    counts = campaign["treatment"].value_counts().sort_index().astype(int).to_dict()
    return DataSummary(
        rows=len(campaign),
        columns=list(campaign.columns),
        treatment_counts={str(key): value for key, value in counts.items()},
        purchase_rate=float(campaign["purchase"].mean()),
    )


@router.post("/generate", response_model=DataSummary, status_code=status.HTTP_201_CREATED)
def generate_data(request: GenerateDataRequest) -> DataSummary:
    try:
        with NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
            path = Path(temporary.name)
        generate_sample_campaign(str(path), request.n_customers, request.random_seed)
        state.campaign = load_csv(path)
        state.clear_analysis()
        path.unlink(missing_ok=True)
        return _summary()
    except (ValueError, OSError) as exc:
        logger.exception("Synthetic campaign generation failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/upload", response_model=DataSummary, status_code=status.HTTP_201_CREATED)
async def upload_data(file: UploadFile) -> DataSummary:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload must be a CSV file")
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(suffix=".csv", delete=False) as temporary:
            temporary.write(await file.read())
            temporary_path = Path(temporary.name)
        state.campaign = load_csv(temporary_path)
        state.clear_analysis()
        return _summary()
    except (ValueError, OSError) as exc:
        logger.exception("Campaign upload validation failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)


@router.get("/summary", response_model=DataSummary)
def data_summary() -> DataSummary:
    return _summary()
