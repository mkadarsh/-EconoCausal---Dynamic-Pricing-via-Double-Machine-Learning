"""Causal training, results, propensity, uplift, and audit endpoints."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.schemas.api import CustomerResult, TrainRequest, TrainResponse
from app.api.state import state
from app.causal.dml import train_dml_models
from app.causal.propensity import analyze_propensity
from app.causal.refutation import audit_all_refutations
from app.causal.uplift import evaluate_all_uplift

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/causal", tags=["causal"])


def _require_campaign() -> pd.DataFrame:
    if state.campaign is None:
        raise HTTPException(status_code=409, detail="Generate or upload campaign data first")
    return state.campaign


def _records(frame: pd.DataFrame) -> list[dict]:
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


@router.post("/train", response_model=TrainResponse)
def train_models(request: TrainRequest) -> TrainResponse:
    campaign = _require_campaign()
    try:
        result_10, result_20, table = train_dml_models(
            campaign,
            random_state=request.random_state,
            cv=request.cv,
            n_estimators=request.n_estimators,
        )
        state.dml_results = {10: result_10, 20: result_20}
        state.ite_results = table
        state.propensity_results.clear()
        state.audit_results.clear()
        state.optimization_result = None
        return TrainResponse(
            status="trained",
            models=[10, 20],
            rows=len(table),
            columns=list(table.columns),
        )
    except ImportError as exc:
        logger.exception("Causal training dependency unavailable")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        logger.exception("Causal training failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/results", response_model=list[CustomerResult])
def causal_results() -> list[dict]:
    if state.ite_results is None:
        raise HTTPException(status_code=404, detail="Causal results are not available; train models first")
    return _records(state.ite_results)


@router.get("/customer/{customer_id}", response_model=CustomerResult)
def customer_result(customer_id: str) -> CustomerResult:
    if state.ite_results is None:
        raise HTTPException(status_code=404, detail="Causal results are not available; train models first")
    match = state.ite_results[state.ite_results["customer_id"].astype(str) == customer_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Customer '{customer_id}' was not found")
    return CustomerResult(**_records(match)[0])


@router.get("/uplift", response_model=dict[str, object])
def uplift_results() -> dict:
    campaign = _require_campaign()
    if state.ite_results is None:
        raise HTTPException(status_code=404, detail="Causal results are not available; train models first")
    frame = campaign.merge(state.ite_results[["customer_id", "ITE_10", "ITE_20"]], on="customer_id")
    try:
        evaluations = evaluate_all_uplift(frame)
        return {str(discount): {"discount_amount": value.discount_amount, **value.plot_data,
                                "metric": value.metric, "area_under_curve": value.area_under_curve,
                                "final_incremental_gain": value.final_incremental_gain}
                for discount, value in evaluations.items()}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/propensity", response_model=dict[str, object])
def propensity_results() -> dict:
    campaign = _require_campaign()
    try:
        analyses = analyze_propensity(campaign)
        return {
            str(discount): {
                "discount_amount": result.discount_amount,
                "propensity_scores": _records(result.propensity_scores),
                "treated_count": result.treated_count,
                "control_count": result.control_count,
                "mean_propensity_treated": result.mean_propensity_treated,
                "mean_propensity_control": result.mean_propensity_control,
                "overlap_statistics": result.overlap_statistics,
                "matching_statistics": result.matching_statistics.__dict__,
                "matched_treatment_rows": _records(result.matched_treatment_rows),
                "matched_control_rows": _records(result.matched_control_rows),
            }
            for discount, result in analyses.items()
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/audit", response_model=dict[str, object])
def causal_audit() -> dict:
    campaign = _require_campaign()
    try:
        audits = audit_all_refutations(campaign)
        return {str(discount): audit.to_dict() for discount, audit in audits.items()}
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
