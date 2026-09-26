"""Budget optimization endpoints."""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.schemas.api import OptimizationRequest, OptimizationResponse
from app.api.state import state
from app.optimization.budget_optimizer import optimize_campaign

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/optimization", tags=["optimization"])


@router.post("/run", response_model=OptimizationResponse, status_code=status.HTTP_200_OK)
def run_optimization(request: OptimizationRequest) -> OptimizationResponse:
    if state.ite_results is None:
        raise HTTPException(status_code=409, detail="Train causal models before optimization")
    data = state.ite_results.copy()
    data["expected_order_value"] = request.expected_order_value
    data["discount_cost_10"] = request.discount_cost_10
    data["discount_cost_20"] = request.discount_cost_20
    data["discount_cost"] = request.discount_cost_10
    try:
        result = optimize_campaign(data, request.budget)
        state.optimization_result = result
        return OptimizationResponse(
            recommendations=result.recommendations.to_dict(orient="records"),
            summary=result.summary.to_dict(),
        )
    except (ValueError, RuntimeError) as exc:
        logger.exception("Campaign optimization failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/results", response_model=OptimizationResponse)
def optimization_results() -> OptimizationResponse:
    if state.optimization_result is None:
        raise HTTPException(status_code=404, detail="Optimization results are not available; run optimization first")
    result = state.optimization_result
    return OptimizationResponse(
        recommendations=result.recommendations.to_dict(orient="records"),
        summary=result.summary.to_dict(),
    )
