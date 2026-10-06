"""Pydantic contracts for the EconoCausal REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateDataRequest(BaseModel):
    n_customers: int = Field(default=5000, ge=1, le=100_000)
    random_seed: int | None = 42


class DataSummary(BaseModel):
    rows: int
    columns: list[str]
    treatment_counts: dict[str, int]
    purchase_rate: float
    has_data: bool = True


class TrainRequest(BaseModel):
    random_state: int = 42
    cv: int = Field(default=3, ge=2, le=10)
    n_estimators: int = Field(default=200, ge=10, le=1000)


class TrainResponse(BaseModel):
    status: str
    models: list[int]
    rows: int
    columns: list[str]


class OptimizationRequest(BaseModel):
    budget: float = Field(ge=0)
    expected_order_value: float = Field(default=100.0, ge=0)
    discount_cost_10: float = Field(default=10.0, ge=0)
    discount_cost_20: float = Field(default=20.0, ge=0)
    max_customers: int | None = Field(default=None, ge=0)


class ErrorResponse(BaseModel):
    detail: str


class CustomerResult(BaseModel):
    customer_id: str
    baseline_purchase_probability: float
    purchase_probability_10: float
    purchase_probability_20: float
    ITE_10: float
    ITE_20: float


class OptimizationResponse(BaseModel):
    recommendations: list[dict[str, Any]]
    summary: dict[str, Any]
