"""In-memory MVP application state."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.causal.dml import DMLResult
from app.optimization.budget_optimizer import OptimizationResult


@dataclass
class ApplicationState:
    campaign: pd.DataFrame | None = None
    dml_results: dict[int, DMLResult] = field(default_factory=dict)
    ite_results: pd.DataFrame | None = None
    propensity_results: dict[int, object] = field(default_factory=dict)
    audit_results: dict[int, object] = field(default_factory=dict)
    optimization_result: OptimizationResult | None = None

    def clear_analysis(self) -> None:
        self.dml_results.clear()
        self.ite_results = None
        self.propensity_results.clear()
        self.audit_results.clear()
        self.optimization_result = None


state = ApplicationState()
