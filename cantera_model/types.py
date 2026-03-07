from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CaseCondition:
    case_id: str
    T0: float
    P0_atm: float
    phi: float
    t_end: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CaseTrace:
    case_id: str
    time: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    X: np.ndarray
    wdot: np.ndarray
    rop: np.ndarray
    species_names: list[str]
    reaction_eqs: list[str]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float)
        self.temperature = np.asarray(self.temperature, dtype=float)
        self.pressure = np.asarray(self.pressure, dtype=float)
        self.X = np.asarray(self.X, dtype=float)
        self.wdot = np.asarray(self.wdot, dtype=float)
        self.rop = np.asarray(self.rop, dtype=float)

        n_t = self.time.shape[0]
        if self.temperature.shape != (n_t,):
            raise ValueError("temperature shape must be (T,)")
        if self.pressure.shape != (n_t,):
            raise ValueError("pressure shape must be (T,)")
        if self.X.shape[0] != n_t:
            raise ValueError("X first dimension must match time")
        if self.wdot.shape[0] != n_t:
            raise ValueError("wdot first dimension must match time")
        if self.rop.shape[0] != n_t:
            raise ValueError("rop first dimension must match time")


@dataclass(slots=True)
class CaseBundle:
    mechanism_path: str
    phase: str
    species_names: list[str]
    reaction_eqs: list[str]
    cases: list[CaseTrace] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReductionMapping:
    S: np.ndarray
    pool_meta: list[dict[str, Any]]
    keep_reactions: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.S = np.asarray(self.S, dtype=float)
        if self.S.ndim != 2:
            raise ValueError("S must be 2-D")
        if self.keep_reactions is not None:
            self.keep_reactions = np.asarray(self.keep_reactions, dtype=bool)


@dataclass(slots=True)
class EvalSummary:
    cases: int
    pass_rate: float
    pass_cases: int
    failed_cases: int
    qoi_metrics_count: int
    max_rel_diff: float | None
    mean_rel_diff: float | None
    worst_case: dict[str, Any] | None
    rel_tolerance: float
    rel_eps: float
    mandatory_total_metric_count: int | None = None
    valid_mandatory_metric_count: int | None = None
    min_valid_mandatory_count_effective: int | None = None
    mandatory_validity_passed: bool | None = None
    invalid_mandatory_metric_count: int | None = None
    inactive_mandatory_metric_count: int | None = None
    active_invalid_mandatory_metric_count: int | None = None
    active_invalid_mandatory_gate_unit_keys: list[str] | None = None
    mandatory_metric_case_pass_rates: dict[str, float] | None = None
    mandatory_gate_unit_case_pass_rates: dict[str, float] | None = None
    mandatory_gate_unit_mode_effective: str | None = None
    mandatory_species_family_score_mode_effective: str | None = None
    mandatory_quality_scope_effective: str | None = None
    mandatory_tail_scope_effective: str | None = None
    mandatory_total_gate_unit_count: int | None = None
    valid_mandatory_gate_unit_count: int | None = None
    mandatory_quality_gate_unit_count: int | None = None
    mandatory_quality_metric_count: int | None = None
    mandatory_species_family_case_pass_min_effective: float | None = None
    mandatory_metric_validity_mode_effective: str | None = None
    mandatory_validity_basis_effective: str | None = None
    mandatory_gate_unit_evaluable_case_rates: dict[str, float] | None = None
    valid_mandatory_gate_unit_count_coverage: int | None = None
    valid_mandatory_gate_unit_count_case_rate: int | None = None
    mandatory_gate_unit_valid_count_shadow_evaluable_ratio: int | None = None
    mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective: float | None = None
    pass_rate_mandatory_case: float | None = None
    pass_rate_mandatory_case_all_units: float | None = None
    pass_rate_mandatory_case_all_required: float | None = None
    pass_rate_mandatory_case_ratio_mean: float | None = None
    pass_rate_mandatory_case_all_required_all_units: float | None = None
    pass_rate_mandatory_case_ratio_mean_all_units: float | None = None
    mandatory_case_unit_weight_mode_effective: str | None = None
    pass_rate_optional_case: float | None = None
    pass_rate_optional_metric_mean: float | None = None
    pass_rate_all_metric_legacy: float | None = None
    mean_rel_diff_mandatory: float | None = None
    mean_rel_diff_mandatory_all_units: float | None = None
    mean_rel_diff_mandatory_raw: float | None = None
    mean_rel_diff_mandatory_family_weighted: float | None = None
    mean_rel_diff_mandatory_winsorized: float | None = None
    mandatory_rel_outlier_ratio: float | None = None
    mandatory_rel_outlier_ratio_all_units: float | None = None
    mandatory_rel_outlier_ratio_max_effective: float | None = None
    mandatory_rel_diff_p95: float | None = None
    mandatory_rel_diff_p95_all_units: float | None = None
    mandatory_tail_guard_passed: bool | None = None
    mandatory_tail_guard_triggered: bool | None = None
    mandatory_tail_guard_hard_applied: bool | None = None
    mandatory_tail_guard_mode_effective: str | None = None
    mandatory_tail_guard_policy_effective: str | None = None
    mandatory_tail_activation_ratio_min_effective: float | None = None
    mandatory_tail_exceed_ref_effective: str | None = None
    mandatory_tail_exceed_ratio: float | None = None
    mandatory_tail_rel_diff_max_effective: float | None = None
    mandatory_quality_scope_empty: bool | None = None
    mandatory_metric_valid_case_pass_min_effective: float | None = None
    mean_rel_diff_optional: float | None = None
    mean_rel_diff_all_metric_legacy: float | None = None
    coverage_gate_passed: bool | None = None
    mandatory_quality_passed: bool | None = None
    optional_quality_passed: bool | None = None
    mandatory_error_passed: bool | None = None
    optional_error_passed: bool | None = None
    error_fail_reason_primary: str | None = None
    mandatory_error_include_validity_effective: bool | None = None
    error_gate_score: float | None = None
    effective_metric_count: int | None = None
    suppressed_low_signal_metric_count: int | None = None
    error_gate_passed: bool | None = None
    mandatory_case_mode_effective: str | None = None
    mandatory_mean_aggregation_effective: str | None = None
    mandatory_mean_mode_effective: str | None = None
    selection_pool_kind: str | None = None
    structure_deficit_score: float | None = None
    metric_drift_raw: float | None = None
    metric_drift_effective_cap: float | None = None
    selection_quality_score_raw_drift: float | None = None
    compression_refine_applied: bool | None = None
    compression_refine_trials: int | None = None
    compression_refine_reaction_delta: int | None = None
    compression_refine_species_delta: int | None = None
    compression_refine_mode_effective: str | None = None
    compression_refine_guard_passed: bool | None = None
    mode_collapse_warning: bool | None = None
    evaluation_contract_version: str | None = None
    metric_taxonomy_profile_effective: str | None = None
    diagnostic_schema_ok: bool | None = None
