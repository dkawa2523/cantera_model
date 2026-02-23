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
