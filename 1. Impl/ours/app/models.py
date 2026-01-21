from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import time


@dataclass
class JobSpec:
    job_id: str
    model: str
    dataset: str
    epochs: int = 20
    batch_size: int = 32
    user_id: Optional[str] = None
    policy_text: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClusterTelemetry:
    cluster_id: str
    used_gpus: int
    total_gpus: int
    util: float
    power_current_w: float
    queue_len: Optional[int] = None
    ts: float = field(default_factory=lambda: time.time())

@dataclass
class ScheduleDecision:
    ts: float
    job_id: str
    cluster_id: str
    g_alloc: int
    score: float
    policy: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ElasticReallocEvent:
    ts: float
    cluster_id: str
    donor_job_id: str
    receiver_job_id: str
    donor_g_before: int
    donor_g_after: int
    receiver_g_before: int
    receiver_g_after: int
    U_before: float
    U_after: float

@dataclass
class PolicyVector:
    lambda_time: float = 1.0 / 3.0
    lambda_cost: float = 1.0 / 3.0
    lambda_energy: float = 1.0 / 3.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "lambda_time": float(self.lambda_time),
            "lambda_cost": float(self.lambda_cost),
            "lambda_energy": float(self.lambda_energy),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PolicyVector":
        return cls(
            lambda_time=float(d.get("lambda_time", 1.0 / 3.0)),
            lambda_cost=float(d.get("lambda_cost", 1.0 / 3.0)),
            lambda_energy=float(d.get("lambda_energy", 1.0 / 3.0)),
        )