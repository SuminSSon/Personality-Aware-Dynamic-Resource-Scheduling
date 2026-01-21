from __future__ import annotations

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class OursBaseModel(BaseModel):
    class Config:
        extra = "allow"

class SubmitReq(OursBaseModel):
    job_id: Optional[str] = Field(
        default=None,
        description="Optional job ID (if client wants to specify).",
    )
    model: str = Field(..., description="Model name to train/evaluate.")
    dataset: str = Field(..., description="Dataset name.")
    epochs: int = Field(20, description="Number of epochs.")
    world_size: int = Field(1, description="Number of world size = #g")
    batch_size: int = Field(32, description="Batch size per GPU.")
    user_id: Optional[str] = Field(
        default=None, description="User identifier (for fairness/accounting)."
    )

    cluster_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional cluster pin. If set, the scheduler will only consider this "
            "cluster (used for re-submitting preempted jobs with local checkpoints)."
        ),
    )

    policy_text: Optional[str] = Field(
        default=None,
        description="Natural-language intent for LLM-based policy generator.",
    )
    intent: Optional[str] = Field(
        default=None, description="Alias of policy_text for legacy clients."
    )
    user_request: Optional[str] = Field(
        default=None, description="Another alias for natural-language policy text."
    )

class TelemetryIn(OursBaseModel):
    # ---- cluster-level (새로 추가) ----
    cluster_id: Optional[str] = None
    used_gpus: Optional[int] = None
    total_gpus: Optional[int] = None
    util: Optional[float] = None          # cluster util (0~1 or 0~100 중 하나로 통일 권장)
    power_cluster_w: Optional[float] = None

    # ---- node/gpu-level (기존 유지) ----
    node_id: Optional[str] = None
    gpu_index: Optional[int] = None
    gpu_util: Optional[float] = None
    power_w: Optional[float] = None       # per-GPU power
    mem_used_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None

class JobCompleteReport(OursBaseModel):
    job_id: str
    status: Optional[str] = None          # "FINISHED"/"FAILED"/"PREEMPTED"/...
    reason: Optional[str] = None          # "PREEMPT", "RESIZE", ...
    run_id: Optional[str] = None
    attempt: Optional[int] = None
    end_ts: Optional[float] = None

    exit_code: int = 0
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None

class SubmitReqFixed(BaseModel):
    """
    S3 / 고정 λ 실험용 submit 요청 스키마
    """
    model: str
    dataset: str
    epochs: int = 20
    batch_size_per_gpu: int = 32

    user_id: Optional[str] = None
    policy_text: Optional[str] = None  # 있어도 되고, 없으면 빈 문자열 취급
    cluster_id: Optional[str] = None   # pinned cluster 용

    lambda_time: float
    lambda_cost: float
    lambda_energy: float