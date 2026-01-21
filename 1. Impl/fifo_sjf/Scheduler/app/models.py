from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal

G = Literal[1,2,3,4]

@dataclass
class ProfilingCompact:
    cluster: str                   # "clusterA" | "clusterB" ...
    g: List[G]                     # [1,2,3,4]
    sps: List[float]               # len=4
    cost: List[float]              # len=4

@dataclass
class ClusterCSP:
    cluster: str
    p_fair: float                  # 0..1
    U_t: float                     # 0..1
    E_t: float                     # 0..1

@dataclass
class JobSpec:
    job_id: str
    user_intent_text: str
    profiling_compact: List[ProfilingCompact]
    csp: List[ClusterCSP]

@dataclass
class ClusterPolicyHint:
    cluster: str
    pref_g: G
    g_rank: List[G]                # permutation of [1,2,3,4]

@dataclass
class PolicyCard:
    job_id: str
    eta_perf: float                # η_perf ∈ [0,1]
    clusters: List[ClusterPolicyHint]
    rationale: List[str] = field(default_factory=list)

# 런타임 내부 구조
@dataclass
class JobRuntime:
    job_id: str
    model: str
    dataset: str
    eta_perf: float
    pref_g: int
    g_rank: List[int]
    cluster_selected: str          # 실제 배치된 클러스터
    current_g: int = 0

