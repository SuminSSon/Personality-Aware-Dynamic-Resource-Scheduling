# [pollux_scheduler.py]

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

# === [Lucid Constants] ===
T_PROF = 60.0        
G_SS_LIMIT = 2       
DT_GPU_UTIL_1 = 53
DT_GPU_UTIL_2 = 25
DT_MEM_USED_1 = 3041
DT_MEM_USED_2 = 2734
DT_MEM_UTIL_1 = 13.5

# === [Gang Scheduling Configuration] ===
# 특정 모델은 무조건 고정된 GPU 개수로만 실행되도록 설정
GANG_JOBS = {
    ("DenseNet-121", "TinyImageNet"): 4,
}

def is_gang_model(model_name: str, dataset: str) -> bool:
    return (model_name, dataset) in GANG_JOBS


@dataclass
class RuntimeJobState:
    job_id: str
    model_name: str
    dataset: str
    cluster_id: str

    # Lucid Fields
    current_gpus: int
    current_local_batch: int
    current_grad_accum: int = 1
    
    start_ts: float = 0.0
    is_profiled: bool = False  
    
    # Decision Tree Accumulators
    sum_gpu_util: float = 0.0
    sum_gpu_mem_util: float = 0.0
    sum_gpu_mem_used: float = 0.0
    metric_count: int = 0
    
    sharing_score: int = 2        
    priority_score: float = 0.0   
    
    # Pollux Legacy Fields (호환성 유지)
    progress: float = 0.0
    attained_service: float = 0.0
    last_scaled_at_ts: float = 0.0
    min_gpus: int = 1
    max_gpus: int = 4
    last_sps: float = 0.0
    last_metric_ts: float = 0.0


class LucidProfiler:
    def update_metrics(self, job: RuntimeJobState, metrics: Dict):
        """Telemetry 데이터 수집"""
        g_util = float(metrics.get("gpu_util", 0.0))
        m_used = float(metrics.get("mem_used_mb", 0.0))
        
        # Agent가 보낸 실제 메모리 I/O 사용률(gpu_mem_util) 사용
        m_util = float(metrics.get("gpu_mem_util", 0.0)) 
        
        job.sum_gpu_util += g_util
        job.sum_gpu_mem_used += m_used
        job.sum_gpu_mem_util += m_util
        job.metric_count += 1

    def classify_job(self, job: RuntimeJobState) -> int:
        ## [Gang Job 처리] Gang Job은 간섭 방지를 위해 무조건 Jumbo(2)로 분류 권장
        #if is_gang_model(job.model_name, job.dataset):
        #    return 2

        if job.metric_count == 0:
            return 2 
        
        # 평균 계산
        Ug = job.sum_gpu_util / job.metric_count
        Mg = job.sum_gpu_mem_used / job.metric_count
        Um = job.sum_gpu_mem_util / job.metric_count
        
        # Decision Tree Logic (Figure 6)
        if Ug <= DT_GPU_UTIL_1:
            if Ug <= DT_GPU_UTIL_2:
                return 0 # Tiny
            else:
                if Mg <= DT_MEM_USED_1:
                    if Mg <= DT_MEM_USED_2:
                         if Um <= DT_MEM_UTIL_1: return 0
                         else: return 1
                    else: return 1 # Medium
                else: return 2 # Jumbo
        else:
            return 2 # Jumbo

_profiler = LucidProfiler()

def update_job_metrics_for_lucid(job: RuntimeJobState, metrics: Dict):
    if "attained_service" in metrics:
        job.attained_service = float(metrics["attained_service"])
    _profiler.update_metrics(job, metrics)

def lucid_classify_job(job: RuntimeJobState):
    ss = _profiler.classify_job(job)
    job.sharing_score = ss
    job.is_profiled = True
    # Priority 계산 (간소화: GPU 시간)
    job.priority_score = job.current_gpus * 3600.0 
    return ss

# --- [Legacy Adapters for Compatibility] ---

def record_goodput_sample(
    job: RuntimeJobState,
    g: int,
    local_batch: int,
    accum: int,
    sps: float,
    gns: Optional[float],
    stat_eff: Optional[float]
):
    """
    기존 Pollux 서버 호환용 (Lucid는 GNS 미사용 -> sps만 업데이트)
    """
    job.last_sps = sps
    pass

def pollux_initial_g(
    job_id: str,
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size_per_gpu: int,
    learning_rate: float,
    cluster_free_gpus: int,
    queue_len: int,
    pollux_desired_gpus: Optional[int] = None,
    initial_g: Optional[int] = None,
) -> int:
    """
    Pollux-style initial world_size 결정 함수.

    - API는 기존 GlobalServer 호출 방식 그대로 유지:
        job_id, model_name, dataset, epochs, batch_size_per_gpu,
        learning_rate, cluster_free_gpus, queue_len, ...
    - 구현은 우리가 논의한 Pollux 스타일로:
        * queue_len은 사용하지 않음 (admission에서 queue pressure 반영 X)
        * profiling/LLM이 정한 pollux_desired_gpus를 우선 존중
        * DenseNet-121/TinyImageNet 같은 gang job은 g=4로만 시작
    """

    # GPU 여유 없으면 시작하지 않고 큐에서 대기
    if cluster_free_gpus <= 0:
        return 0

    # 1) gang job인지 체크
    if is_gang_model(model_name, dataset):
        # GANG_JOBS[(model,dataset)] = 4 이런 형태라고 가정
        gang_g = int(GANG_JOBS[(model_name, dataset)])
        # free가 모자르면 시작하지 않고 큐에서 대기
        return gang_g if gang_g <= cluster_free_gpus else 0

    # 2) non-gang job: desired g 결정
    #    - pollux_desired_gpus > initial_g > 1 순서로 우선
    if pollux_desired_gpus is not None:
        desired = int(pollux_desired_gpus)
    elif initial_g is not None:
        desired = int(initial_g)
    else:
        desired = 1

    desired = max(desired, 1)

    # 3) 클러스터 여유 GPU와 desired 중 작은 값 선택
    g_init = min(desired, cluster_free_gpus)

    # 최소 1장은 들고 시작 (preemption 없는 Pollux 기본 철학)
    return max(1, g_init)

def pollux_best_local_config_for_g(
    job_id: str,
    model_name: str,
    dataset: str,
    g: int,
    default_batch: int,
    default_accum: int,
    default_sps: float,
) -> Tuple[int, int, float, float, float]:
    # Lucid는 배치 사이즈 튜닝 안 함 -> 입력값 그대로 리턴
    return int(default_batch), int(default_accum), 1.0, 1.0, 1.0