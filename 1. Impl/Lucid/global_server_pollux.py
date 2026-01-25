from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, ValidationError
import httpx
import time
import threading
import itertools
import logging
import csv
import os
import datetime
import asyncio  
from typing import Optional, Dict, List, Any
from collections import defaultdict
import argparse
from contextlib import asynccontextmanager
import json

from pollux_scheduler import (
    RuntimeJobState,
    update_job_metrics_for_lucid,
    record_goodput_sample,
    pollux_initial_g,
    pollux_best_local_config_for_g,
    lucid_classify_job,
    T_PROF,
    G_SS_LIMIT,
    is_gang_model,
    GANG_JOBS,
)


# --- 로깅 설정 (콘솔 출력용) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (GlobalServer) %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

MODEL_BASENAME = {
    "ResNet-50":        "Resnet50",
    "ResNet-18":        "Resnet18",
    "DenseNet-121":     "Densenet121",
    "MobileNetV2":      "MobileNetV2",
    "EfficientNetV2-S": "EfficientNetV2",
    "EfficientNetV2":   "EfficientNetV2",
    "VGG-11":           "VGG11",
    "DistilBERT":       "DistilBert",
    "DeepSpeech2":      "DeepSpeech2",
}

DS_PREFIX = {
    "CIFAR-100":     "DDP_CIFAR100_",
    "CIFAR-10":      "DDP_CIFAR10_",
    "TinyImageNet":  "DDP_TinyImageNet_",
    "Fashion-MNIST": "DDP_FashoinMNIST_",
    "MNIST":         "DDP_MNIST_",
    "SST2":          "DDP_SST2_",
    "SST-2":         "DDP_SST2_",
    "ARCTIC":        "DDP_CMU_ARCTIC_",
}

def make_filename(model: str, dataset: str) -> str:
    if model not in MODEL_BASENAME:
        raise KeyError(f"Unknown model name: {model}")
    if dataset not in DS_PREFIX:
        raise KeyError(f"Unknown dataset name: {dataset}")
    return f"{DS_PREFIX[dataset]}{MODEL_BASENAME[model]}.py"

MODEL_DS_MAP = {
    ("ResNet-50",        "CIFAR-100"):     make_filename("ResNet-50",        "CIFAR-100"),
    ("ResNet-18",        "CIFAR-100"):     make_filename("ResNet-18",        "CIFAR-100"),
    ("DenseNet-121",     "CIFAR-100"):     make_filename("DenseNet-121",     "CIFAR-100"),
    ("MobileNetV2",      "CIFAR-100"):     make_filename("MobileNetV2",      "CIFAR-100"),
    ("EfficientNetV2-S", "CIFAR-100"):     make_filename("EfficientNetV2-S", "CIFAR-100"),
    ("VGG-11",           "CIFAR-100"):     make_filename("VGG-11",           "CIFAR-100"),

    ("ResNet-50",        "CIFAR-10"):      make_filename("ResNet-50",        "CIFAR-10"),
    ("DenseNet-121",     "CIFAR-10"):      make_filename("DenseNet-121",     "CIFAR-10"),
    ("MobileNetV2",      "CIFAR-10"):      make_filename("MobileNetV2",      "CIFAR-10"),
    ("EfficientNetV2-S", "CIFAR-10"):      make_filename("EfficientNetV2-S", "CIFAR-10"),
    ("VGG-11",           "CIFAR-10"):      make_filename("VGG-11",           "CIFAR-10"),

    ("ResNet-50",        "TinyImageNet"):  make_filename("ResNet-50",        "TinyImageNet"),
    ("DenseNet-121",     "TinyImageNet"):  make_filename("DenseNet-121",     "TinyImageNet"),
    ("MobileNetV2",      "TinyImageNet"):  make_filename("MobileNetV2",      "TinyImageNet"),
    ("EfficientNetV2-S", "TinyImageNet"):  make_filename("EfficientNetV2-S", "TinyImageNet"),
    ("VGG-11",           "TinyImageNet"):  make_filename("VGG-11",           "TinyImageNet"),

    ("ResNet-50",        "Fashion-MNIST"): make_filename("ResNet-50",        "Fashion-MNIST"),
    ("DenseNet-121",     "Fashion-MNIST"): make_filename("DenseNet-121",     "Fashion-MNIST"),
    ("MobileNetV2",      "Fashion-MNIST"): make_filename("MobileNetV2",      "Fashion-MNIST"),
    ("EfficientNetV2-S", "Fashion-MNIST"): make_filename("EfficientNetV2-S", "Fashion-MNIST"),
    ("VGG-11",           "Fashion-MNIST"): make_filename("VGG-11",           "Fashion-MNIST"),

    ("ResNet-50",        "MNIST"):         make_filename("ResNet-50",        "MNIST"),
    ("DenseNet-121",     "MNIST"):         make_filename("DenseNet-121",     "MNIST"),
    ("MobileNetV2",      "MNIST"):         make_filename("MobileNetV2",      "MNIST"),
    ("EfficientNetV2-S", "MNIST"):         make_filename("EfficientNetV2-S", "MNIST"),
    ("VGG-11",           "MNIST"):         make_filename("VGG-11",           "MNIST"),

    ("DistilBERT",       "SST2"):          make_filename("DistilBERT",       "SST2"),
    ("DistilBERT",       "SST-2"):         make_filename("DistilBERT",       "SST-2"),

    ("DeepSpeech2",      "ARCTIC"):        make_filename("DeepSpeech2",      "ARCTIC"),
}

NODE_REGISTRY = {
    # Cluster A
    "node_a": {"ip": "163.180.117.216", "agent_port": 8001, "gpu_id": 0,
               "prefix": "/home/ubuntu216/SUN", "status": "idle",
               "current_job_id": None, "cluster": "clusterA"},
    "node_b": {"ip": "163.180.117.216", "agent_port": 8002, "gpu_id": 1,
               "prefix": "/home/ubuntu216/SUN", "status": "idle",
               "current_job_id": None, "cluster": "clusterA"},
    "node_c": {"ip": "163.180.117.216", "agent_port": 8003, "gpu_id": 2,
               "prefix": "/home/ubuntu216/SUN", "status": "idle",
               "current_job_id": None, "cluster": "clusterA"},
    "node_d": {"ip": "163.180.117.216", "agent_port": 8004, "gpu_id": 3,
               "prefix": "/home/ubuntu216/SUN", "status": "idle",
               "current_job_id": None, "cluster": "clusterA"},

    # Cluster B
    "node_e": {"ip": "163.180.160.62", "agent_port": 8005, "gpu_id": 0,
               "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "clusterB"},
    "node_f": {"ip": "163.180.160.62", "agent_port": 8006, "gpu_id": 0,
               "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "clusterB"},
    "node_g": {"ip": "163.180.160.62", "agent_port": 8007, "gpu_id": 0,
               "prefix": "/data/breath12/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "clusterB"},
    "node_h": {"ip": "163.180.160.62", "agent_port": 8008, "gpu_id": 0,
               "prefix": "/data/breath12/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "clusterB"},
}

# --- 전역 상태 변수 ---
GLOBAL_SERVER_ADDRESS: Optional[str] = None
MASTER_PORT_COUNTER = itertools.count(20000)

LAMBDA_TIME = 0.0
LAMBDA_COST = 0.0
LAMBDA_FAIR = 0.0

POLLUX_DELTA_GOODPUT_THRESH_IDLE = 0.0
POLLUX_DELTA_GOODPUT_THRESH_BUSY = 0.1
POLLUX_SCALE_COOLDOWN_SEC = 60.0

NODE_REGISTRY_LOCK = threading.Lock()

JOB_QUEUE: List[Dict[str, Any]] = []
JOB_QUEUE_LOCK = threading.Lock()

ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}
ACTIVE_JOBS_LOCK = threading.Lock()

FINISHED_JOB_IDS: set[str] = set()
JOB_LATEST_METRICS: Dict[str, Dict[str, Any]] = {}
RUNTIME_JOBS: Dict[str, RuntimeJobState] = {}
LAST_SCHED_TICK_TS: Optional[float] = None

# --- CSV 및 파일 로깅 관리자 ---
class MetricsLogger:
    def __init__(self, base_log_dir: str = "./logs"):
        # 런 단위 타임스탬프 (폴더 이름)
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 예: ./logs/20251123_213210
        self.log_dir = os.path.join(base_log_dir, run_ts)

        # 디렉토리 생성 (부모 ./logs가 없어도 같이 생성됨)
        os.makedirs(self.log_dir, exist_ok=True)

        # 각 파일은 이 run_dir 아래에 생성
        self.files = {
            "scheduler":   os.path.join(self.log_dir, "scheduler.log"),
            "csp_metrics": os.path.join(self.log_dir, "csp_metrics.csv"),
            "job_metrics": os.path.join(self.log_dir, "job_metrics.csv"),
            "job_events":  os.path.join(self.log_dir, "job_events.csv"),
            "queue_events":os.path.join(self.log_dir, "queue_events.csv"),
            "telemetry":   os.path.join(self.log_dir, "telemetry.csv"),
        }

        # 스케줄러 전용 텍스트 로거 설정
        self.sched_log = logging.getLogger("SchedulerFileLog")
        self.sched_log.setLevel(logging.INFO)
        if self.sched_log.hasHandlers():
            self.sched_log.handlers.clear()

        file_handler = logging.FileHandler(self.files["scheduler"])
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.sched_log.addHandler(file_handler)

        self._init_csv_headers()

    def _init_csv_headers(self):
        headers = {
            "csp_metrics": [
                "ts", "cluster", "queue_len", "slots_total", "slots_used", "free",
                "U_t", "E_t", "p_fair",
                "lam_time", "lam_cost", "lam_fair"
            ],
            "job_metrics": [
                "job_id", "cluster", "model", "dataset", "world_size",
                "submitted_ts", "started_ts", "end_ts", "queued_sec", "jct_sec", "status",
                "final_accuracy",
            ],
            "job_events": ["ts", "event", "job_id", "cluster", "world_size", "note", "metadata_json"],
            "queue_events": ["ts", "event", "job_id", "queue_len_after", "note"],
            "telemetry": ["ts", "node_id", "gpu_index", "gpu_util", "power_w", "mem_used_mb", "mem_total_mb"],
        }

        for name, path in self.files.items():
            if name == "scheduler":
                continue
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers[name])

    def log_scheduler(self, message: str):
        self.sched_log.info(message)

    def log_csv(self, file_key: str, data_list: List[Any]):
        try:
            with open(self.files[file_key], 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(data_list)
        except Exception as e:
            log.error(f"Failed to write to {file_key}: {e}")


metrics_logger = MetricsLogger()

def log_scheduler_state(reason: str):
    with JOB_QUEUE_LOCK:
        q_len = len(JOB_QUEUE)
    with ACTIVE_JOBS_LOCK:
        active_jobs = len(ACTIVE_JOBS)
    with NODE_REGISTRY_LOCK:
        cluster_counts = defaultdict(lambda: {"total": 0, "busy": 0, "idle": 0})
        for node_id, info in NODE_REGISTRY.items():
            c = info.get("cluster", "Cluster_Default")
            cluster_counts[c]["total"] += 1
            if info["status"] == "busy":
                cluster_counts[c]["busy"] += 1
            elif info["status"] == "idle":
                cluster_counts[c]["idle"] += 1

    cluster_summaries = []
    for c_name, st in cluster_counts.items():
        cluster_summaries.append(
            f"{c_name}: total={st['total']} busy={st['busy']} idle={st['idle']}"
        )

    metrics_logger.log_scheduler(
        f"[{reason}] queue={q_len} active_jobs={active_jobs} | "
        + " | ".join(cluster_summaries)
    )


def get_now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _compute_cluster_free_gpus() -> int:
    """
    현재 전체 클러스터(노드 레지스트리 기준)의 free GPU 개수 추정.
    - NODE_REGISTRY: 전체 슬롯 수
    - ACTIVE_JOBS: 이미 사용 중인 슬롯 수
    """
    with NODE_REGISTRY_LOCK:
        total_slots = len(NODE_REGISTRY)

    with ACTIVE_JOBS_LOCK:
        used_slots = sum(len(info.get("nodes", [])) for info in ACTIVE_JOBS.values())

    free_slots = max(0, total_slots - used_slots)
    return free_slots

def _decide_initial_g_for_job(
    job_id: str,
    model_name: str,
    dataset: str,
    base_batch: int,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    min_gpus: int = 1,
    cluster_free_gpus: Optional[int] = None,
) -> int:
    """
    Lucid/Pollux compatibility initial g chooser.

    - pollux_scheduler.pollux_initial_g()의 "새 시그니처"에 맞춰 호출
    - cluster_free_gpus가 None이면 전체 free 계산
    - free가 없으면 0(대기) 반환
    - gang job 처리(0 또는 4)는 pollux_initial_g 내부 로직을 따른다
    """

    # 1) free_slots
    if cluster_free_gpus is None:
        free_slots = int(_compute_cluster_free_gpus())
    else:
        free_slots = max(0, int(cluster_free_gpus))

    # free가 없으면 대기
    if free_slots <= 0:
        return 0

    # 2) queue_len
    with JOB_QUEUE_LOCK:
        qlen = int(len(JOB_QUEUE))

    # 3) pollux_scheduler.py의 "새 시그니처"에 맞춘 호출
    g = pollux_initial_g(
        job_id=str(job_id),
        model_name=str(model_name),
        dataset=str(dataset),
        epochs=int(epochs),
        batch_size_per_gpu=int(base_batch),
        learning_rate=float(learning_rate),
        cluster_free_gpus=int(free_slots),
        queue_len=int(qlen),
        pollux_desired_gpus=None,
        initial_g=int(min_gpus),
    )

    # 4) 안전 클램프
    try:
        g = int(g)
    except Exception:
        g = int(min_gpus)

    # g==0: "대기" 의미 보존
    if g <= 0:
        return 0

    # 실험 환경 상한 4, free_slots 반영
    g = max(1, min(g, 4, free_slots))
    return g

# --- Co-Adaptive Tuner (batch size / learning rate) ---
class CoAdaptiveTuner:
    """
    간단한 Pollux-style co-adaptive 튜너 (per-job).
    - 제출 시 몇 개의 (g, batch, lr) 후보를 등록
    - checkpoint마다 reward(accuracy - loss) 기록
    - 일정 이상 관측되면 best 후보를 고르고 이후 exploit 단계에서 g에 맞춰 scaling
    """
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def init_job(self, job_id: str, base_batch: int, base_lr: float):
        with self._lock:
            # g는 Pollux allocator가 따로 결정하므로, 여기서는 대표 g만 사용
            candidates = [
                {"g": 1, "batch": base_batch,       "lr": base_lr},
                {"g": 2, "batch": base_batch * 2,   "lr": base_lr * 2},
                {"g": 4, "batch": base_batch * 4,   "lr": base_lr * 4},
            ]
            self.jobs[job_id] = {
                "base_batch": base_batch,
                "base_lr": base_lr,
                "candidates": candidates,
                "metrics": {i: [] for i in range(len(candidates))},
                "current_idx": 0,
                "best_idx": 0,
                "phase": "explore",  # explore -> exploit
            }

    def on_checkpoint(self, job_id: str, epoch: int, loss: float, acc: float):
        with self._lock:
            st = self.jobs.get(job_id)
            if not st:
                return

            idx = st["current_idx"]
            reward = acc - loss
            st["metrics"][idx].append(reward)

            explored_all = all(len(v) >= 1 for v in st["metrics"].values())
            if explored_all and st["phase"] == "explore":
                avg_rewards = {i: (sum(v) / len(v)) for i, v in st["metrics"].items()}
                best_idx = max(avg_rewards, key=avg_rewards.get)
                st["best_idx"] = best_idx
                st["phase"] = "exploit"

    def choose_config(self, job_id: str, new_g: int) -> Dict[str, Any]:
        with self._lock:
            st = self.jobs.get(job_id)
            if not st:
                # 튜너 state가 없으면 fallback: 단순 g비례 scaling
                base_batch = 64
                base_lr = 1e-3
                return {
                    "g": new_g,
                    "batch": max(1, int(base_batch * new_g)),
                    "lr": base_lr * new_g,
                }

            if st["phase"] == "explore":
                idx = st["current_idx"]
                cand = st["candidates"][idx]
                st["current_idx"] = (idx + 1) % len(st["candidates"])
                return {
                    "g": new_g,
                    "batch": cand["batch"],
                    "lr": cand["lr"],
                }

            # exploit 단계
            best = st["candidates"][st["best_idx"]]
            base_g = max(best["g"], 1)
            scale = new_g / base_g
            return {
                "g": new_g,
                "batch": max(1, int(best["batch"] * scale)),
                "lr": best["lr"] * scale,
            }

    def remove_job(self, job_id: str):
        with self._lock:
            self.jobs.pop(job_id, None)

tuner = CoAdaptiveTuner()

# --- API 데이터 모델 ---
class ProgressReport(BaseModel):
    job_id: str
    world_size: int
    local_batch: int
    grad_accum: int
    steps_per_sec: float
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    gns: Optional[float] = None
    stat_eff: Optional[float] = None


class JobCheckpointReport(BaseModel):
    job_id: str
    current_epoch: int
    total_epochs: int
    latest_accuracy: Optional[float] = None
    latest_eval_loss: Optional[float] = None
    checkpoint_path: str

class JobStopReport(BaseModel):
    job_id: str

# [수정] JobSubmitRequest 클래스
class JobSubmitRequest(BaseModel):
    job_id: str
    model_name: str
    dataset: str
    epochs: int = 20
    batch_size_per_gpu: int = 64
    learning_rate: float = 1e-3
    min_gpus: int = 1  # [NEW] Lucid: 사용자가 최소 GPU 지정 가능

class TelemetryData(BaseModel):
    node_id: str
    gpu_index: int
    gpu_util: float
    gpu_mem_util: float = 0.0 # [NEW] Lucid 논문의 Um (Memory I/O Utilization)
    power_w: float
    mem_used_mb: float
    mem_total_mb: float

class JobStatusReport(BaseModel):
    job_id: str
    status: Optional[str] = None
    exit_code: Optional[int] = None
    stderr_tail: Optional[str] = None
    final_accuracy: Optional[float] = None

class JobStoppedReport(BaseModel):
    job_id: str

class JobMetrics(BaseModel):
    job_id: str
    attained_service: float

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting background scheduling loop...")
    metrics_logger.log_scheduler("Global Server Started. Scheduler loop initiated.")
    asyncio.create_task(schedule_and_dispatch_jobs())
    yield
    metrics_logger.log_scheduler("Global Server Shutting Down.")

app = FastAPI(lifespan=lifespan)

# --- Helper Functions ---
def _set_node_status(node_ids: List[str], status: str, job_id: Optional[str]):
    with NODE_REGISTRY_LOCK:
        for node_id in node_ids:
            if node_id in NODE_REGISTRY:
                NODE_REGISTRY[node_id]["status"] = status
                NODE_REGISTRY[node_id]["current_job_id"] = job_id

# [수정] calculate_cluster_metrics 함수
def calculate_cluster_metrics():
    cluster_stats = defaultdict(lambda: {"total": 0, "used": 0})
    with NODE_REGISTRY_LOCK:
        for node_info in NODE_REGISTRY.values():
            c_name = node_info.get("cluster", "Cluster_Default")
            cluster_stats[c_name]["total"] += 1
            if node_info["status"] == "busy":
                cluster_stats[c_name]["used"] += 1
    with JOB_QUEUE_LOCK:
        queue_len = len(JOB_QUEUE)
    ts = get_now_str()
    for c_name, stats in cluster_stats.items():
        total = stats["total"]
        used = stats["used"]
        free = total - used
        u_t = used / total if total > 0 else 0.0
        
        # [중요] Pollux 전용 컬럼(U_t, E_t, p_fair 등) 자리에 Lucid는 
        # 개념이 없으므로 1.0(효율), 0.5(공정성), 0.0(라그랑주) 등 더미 값을 채워 로그 포맷 유지
        metrics_logger.log_csv("csp_metrics", [
            ts, c_name, queue_len, total, used, free, f"{u_t:.2f}", 
            1.0, 0.5, 0.0, 0.0, 0.0
        ])

def _count_cluster_gpus(cluster_id: str) -> int:
    with NODE_REGISTRY_LOCK:
        return sum(
            1
            for info in NODE_REGISTRY.values()
            if info.get("cluster") == cluster_id
        )

def _reconcile_node_states():
    with ACTIVE_JOBS_LOCK, NODE_REGISTRY_LOCK:
        active_job_ids = set(ACTIVE_JOBS.keys())

        # 1) ACTIVE_JOBS 기준으로 busy 상태 보정
        for jid, info in ACTIVE_JOBS.items():
            nodes = info.get("nodes", [])
            for nid in nodes:
                if nid not in NODE_REGISTRY:
                    continue
                ninfo = NODE_REGISTRY[nid]
                changed = False
                if ninfo.get("current_job_id") != jid:
                    log.warning(
                        f"[Reconcile] Node {nid} current_job_id={ninfo.get('current_job_id')} "
                        f"→ {jid} (from ACTIVE_JOBS)"
                    )
                    ninfo["current_job_id"] = jid
                    changed = True
                if ninfo.get("status") != "busy":
                    log.warning(
                        f"[Reconcile] Node {nid} status={ninfo.get('status')} → busy "
                        f"(job_id={jid})"
                    )
                    ninfo["status"] = "busy"
                    changed = True
                if changed:
                    NODE_REGISTRY[nid] = ninfo

        # 2) NODE_REGISTRY 기준으로 "유령 busy" 정리
        for nid, ninfo in NODE_REGISTRY.items():
            jid = ninfo.get("current_job_id")
            if jid and jid not in active_job_ids and ninfo.get("status") == "busy":
                log.warning(
                    f"[Reconcile] Node {nid} busy with non-active job_id={jid} → idle/None"
                )
                ninfo["current_job_id"] = None
                ninfo["status"] = "idle"
                NODE_REGISTRY[nid] = ninfo

# [수정] build_runtime_jobs 함수
def build_runtime_jobs() -> List[RuntimeJobState]:
    jobs = []
    with ACTIVE_JOBS_LOCK, NODE_REGISTRY_LOCK:
        for jid, info in ACTIVE_JOBS.items():
            nodes = info.get("nodes", [])
            if not nodes: continue
            
            cfg = info.get("config", {})
            rj = RuntimeJobState(
                job_id=jid,
                model_name=cfg.get("model_name", "default"),
                dataset=cfg.get("dataset", "default"),
                cluster_id=NODE_REGISTRY[nodes[0]].get("cluster", "default"),
                current_gpus=len(nodes),
                current_local_batch=int(cfg.get("batch_size_per_gpu", 64)),
                start_ts=info.get("start_ts", datetime.datetime.now()).timestamp(),
                # [Lucid] 상태 복원
                is_profiled=info.get("is_profiled", False),
                sharing_score=info.get("sharing_score", 2)
            )
            
            # [Lucid] Profiler가 누적한 메트릭이 있다면 캐시에서 복원
            if jid in RUNTIME_JOBS:
                cached = RUNTIME_JOBS[jid]
                rj.sum_gpu_util = cached.sum_gpu_util
                rj.sum_gpu_mem_used = cached.sum_gpu_mem_used
                rj.sum_gpu_mem_util = cached.sum_gpu_mem_util
                rj.metric_count = cached.metric_count
            
            jobs.append(rj)
    return jobs

async def _launch_job_on_nodes(job_config: Dict[str, Any], assigned_nodes: List[str]):
    job_id = job_config["job_id"]
    world_size = len(assigned_nodes)
    if world_size == 0:
        return

    # Master 노드 정보 조회
    master_node_id = assigned_nodes[0]
    with NODE_REGISTRY_LOCK:
        master_info = NODE_REGISTRY[master_node_id]
        master_ip = master_info["ip"]
        cluster_name = master_info.get("cluster", "Cluster_Default")

    master_port = next(MASTER_PORT_COUNTER)

    # 비동기 Launch 요청 헬퍼 함수
    async def _send_launch(node_id: str, url: str, payload: Dict):
        try:
            async with httpx.AsyncClient() as c:
                resp = await c.post(url, json=payload, timeout=10)
            return node_id, resp.status_code, resp.text
        except Exception as e:
            return node_id, None, str(e)

    tasks = []
    for rank, node_id in enumerate(assigned_nodes):
        with NODE_REGISTRY_LOCK:
            node_info = NODE_REGISTRY[node_id]
            prefix = node_info["prefix"]
            script_file = job_config["script_file"]
            # script_path 구성 (node_info의 prefix 활용)
            script_path = f"{prefix}/{script_file}"
            url = f"http://{node_info['ip']}:{node_info['agent_port']}/launch_task"

        payload = {
            "job_id": job_id,
            "script_path": script_path,
            "master_addr": master_ip,
            "master_port": master_port,
            "world_size": world_size,
            "rank": rank,
            "local_rank": 0,
            "gpu_id": node_info["gpu_id"],

            "epochs": job_config.get("epochs", 20),
            "checkpoint_dir": f"./checkpoints/{job_id}",
            "global_server_addr": GLOBAL_SERVER_ADDRESS,
            "resume_from_checkpoint": job_config.get("latest_checkpoint_path"),

            "dataset": job_config.get("dataset", "default"),
            "model_name": job_config.get("model_name", "default"),
            
            # [LUCID 변경점] 
            # Pollux처럼 동적 튜닝된 배치가 아니라, 사용자가 제출한 고정 배치 크기를 사용합니다.
            "batch_size_per_gpu": job_config.get("batch_size_per_gpu", 64),
            "learning_rate": job_config.get(
                "learning_rate",
                job_config.get("base_lr", 1e-3),
            ),
        }

        tasks.append(asyncio.create_task(_send_launch(node_id, url, payload)))

    # 모든 노드에 Launch 요청 전송 및 대기
    results = await asyncio.gather(*tasks)

    # --- 응답 상태 체크 ---
    failed_nodes = []
    for node_id, status_code, body in results:
        if status_code != 200:
            failed_nodes.append((node_id, status_code, body))

    if failed_nodes:
        # 하나라도 실패하면 전체 실패로 간주 및 로그 기록
        msg_lines = [
            f"{node_id}: status={status_code}, body={str(body)[:200]}"
            for (node_id, status_code, body) in failed_nodes
        ]
        msg = f"[LAUNCH_FAIL] job={job_id}, errors: " + " | ".join(msg_lines)
        log.error(msg)
        metrics_logger.log_scheduler(msg)

        metadata = {
            "assigned_nodes": assigned_nodes,
            "errors": msg_lines,
        }
        metrics_logger.log_csv("job_events", [
            get_now_str(),
            "launch_failed",
            job_id,
            cluster_name,
            world_size,
            "launch_task_failed",
            json.dumps(metadata),
        ])

        # 실패한 job은 다시 큐 앞쪽에 넣어서 유실 방지 (Retry 로직)
        with JOB_QUEUE_LOCK:
            already = any(j.get("job_id") == job_id for j in JOB_QUEUE)
            if not already:
                JOB_QUEUE.insert(0, job_config)
        return

    # --- [LUCID 핵심] 모든 노드 성공 시 상태 등록 ---
    start_ts = datetime.datetime.now()
    sub_ts = job_config.get("submitted_ts", start_ts)

    with ACTIVE_JOBS_LOCK:
        ACTIVE_JOBS[job_id] = {
            "config": job_config,
            "status": "running",
            "nodes": assigned_nodes,
            "stop_event": asyncio.Event(),
            "latest_progress": 0.0,
            "submitted_ts": sub_ts,
            "start_ts": start_ts,
            "attained_service": float(job_config.get("attained_service", 0.0)),
            
            # [LUCID STATE] 스케줄러가 판단한 프로파일링 및 공유 점수 상태를 유지
            "is_profiled": job_config.get("is_profiled", False),
            "sharing_score": job_config.get("sharing_score", 2),
            
            "latest_checkpoint_path": job_config.get("latest_checkpoint_path"),
        }

    _set_node_status(assigned_nodes, "busy", job_id)

    # 이벤트 로깅
    metadata = {"nodes": assigned_nodes, "started_ts": start_ts.isoformat()}
    metrics_logger.log_csv("job_events", [
        get_now_str(),
        "started",
        job_id,
        cluster_name,
        world_size,
        f"started_on={cluster_name}, g={world_size}",
        json.dumps(metadata),
    ])

    metrics_logger.log_csv("queue_events", [
        get_now_str(),
        "dequeue_start",
        job_id,
        len(JOB_QUEUE), # 큐에서 빠진 직후의 길이 (호출처에서 pop 수행됨)
        f"g={world_size}, cluster={cluster_name}",
    ])
    
    metrics_logger.log_scheduler(f"[LUCID] Launched {job_id} on {assigned_nodes}")

async def _stop_job(job_id: str) -> bool:
    """
    단순화된 stop 로직:
    - ACTIVE_JOBS에서 job을 제거
    - 해당 노드들에 /stop_task 전송 (best-effort)
    - NODE_REGISTRY 상태를 idle로 되돌림
    - job_events에 stopped(Preempted) 찍고 끝

    /report_job_stopped 핸드셰이크는 더 이상 의존하지 않음.
    """
    with ACTIVE_JOBS_LOCK:
        if job_id not in ACTIVE_JOBS:
            # 이미 정리된 job이면 OK로 본다
            return True
        job_info = ACTIVE_JOBS.pop(job_id)
        assigned_nodes = list(job_info.get("nodes", []))
        world_size = len(assigned_nodes)

    cluster_name = "Cluster_Default"
    with NODE_REGISTRY_LOCK:
        if assigned_nodes and assigned_nodes[0] in NODE_REGISTRY:
            cluster_name = NODE_REGISTRY[assigned_nodes[0]].get("cluster", "Cluster_Default")

    async def send_stop(node_id: str):
        with NODE_REGISTRY_LOCK:
            info = NODE_REGISTRY.get(node_id)
            if not info:
                return
            url = f"http://{info['ip']}:{info['agent_port']}/stop_task"

        try:
            async with httpx.AsyncClient() as c:
                await c.post(url, json={"job_id": job_id}, timeout=5)
        except Exception:
            # 워커가 죽어 있어도 여기서 실패하면 그냥 넘어감
            pass

    tasks = [asyncio.create_task(send_stop(nid)) for nid in assigned_nodes]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # 노드 상태를 즉시 idle로 되돌린다
    _set_node_status(assigned_nodes, "idle", None)

    # 이벤트 로그
    metrics_logger.log_csv("job_events", [
        get_now_str(),
        "stopped",
        job_id,
        cluster_name,
        world_size,
        "Preempted",
        "{}",
    ])

    log_scheduler_state(f"stop job={job_id}")
    return True

async def _scale_and_requeue_job(rj: RuntimeJobState, desired_g: int, suggested_local_b: int):
    jid = rj.job_id

    log.info(
        f"[POLLUX][SCALE_EXEC] start job={jid} "
        f"desired_g={desired_g}, suggested_local_b={suggested_local_b}"
    )

    # g=0이면 순수 preemption (다시 큐에 넣지 않음)
    if desired_g <= 0:
        log.info(f"[POLLUX][SCALE_EXEC] job={jid} new_g=0 → stop only, no requeue")
        stop_ok = await _stop_job(jid)
        if not stop_ok:
            log.error(f"[POLLUX][SCALE_EXEC] job={jid} preempt(stop) failed")
        return

    # 1) 현재 config / attained_service / checkpoint 먼저 복사
    with ACTIVE_JOBS_LOCK:
        info = ACTIVE_JOBS.get(jid)
        if not info:
            log.warning(f"[POLLUX][SCALE_EXEC] job={jid} not found in ACTIVE_JOBS")
            return
        base_cfg = dict(info.get("config", {}))
        prev_attained = float(info.get("attained_service", 0.0))
        latest_ckpt = info.get("latest_checkpoint_path")

    # 2) stop 후 재큐잉
    log.info(f"[POLLUX][SCALE_EXEC] job={jid} calling _stop_job()")
    stop_ok = await _stop_job(jid)
    if not stop_ok:
        log.error(f"[POLLUX][SCALE_EXEC] job={jid} stop failed, skip requeue")
        return
    log.info(f"[POLLUX][SCALE_EXEC] job={jid} stop_ok, proceed to requeue")

    # 3) Pollux-style local config 선택 (batch, accum, lr)
    base_batch = int(base_cfg.get("batch_size_per_gpu", rj.current_local_batch or 64))
    base_accum = int(base_cfg.get("grad_accum", getattr(rj, "current_grad_accum", 1) or 1))
    base_sps = float(rj.last_sps) if rj.last_sps > 0 else 1.0

    if suggested_local_b is not None and suggested_local_b > 0:
        default_batch = int(suggested_local_b)
    else:
        default_batch = base_batch
    default_accum = base_accum

    best_b, best_a, best_sps, best_se, best_gp = pollux_best_local_config_for_g(
        job_id=jid,
        model_name=rj.model_name,
        dataset=rj.dataset,
        g=int(desired_g),
        default_batch=default_batch,
        default_accum=default_accum,
        default_sps=base_sps,
    )

    base_lr = float(base_cfg.get("base_lr", base_cfg.get("learning_rate", 1e-3)))
    base_global_B = float(max(1, base_batch) * max(1, base_accum))
    new_global_B = float(max(1, best_b) * max(1, best_a) * max(1, desired_g))

    if base_global_B <= 0:
        new_lr = base_lr
    else:
        scale = new_global_B / base_global_B
        new_lr = base_lr * scale

    cfg = dict(base_cfg)
    cfg["preferred_cluster"] = rj.cluster_id
    cfg["pollux_desired_gpus"] = int(desired_g)
    cfg["batch_size_per_gpu"] = int(best_b)
    cfg["grad_accum"] = int(best_a)
    cfg["learning_rate"] = float(new_lr)
    cfg["attained_service"] = prev_attained
    if latest_ckpt:
        cfg["latest_checkpoint_path"] = latest_ckpt

    # 5) 큐에 다시 넣기 (중복 방지)
    with JOB_QUEUE_LOCK:
        already = any(j["job_id"] == jid for j in JOB_QUEUE)
        if not already:
            msg = (
                f"[POLLUX][REQUEUE] job={jid} "
                f"g={desired_g}, batch={cfg['batch_size_per_gpu']}, accum={cfg.get('grad_accum', 1)}, "
                f"pref_cluster={cfg.get('preferred_cluster')}, "
                f"prev_attained={prev_attained}, new_lr={cfg['learning_rate']:.6f}"
            )
            log.info(msg)
            metrics_logger.log_scheduler(msg)

            JOB_QUEUE.insert(0, cfg)
        else:
            log.warning(f"[POLLUX][REQUEUE] job={jid} already in queue, skip duplicate insert")

    log_scheduler_state(
        f"scale_requeue job={jid} g={desired_g} batch={cfg['batch_size_per_gpu']} accum={cfg.get('grad_accum', 1)}"
    )

def _make_scale_job_fn():
    def _scale_job(rj: RuntimeJobState, new_g: int, new_local_batch: int) -> bool:
        now_ts = time.time()

        msg = (
            f"[POLLUX][SCALE_REQ] job={rj.job_id} "
            f"cluster={rj.cluster_id} "
            f"g: {rj.current_gpus} -> {new_g}, "
            f"batch: {rj.current_local_batch} -> {new_local_batch}, "
            f"attained={rj.attained_service:.2f}, "
            f"last_sps={rj.last_sps:.2f}"
        )
        log.info(msg)
        metrics_logger.log_scheduler(msg)

        # last_scaled_at_ts 갱신 (쿨다운에 사용)
        with ACTIVE_JOBS_LOCK:
            if rj.job_id in ACTIVE_JOBS:
                ACTIVE_JOBS[rj.job_id]["last_scaled_at_ts"] = now_ts

        # 실제 scale 동작은 비동기로 처리 (stop + requeue or preempt)
        asyncio.create_task(_scale_and_requeue_job(rj, new_g, new_local_batch))
        return True

    return _scale_job

async def schedule_and_dispatch_jobs():
    global RUNTIME_JOBS
    metrics_logger.log_scheduler("Lucid Scheduler Loop Started")

    while True:
        try:
            _reconcile_node_states()
            calculate_cluster_metrics()

            # 1) Update Runtime State & Profiling
            current_jobs = build_runtime_jobs()
            for rj in current_jobs:
                RUNTIME_JOBS[rj.job_id] = rj

                if not rj.is_profiled:
                    elapsed = time.time() - rj.start_ts
                    if elapsed >= T_PROF:
                        lucid_classify_job(rj)
                        with ACTIVE_JOBS_LOCK:
                            if rj.job_id in ACTIVE_JOBS:
                                ACTIVE_JOBS[rj.job_id]["is_profiled"] = True
                                ACTIVE_JOBS[rj.job_id]["sharing_score"] = rj.sharing_score

                        metrics_logger.log_scheduler(
                            f"[LUCID-PROFILE] Job {rj.job_id} Classifed -> SS={rj.sharing_score} (Tiny/Med/Jumbo)"
                        )

            # 2) Snapshot queue (sorted) - only once
            with JOB_QUEUE_LOCK:
                JOB_QUEUE.sort(key=lambda j: int(j.get("epochs", 20)) * int(j.get("min_gpus", 1)))
                queue_snapshot = [dict(j) for j in JOB_QUEUE]  # defensive copy

            # 3) Build node_ss_map from ACTIVE_JOBS
            node_ss_map = defaultdict(int)
            with ACTIVE_JOBS_LOCK:
                for jid, info in ACTIVE_JOBS.items():
                    ss = int(info.get("sharing_score", 2))
                    for nid in info.get("nodes", []) or []:
                        node_ss_map[nid] += ss

            # 4) Decide placements (no JOB_QUEUE mutation here)
            jobs_to_launch = []  # List[Tuple[job_id, candidate_nodes]]
            reserved_nodes = set()
            reserved_job_ids = set()

            with NODE_REGISTRY_LOCK:
                # 미리 idle_by_cluster 만들어두면 gang 선택이 쉬움
                idle_by_cluster = defaultdict(list)
                for nid, ninfo in NODE_REGISTRY.items():
                    if ninfo.get("status") == "idle":
                        idle_by_cluster[ninfo.get("cluster", "default")].append(nid)

            for job_cfg in queue_snapshot:
                jid = str(job_cfg.get("job_id", "")).strip()
                if not jid or jid in reserved_job_ids:
                    continue

                needed_gpus = int(job_cfg.get("min_gpus", 1))
                is_gang = (needed_gpus >= 4)

                candidate_nodes = []

                if is_gang:
                    # ✅ gang은 단일 클러스터에서만 4 idle 확보 (교차 금지)
                    cluster_order = ["clusterB", "clusterA"]
                    chosen = None
                    for cid in cluster_order:
                        # 이미 이 tick에서 예약한 노드는 제외
                        idle_list = [n for n in idle_by_cluster.get(cid, []) if n not in reserved_nodes]
                        if len(idle_list) >= needed_gpus:
                            chosen = idle_list[:needed_gpus]
                            break
                    if chosen is None:
                        continue
                    candidate_nodes = chosen

                else:
                    # ✅ non-gang packing (SS<=2)
                    with NODE_REGISTRY_LOCK:
                        for nid, ninfo in NODE_REGISTRY.items():
                            if nid in reserved_nodes:
                                continue

                            current_ss = int(node_ss_map.get(nid, 0))
                            new_job_ss = 2  # unprofiled=2 가정

                            if current_ss + new_job_ss <= G_SS_LIMIT:
                                candidate_nodes.append(nid)

                            if len(candidate_nodes) == needed_gpus:
                                break

                    if len(candidate_nodes) != needed_gpus:
                        continue

                # reserve
                reserved_job_ids.add(jid)
                for nid in candidate_nodes:
                    reserved_nodes.add(nid)
                    node_ss_map[nid] = int(node_ss_map.get(nid, 0)) + 2

                jobs_to_launch.append((jid, candidate_nodes))

            # 5) Remove selected jobs from JOB_QUEUE by job_id, then launch
            launch_tasks = []
            with JOB_QUEUE_LOCK:
                for jid, nodes in jobs_to_launch:
                    pos = next((k for k, j in enumerate(JOB_QUEUE) if j.get("job_id") == jid), None)
                    if pos is None:
                        continue
                    job = JOB_QUEUE.pop(pos)
                    launch_tasks.append(_launch_job_on_nodes(job, nodes))

            if launch_tasks:
                await asyncio.gather(*launch_tasks)

        except Exception as e:
            log.error(f"Lucid Loop Error: {e}", exc_info=True)

        await asyncio.sleep(5)

# --- API Endpoints ---
@app.post("/submit_job")
async def submit_job(req: JobSubmitRequest):
    req_data = req.model_dump()
    log.info(f"[Job Submission] Received Request Data: {req_data}")

    ts = datetime.datetime.now()
    data = req_data
    data["submitted_ts"] = ts

    model = data["model_name"]
    dataset = data["dataset"]

    if is_gang_model(model, dataset):
        gang_g = int(GANG_JOBS[(model, dataset)])  # 보통 4
        data["min_gpus"] = gang_g
        # (선택) pollux_desired_gpus도 같이 맞춰서 혼동 제거
        data["pollux_desired_gpus"] = gang_g

    # 1) 스크립트 파일명 결정
    try:
        script_file = MODEL_DS_MAP[(model, dataset)]
    except KeyError:
        script_file = make_filename(model, dataset)
    data["script_file"] = script_file

    base_batch = data.get("batch_size_per_gpu", 64)
    base_lr = data.get("learning_rate", 1e-3)
    data["base_lr"] = base_lr

    #  2) Pollux-style admission으로 초기 g 결정
    initial_g = _decide_initial_g_for_job(
        job_id=req.job_id,
        model_name=model,
        dataset=dataset,
        base_batch=base_batch,
    )
    data["pollux_desired_gpus"] = initial_g

    #  로그용 free_slots도 같이 찍어보고 싶으면 여기서 다시 계산
    free_slots = _compute_cluster_free_gpus()
    log.info(
        f"[{req.job_id}] initial pollux_desired_gpus={initial_g} "
        f"(free_slots={free_slots})"
    )

    # 3) 큐에 넣기 전에 중복 검사
    with JOB_QUEUE_LOCK:
        if any(j["job_id"] == req.job_id for j in JOB_QUEUE):
            raise HTTPException(400, "Already in queue")
        with ACTIVE_JOBS_LOCK:
            if req.job_id in ACTIVE_JOBS:
                raise HTTPException(400, "Running")
        JOB_QUEUE.append(data)

    metadata = {
        "model": model,
        "dataset": dataset,
        "epochs": req.epochs,
        "batch_size_per_gpu": req.batch_size_per_gpu,
        "learning_rate": base_lr,
        "initial_g": initial_g,
    }

    metrics_logger.log_csv("job_events", [
        get_now_str(),
        "submitted",
        req.job_id,
        "-",
        0,
        "",
        json.dumps(metadata),
    ])

    metrics_logger.log_csv("queue_events", [
        get_now_str(),
        "enqueue",
        req.job_id,
        len(JOB_QUEUE),
        f"g_target={initial_g}, batch={req.batch_size_per_gpu}",
    ])

    log_scheduler_state(f"enqueue job={req.job_id}, g_target={initial_g}")

    return {"status": "queued", "job_id": req.job_id}

@app.post("/report_checkpoint")
async def report_checkpoint(report: JobCheckpointReport):
    job_id = report.job_id

    with ACTIVE_JOBS_LOCK:
        if job_id in ACTIVE_JOBS:
            ACTIVE_JOBS[job_id]["latest_checkpoint_path"] = report.checkpoint_path
            log.info(f"[{job_id}] Checkpoint: {report.checkpoint_path}")

    global JOB_LATEST_METRICS
    JOB_LATEST_METRICS[job_id] = {
        "epoch": report.current_epoch,
        "total_epochs": report.total_epochs,
        "accuracy": report.latest_accuracy,
        "eval_loss": report.latest_eval_loss,
        "checkpoint_path": report.checkpoint_path,
    }

    # Pollux-style goodput 튜너는 /report_progress 기반으로 동작하므로
    # 여기서는 acc/loss를 사용한 추가 튜닝을 수행하지 않는다.
    return {"status": "checkpoint_acked"}

@app.post("/report_job_status")
async def report_job_status(rep: JobStatusReport):
    job_id = rep.job_id
    status_raw = rep.status or ""
    status = status_raw.upper()
    exit_code = rep.exit_code

    # ==== 0) latest accuracy 결정 (Worker vs Checkpoint 중 우선순위 적용) ====
    global JOB_LATEST_METRICS
    m = JOB_LATEST_METRICS.get(job_id) if "JOB_LATEST_METRICS" in globals() else None
    acc_from_ckpt = None
    if m is not None:
        acc_from_ckpt = m.get("accuracy")

    # 우선순위: 1) WorkerAgent에서 온 final_accuracy  2) checkpoint 기반 accuracy
    final_acc: Optional[float] = None
    if rep.final_accuracy is not None:
        final_acc = rep.final_accuracy
    elif acc_from_ckpt is not None:
        final_acc = acc_from_ckpt

    job_info = None
    assigned_nodes: List[str] = []
    cluster_name = "Cluster_Default"
    world_size = 0

    # 0) ACTIVE_JOBS에서 가능한 정보 먼저 긁어오기
    with ACTIVE_JOBS_LOCK:
        if job_id in ACTIVE_JOBS:
            job_info = ACTIVE_JOBS[job_id]
            assigned_nodes = list(job_info.get("nodes", []))

    # 0-1) NODE_REGISTRY 기반 fallback
    if not assigned_nodes:
        assigned_nodes = _get_nodes_for_job(job_id)

    if assigned_nodes:
        with NODE_REGISTRY_LOCK:
            first = assigned_nodes[0]
            if first in NODE_REGISTRY:
                cluster_name = NODE_REGISTRY[first].get("cluster", "Cluster_Default")
        world_size = len(assigned_nodes)

    # ---------- 1) preempt/kill (SIGTERM -15) 처리 ----------
    if status in ["FINISHED", "FAILED"] and exit_code == -15:
        return {"status": "ok"}

    # ---------- 2) 최종 완료 / 실패 ----------
    if status in ["FINISHED", "FAILED"]:
        global FINISHED_JOB_IDS

        # 2-1) 중복 방지 – 이미 처리된 job이면 duplicate 이벤트만 남김
        if job_id in FINISHED_JOB_IDS:
            md: Dict[str, Any] = {}
            if exit_code is not None:
                md["exit_code"] = exit_code
            if rep.stderr_tail:
                md["stderr_tail"] = rep.stderr_tail[:200]
            if final_acc is not None:
                md["final_accuracy"] = float(final_acc)

            metrics_logger.log_csv("job_events", [
                get_now_str(),
                f"{status.lower()}_duplicate",
                job_id,
                cluster_name,
                world_size,
                "Duplicate final status ignored",
                json.dumps(md) if md else "{}",
            ])
            return {"status": "ok"}

        FINISHED_JOB_IDS.add(job_id)

        # 2-2) ACTIVE_JOBS / 노드 상태 정리
        with ACTIVE_JOBS_LOCK:
            if job_id in ACTIVE_JOBS:
                job_info = ACTIVE_JOBS.pop(job_id)
                assigned_nodes = job_info.get("nodes", assigned_nodes)

        if assigned_nodes:
            _set_node_status(assigned_nodes, "idle", None)
            world_size = len(assigned_nodes)

        # 2-3) job_metrics 작성
        sub_ts_str = ""
        start_ts_str = ""
        end_ts = datetime.datetime.now()
        queued_str = ""
        jct_str = ""
        model_name = ""
        dataset = ""

        if job_info:
            cfg = job_info.get("config", {})
            sub_ts = cfg.get("submitted_ts")
            start_ts = job_info.get("start_ts")

            model_name = cfg.get("model_name", "")
            dataset = cfg.get("dataset", "")

            if isinstance(sub_ts, datetime.datetime):
                sub_ts_str = sub_ts.strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(start_ts, datetime.datetime):
                start_ts_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(sub_ts, datetime.datetime):
                    queued = (start_ts - sub_ts).total_seconds()
                    queued_str = f"{queued:.2f}"
                    jct = (end_ts - sub_ts).total_seconds()
                    jct_str = f"{jct:.2f}"

        metrics_logger.log_csv("job_metrics", [
            job_id,
            cluster_name,
            model_name,
            dataset,
            world_size,
            sub_ts_str,
            start_ts_str,
            end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            queued_str,
            jct_str,
            status,
            float(final_acc) if final_acc is not None else "",
        ])

        # 2-4) job_events에 최종 이벤트 남기기
        md: Dict[str, Any] = {}
        if exit_code is not None:
            md["exit_code"] = exit_code
        if rep.stderr_tail:
            md["stderr_tail"] = rep.stderr_tail[:200]
        if final_acc is not None:
            md["final_accuracy"] = float(final_acc)

        metrics_logger.log_csv("job_events", [
            get_now_str(),
            status.lower(),
            job_id,
            cluster_name,
            world_size,
            "Done",
            json.dumps(md) if md else "{}",
        ])

        log_scheduler_state(f"finish job={job_id} status={status} exit_code={exit_code}")
        return {"status": "ok"}

    # ---------- 3) 그 외 (비-터미널 status) ----------
    log_scheduler_state(f"report_job_status non-terminal job={job_id} status={status_raw}")
    return {"status": "ok"}

# [수정] global_server_pollux.py 내부 report_progress 함수

# [global_server_pollux.py의 report_progress 함수만 교체]

@app.post("/report_progress")
async def report_progress(req: ProgressReport):
    # 값 파싱
    g = max(1, int(req.world_size))
    local_b = max(1, int(req.local_batch))
    accum = max(1, int(req.grad_accum))
    sps = float(req.steps_per_sec)

    # 로그 기록 (기존 유지)
    msg = (
        f"[PROGRESS] job={req.job_id} "
        f"g={g}, batch={local_b}, accum={accum}, sps={sps:.3f}, "
        f"loss={req.loss}, acc={req.accuracy}"
    )
    # log.info(msg) # 필요시 주석 해제
    metrics_logger.log_scheduler(msg)

    # ACTIVE_JOBS 상태 업데이트
    with ACTIVE_JOBS_LOCK:
        info = ACTIVE_JOBS.get(req.job_id)
        if info is not None:
            info["last_sps"] = sps
            cfg = dict(info.get("config", {}))
            cfg["batch_size_per_gpu"] = local_b
            cfg["grad_accum"] = accum
            info["config"] = cfg
            ACTIVE_JOBS[req.job_id] = info
        
        # RUNTIME_JOBS 업데이트 (Lucid 스케줄러용 객체)
        if req.job_id in RUNTIME_JOBS:
            job_state = RUNTIME_JOBS[req.job_id]
            
            # [중요] record_goodput_sample을 호출하여 "기록" 행위 자체는 유지함.
            # 단, Lucid에서는 GNS/StatEff가 없으므로(None), 0.0 등 빈 값으로 채워서 보냄.
            # 이렇게 하면 함수 호출 구조가 유지되므로 에러가 나지 않음.
            record_goodput_sample(
                job_state,
                g,
                local_b,
                accum,
                sps,
                req.gns if req.gns is not None else 0.0,      # 없으면 0.0
                req.stat_eff if req.stat_eff is not None else 0.0 # 없으면 0.0
            )

    return {"status": "ok"}

@app.post("/report_job_stopped")
async def report_job_stopped(rep: JobStoppedReport):
    job_id = rep.job_id
    with ACTIVE_JOBS_LOCK:
        info = ACTIVE_JOBS.get(job_id)
        if not info:
            return {"status": "ok"}
        stop_event = info.get("stop_event")

    if isinstance(stop_event, asyncio.Event):
        stop_event.set()

    log_scheduler_state(f"report_job_stopped job={job_id}")
    return {"status": "ok"}

@app.post("/report_telemetry")
async def report_telemetry(data: TelemetryData):
    metrics_logger.log_csv("telemetry", [
        get_now_str(),
        data.node_id,
        data.gpu_index,
        data.gpu_util,
        data.power_w,
        data.mem_used_mb,
        data.mem_total_mb,
    ])
    return {"status": "ok"}

def _get_nodes_for_job(job_id: str) -> List[str]:
    nodes: List[str] = []
    with NODE_REGISTRY_LOCK:
        for nid, info in NODE_REGISTRY.items():
            if info["current_job_id"] == job_id:
                nodes.append(nid)
    return nodes

@app.get("/debug_state")
async def debug_state():
    # 1) 큐 상태
    with JOB_QUEUE_LOCK:
        queue_snapshot = [
            {
                "job_id": j.get("job_id"),
                "model_name": j.get("model_name"),
                "dataset": j.get("dataset"),
                "epochs": j.get("epochs"),
                "batch_size_per_gpu": j.get("batch_size_per_gpu"),
                "learning_rate": j.get("learning_rate"),
                "preferred_cluster": j.get("preferred_cluster"),
                "pollux_desired_gpus": j.get("pollux_desired_gpus", 1),
                "submitted_ts": str(j.get("submitted_ts")),
            }
            for j in JOB_QUEUE
        ]

    # 2) 실행 중인 잡 상태
    with ACTIVE_JOBS_LOCK:
        active_snapshot = []
        for jid, info in ACTIVE_JOBS.items():
            cfg = info.get("config", {})
            active_snapshot.append({
                "job_id": jid,
                "status": info.get("status"),
                "model_name": cfg.get("model_name"),
                "dataset": cfg.get("dataset"),
                "nodes": info.get("nodes", []),
                "world_size": len(info.get("nodes", [])),
                "submitted_ts": str(cfg.get("submitted_ts")),
                "start_ts": str(info.get("start_ts")),
                "attained_service": info.get("attained_service"),
                "last_scaled_at_ts": info.get("last_scaled_at_ts"),
                "last_sps": info.get("last_sps", 0.0),
            })

    # 3) 노드 상태
    with NODE_REGISTRY_LOCK:
        node_snapshot = {
            nid: {
                "ip": ninfo.get("ip"),
                "agent_port": ninfo.get("agent_port"),
                "cluster": ninfo.get("cluster"),
                "gpu_id": ninfo.get("gpu_id"),
                "status": ninfo.get("status"),
                "current_job_id": ninfo.get("current_job_id"),
            }
            for nid, ninfo in NODE_REGISTRY.items()
        }

    return {
        "queue": queue_snapshot,
        "active_jobs": active_snapshot,
        "nodes": node_snapshot,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Server Manager with Pollux Scheduler (Co-Adaptive)")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--public_ip', type=str, required=True)

    args = parser.parse_args()

    GLOBAL_SERVER_ADDRESS = f"http://{args.public_ip}:{args.port}"
    log.info(f"Global Server Address: {GLOBAL_SERVER_ADDRESS}")

    if args.public_ip == "127.0.0.1":
        with NODE_REGISTRY_LOCK:
            for node_id in NODE_REGISTRY:
                NODE_REGISTRY[node_id]["ip"] = "127.0.0.1"

    uvicorn.run(app, host=args.host, port=args.port)