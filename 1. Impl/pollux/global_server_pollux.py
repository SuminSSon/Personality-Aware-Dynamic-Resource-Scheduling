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
    pollux_reallocation_tick,
    record_goodput_sample,
    pollux_initial_g,
    update_job_metrics_from_telemetry,
    is_gang_model,
    GANG_JOBS,
    pollux_best_local_config_for_g,
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
    "node_e": {"ip": "163.180.160.62", "agent_port": 8301, "gpu_id": 0,
               "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "clusterB"},
    "node_f": {"ip": "163.180.160.62", "agent_port": 8302, "gpu_id": 0,
               "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "clusterB"},
    "node_g": {"ip": "163.180.160.62", "agent_port": 8303, "gpu_id": 0,
               "prefix": "/data/breath12/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "clusterB"},
    "node_h": {"ip": "163.180.160.62", "agent_port": 8304, "gpu_id": 0,
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
POLLUX_DELTA_GOODPUT_THRESH_BUSY = 0.01
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
    cluster_free_gpus: Optional[int] = None,
    cluster_total_gpus: Optional[int] = None,  # ✅ 추가
) -> int:
    """
    launch 시점 free 기준으로 Pollux admission 재평가.
    - queue_len 같은 간접 신호 없이, free/total(=scarcity)로 initial_g를 결정.

    반환:
      - submit/admission 단계에서는 최소 1로 clamp해서 큐에 넣고,
        launch 시점에 다시 평가해 실제 desired_g로 사용.
    """
    if cluster_free_gpus is None:
        free_slots = _compute_cluster_free_gpus()
    else:
        free_slots = max(0, int(cluster_free_gpus))

    # total이 안 들어오면 registry로 계산
    if cluster_total_gpus is None:
        # 전체 NODE_REGISTRY 기준 total(전역) 말고, 여기서는 "현재 고려 클러스터 total"이 더 정확하지만
        # 이 함수 호출부에서 cluster_id를 알고 있으니 보통은 호출부에서 넘겨주는 걸 권장합니다.
        # fallback으로는 전역 total 사용(보수적/대략치)
        with NODE_REGISTRY_LOCK:
            total_slots = len(NODE_REGISTRY)
        cluster_total_gpus = int(total_slots)

    # submit 시점 free=0이어도 큐는 들어가야 하므로 1
    if free_slots <= 0:
        return 1

    max_gpus = min(4, free_slots)

    g = pollux_initial_g(
        job_id=job_id,
        model_name=model_name,
        dataset=dataset,
        min_gpus=1,
        max_gpus=max_gpus,
        cluster_free_gpus=free_slots,
        cluster_total_gpus=int(cluster_total_gpus) if cluster_total_gpus is not None else None,
        default_batch=int(base_batch),
        default_accum=1,
        default_sps=1.0,
        # 원하면 여기서 price/shape도 실험적으로 박아도 됨:
        # pollux_price_min=0.01, pollux_price_max=0.20, pollux_price_shape=2.0
    )

    # submit/admission 단계에서는 최소 1로 clamp
    g = max(1, min(int(g), max_gpus, free_slots))
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

class JobSubmitRequest(BaseModel):
    job_id: str
    model_name: str
    dataset: str
    epochs: int = 20
    batch_size_per_gpu: int = 64
    learning_rate: float = 1e-3

class TelemetryData(BaseModel):
    node_id: str
    gpu_index: int
    gpu_util: float
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
        total_slots = stats["total"]
        used_slots = stats["used"]
        free_slots = total_slots - used_slots
        u_t = used_slots / total_slots if total_slots > 0 else 0.0

        metrics_logger.log_csv("csp_metrics", [
            ts,
            c_name,
            queue_len,
            total_slots,
            used_slots,
            free_slots,
            f"{u_t:.2f}",
            0.5,
            0.5,
            LAMBDA_TIME,
            LAMBDA_COST,
            LAMBDA_FAIR,
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
            st = ninfo.get("status")

            # ✅ stopping은 stop 핸드셰이크/타임아웃 경로에서 정리되므로 여기서 건드리지 않는다
            if st == "stopping":
                continue

            if jid and jid not in active_job_ids and st == "busy":
                log.warning(
                    f"[Reconcile] Node {nid} busy with non-active job_id={jid} → idle/None"
                )
                ninfo["current_job_id"] = None
                ninfo["status"] = "idle"
                NODE_REGISTRY[nid] = ninfo

def build_runtime_jobs_for_cluster(cluster_id: str) -> List[RuntimeJobState]:
    jobs: List[RuntimeJobState] = []
    with ACTIVE_JOBS_LOCK, NODE_REGISTRY_LOCK:
        for jid, info in ACTIVE_JOBS.items():
            nodes = info.get("nodes", [])
            if not nodes:
                # 노드가 하나도 없으면 현재 cluster에서 돌아가는 job은 아님 (preempted 등)
                continue

            first_node = nodes[0]
            node_info = NODE_REGISTRY.get(first_node)
            if not node_info:
                continue

            node_cluster = node_info.get("cluster", "Cluster_Default")
            if node_cluster != cluster_id:
                continue

            cfg = info.get("config", {})

            current_g = len(nodes)
            current_batch = int(cfg.get("batch_size_per_gpu", 64))
            current_grad_accum = int(cfg.get("grad_accum", 1))
            last_sps = float(info.get("last_sps", 0.0))
            attained_service = float(info.get("attained_service", 0.0))
            last_scaled_at_ts = float(info.get("last_scaled_at_ts", 0.0))

            rj = RuntimeJobState(
                job_id=jid,
                model_name=cfg.get("model_name", "default"),
                dataset=cfg.get("dataset", "default"),
                cluster_id=cluster_id,
                current_gpus=current_g,
                current_local_batch=current_batch,
                current_grad_accum=current_grad_accum,
                progress=info.get("latest_progress", 0.0),
                attained_service=attained_service,
                last_scaled_at_ts=last_scaled_at_ts,

                # 🔹 Pollux 철학: elastic 경로에서는 기본적으로 g>=1 유지
                min_gpus=1,
                # 🔹 한 job이 최대 몇 GPU까지 쓸 수 있는지 상한
                max_gpus=4,

                last_sps=last_sps,
            )

            # 🔹 warmup용 start_ts (초 단위) 동적으로 부여
            start_dt = info.get("start_ts")
            if isinstance(start_dt, datetime.datetime):
                rj.start_ts = start_dt.timestamp()
            else:
                start_dt = info.get("start_ts")
                if isinstance(start_dt, datetime.datetime):
                    rj.start_ts = start_dt.timestamp()
                else:
                    rj.start_ts = 0.0


            jobs.append(rj)

    return jobs

async def _launch_job_on_nodes(job_config: Dict[str, Any], assigned_nodes: List[str]):
    job_id = job_config["job_id"]
    world_size = len(assigned_nodes)
    if world_size == 0:
        return

    # launch 전에 idle invariant 체크
    with NODE_REGISTRY_LOCK:
        for nid in assigned_nodes:
            info = NODE_REGISTRY.get(nid)
            if not info:
                log.error(f"[{job_id}] launch aborted: node {nid} not found in NODE_REGISTRY")
                return
            if info.get("status") != "idle" or info.get("current_job_id") not in (None, ""):
                log.error(
                    f"[{job_id}] launch aborted: node {nid} is not idle "
                    f"(status={info.get('status')}, current_job_id={info.get('current_job_id')})"
                )
                return

    master_node_id = assigned_nodes[0]
    with NODE_REGISTRY_LOCK:
        master_info = NODE_REGISTRY[master_node_id]
        master_ip = master_info["ip"]
        cluster_name = master_info.get("cluster", "Cluster_Default")

    master_port = next(MASTER_PORT_COUNTER)

    async def _send_launch(node_id: str, url: str, payload: Dict[str, Any]):
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
            "batch_size_per_gpu": job_config.get("batch_size_per_gpu", 64),
            "learning_rate": job_config.get(
                "learning_rate",
                job_config.get("base_lr", 1e-3),
            ),
        }

        tasks.append(asyncio.create_task(_send_launch(node_id, url, payload)))

    results = await asyncio.gather(*tasks)

    # --- 응답 상태 체크 ---
    failed_nodes = []
    for node_id, status_code, body in results:
        if status_code != 200:
            failed_nodes.append((node_id, status_code, body))

    if failed_nodes:
        # launch 전체를 실패로 본다
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

        # 실패한 job은 다시 큐에 넣어서 유실 방지
        with JOB_QUEUE_LOCK:
            already = any(j.get("job_id") == job_id for j in JOB_QUEUE)
            if not already:
                JOB_QUEUE.insert(0, job_config)

        # 노드 상태는 idle 그대로 유지
        return

    # --- 여기까지 왔으면 모든 노드에서 200 OK ---
    start_ts = datetime.datetime.now()
    sub_ts = job_config["submitted_ts"]

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
            "last_scaled_at_ts": time.time(),
            "latest_checkpoint_path": job_config.get("latest_checkpoint_path"),
        }

    _set_node_status(assigned_nodes, "busy", job_id)

    try:
        log_scheduler_state(f"start job={job_id}")
    except NameError:
        pass

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
        len(JOB_QUEUE),
        f"g={world_size}, cluster={cluster_name}",
    ])

async def _stop_job(job_id: str, stop_timeout_sec: float = 15.0) -> bool:
    """
    안전한 stop 로직:
    - ACTIVE_JOBS에서 제거하되, 노드를 즉시 idle로 풀지 않는다.
    - 노드를 'stopping'으로 표기하고 stop_task 전송
    - /report_job_stopped(stop_event) ACK를 최대 stop_timeout_sec까지 기다린다.
      (ACK가 없으면 강제 idle로 전환하되, 로그에 "FORCE"를 남긴다.)

    이렇게 해야 "같은 GPU에 두 잡이 동시에 뜨는" 최악을 막는다.
    """
    # 0) ACTIVE_JOBS에서 job_info 확보 + stop_event 확보 (먼저)
    with ACTIVE_JOBS_LOCK:
        job_info = ACTIVE_JOBS.get(job_id)
        if not job_info:
            return True
        assigned_nodes = list(job_info.get("nodes", []))
        world_size = len(assigned_nodes)
        stop_event = job_info.get("stop_event")

        # ACTIVE_JOBS에서 제거 (스케줄러가 다시 건드리지 않게)
        ACTIVE_JOBS.pop(job_id, None)

    cluster_name = "Cluster_Default"
    with NODE_REGISTRY_LOCK:
        if assigned_nodes and assigned_nodes[0] in NODE_REGISTRY:
            cluster_name = NODE_REGISTRY[assigned_nodes[0]].get("cluster", "Cluster_Default")

    # 1) 노드를 즉시 idle로 풀지 말고 stopping으로 전환
    if assigned_nodes:
        _set_node_status(assigned_nodes, "stopping", job_id)

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

    # 2) stop ack 기다리기 (가능하면)
    acked = False
    try:
        if isinstance(stop_event, asyncio.Event):
            await asyncio.wait_for(stop_event.wait(), timeout=stop_timeout_sec)
            acked = True
    except asyncio.TimeoutError:
        acked = False
    except Exception:
        acked = False

    # 3) 최종적으로 idle 전환 (ACK 여부에 따라 로그)
    if assigned_nodes:
        _set_node_status(assigned_nodes, "idle", None)

    note = "Preempted" if acked else f"Preempted(FORCE after {stop_timeout_sec}s)"
    metrics_logger.log_csv("job_events", [
        get_now_str(),
        "stopped",
        job_id,
        cluster_name,
        world_size,
        note,
        "{}",
    ])

    log_scheduler_state(f"stop job={job_id} acked={acked}")
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

        # ✅ 최종 안전 가드: 스케일 최소 간격(글로벌)
        MIN_SCALE_INTERVAL_SEC = 240.0  # 4분 (pollux_reallocation_tick의 min_residency와 맞추기)

        with ACTIVE_JOBS_LOCK:
            info = ACTIVE_JOBS.get(rj.job_id)
            if info is not None:
                last = float(info.get("last_scaled_at_ts", 0.0) or 0.0)
                if last > 0.0 and (now_ts - last) < MIN_SCALE_INTERVAL_SEC:
                    return False

                # last_scaled_at_ts 갱신 (쿨다운/가드용)
                info["last_scaled_at_ts"] = now_ts
                ACTIVE_JOBS[rj.job_id] = info

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

        asyncio.create_task(_scale_and_requeue_job(rj, new_g, new_local_batch))
        return True

    return _scale_job

def build_queued_runtime_jobs_for_cluster(cluster_id: str) -> List[RuntimeJobState]:
    out: List[RuntimeJobState] = []
    with JOB_QUEUE_LOCK:
        snap = list(JOB_QUEUE)

    now = time.time()
    for j in snap:
        pref = j.get("preferred_cluster")
        if pref is not None and pref != cluster_id:
            continue

        jid = str(j.get("job_id"))
        model = str(j.get("model_name", "default"))
        dataset = str(j.get("dataset", "default"))
        lb = int(j.get("batch_size_per_gpu", 64))
        ga = int(j.get("grad_accum", 1))

        # gang job은 여기서도 min=max=4로 두되, current=0(대기) 상태
        if is_gang_model(model, dataset):
            min_g = max_g = int(GANG_JOBS.get((model, dataset), 4))
        else:
            min_g, max_g = 0, 4  # ✅ Pollux에서 pending job은 min_g=0이 자연스럽습니다.

        out.append(RuntimeJobState(
            job_id=jid,
            model_name=model,
            dataset=dataset,
            cluster_id=cluster_id,
            current_gpus=0,
            current_local_batch=lb,
            current_grad_accum=ga,
            progress=0.0,
            attained_service=0.0,
            last_scaled_at_ts=0.0,
            start_ts=0.0,
            min_gpus=min_g,
            max_gpus=max_g,
            last_sps=0.0,
            last_metric_ts=now,
            status="queued",
        ))
    return out

async def schedule_and_dispatch_jobs():
    global JOB_QUEUE, LAST_SCHED_TICK_TS
    metrics_logger.log_scheduler("Scheduler Loop Started")

    scale_job_fn = _make_scale_job_fn()

    while True:
        try:
            _reconcile_node_states()

            # 1) 클러스터 메트릭 기록
            calculate_cluster_metrics()

            now_ts = time.time()
            if LAST_SCHED_TICK_TS is None:
                dt = 5.0
            else:
                dt = max(1e-3, now_ts - LAST_SCHED_TICK_TS)
            LAST_SCHED_TICK_TS = now_ts

            # 2) attained_service 업데이트 (Pollux fairness용)
            # ✅ attained_service = GPU-seconds (world_size * dt) 로만 누적 (SSOT=GlobalServer)
            with ACTIVE_JOBS_LOCK:
                for jid, info in ACTIVE_JOBS.items():
                    world_size = len(info.get("nodes", []))
                    if world_size <= 0:
                        continue
                    prev = float(info.get("attained_service", 0.0))
                    info["attained_service"] = prev + float(world_size) * float(dt)

            # 3) 큐 길이 스냅샷
            with JOB_QUEUE_LOCK:
                global_queue_len = len(JOB_QUEUE)

            # ============================================================
            # ✅ 핵심 변경: BUSY(큐 존재)에서는 reallocation 호출 자체를 스킵
            #   - stop+requeue 기반 시스템에서 realloc은 진동/오버헤드 폭탄
            #   - 당신 환경(이기종/클러스터 내 donor-receiver 확률 낮음)에서는 더더욱 손해
            # ============================================================
            if global_queue_len == 0:
                # ======= IDLE일 때만 reallocation 고려 =======
                for cluster_id in ["clusterA", "clusterB"]:
                    runtime_jobs = build_runtime_jobs_for_cluster(cluster_id)
                    if not runtime_jobs:
                        continue

                    # 전역 RuntimeJobState 캐시 갱신
                    global RUNTIME_JOBS
                    for rj in runtime_jobs:
                        RUNTIME_JOBS[rj.job_id] = rj

                    cluster_total_gpus = _count_cluster_gpus(cluster_id)
                    if cluster_total_gpus <= 1:
                        continue

                    msg = (
                        f"[POLLUX][REALLOC_CALL] cluster={cluster_id}, "
                        f"total_gpus={cluster_total_gpus}, "
                        f"num_jobs={len(runtime_jobs)}, "
                        f"min_delta_gain={POLLUX_DELTA_GOODPUT_THRESH_IDLE:.3f}, "
                        f"queue_len={global_queue_len}"
                    )
                    log.info(msg)
                    metrics_logger.log_scheduler(msg)

                    queued_jobs = build_queued_runtime_jobs_for_cluster(cluster_id)
                    runtime_jobs_all = runtime_jobs + queued_jobs

                    pollux_reallocation_tick(
                        cluster_id=cluster_id,
                        cluster_total_gpus=cluster_total_gpus,
                        runtime_jobs=runtime_jobs_all,
                        scale_job_fn=scale_job_fn,
                        now_ts=now_ts,
                        cooldown_sec=POLLUX_SCALE_COOLDOWN_SEC,
                        min_delta_gain=POLLUX_DELTA_GOODPUT_THRESH_IDLE,  # 0.0
                    )
            else:
                # ======= BUSY면 realloc 완전 중지(로그만 남김) =======
                msg = f"[POLLUX][REALLOC_SKIP_BUSY] queue_len={global_queue_len} -> realloc disabled"
                log.info(msg)
                metrics_logger.log_scheduler(msg)

            launch_tasks: List[asyncio.Task] = []

            # 5) 클러스터별 idle 노드 목록 수집
            with NODE_REGISTRY_LOCK:
                avail_by_cluster: Dict[str, List[str]] = defaultdict(list)
                for node_id, info in NODE_REGISTRY.items():
                    if info["status"] == "idle":
                        c_name = info.get("cluster", "Cluster_Default")
                        avail_by_cluster[c_name].append(node_id)

            # 6) 멀티 클러스터 placement (launch 시점에 g 재결정)
            while True:
                with JOB_QUEUE_LOCK:
                    if not JOB_QUEUE:
                        break

                    chosen_idx: Optional[int] = None
                    chosen_cluster: Optional[str] = None
                    chosen_nodes: List[str] = []
                    chosen_desired_g: int = 0

                    # --- HoL + gang blocking 검사 ---
                    first_job = JOB_QUEUE[0]
                    first_model = first_job.get("model_name")
                    first_dataset = first_job.get("dataset")
                    first_pref = first_job.get("preferred_cluster")

                    if is_gang_model(first_model, first_dataset):
                        gang_g = GANG_JOBS.get((first_model, first_dataset), 4)

                        can_place_gang = False
                        for c_id in ["clusterA", "clusterB"]:
                            if first_pref is not None and first_pref != c_id:
                                continue
                            if len(avail_by_cluster.get(c_id, [])) >= gang_g:
                                can_place_gang = True
                                break

                        if not can_place_gang:
                            # HOL이 gang job이고, 아직 gang_g장이 한 클러스터에 안 모였으면 전체 block
                            break

                    # --- FIFO 순서 유지하며 배치 가능한 job 찾기 ---
                    for i, j in enumerate(JOB_QUEUE):
                        model = j.get("model_name")
                        dataset = j.get("dataset")
                        pref = j.get("preferred_cluster")

                        job_id = j.get("job_id")
                        base_batch = j.get("batch_size_per_gpu", 64)

                        for c_id in ["clusterA", "clusterB"]:
                            if pref is not None and pref != c_id:
                                continue

                            avail_nodes = avail_by_cluster.get(c_id, [])
                            free_in_cluster = len(avail_nodes)
                            if free_in_cluster <= 0:
                                continue

                            # launch 시점 free GPU 기준으로 Pollux admission 다시 평가
                            if is_gang_model(model, dataset):
                                desired_g = GANG_JOBS.get((model, dataset), 4)
                            else:
                                desired_g = _decide_initial_g_for_job(
                                    job_id=job_id,
                                    model_name=model,
                                    dataset=dataset,
                                    base_batch=base_batch,
                                    cluster_free_gpus=free_in_cluster,
                                )

                            desired_g = max(1, int(desired_g))
                            if desired_g > free_in_cluster:
                                continue

                            j["pollux_desired_gpus"] = desired_g

                            chosen_idx = i
                            chosen_cluster = c_id
                            chosen_nodes = avail_nodes[:desired_g]
                            chosen_desired_g = desired_g
                            break

                        if chosen_idx is not None:
                            break

                if chosen_idx is None or chosen_cluster is None or not chosen_nodes:
                    break

                with JOB_QUEUE_LOCK:
                    job_cfg = JOB_QUEUE.pop(chosen_idx)

                job_cfg["pollux_desired_gpus"] = chosen_desired_g

                # 선택된 클러스터에서 노드 소비
                remain = avail_by_cluster[chosen_cluster]
                avail_by_cluster[chosen_cluster] = remain[chosen_desired_g:]

                launch_tasks.append(
                    asyncio.create_task(_launch_job_on_nodes(job_cfg, chosen_nodes))
                )

            # 7) 실제 런치 요청 송신
            if launch_tasks:
                await asyncio.gather(*launch_tasks)

        except Exception as e:
            log.error(f"Error in scheduling loop: {e}", exc_info=True)
            metrics_logger.log_scheduler(f"Error in loop: {e}")

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

@app.post("/report_job_metrics")
async def report_job_metrics(req: JobMetrics):
    # ✅ fairness용 attained_service는 GlobalServer가 world_size * dt로 SSOT 관리함.
    # worker가 보내는 attained_service는 무시 (혼선/왜곡 방지)
    return {"ok": True, "ignored": True}

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

@app.post("/report_progress")
async def report_progress(req: ProgressReport):
    g = max(1, int(req.world_size))
    local_b = max(1, int(req.local_batch))
    accum = max(1, int(req.grad_accum))
    sps = float(req.steps_per_sec)

    msg = (
        f"[PROGRESS] job={req.job_id} "
        f"g={g}, batch={local_b}, accum={accum}, sps={sps:.3f}, "
        f"loss={req.loss}, acc={req.accuracy}, gns={req.gns}, stat_eff={req.stat_eff}"
    )
    log.info(msg)
    metrics_logger.log_scheduler(msg)

    # 1) ACTIVE_JOBS에 최신 throughput / batch / accum 반영
    with ACTIVE_JOBS_LOCK:
        info = ACTIVE_JOBS.get(req.job_id)
        if info is not None:
            info["last_sps"] = sps

            cfg = dict(info.get("config", {}))
            cfg["batch_size_per_gpu"] = local_b
            cfg["grad_accum"] = accum
            info["config"] = cfg

            ACTIVE_JOBS[req.job_id] = info
        else:
            log.warning(f"[PROGRESS] job={req.job_id} not found in ACTIVE_JOBS")
            metrics_logger.log_scheduler(
                f"[PROGRESS_WARN] job={req.job_id} not in ACTIVE_JOBS (g={g}, sps={sps:.3f})"
            )

    # 2) Pollux goodput profiler에 샘플 기록
    record_goodput_sample(
        job_id=req.job_id,
        g=g,
        local_batch=local_b,
        accum=accum,
        sps=sps,
        gns=req.gns,
        stat_eff=req.stat_eff,
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