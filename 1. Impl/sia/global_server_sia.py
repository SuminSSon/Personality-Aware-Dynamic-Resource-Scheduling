from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, ValidationError
import httpx, time, threading, itertools, logging, csv, os, datetime, asyncio
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict
import argparse
from contextlib import asynccontextmanager
import json
import re

from sia_scheduler import (
    RuntimeJobState,
    sia_reallocation_tick,
    record_goodput_sample,
    update_job_metrics_from_telemetry,
    is_gang_model,
    GANG_JOBS,
    sia_best_local_config_for_g,
    GoodputFunction,
    SiaProblem,
    SiaScheduler,
    _GOODPUT_SAMPLES,
    load_sps_surface,
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
    cluster_free_gpus: Optional[int] = None,
) -> int:
    # 1) 이 job이 쓸 수 있는 free_slots 계산
    if cluster_free_gpus is None:
        free_slots = _compute_cluster_free_gpus()
    else:
        free_slots = max(0, int(cluster_free_gpus))

    if free_slots <= 0:
        # capacity 없으면 최소 1장으로 표기해 두고,
        # 나중에 launch / elastic에서 다시 조정
        return 1

    # 이 job이 이론상 쓸 수 있는 최대 GPU는 4로 제한 (실험 환경 맞춤)
    max_gpus = min(4, free_slots)

    # default_sps는 아직 관측치가 없으니 1.0 정도의 placeholder 사용
    initial_g = sia_initial_g(
        job_id=job_id,
        model_name=model_name,
        dataset=dataset,
        min_gpus=1,
        max_gpus=max_gpus,
        cluster_free_gpus=free_slots,
        default_batch=base_batch,
        default_accum=1,
        default_sps=1.0,
    )

    # 안전 가드: [1, max_gpus, free_slots] 범위 안으로 클램프
    initial_g = max(1, min(int(initial_g), max_gpus, free_slots))
    return initial_g

def sia_initial_g(
    job_id: str,
    model_name: str,
    dataset: str,
    min_gpus: int,
    max_gpus: int,
    cluster_free_gpus: int,
    default_batch: int,
    default_accum: int,
    default_sps: float,
    cluster_id: Optional[str] = None,
) -> int:
    """
    가능한 g 중에서 예측 goodput이 최대가 되는 g 선택.
    - cluster_id를 주면 이기종에 맞게 surface를 고름
    """
    cid = str(cluster_id or "clusterA")
    free = max(0, int(cluster_free_gpus))
    lo = max(1, int(min_gpus))
    hi = max(lo, int(max_gpus))
    hi = min(hi, free) if free > 0 else hi

    gp_fn = GoodputFunction(model_name=model_name, dataset=dataset, cluster_id=cid, scale_factor=1.0)

    best_g = lo
    best_gp = -1.0
    for g in range(lo, hi + 1):
        gp, _ = gp_fn.optimize(g)
        if gp > best_gp:
            best_gp = gp
            best_g = g

    return int(best_g)

def sia_predict_goodput(
    job_id: str,
    model_name: str,
    dataset: str,
    cluster_id: str,
    g: int,
    default_batch: int,
    default_accum: int,
    default_sps: float,
) -> float:
    cid = str(cluster_id)
    gp_fn = GoodputFunction(model_name=model_name, dataset=dataset, cluster_id=cid, scale_factor=1.0)
    gp, _ = gp_fn.optimize(int(g))
    return float(gp)

SIA_COOLDOWN_SEC = 180.0
def sia_global_reallocation_tick(
    cluster_caps: Dict[str, int],
    runtime_jobs: List[RuntimeJobState],
    scale_job_fn,
    now_ts: Optional[float] = None,
    cooldown_sec: Optional[float] = None,
) -> Dict[str, Tuple[str, int, int]]:
    """
    전역 Sia tick:
    - job별 target cluster/type까지 포함
    - 실제 적용은 scale_job_fn으로 (stop+requeue) 실행
    """
    if now_ts is None:
        now_ts = time.time()
    if cooldown_sec is None:
        cooldown_sec = float(SIA_COOLDOWN_SEC)

    if not runtime_jobs:
        return {}

    sched = SiaGlobalScheduler()
    targets = sched.optimize_global(runtime_jobs, cluster_caps)
    if not targets:
        return {}

    final_targets: Dict[str, Tuple[str, int, int]] = {}

    for rj in runtime_jobs:
        jid = rj.job_id
        tgt = targets.get(jid)
        if not tgt:
            final_targets[jid] = (rj.cluster_id, rj.current_gpus, rj.current_local_batch or 0)
            continue

        target_cluster, desired_g, desired_b = tgt
        desired_g = int(max(rj.min_gpus, min(desired_g, rj.max_gpus)))
        desired_b = int(desired_b) if int(desired_b) > 0 else int(rj.current_local_batch or 0)

        # 변화 없으면 스킵
        if (str(target_cluster) == str(rj.cluster_id)) and (desired_g == int(rj.current_gpus)):
            final_targets[jid] = (rj.cluster_id, rj.current_gpus, rj.current_local_batch or desired_b)
            continue

        # 쿨다운
        elapsed = float(now_ts - float(rj.last_scaled_at_ts or 0.0))
        if elapsed < float(cooldown_sec):
            final_targets[jid] = (rj.cluster_id, rj.current_gpus, rj.current_local_batch or desired_b)
            continue

        # 실제 scale 실행 (여기서 "이동"도 처리해야 함: scale_job_fn이 requeue cfg에 preferred_cluster 반영하도록)
        try:
            ok = scale_job_fn(rj, desired_g, desired_b, target_cluster)
        except TypeError:
            # 기존 scale_job_fn 시그니처면 fallback (이동 불가)
            ok = scale_job_fn(rj, desired_g, desired_b)
            target_cluster = rj.cluster_id
        except Exception:
            ok = False

        if ok:
            rj.last_scaled_at_ts = float(now_ts)
            final_targets[jid] = (str(target_cluster), desired_g, desired_b)
        else:
            final_targets[jid] = (rj.cluster_id, rj.current_gpus, rj.current_local_batch or desired_b)

    return final_targets

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
            if jid and jid not in active_job_ids and ninfo.get("status") == "busy":
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
                continue

            first_node = nodes[0]
            node_info = NODE_REGISTRY.get(first_node)
            if not node_info:
                continue

            node_cluster = node_info.get("cluster", "Cluster_Default")
            if node_cluster != cluster_id:
                continue

            cfg = info.get("config", {}) or {}

            model = str(cfg.get("model_name", "default"))
            dataset = str(cfg.get("dataset", "default"))

            current_g = int(len(nodes))
            current_batch = int(cfg.get("batch_size_per_gpu", 64) or 64)
            current_grad_accum = int(cfg.get("grad_accum", 1) or 1)
            last_sps = float(info.get("last_sps", 0.0) or 0.0)
            attained_service = float(info.get("attained_service", 0.0) or 0.0)
            last_scaled_at_ts = float(info.get("last_scaled_at_ts", 0.0) or 0.0)

            # ✅ gang job이면 min=max=gang_g로 고정 + current_g도 최소 gang_g로 강제
            if is_gang_model(model, dataset):
                gang_g = int(GANG_JOBS.get((model, dataset), 4))
                if gang_g <= 0:
                    gang_g = 4
                min_g = gang_g
                max_g = gang_g

                # 🔥 중요: 이미 1로 떠버린 running job이 있으면
                # runtime view에서라도 "원래는 4 고정"임을 강제.
                # (실제 실행은 launch 단계에서 이미 4로 막아야 함)
                current_g = max(int(current_g), int(gang_g))
            else:
                min_g = 1
                max_g = 4

            rj = RuntimeJobState(
                job_id=jid,
                model_name=model,
                dataset=dataset,
                cluster_id=cluster_id,
                current_gpus=int(current_g),
                current_local_batch=int(current_batch),
                current_grad_accum=int(current_grad_accum),
                progress=info.get("latest_progress", 0.0),
                attained_service=float(attained_service),
                last_scaled_at_ts=float(last_scaled_at_ts),
                min_gpus=int(min_g),
                max_gpus=int(max_g),
                last_sps=float(last_sps),
            )

            # warmup start_ts
            start_dt = info.get("start_ts")
            if isinstance(start_dt, datetime.datetime):
                rj.start_ts = float(start_dt.timestamp())
            else:
                rj.start_ts = 0.0

            jobs.append(rj)

    return jobs

async def _launch_job_on_nodes(job_config: Dict[str, Any], assigned_nodes: List[str]):
    job_id = job_config["job_id"]
    model = str(job_config.get("model_name", ""))
    dataset = str(job_config.get("dataset", ""))

    world_size = len(assigned_nodes)
    if world_size == 0:
        return

    # =========================================================
    # ✅ SSOT: gang invariant를 "launch 직전"에 최종 강제
    # - scheduler가 실수로 1개 노드를 넘겨도 여기서 막아야 함
    # =========================================================
    if is_gang_model(model, dataset):
        desired_g = int(GANG_JOBS.get((model, dataset), 4))
        if desired_g <= 0:
            desired_g = 4

        if world_size != desired_g:
            log.error(
                f"[GANG_VIOLATION][LAUNCH_ABORT] job={job_id} model={model} dataset={dataset} "
                f"assigned_nodes={world_size} expected={desired_g} nodes={assigned_nodes}"
            )
            metrics_logger.log_scheduler(
                f"[GANG_VIOLATION][LAUNCH_ABORT] job={job_id} g={world_size} expected={desired_g}"
            )

            # ✅ launch하지 않고 HoL 유지 위해 맨 앞에 requeue
            with JOB_QUEUE_LOCK:
                already = any(j.get("job_id") == job_id for j in JOB_QUEUE)
                if not already:
                    JOB_QUEUE.insert(0, job_config)
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

    # --- script args ---
    epochs = int(job_config.get("epochs", 20) or 20)
    bsz_per_gpu = int(job_config.get("batch_size_per_gpu", 64) or 64)
    lr = float(job_config.get("learning_rate", job_config.get("base_lr", 1e-3)) or 1e-3)

    # grad_accum은 실험에서 랜덤하게 들어온다고 했으니 그대로 받아야 함
    grad_accum = int(job_config.get("grad_accum", 1) or 1)
    grad_accum = max(1, grad_accum)

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

            "epochs": epochs,
            "checkpoint_dir": f"./checkpoints/{job_id}",
            "global_server_addr": GLOBAL_SERVER_ADDRESS,
            "resume_from_checkpoint": job_config.get("latest_checkpoint_path"),

            "dataset": job_config.get("dataset", "default"),
            "model_name": job_config.get("model_name", "default"),

            # ✅ workeragent가 실제로 사용하는 키들(=학습 스크립트에 전달되는 키들)
            "batch_size_per_gpu": bsz_per_gpu,
            "learning_rate": lr,
            "grad_accum": grad_accum,
        }

        # ---------------------------------------------------------
        # (선택) "호환 키"는 TaskConfig가 extra를 막는다면 422 원인입니다.
        # 지금 DistilBERT exit_code=2는 학습코드 argparse mismatch가 원인이었지,
        # 여기서 키를 더 보내서 해결되는 게 아닙니다.
        #
        # 정말 필요하면 TaskConfig가 허용할 때만 아래를 켜세요.
        # ---------------------------------------------------------
        # payload.update({
        #     "batch_size": bsz_per_gpu,
        #     "init_lr": lr,
        #     "nproc_per_node": world_size,
        # })

        tasks.append(asyncio.create_task(_send_launch(node_id, url, payload)))

    results = await asyncio.gather(*tasks)

    failed_nodes = []
    for node_id, status_code, body in results:
        if status_code != 200:
            failed_nodes.append((node_id, status_code, body))

    if failed_nodes:
        msg_lines = [
            f"{node_id}: status={status_code}, body={str(body)[:200]}"
            for (node_id, status_code, body) in failed_nodes
        ]
        msg = f"[LAUNCH_FAIL] job={job_id}, errors: " + " | ".join(msg_lines)
        log.error(msg)
        metrics_logger.log_scheduler(msg)

        metadata = {"assigned_nodes": assigned_nodes, "errors": msg_lines}
        metrics_logger.log_csv("job_events", [
            get_now_str(),
            "launch_failed",
            job_id,
            cluster_name,
            world_size,
            "launch_task_failed",
            json.dumps(metadata),
        ])

        with JOB_QUEUE_LOCK:
            already = any(j.get("job_id") == job_id for j in JOB_QUEUE)
            if not already:
                JOB_QUEUE.insert(0, job_config)
        return

    # ✅ 성공: 배치된 순간 클러스터 확정(= migration 금지)
    job_config = dict(job_config)

    pref = job_config.get("preferred_cluster")
    if pref is None or str(pref).strip() == "":
        job_config["preferred_cluster"] = str(cluster_name)
    else:
        if str(pref) != str(cluster_name):
            log.error(
                f"[SIA][MIGRATION_FORBIDDEN] job={job_id} pref={pref} but placed_on={cluster_name}."
            )
            _set_node_status(assigned_nodes, "idle", None)
            with JOB_QUEUE_LOCK:
                already = any(j.get("job_id") == job_id for j in JOB_QUEUE)
                if not already:
                    JOB_QUEUE.insert(0, job_config)
            return

    start_ts = datetime.datetime.now()
    sub_ts = job_config["submitted_ts"]

    # ✅ SIA restart stats: cfg 우선 복원
    cfg_num_restarts = int(job_config.get("num_restarts", 0) or 0)
    cfg_total_run_time = float(job_config.get("total_run_time", 0.0) or 0.0)
    cfg_total_overhead = float(job_config.get("total_restart_overhead", 0.0) or 0.0)
    cfg_attained = float(job_config.get("attained_service", 0.0) or 0.0)

    # ✅ 이번 재시작 overhead 누적 (stop→재실행 완료까지)
    now_epoch = time.time()
    restart_begin = job_config.get("restart_begin_ts_epoch")
    restart_overhead_this = 0.0
    if restart_begin is not None:
        try:
            rb = float(restart_begin)
            if rb > 0 and now_epoch > rb:
                restart_overhead_this = now_epoch - rb
        except Exception:
            restart_overhead_this = 0.0

    if restart_overhead_this > 0.0:
        cfg_total_overhead = float(cfg_total_overhead) + float(restart_overhead_this)
        job_config["total_restart_overhead"] = float(cfg_total_overhead)
        job_config.pop("restart_begin_ts_epoch", None)

    with ACTIVE_JOBS_LOCK:
        prev = ACTIVE_JOBS.get(job_id, {}) or {}

        prev_attained = float(prev.get("attained_service", 0.0) or 0.0)
        attained_service = cfg_attained if cfg_attained > 0.0 else prev_attained

        prev_restarts = int(prev.get("num_restarts", 0) or 0)
        prev_run_time = float(prev.get("total_run_time", 0.0) or 0.0)
        prev_overhead = float(prev.get("total_restart_overhead", 0.0) or 0.0)

        num_restarts = cfg_num_restarts if cfg_num_restarts >= 0 else prev_restarts
        total_run_time = cfg_total_run_time if cfg_total_run_time >= 0.0 else prev_run_time
        total_overhead = cfg_total_overhead if cfg_total_overhead >= 0.0 else prev_overhead

        cfg = dict(job_config)
        cfg["attained_service"] = float(attained_service)
        cfg["total_run_time"] = float(total_run_time)
        cfg["num_restarts"] = int(num_restarts)
        cfg["total_restart_overhead"] = float(total_overhead)

        ACTIVE_JOBS[job_id] = {
            "config": cfg,
            "status": "running",
            "nodes": assigned_nodes,
            "stop_event": asyncio.Event(),
            "latest_progress": float(prev.get("latest_progress", 0.0) or 0.0),
            "submitted_ts": sub_ts,
            "start_ts": start_ts,
            "attained_service": float(attained_service),
            "last_scaled_at_ts": time.time(),
            "latest_checkpoint_path": cfg.get("latest_checkpoint_path"),

            # ✅ SIA SSOT fields
            "num_restarts": int(num_restarts),
            "total_run_time": float(total_run_time),
            "total_restart_overhead": float(total_overhead),
            "last_started_ts_epoch": float(now_epoch),
        }

    _set_node_status(assigned_nodes, "busy", job_id)

    try:
        log_scheduler_state(f"start job={job_id}")
    except NameError:
        pass

    metadata = {
        "nodes": assigned_nodes,
        "started_ts": start_ts.isoformat(),
        "restart_overhead_this": float(restart_overhead_this),
    }
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

async def _stop_job(job_id: str) -> bool:
    with ACTIVE_JOBS_LOCK:
        if job_id not in ACTIVE_JOBS:
            return True

        job_info = ACTIVE_JOBS.pop(job_id)
        assigned_nodes = list(job_info.get("nodes", []))
        world_size = len(assigned_nodes)

        # ---- Sia restart stats update (SSOT) ----
        now = time.time()
        last_started = float(job_info.get("last_started_ts_epoch", 0.0) or 0.0)

        total_run_time = float(job_info.get("total_run_time", 0.0) or 0.0)
        total_overhead = float(job_info.get("total_restart_overhead", 0.0) or 0.0)
        num_restarts = int(job_info.get("num_restarts", 0) or 0)

        if last_started > 0.0 and now > last_started:
            total_run_time += (now - last_started)

        # 현실 baseline: restart overhead는 고정 상수로 근사
        # (체크포인트/로드가 실제로 걸리는 시간은 워커에서 측정해야 정확하지만, 지금은 논문 구현형 baseline 우선)
        RESTART_OVERHEAD_SEC = float(os.getenv("SIA_RESTART_OVERHEAD_SEC", "10.0"))
        total_overhead += max(0.0, RESTART_OVERHEAD_SEC)

        num_restarts += 1

        # stop 후 requeue에서 이 값을 config에 실어서 다시 ACTIVE_JOBS로 들어오게 해야 함
        # -> _scale_and_requeue_job()에서 config에 실어주도록 아래에서 처리(이미 해둘 수도 있음)
        job_info["_sia_stats"] = {
            "num_restarts": num_restarts,
            "total_run_time": total_run_time,
            "total_restart_overhead": total_overhead,
        }

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
            pass

    tasks = [asyncio.create_task(send_stop(nid)) for nid in assigned_nodes]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    _set_node_status(assigned_nodes, "idle", None)

    # 이벤트 로그
    metrics_logger.log_csv("job_events", [
        get_now_str(),
        "stopped",
        job_id,
        cluster_name,
        world_size,
        "Preempted/Scaled",
        json.dumps(job_info.get("_sia_stats", {})),
    ])

    log_scheduler_state(f"stop job={job_id}")
    return True

async def _scale_and_requeue_job(
    rj,
    new_g: int,
    new_local_batch: int,
    preferred_cluster: Optional[str] = None,
    **kwargs,
):
    """
    Stop current run of rj, then requeue it with updated (g, local_batch).
    NOTE: This should NOT migrate clusters; preferred_cluster is only recorded for SSOT/debug.
    """
    job_id = str(getattr(rj, "job_id", ""))

    # ---- 0) sanitize ----
    try:
        new_g = int(new_g)
    except Exception:
        new_g = 1
    if new_g <= 0:
        new_g = 1

    try:
        new_local_batch = int(new_local_batch)
    except Exception:
        new_local_batch = 0
    if new_local_batch < 0:
        new_local_batch = 0

    # ---- 1) fetch current ACTIVE_JOBS snapshot ----
    with ACTIVE_JOBS_LOCK:
        info = dict(ACTIVE_JOBS.get(job_id, {}) or {})
        if not info:
            log.warning(f"[SCALE] job={job_id} not found in ACTIVE_JOBS; skip")
            return False

        cfg = dict(info.get("config", {}) or {})

        # ✅ SSOT: 기록만 (migration은 여기서 만들지 않음)
        if preferred_cluster is not None:
            cfg["preferred_cluster"] = str(preferred_cluster)

        # 목표값 기록
        cfg["sia_desired_gpus"] = int(new_g)
        # batch는 “요청한 batch 그대로” 정책이면, 여기서 new_local_batch를 억지로 덮지 마세요.
        # 다만 실험상 local_batch를 바꾸고 싶으면 아래 주석 해제:
        # if new_local_batch > 0:
        #     cfg["batch_size_per_gpu"] = int(new_local_batch)

        # restart bookkeeping (있으면 carry)
        cfg["attained_service"] = float(info.get("attained_service", cfg.get("attained_service", 0.0) or 0.0) or 0.0)
        cfg["total_run_time"] = float(info.get("total_run_time", cfg.get("total_run_time", 0.0) or 0.0) or 0.0)
        cfg["num_restarts"] = int(info.get("num_restarts", cfg.get("num_restarts", 0) or 0) or 0)
        cfg["total_restart_overhead"] = float(info.get("total_restart_overhead", cfg.get("total_restart_overhead", 0.0) or 0.0) or 0.0)

        # ✅ 이번 stop→start overhead 측정 시작점
        cfg["restart_begin_ts_epoch"] = float(time.time())

        info["config"] = cfg
        ACTIVE_JOBS[job_id] = info

        # stop_event 확보
        stop_event = info.get("stop_event")

    # ---- 2) request stop to nodes (이미 구현된 stop 경로를 호출한다고 가정) ----
    try:
        await _stop_job(job_id)  # ✅ 기존 코드에 있는 stop 루틴을 그대로 사용하세요
    except Exception as e:
        log.error(f"[SCALE] stop failed job={job_id}: {e}", exc_info=True)
        return False

    # ---- 3) wait job to finish (or a short timeout) ----
    # stop_event가 asyncio.Event면 기다려서 “실제로 내려간 후” requeue
    try:
        if stop_event is not None:
            await asyncio.wait_for(stop_event.wait(), timeout=60)
    except Exception:
        # timeout이어도 requeue는 진행 (환경에 따라 stop_event가 안 울리는 경우가 있음)
        pass

    # ---- 4) requeue with updated cfg ----
    with ACTIVE_JOBS_LOCK:
        info2 = dict(ACTIVE_JOBS.get(job_id, {}) or {})
        cfg2 = dict(info2.get("config", {}) or {})

        # 이 시점에서 ACTIVE에서 제거(중복 방지)
        if job_id in ACTIVE_JOBS:
            ACTIVE_JOBS.pop(job_id, None)

    # ✅ 큐에 다시 넣기
    with JOB_QUEUE_LOCK:
        already = any(j.get("job_id") == job_id for j in JOB_QUEUE)
        if not already:
            # FIFO 유지하려면 앞에 넣는 게 맞음(“scale로 인해 재시작”은 원래 job의 연속이니까)
            JOB_QUEUE.insert(0, cfg2)

    metrics_logger.log_csv("queue_events", [
        get_now_str(),
        "requeue_by_scale",
        job_id,
        len(JOB_QUEUE),
        f"new_g={new_g}, new_local_batch={new_local_batch}, pref={preferred_cluster}",
    ])

    log.info(
        f"[SCALE][REQUEUE] job={job_id} -> queued with target_g={new_g}, "
        f"local_batch={new_local_batch}, preferred_cluster={preferred_cluster}"
    )
    return True

def _make_scale_job_fn():
    """
    Returns a scale_job_fn(rj, new_g, new_local_batch) -> bool
    Fire-and-forget style, but ensures task exceptions are retrieved.
    """

    def _consume_task_exception(t: "asyncio.Task"):
        try:
            _ = t.exception()  # ✅ exception 회수 (로그 방지)
        except asyncio.CancelledError:
            return
        except Exception:
            return

    def scale_job_fn(rj, new_g: int, new_local_batch: int) -> bool:
        job_id = str(getattr(rj, "job_id", ""))

        # ✅ 같은 클러스터에서만 scale/requeue 하도록 SSOT용 preferred_cluster를 넘김
        tgt = str(getattr(rj, "cluster_id", "") or "")

        async def _do():
            # 여기서 예외가 나도, done_callback에서 회수됨
            return await _scale_and_requeue_job(
                rj,
                int(new_g),
                int(new_local_batch),
                preferred_cluster=tgt,  # ✅ 이제 _scale_and_requeue_job이 받음
            )

        try:
            task = asyncio.create_task(_do())
            task.add_done_callback(_consume_task_exception)  # ✅ 핵심
            return True
        except Exception as e:
            log.error(f"[SCALE] create_task failed job={job_id}: {e}", exc_info=True)
            return False

    return scale_job_fn

async def schedule_and_dispatch_jobs():
    global JOB_QUEUE, LAST_SCHED_TICK_TS, RUNTIME_JOBS

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

            # 2) attained_service + total_run_time 업데이트
            with ACTIVE_JOBS_LOCK:
                for jid, info in list(ACTIVE_JOBS.items()):
                    world_size = len(info.get("nodes", []) or [])
                    if world_size <= 0:
                        continue

                    # ---- attained_service (fairness용) ----
                    prev_attained = float(info.get("attained_service", 0.0) or 0.0)
                    sps = float(info.get("last_sps", 0.0) or 0.0)
                    if sps > 0.0:
                        dservice = sps * dt
                    else:
                        dservice = world_size * dt
                    info["attained_service"] = prev_attained + dservice

                    # ---- total_run_time (SIA r_i 입력) ----
                    prev_run = float(info.get("total_run_time", 0.0) or 0.0)
                    info["total_run_time"] = prev_run + float(dt)

                    # config에도 carry (stop→requeue 안정)
                    cfg = dict(info.get("config", {}) or {})
                    cfg["attained_service"] = float(info["attained_service"])
                    cfg["total_run_time"] = float(info["total_run_time"])
                    cfg["num_restarts"] = int(info.get("num_restarts", cfg.get("num_restarts", 0) or 0) or 0)
                    cfg["total_restart_overhead"] = float(
                        info.get("total_restart_overhead", cfg.get("total_restart_overhead", 0.0) or 0.0) or 0.0
                    )
                    info["config"] = cfg

                    ACTIVE_JOBS[jid] = info

            # 3) 큐 길이 (idle/busy)
            with JOB_QUEUE_LOCK:
                global_queue_len = len(JOB_QUEUE)

            if global_queue_len == 0:
                delta_gain_thresh = POLLUX_DELTA_GOODPUT_THRESH_IDLE
            else:
                delta_gain_thresh = POLLUX_DELTA_GOODPUT_THRESH_BUSY

            # 4) 클러스터별 SIA 재할당 (running job들만 대상)
            for cluster_id in ["clusterA", "clusterB"]:
                runtime_jobs = build_runtime_jobs_for_cluster(cluster_id)
                if not runtime_jobs:
                    continue

                # 전역 RuntimeJobState 캐시 갱신
                for rj in runtime_jobs:
                    RUNTIME_JOBS[rj.job_id] = rj

                cluster_total_gpus = _count_cluster_gpus(cluster_id)
                if cluster_total_gpus <= 1:
                    continue

                msg = (
                    f"[SIA][REALLOC_CALL] cluster={cluster_id}, "
                    f"total_gpus={cluster_total_gpus}, "
                    f"num_jobs={len(runtime_jobs)}, "
                    f"min_obj_gain={float(delta_gain_thresh)}, "
                    f"queue_len={global_queue_len}"
                )
                log.info(msg)
                metrics_logger.log_scheduler(msg)

                # ✅ reallocation은 gang 제외 버전이 이미 적용돼 있어야 함
                sia_reallocation_tick(
                    cluster_id=cluster_id,
                    cluster_total_gpus=int(cluster_total_gpus),
                    runtime_jobs=runtime_jobs,
                    scale_job_fn=scale_job_fn,
                    now_ts=now_ts,
                )

            launch_tasks: List[asyncio.Task] = []

            # 5) 클러스터별 idle 노드 목록 수집
            with NODE_REGISTRY_LOCK:
                avail_by_cluster: Dict[str, List[str]] = defaultdict(list)
                for node_id, info in (NODE_REGISTRY or {}).items():
                    if info.get("status") == "idle":
                        c_name = str(info.get("cluster", "Cluster_Default"))
                        avail_by_cluster[c_name].append(str(node_id))

            # ✅ 안정화: 중복 제거 + 정렬 (디버깅/결정론)
            for cid in list(avail_by_cluster.keys()):
                avail_by_cluster[cid] = sorted(set(avail_by_cluster[cid]))

            # 6) 멀티 클러스터 placement (FIFO 순서 유지 + 배치 가능하면 실행)
            while True:
                with JOB_QUEUE_LOCK:
                    if not JOB_QUEUE:
                        break

                    # ============================
                    # HoL + gang blocking (강제)
                    # ============================
                    first_job = JOB_QUEUE[0]
                    first_model = str(first_job.get("model_name"))
                    first_dataset = str(first_job.get("dataset"))
                    first_pref = first_job.get("preferred_cluster")

                    if is_gang_model(first_model, first_dataset):
                        gang_g = int(GANG_JOBS.get((first_model, first_dataset), 4))
                        if gang_g <= 0:
                            gang_g = 4

                        can_place_gang = False
                        for c_id in ["clusterA", "clusterB"]:
                            if first_pref is not None and str(first_pref) != c_id:
                                continue
                            if len(avail_by_cluster.get(c_id, [])) >= gang_g:
                                can_place_gang = True
                                break

                        if not can_place_gang:
                            # ✅ HoL(gang)이 막히면 뒤를 절대 건드리지 않음
                            break

                    chosen_idx: Optional[int] = None
                    chosen_cluster: Optional[str] = None
                    chosen_nodes: List[str] = []
                    chosen_desired_g: int = 0

                    # FIFO: 앞에서부터 배치 가능한 첫 job을 고른다
                    for i, j in enumerate(JOB_QUEUE):
                        model = str(j.get("model_name"))
                        dataset = str(j.get("dataset"))

                        c_id, nodes, g = _placement_choose_best_cluster(j, avail_by_cluster)
                        if c_id is None:
                            continue

                        free_in_cluster = len(avail_by_cluster.get(c_id, []))

                        # ✅ gang이면 무조건 g=4(또는 정의값) 강제
                        if is_gang_model(model, dataset):
                            desired_g = int(GANG_JOBS.get((model, dataset), 4))
                            if desired_g <= 0:
                                desired_g = 4

                            if free_in_cluster < desired_g:
                                continue

                            nodes = list(avail_by_cluster.get(c_id, []))[:desired_g]
                            if len(nodes) < desired_g:
                                continue

                            g = desired_g
                        else:
                            g = int(g)
                            if g <= 0:
                                continue
                            if free_in_cluster < g:
                                continue
                            if (not nodes) or (len(nodes) < g):
                                nodes = list(avail_by_cluster.get(c_id, []))[:g]
                                if len(nodes) < g:
                                    continue

                        # ✅ SSOT: launch에 쓰는 g는 항상 len(nodes)
                        g = int(len(nodes))
                        if g <= 0:
                            continue

                        chosen_idx = i
                        chosen_cluster = c_id
                        chosen_nodes = nodes
                        chosen_desired_g = g

                        # 기록
                        j["sia_desired_gpus"] = int(chosen_desired_g)
                        break

                if chosen_idx is None or chosen_cluster is None or not chosen_nodes or chosen_desired_g <= 0:
                    break

                # 큐에서 제거
                with JOB_QUEUE_LOCK:
                    job_cfg = JOB_QUEUE.pop(chosen_idx)

                # ✅ launch SSOT 최종 강제
                job_cfg["sia_desired_gpus"] = int(len(chosen_nodes))

                # ✅ gang이면 여기서도 마지막으로 못 박기 (절대 흔들리면 안 됨)
                model = str(job_cfg.get("model_name"))
                dataset = str(job_cfg.get("dataset"))
                if is_gang_model(model, dataset):
                    desired_g = int(GANG_JOBS.get((model, dataset), 4))
                    if desired_g <= 0:
                        desired_g = 4

                    if len(chosen_nodes) != desired_g:
                        log.error(
                            f"[GANG_VIOLATION] job={job_cfg.get('job_id')} "
                            f"model={model} dataset={dataset} "
                            f"nodes={len(chosen_nodes)} expected={desired_g} "
                            f"cluster={chosen_cluster} avail={len(avail_by_cluster.get(chosen_cluster, []))}"
                        )
                        # 큐 맨 앞에 되돌려 HoL 유지
                        with JOB_QUEUE_LOCK:
                            JOB_QUEUE.insert(0, job_cfg)
                        break

                # 선택된 클러스터에서 노드 소비
                remain = list(avail_by_cluster.get(chosen_cluster, []))
                avail_by_cluster[chosen_cluster] = remain[chosen_desired_g:]

                log.info(
                    f"[LAUNCH_SSOT] job={job_cfg.get('job_id')} "
                    f"model={job_cfg.get('model_name')} dataset={job_cfg.get('dataset')} "
                    f"cluster={chosen_cluster} g={job_cfg.get('sia_desired_gpus')} "
                    f"nodes={chosen_nodes}"
                )

                launch_tasks.append(asyncio.create_task(_launch_job_on_nodes(job_cfg, chosen_nodes)))

            # 7) 실제 런치 요청 송신
            if launch_tasks:
                await asyncio.gather(*launch_tasks)

        except Exception as e:
            log.error(f"Error in scheduling loop: {e}", exc_info=True)
            metrics_logger.log_scheduler(f"Error in loop: {e}")

        await asyncio.sleep(5)

class SiaGlobalScheduler:
    def __init__(self):
        self.log = logging.getLogger("Sia.GLOBAL")
        self.log.setLevel(logging.INFO)

    def _job_utility_on_cluster(
        self,
        rj: RuntimeJobState,
        cluster_id: str,
        g: int,
    ) -> Tuple[float, int]:
        """
        u = w * (r * (gp/min_gp))^rho  형태를 따르되,
        여기선 "해당 cluster에서의 JobInfo"로 계산.
        """
        # cluster_id만 바꿔서 goodput fn 만들기
        tmp = RuntimeJobState(
            job_id=rj.job_id,
            model_name=rj.model_name,
            dataset=rj.dataset,
            cluster_id=cluster_id,
            current_gpus=max(1, g),
            progress=rj.progress,
            attained_service=rj.attained_service,
            last_scaled_at_ts=rj.last_scaled_at_ts,
            min_gpus=rj.min_gpus,
            max_gpus=rj.max_gpus,
            current_local_batch=rj.current_local_batch,
            recent_sps=rj.recent_sps,
            num_restarts=rj.num_restarts,
            total_run_time=rj.total_run_time,
            total_restart_overhead=rj.total_restart_overhead,
            last_started_ts=rj.last_started_ts,
            current_gpu_type=rj.current_gpu_type,
        )

        sched = SiaScheduler()
        job_infos = sched.build_job_infos([tmp])
        ji = job_infos[0]

        gp, best_b = ji.goodput_fn.optimize(int(g))
        gp = max(float(gp), 0.0)
        if gp <= 0.0:
            return 0.0, int(best_b)

        base = float(ji.min_goodput) if ji.min_goodput > 0 else 1.0
        norm_gp = gp / base
        eff = float(ji.restart_factor) * norm_gp
        if eff <= 0:
            return 0.0, int(best_b)

        rho = SiaProblem.RHO
        u = float(ji.fairness_weight) * (eff ** rho)
        return float(u), int(best_b)

    def optimize_global(
        self,
        runtime_jobs: List[RuntimeJobState],
        cluster_caps: Dict[str, int],
    ) -> Dict[str, Tuple[str, int, int]]:
        if not runtime_jobs:
            return {}

        clusters = [str(c) for c in cluster_caps.keys() if cluster_caps.get(c, 0) > 0]
        if not clusters:
            return {}

        # 1) 초기: 각 job은 현재 cluster 유지 + 현재 g
        alloc: Dict[str, Tuple[str, int, int]] = {}
        usage: Dict[str, int] = {c: 0 for c in clusters}

        for rj in runtime_jobs:
            c0 = str(rj.cluster_id)
            if c0 not in usage:
                # 현재 cluster가 cap dict에 없으면 가장 큰 cap cluster로 보냄
                c0 = max(clusters, key=lambda x: cluster_caps.get(x, 0))
            g0 = max(int(rj.min_gpus), min(int(rj.current_gpus), int(rj.max_gpus)))
            u0, b0 = self._job_utility_on_cluster(rj, c0, g0)
            alloc[rj.job_id] = (c0, g0, b0)
            usage[c0] += g0

        # 2) 타입 선택 개선(1-step local search):
        #    각 job에 대해 "다른 cluster로 옮겼을 때 효용이 좋아지는지"를 보고 교체
        for rj in runtime_jobs:
            cur_c, cur_g, cur_b = alloc[rj.job_id]
            cur_u, _ = self._job_utility_on_cluster(rj, cur_c, cur_g)

            best = (cur_u, cur_c, cur_g, cur_b)
            for c in clusters:
                # 같은 cluster면 스킵
                if c == cur_c:
                    continue

                # 일단 g는 현재 유지(이동만)
                cand_u, cand_b = self._job_utility_on_cluster(rj, c, cur_g)
                if cand_u > best[0]:
                    best = (cand_u, c, cur_g, cand_b)

            if best[1] != cur_c:
                # usage 갱신 (capacity는 다음 단계에서 맞춤)
                usage[cur_c] -= cur_g
                usage[best[1]] += cur_g
                alloc[rj.job_id] = (best[1], cur_g, best[3])

        # 3) cluster별 capacity 맞추기: 초과면 shrink, 남으면 expand
        #    - shrink: "GPU 1개 줄였을 때 손해가 가장 작은 job"부터 줄임
        #    - expand: "GPU 1개 늘렸을 때 이득이 가장 큰 job"부터 늘림
        #    - scale-up은 라운드당 2x 제한 (논문 정책 반영)
        jobs_by_cluster: Dict[str, List[RuntimeJobState]] = {c: [] for c in clusters}
        by_id = {rj.job_id: rj for rj in runtime_jobs}
        for jid, (c, g, b) in alloc.items():
            jobs_by_cluster[c].append(by_id[jid])

        for c in clusters:
            cap = int(cluster_caps[c])
            # ---- shrink ----
            while usage[c] > cap:
                best_jid = None
                best_delta = None

                for rj in jobs_by_cluster[c]:
                    jid = rj.job_id
                    cc, g, b = alloc[jid]
                    if g <= int(rj.min_gpus):
                        continue

                    # delta = u(curr) - u(curr-1)  (손해)
                    u1, _ = self._job_utility_on_cluster(rj, c, g)
                    u0, _ = self._job_utility_on_cluster(rj, c, g - 1)
                    delta = u1 - u0

                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_jid = jid

                if best_jid is None:
                    break

                cc, g, b = alloc[best_jid]
                alloc[best_jid] = (cc, g - 1, b)
                usage[c] -= 1

            # ---- expand ----
            while usage[c] < cap:
                best_jid = None
                best_gain = 0.0

                for rj in jobs_by_cluster[c]:
                    jid = rj.job_id
                    cc, g, b = alloc[jid]
                    if g >= int(rj.max_gpus):
                        continue

                    # ✅ 라운드당 스케일업 2x 제한
                    max_up = max(1, int(rj.current_gpus) * 2)
                    if (g + 1) > max_up:
                        continue

                    u0, _ = self._job_utility_on_cluster(rj, c, g)
                    u1, _ = self._job_utility_on_cluster(rj, c, g + 1)
                    gain = u1 - u0
                    if gain > best_gain:
                        best_gain = gain
                        best_jid = jid

                if best_jid is None or best_gain <= 0.0:
                    break

                cc, g, b = alloc[best_jid]
                alloc[best_jid] = (cc, g + 1, b)
                usage[c] += 1

        return alloc

def _freeze_runtime_job_at_current_g(rj: "RuntimeJobState") -> "RuntimeJobState":
    """
    Start-placement에서는 기존 running job을 건드리지 않는(즉시 scale/preempt 금지) 가정으로
    min_gpus=max_gpus=current_gpus로 고정한 복사본을 만든다.
    """
    return RuntimeJobState(
        job_id=str(rj.job_id),
        model_name=str(rj.model_name),
        dataset=str(rj.dataset),
        cluster_id=str(rj.cluster_id),
        current_gpus=int(rj.current_gpus),

        progress=float(getattr(rj, "progress", 0.0) or 0.0),
        attained_service=float(getattr(rj, "attained_service", 0.0) or 0.0),
        last_scaled_at_ts=float(getattr(rj, "last_scaled_at_ts", 0.0) or 0.0),

        # ✅ freeze
        min_gpus=int(rj.current_gpus),
        max_gpus=int(rj.current_gpus),

        current_local_batch=getattr(rj, "current_local_batch", None),
        recent_sps=getattr(rj, "recent_sps", None),

        num_restarts=int(getattr(rj, "num_restarts", 0) or 0),
        total_run_time=float(getattr(rj, "total_run_time", 0.0) or 0.0),
        total_restart_overhead=float(getattr(rj, "total_restart_overhead", 0.0) or 0.0),
        last_started_ts=float(getattr(rj, "last_started_ts", 0.0) or 0.0),

        current_gpu_type=getattr(rj, "current_gpu_type", None),
    )


def _make_new_runtime_job_for_placement(
    job_cfg: Dict[str, Any],
    cluster_id: str,
    max_g_for_new: int,
) -> "RuntimeJobState":
    job_id = str(job_cfg.get("job_id"))
    model = str(job_cfg.get("model_name"))
    dataset = str(job_cfg.get("dataset"))
    req_local_b = int(job_cfg.get("batch_size_per_gpu", 64) or 64)

    # 새로 들어오는 job은 attained_service=0에서 시작 (SIA fairness 상 자연스러움)
    return RuntimeJobState(
        job_id=job_id,
        model_name=model,
        dataset=dataset,
        cluster_id=str(cluster_id),

        # optimizer initial alloc seed로 쓰임 (어차피 optimize가 다시 결정)
        current_gpus=1,

        progress=0.0,
        attained_service=float(job_cfg.get("attained_service", 0.0) or 0.0),
        last_scaled_at_ts=0.0,

        min_gpus=1,
        max_gpus=max(1, int(max_g_for_new)),

        current_local_batch=req_local_b,
        recent_sps=None,

        num_restarts=int(job_cfg.get("num_restarts", 0) or 0),
        total_run_time=float(job_cfg.get("total_run_time", 0.0) or 0.0),
        total_restart_overhead=float(job_cfg.get("total_restart_overhead", 0.0) or 0.0),
        last_started_ts=0.0,

        current_gpu_type=None,
    )


def _placement_choose_best_cluster(
    job_cfg: Dict[str, Any],
    avail_by_cluster: Dict[str, List[str]],
) -> Tuple[Optional[str], List[str], int]:
    """
    ✅ Start-placement에서 'SIA답게' 배치:
      - 후보 cluster마다: (기존 running jobs 고정) + (새 job 추가) 후 SiaScheduler.optimize 실행
      - 새 job에게 배정된 g를 뽑아, objective가 가장 큰 cluster를 선택
      - gang job은 무조건 g=4 강제 (free<4면 HoL)
    """
    job_id = str(job_cfg.get("job_id"))
    model = str(job_cfg.get("model_name"))
    dataset = str(job_cfg.get("dataset"))

    pref = job_cfg.get("preferred_cluster")
    candidate_clusters = ["clusterA", "clusterB"]
    if pref is not None and str(pref).strip() != "":
        candidate_clusters = [str(pref)]

    # ============================
    # 1) GANG JOB: 무조건 g=4 고정
    # ============================
    if is_gang_model(model, dataset):
        desired_g = int(GANG_JOBS.get((model, dataset), 4))
        if desired_g <= 0:
            desired_g = 4

        job_cfg["sia_desired_gpus"] = desired_g

        for cid in candidate_clusters:
            avail_nodes = list(avail_by_cluster.get(cid, []) or [])
            if len(avail_nodes) >= desired_g:
                return cid, avail_nodes[:desired_g], desired_g

        # 어느 클러스터도 4 확보 못하면 HoL
        return None, [], 0

    # ============================
    # 2) NON-GANG: SIA optimize로 (cluster, g) 선택
    # ============================
    best_cluster: Optional[str] = None
    best_nodes: List[str] = []
    best_g: int = 0
    best_obj: float = -1e18

    for cid in candidate_clusters:
        avail_nodes = list(avail_by_cluster.get(cid, []) or [])
        free = len(avail_nodes)
        if free <= 0:
            continue

        # 이 클러스터 total GPUs
        cluster_total_gpus = int(_count_cluster_gpus(cid))
        if cluster_total_gpus <= 0:
            continue

        # 현재 running jobs (cluster 내부) 가져오기
        runtime_jobs = build_runtime_jobs_for_cluster(cid) or []

        # ✅ start-placement에서는 기존 running jobs를 지금 당장 못 줄이므로 freeze
        frozen = [_freeze_runtime_job_at_current_g(rj) for rj in runtime_jobs]

        # 새 job이 가질 수 있는 최대 g는 free 범위 내에서만 (그리고 1..4)
        max_g_for_new = min(4, free)
        new_rj = _make_new_runtime_job_for_placement(job_cfg, cid, max_g_for_new=max_g_for_new)

        jobs_plus = frozen + [new_rj]

        # SIA optimize 실행
        sched = SiaScheduler()
        alloc_dict = sched.optimize(jobs_plus, cluster_total_gpus=int(cluster_total_gpus))
        if not alloc_dict:
            continue

        # 새 job의 g 추출
        desired_g, _best_b = alloc_dict.get(job_id, (0, 0))
        try:
            desired_g = int(desired_g)
        except Exception:
            desired_g = 0

        # free 범위를 벗어나면 배치 불가 (freeze 했으면 보통 안 벗어남)
        if desired_g <= 0 or desired_g > free:
            continue

        # ✅ 이 cluster에서의 objective를 직접 평가해서 비교 (새 job만 뽑으면 위험)
        # - alloc_dict는 (g, local_b)인데 objective는 g만 필요
        alloc_g = {jid: int(v[0]) for jid, v in alloc_dict.items()}
        job_infos = sched.build_job_infos(jobs_plus)
        obj = float(sched._compute_objective(job_infos, alloc_g))

        if obj > best_obj:
            best_obj = obj
            best_cluster = str(cid)
            best_g = int(desired_g)
            best_nodes = avail_nodes[:best_g]

    if best_cluster is None or best_g <= 0 or not best_nodes:
        return None, [], 0

    # SSOT 기록
    job_cfg["sia_desired_gpus"] = int(best_g)
    return best_cluster, best_nodes, best_g

def _canon(s: Optional[str]) -> str:
    """모델/데이터셋 이름 정규화: 하이픈/언더스코어/공백/대소문자 차이 흡수"""
    if s is None:
        return ""
    x = str(s).strip().lower()
    x = x.replace("–", "-").replace("—", "-").replace("-", "-")  # 다양한 하이픈 문자 통일
    x = re.sub(r"[\s_]+", "-", x)                                # 공백/언더스코어 -> 하이픈
    x = re.sub(r"[^a-z0-9\-]+", "", x)                          # 나머지 기호 제거
    return x

# ✅ 여기만 “진실의 원천(SSOT)”으로 두세요.
GANG_JOBS_CANON: Dict[Tuple[str, str], int] = {
    (_canon("DenseNet-121"), _canon("TinyImageNet")): 4,
    # 필요하면 여기 추가
}

def is_gang_model(model_name: Optional[str], dataset: Optional[str]) -> bool:
    key = (_canon(model_name), _canon(dataset))
    return key in GANG_JOBS_CANON

def gang_required_g(model_name: Optional[str], dataset: Optional[str]) -> int:
    key = (_canon(model_name), _canon(dataset))
    return int(GANG_JOBS_CANON.get(key, 0) or 0)

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

    base_batch = int(data.get("batch_size_per_gpu", 64) or 64)
    base_lr = float(data.get("learning_rate", 1e-3) or 1e-3)
    data["base_lr"] = base_lr

    # 2) 초기 g 결정
    initial_g = _decide_initial_g_for_job(
        job_id=req.job_id,
        model_name=model,
        dataset=dataset,
        base_batch=base_batch,
    )
    if is_gang_model(model, dataset):
        initial_g = gang_required_g(model, dataset) or 4

    data["sia_desired_gpus"] = int(initial_g)

    # 로그용 free_slots도 같이 찍기
    free_slots = _compute_cluster_free_gpus()
    log.info(
        f"[{req.job_id}] initial sia_desired_gpus={initial_g} "
        f"(free_slots={free_slots}) model={model} dataset={dataset}"
    )

    # 3) 큐에 넣기 전에 중복 검사
    with JOB_QUEUE_LOCK:
        if any(j.get("job_id") == req.job_id for j in JOB_QUEUE):
            raise HTTPException(400, "Already in queue")
        with ACTIVE_JOBS_LOCK:
            if req.job_id in ACTIVE_JOBS:
                raise HTTPException(400, "Running")
        JOB_QUEUE.append(data)

        # queue 길이는 lock 안에서 찍는 게 정확함
        q_len = len(JOB_QUEUE)

    metadata = {
        "model": model,
        "dataset": dataset,
        "epochs": int(req.epochs),
        "batch_size_per_gpu": int(req.batch_size_per_gpu),
        "learning_rate": float(base_lr),
        "initial_g": int(initial_g),
        "is_gang": bool(is_gang_model(model, dataset)),
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
        q_len,
        f"g_target={initial_g}, batch={req.batch_size_per_gpu}, gang={int(is_gang_model(model, dataset))}",
    ])

    log_scheduler_state(f"enqueue job={req.job_id}, g_target={initial_g}")

    return {"status": "queued", "job_id": req.job_id, "initial_g": int(initial_g)}

@app.post("/report_job_metrics")
async def report_job_metrics(req: JobMetrics):
    job = RUNTIME_JOBS.get(req.job_id)
    if job is None:
        # 아직 등록 안 된 job이면 무시
        return {"ok": False, "reason": "job_not_found"}

    metrics = {
        "attained_service": req.attained_service,
        # 나중에 gns, loss 등 더 넣고 싶으면 여기에 키 추가
    }

    update_job_metrics_from_telemetry(job, metrics)
    return {"ok": True}

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
    """
    Robust progress endpoint:
    - Never crash the server (no 500 due to optional components).
    - Update ACTIVE_JOBS last_sps + config if present.
    - Record goodput samples best-effort (may be used by SIA/Pollux-style profilers).
    """
    # ---- 0) sanitize inputs ----
    try:
        job_id = str(req.job_id)
        g = max(1, int(req.world_size))
        local_b = max(1, int(req.local_batch))
        accum = max(1, int(req.grad_accum))
        sps = float(req.steps_per_sec)
    except Exception as e:
        log.error(f"[PROGRESS] invalid payload: {e}")
        metrics_logger.log_scheduler(f"[PROGRESS_ERR] invalid payload: {e}")
        # Bad payload: still return 200 to avoid worker retry storms
        return {"status": "ok", "warning": "invalid_payload"}

    # ---- 1) infer cluster_id best-effort (optional) ----
    cluster_id: Optional[str] = None
    try:
        assigned_nodes: List[str] = []

        with ACTIVE_JOBS_LOCK:
            info = ACTIVE_JOBS.get(job_id)
            if info:
                assigned_nodes = list(info.get("nodes", []) or [])

        if not assigned_nodes:
            assigned_nodes = _get_nodes_for_job(job_id)

        if assigned_nodes:
            with NODE_REGISTRY_LOCK:
                first = assigned_nodes[0]
                ninfo = NODE_REGISTRY.get(first)
                if ninfo:
                    cluster_id = str(ninfo.get("cluster") or "")
                    if not cluster_id:
                        cluster_id = None
    except Exception:
        # cluster_id is optional; ignore failures
        cluster_id = None

    # ---- 2) logging (do not assume anything exists) ----
    try:
        msg = (
            f"[PROGRESS] job={job_id} "
            f"g={g}, batch={local_b}, accum={accum}, sps={sps:.3f}, "
            f"loss={req.loss}, acc={req.accuracy}, gns={req.gns}, stat_eff={req.stat_eff}"
        )
        log.info(msg)
        metrics_logger.log_scheduler(msg)
    except Exception:
        pass

    # ---- 3) update ACTIVE_JOBS state if present ----
    try:
        with ACTIVE_JOBS_LOCK:
            info = ACTIVE_JOBS.get(job_id)
            if info is not None:
                info["last_sps"] = sps

                cfg = dict(info.get("config", {}) or {})
                cfg["batch_size_per_gpu"] = local_b
                cfg["grad_accum"] = accum
                info["config"] = cfg

                ACTIVE_JOBS[job_id] = info
            else:
                # Not in ACTIVE_JOBS: still OK.
                # This can happen during preempt/requeue transitions.
                try:
                    log.warning(f"[PROGRESS] job={job_id} not found in ACTIVE_JOBS")
                    metrics_logger.log_scheduler(
                        f"[PROGRESS_WARN] job={job_id} not in ACTIVE_JOBS (g={g}, sps={sps:.3f})"
                    )
                except Exception:
                    pass
    except Exception as e:
        # Never crash endpoint
        try:
            log.error(f"[PROGRESS] failed updating ACTIVE_JOBS for job={job_id}: {e}", exc_info=True)
            metrics_logger.log_scheduler(f"[PROGRESS_ERR] update ACTIVE_JOBS failed job={job_id}: {e}")
        except Exception:
            pass

    # ---- 4) record goodput sample best-effort ----
    try:
        # Keep the call compatible with both old/new signatures.
        # If your sia_scheduler.py was updated to accept cluster_id, this will work.
        record_goodput_sample(
            job_id=job_id,
            g=g,
            local_batch=local_b,
            accum=accum,
            sps=sps,
            gns=req.gns,
            stat_eff=req.stat_eff,
            cluster_id=cluster_id,
        )
    except TypeError:
        # Backward compatibility: older record_goodput_sample doesn't accept cluster_id
        try:
            record_goodput_sample(
                job_id=job_id,
                g=g,
                local_batch=local_b,
                accum=accum,
                sps=sps,
                gns=req.gns,
                stat_eff=req.stat_eff,
            )
        except Exception as e2:
            try:
                log.warning(f"[PROGRESS] record_goodput_sample failed (legacy call) job={job_id}: {e2}")
                metrics_logger.log_scheduler(
                    f"[PROGRESS_WARN] record_goodput_sample failed (legacy) job={job_id}: {e2}"
                )
            except Exception:
                pass
    except Exception as e:
        # Never crash endpoint
        try:
            log.warning(f"[PROGRESS] record_goodput_sample failed job={job_id}: {e}")
            metrics_logger.log_scheduler(f"[PROGRESS_WARN] record_goodput_sample failed job={job_id}: {e}")
        except Exception:
            pass

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

def _infer_cluster_for_job(job_id: str) -> Optional[str]:
    """
    현재 ACTIVE_JOBS/NODE_REGISTRY를 이용해 job이 돌아가는 cluster_id를 추론.
    - 없으면 None
    """
    nodes: List[str] = []
    with ACTIVE_JOBS_LOCK:
        info = ACTIVE_JOBS.get(job_id)
        if info:
            nodes = list(info.get("nodes", []) or [])

    if not nodes:
        nodes = _get_nodes_for_job(job_id)

    if not nodes:
        return None

    first = nodes[0]
    with NODE_REGISTRY_LOCK:
        ninfo = NODE_REGISTRY.get(first)
        if not ninfo:
            return None
        return ninfo.get("cluster")

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
                "sia_desired_gpus": j.get("sia_desired_gpus", 1),
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