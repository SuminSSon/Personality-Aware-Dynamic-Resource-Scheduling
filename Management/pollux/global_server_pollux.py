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

from pollux_scheduler import RuntimeJobState, pollux_reallocation_tick

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
               "current_job_id": None, "cluster": "Cluster_A"},
    "node_b": {"ip": "163.180.117.216", "agent_port": 8002, "gpu_id": 1,
               "prefix": "/home/ubuntu216/SUN", "status": "idle",
               "current_job_id": None, "cluster": "Cluster_A"},
    "node_c": {"ip": "163.180.117.216", "agent_port": 8003, "gpu_id": 2,
               "prefix": "/home/ubuntu216/SUN", "status": "idle",
               "current_job_id": None, "cluster": "Cluster_A"},
    "node_d": {"ip": "163.180.117.216", "agent_port": 8004, "gpu_id": 3,
               "prefix": "/home/ubuntu216/SUN", "status": "idle",
               "current_job_id": None, "cluster": "Cluster_A"},

    # Cluster B
    "node_e": {"ip": "163.180.160.62", "agent_port": 8005, "gpu_id": 0,
               "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "Cluster_B"},
    "node_f": {"ip": "163.180.160.62", "agent_port": 8006, "gpu_id": 0,
               "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "Cluster_B"},
    "node_g": {"ip": "163.180.160.62", "agent_port": 8007, "gpu_id": 0,
               "prefix": "/data/breath12/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "Cluster_B"},
    "node_h": {"ip": "163.180.160.62", "agent_port": 8008, "gpu_id": 0,
               "prefix": "/data/breath12/CCGRID/TrainingCode",
               "status": "idle", "current_job_id": None, "cluster": "Cluster_B"},
}

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
                "submitted_ts", "started_ts", "end_ts", "queued_sec", "jct_sec", "status"
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


def get_now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


# --- 전역 상태 변수 ---
GLOBAL_SERVER_ADDRESS: Optional[str] = None
MASTER_PORT_COUNTER = itertools.count(20000)

LAMBDA_TIME = 0.0
LAMBDA_COST = 0.0
LAMBDA_FAIR = 0.0

NODE_REGISTRY_LOCK = threading.Lock()

JOB_QUEUE: List[Dict[str, Any]] = []
JOB_QUEUE_LOCK = threading.Lock()

# ACTIVE_JOBS: {job_id: {config, status, nodes, stop_event, latest_progress, submitted_ts, start_ts, ...}}
ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}
ACTIVE_JOBS_LOCK = threading.Lock()


# --- API 데이터 모델 ---
class JobProgressReport(BaseModel):
    job_id: str
    current_step: int
    current_epoch: int
    total_epochs: int
    latest_eval_loss: float
    latest_accuracy: float


class JobCheckpointReport(BaseModel):
    job_id: str
    current_epoch: int
    total_epochs: int
    latest_eval_loss: float
    latest_accuracy: float
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
    status: str  # STARTED, STOPPED, FINISHED, FAILED


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
def _get_available_nodes() -> List[str]:
    with NODE_REGISTRY_LOCK:
        return [node_id for node_id, info in NODE_REGISTRY.items() if info["status"] == "idle"]


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


def build_runtime_jobs_for_cluster(cluster_id: str) -> List[RuntimeJobState]:
    """
    ACTIVE_JOBS + NODE_REGISTRY를 사용해서 실제 해당 클러스터에서 실행 중인 job만 추출.
    """
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

            cfg = info["config"]
            jobs.append(
                RuntimeJobState(
                    job_id=jid,
                    model_name=cfg.get("model_name", "default"),
                    dataset=cfg.get("dataset", "default"),
                    cluster_id=cluster_id,
                    current_gpus=len(nodes),
                    progress=info.get("latest_progress", 0.0),
                    attained_service=info.get("attained_service", 0.0),
                    last_scaled_at_ts=info.get("last_scaled_at_ts", 0.0),
                    min_gpus=1,
                    max_gpus=4,
                )
            )
    return jobs


async def _launch_job_on_nodes(job_config: Dict[str, Any], assigned_nodes: List[str]):
    job_id = job_config["job_id"]
    world_size = len(assigned_nodes)
    if world_size == 0:
        return

    master_node_id = assigned_nodes[0]
    with NODE_REGISTRY_LOCK:
        master_ip = NODE_REGISTRY[master_node_id]["ip"]

    master_port = next(MASTER_PORT_COUNTER)

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
            "script_path": script_path,  # 여기서 prefix+script_file 조합해서 넘김
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
            "learning_rate": job_config.get("learning_rate", job_config.get("base_lr", 1e-3)),
        }

        async def send(u, p):
            async with httpx.AsyncClient() as c:
                await c.post(u, json=p, timeout=10)

        tasks.append(send(url, payload))

    await asyncio.gather(*tasks)

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
            "attained_service": 0.0,
            "last_scaled_at_ts": time.time(),
        }
    _set_node_status(assigned_nodes, "busy", job_id)

    cluster_name = "Cluster_Default"
    with NODE_REGISTRY_LOCK:
        if assigned_nodes and assigned_nodes[0] in NODE_REGISTRY:
            cluster_name = NODE_REGISTRY[assigned_nodes[0]].get("cluster", "Cluster_Default")

    queued_sec = (start_ts - sub_ts).total_seconds()
    metrics_logger.log_csv("job_metrics", [
        job_id,
        cluster_name,
        job_config.get("model_name", ""),
        job_config.get("dataset", ""),
        world_size,
        sub_ts.strftime("%Y-%m-%d %H:%M:%S"),
        start_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "",
        f"{queued_sec:.6f}",
        "",
        "running",
    ])

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


async def _stop_job(job_id: str) -> bool:
    with ACTIVE_JOBS_LOCK:
        if job_id not in ACTIVE_JOBS:
            return True
        job_info = ACTIVE_JOBS[job_id]
        assigned_nodes = job_info["nodes"]
        stop_event = job_info["stop_event"]
        world_size = len(assigned_nodes)

    cluster_name = "Cluster_Default"
    with NODE_REGISTRY_LOCK:
        if assigned_nodes and assigned_nodes[0] in NODE_REGISTRY:
            cluster_name = NODE_REGISTRY[assigned_nodes[0]].get("cluster", "Cluster_Default")

    tasks = []
    for node_id in assigned_nodes:
        with NODE_REGISTRY_LOCK:
            if node_id in NODE_REGISTRY:
                info = NODE_REGISTRY[node_id]
                url = f"http://{info['ip']}:{info['agent_port']}/stop_task"

                async def send_stop(u):
                    try:
                        async with httpx.AsyncClient() as c:
                            await c.post(u, json={"job_id": job_id}, timeout=5)
                    except:
                        pass

                tasks.append(send_stop(url))

    await asyncio.gather(*tasks)

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=60)
        with ACTIVE_JOBS_LOCK:
            if job_id in ACTIVE_JOBS:
                del ACTIVE_JOBS[job_id]
        _set_node_status(assigned_nodes, "idle", None)

        metrics_logger.log_csv("job_events", [
            get_now_str(),
            "stopped",
            job_id,
            cluster_name,
            world_size,
            "Preempted",
            "{}",
        ])
        return True
    except asyncio.TimeoutError:
        log.error(f"[{job_id}] Stop timeout.")
        with ACTIVE_JOBS_LOCK:
            if job_id in ACTIVE_JOBS:
                del ACTIVE_JOBS[job_id]
        _set_node_status(assigned_nodes, "idle", None)
        return False


async def _scale_and_requeue_job(rj: RuntimeJobState, desired_g: int):
    jid = rj.job_id
    cfg: Optional[Dict[str, Any]] = None

    with ACTIVE_JOBS_LOCK:
        if jid not in ACTIVE_JOBS:
            return
        info = ACTIVE_JOBS[jid]
        cfg = dict(info["config"])

    if cfg is None:
        return

    # co-adaptive tuner에서 (g,batch,lr) 선택
    chosen = tuner.choose_config(jid, desired_g)
    cfg["pollux_desired_gpus"] = int(chosen["g"])
    cfg["batch_size_per_gpu"] = int(chosen["batch"])
    cfg["learning_rate"] = float(chosen["lr"])

    with JOB_QUEUE_LOCK:
        if not any(j["job_id"] == jid for j in JOB_QUEUE):
            JOB_QUEUE.insert(0, cfg)

    await _stop_job(jid)


def _make_scale_job_fn():
    def _scale_job(rj: RuntimeJobState, new_g: int) -> bool:
        asyncio.create_task(_scale_and_requeue_job(rj, new_g))
        return True

    return _scale_job


async def schedule_and_dispatch_jobs():
    global JOB_QUEUE
    metrics_logger.log_scheduler("Scheduler Loop Started")

    scale_job_fn = _make_scale_job_fn()

    while True:
        try:
            calculate_cluster_metrics()

            now_ts = time.time()

            # attained_service 업데이트 (대략 world_size * tick_interval)
            with ACTIVE_JOBS_LOCK:
                for jid, info in ACTIVE_JOBS.items():
                    world_size = len(info.get("nodes", []))
                    if world_size <= 0:
                        continue
                    prev = info.get("attained_service", 0.0)
                    info["attained_service"] = prev + world_size * 5.0  # tick_interval ≈ 5s

            # 클러스터별 Pollux 재할당 tick
            for cluster_id in ["Cluster_A", "Cluster_B"]:
                runtime_jobs = build_runtime_jobs_for_cluster(cluster_id)
                if not runtime_jobs:
                    continue

                cluster_total_gpus = _count_cluster_gpus(cluster_id)
                if cluster_total_gpus <= 0:
                    continue

                pollux_reallocation_tick(
                    cluster_id=cluster_id,
                    cluster_total_gpus=cluster_total_gpus,
                    runtime_jobs=runtime_jobs,
                    scale_job_fn=scale_job_fn,
                    now_ts=now_ts,
                )

            # 빈 슬롯에 큐에서 job 실행 (FIFO + pollux_desired_gpus)
            avail = _get_available_nodes()
            if avail:
                launch_tasks = []
                while avail:
                    with JOB_QUEUE_LOCK:
                        if not JOB_QUEUE:
                            break
                        job_cfg = JOB_QUEUE.pop(0)

                    desired_g = int(job_cfg.get("pollux_desired_gpus", 1))
                    g = min(desired_g, len(avail))
                    if g <= 0:
                        break

                    nodes = avail[:g]
                    avail = avail[g:]

                    launch_tasks.append(
                        asyncio.create_task(_launch_job_on_nodes(job_cfg, nodes))
                    )

                if launch_tasks:
                    await asyncio.gather(*launch_tasks)

            metrics_logger.log_scheduler("--- SCHEDULER TICK END ---")

        except Exception as e:
            log.error(f"Error in scheduling loop: {e}", exc_info=True)
            metrics_logger.log_scheduler(f"Error in loop: {e}")

        await asyncio.sleep(5)


# --- API Endpoints ---
@app.post("/submit_job")
async def submit_job(req: JobSubmitRequest):
    req_data = req.dict()
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

    data["script_file"] = script_file  # 절대경로가 아니라 "DDP_CIFAR100_Resnet50.py" 같은 relative 이름

    base_batch = data.get("batch_size_per_gpu", 64)
    base_lr = data.get("learning_rate", 1e-3)
    data["base_lr"] = base_lr

    with JOB_QUEUE_LOCK:
        if any(j["job_id"] == req.job_id for j in JOB_QUEUE):
            raise HTTPException(400, "Already in queue")
        with ACTIVE_JOBS_LOCK:
            if req.job_id in ACTIVE_JOBS:
                raise HTTPException(400, "Running")
        JOB_QUEUE.append(data)

    # co-adaptive tuner 초기화
    tuner.init_job(req.job_id, base_batch=base_batch, base_lr=base_lr)

    metadata = {
        "model": model,
        "dataset": dataset,
        "epochs": req.epochs,
        "batch_size_per_gpu": req.batch_size_per_gpu,
        "learning_rate": base_lr,
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
        f"g=0, batch={req.batch_size_per_gpu}",
    ])

    return {"status": "queued", "job_id": req.job_id}


@app.post("/report_job_status")
async def report_job_status(rep: JobStatusReport):
    if rep.status in ["FINISHED", "FAILED"]:
        # tuner state 정리
        tuner.remove_job(rep.job_id)

        job_info = None
        with ACTIVE_JOBS_LOCK:
            if rep.job_id in ACTIVE_JOBS:
                job_info = ACTIVE_JOBS[rep.job_id]
                del ACTIVE_JOBS[rep.job_id]

        _set_node_status(_get_nodes_for_job(rep.job_id), "idle", None)

        if job_info:
            assigned_nodes = job_info["nodes"]
            cluster_name = "Cluster_Default"
            with NODE_REGISTRY_LOCK:
                if assigned_nodes and assigned_nodes[0] in NODE_REGISTRY:
                    cluster_name = NODE_REGISTRY[assigned_nodes[0]].get("cluster", "Cluster_Default")

            sub_ts = job_info["config"]["submitted_ts"]
            start_ts = job_info["start_ts"]
            end_ts = datetime.datetime.now()
            queued = (start_ts - sub_ts).total_seconds() if start_ts else 0
            jct = (end_ts - sub_ts).total_seconds()
            world_size = len(assigned_nodes)

            metrics_logger.log_csv("job_metrics", [
                rep.job_id,
                cluster_name,
                job_info["config"].get("model_name"),
                job_info["config"].get("dataset"),
                world_size,
                sub_ts.strftime("%Y-%m-%d %H:%M:%S"),
                start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else "",
                end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                f"{queued:.2f}",
                f"{jct:.2f}",
                rep.status,
            ])

            status_lower = rep.status.lower()
            metrics_logger.log_csv("job_events", [
                get_now_str(),
                status_lower,
                rep.job_id,
                cluster_name,
                world_size,
                "Done",
                "{}",
            ])

    return {"status": "ok"}


@app.post("/report_checkpoint")
async def report_checkpoint(report: JobCheckpointReport):
    job_id = report.job_id
    with ACTIVE_JOBS_LOCK:
        if job_id in ACTIVE_JOBS:
            ACTIVE_JOBS[job_id]["latest_checkpoint_path"] = report.checkpoint_path
            log.info(f"[{job_id}] Checkpoint: {report.checkpoint_path}")

    # co-adaptive tuner 업데이트
    tuner.on_checkpoint(
        job_id,
        epoch=report.current_epoch,
        loss=report.latest_eval_loss,
        acc=report.latest_accuracy,
    )

    return {"status": "checkpoint_acked"}


@app.post("/report_job_stopped")
async def report_job_stopped(report: JobStopReport):
    job_id = report.job_id
    with ACTIVE_JOBS_LOCK:
        if job_id in ACTIVE_JOBS and ACTIVE_JOBS[job_id].get("stop_event"):
            ACTIVE_JOBS[job_id]["stop_event"].set()
            return {"status": "stop_acked"}
    raise HTTPException(status_code=404, detail="Job not found")


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
