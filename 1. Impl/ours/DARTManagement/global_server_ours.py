import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict, Optional, List, Tuple, Set
from collections import defaultdict
import argparse
import httpx, threading, requests, time, itertools, logging, asyncio, uuid, os, json


_COMPLETION_SENT = set()
_COMPLETION_LOCK = threading.Lock()

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

app = FastAPI()

# --- 1. 글로벌 상태 관리 ---


SCHEDULER__ADDRESS = "http://localhost:8082"
GLOBAL_SERVER_ADDRESS = None
NON_SERIALIZABLE_KEYS = ["stop_event"]

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
    "CIFAR-100":     "CIFAR100_",
    "CIFAR-10":      "CIFAR10_",
    "TinyImageNet":  "TinyImageNet_",
    "Fashion-MNIST": "FashoinMNIST_",
    "MNIST":         "MNIST_",
    "SST2":          "SST2_",
    "SST-2":         "SST2_",
    "ARCTIC":        "CMU_ARCTIC_",
}

def make_filename(model: str, dataset: str) -> str:
    """모델/데이터셋에서 실제 DDP 학습 스크립트 파일명을 생성."""
    if model not in MODEL_BASENAME:
        raise KeyError(f"Unknown model name: {model}")
    if dataset not in DS_PREFIX:
        raise KeyError(f"Unknown dataset name: {dataset}")
    return f"{DS_PREFIX[dataset]}{MODEL_BASENAME[model]}.py"

# ----- 정적 매핑(요청한 조합 전부 포함) -----
MODEL_DS_MAP = {
    # CIFAR-100
    ("ResNet-50",        "CIFAR-100"):     make_filename("ResNet-50",        "CIFAR-100"),
    ("ResNet-18",        "CIFAR-100"):     make_filename("ResNet-18",        "CIFAR-100"),
    ("DenseNet-121",     "CIFAR-100"):     make_filename("DenseNet-121",     "CIFAR-100"),
    ("MobileNetV2",      "CIFAR-100"):     make_filename("MobileNetV2",      "CIFAR-100"),
    ("EfficientNetV2-S", "CIFAR-100"):     make_filename("EfficientNetV2-S", "CIFAR-100"),
    ("VGG-11",           "CIFAR-100"):     make_filename("VGG-11",           "CIFAR-100"),

    # CIFAR-10 (추가)
    ("ResNet-50",        "CIFAR-10"):      make_filename("ResNet-50",        "CIFAR-10"),
    ("DenseNet-121",     "CIFAR-10"):      make_filename("DenseNet-121",     "CIFAR-10"),
    ("MobileNetV2",      "CIFAR-10"):      make_filename("MobileNetV2",      "CIFAR-10"),
    ("EfficientNetV2-S", "CIFAR-10"):      make_filename("EfficientNetV2-S", "CIFAR-10"),
    ("VGG-11",           "CIFAR-10"):      make_filename("VGG-11",           "CIFAR-10"),

    # TinyImageNet
    ("ResNet-50",        "TinyImageNet"):  make_filename("ResNet-50",        "TinyImageNet"),
    ("DenseNet-121",     "TinyImageNet"):  make_filename("DenseNet-121",     "TinyImageNet"),
    ("MobileNetV2",      "TinyImageNet"):  make_filename("MobileNetV2",      "TinyImageNet"),
    ("EfficientNetV2-S", "TinyImageNet"):  make_filename("EfficientNetV2-S", "TinyImageNet"),
    ("VGG-11",           "TinyImageNet"):  make_filename("VGG-11",           "TinyImageNet"),

    # Fashion-MNIST (파일명은 FashoinMNIST)
    ("ResNet-50",        "Fashion-MNIST"): make_filename("ResNet-50",        "Fashion-MNIST"),
    ("DenseNet-121",     "Fashion-MNIST"): make_filename("DenseNet-121",     "Fashion-MNIST"),
    ("MobileNetV2",      "Fashion-MNIST"): make_filename("MobileNetV2",      "Fashion-MNIST"),
    ("EfficientNetV2-S", "Fashion-MNIST"): make_filename("EfficientNetV2-S", "Fashion-MNIST"),
    ("VGG-11",           "Fashion-MNIST"): make_filename("VGG-11",           "Fashion-MNIST"),

    # MNIST (추가)
    ("ResNet-50",        "MNIST"):         make_filename("ResNet-50",        "MNIST"),
    ("DenseNet-121",     "MNIST"):         make_filename("DenseNet-121",     "MNIST"),
    ("MobileNetV2",      "MNIST"):         make_filename("MobileNetV2",      "MNIST"),
    ("EfficientNetV2-S", "MNIST"):         make_filename("EfficientNetV2-S", "MNIST"),
    ("VGG-11",           "MNIST"):         make_filename("VGG-11",           "MNIST"),

    ("DistilBERT",       "SST2"):         make_filename("DistilBERT",       "SST2"),
    ("DistilBERT",       "SST-2"):        make_filename("DistilBERT",       "SST-2"),

    ("DeepSpeech2",      "ARCTIC"):        make_filename("DeepSpeech2",     "ARCTIC"),
}

NODE_REGISTRY = {
    # Cluster A
    "node_a": {"ip": "163.180.117.216", "agent_port": 8001, "gpu_id": 0, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None, "cluster": "clusterA"},
    "node_b": {"ip": "163.180.117.216", "agent_port": 8002, "gpu_id": 1, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None, "cluster": "clusterA"},
    "node_c": {"ip": "163.180.117.216", "agent_port": 8003, "gpu_id": 2, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None, "cluster": "clusterA"},
    "node_d": {"ip": "163.180.117.216", "agent_port": 8004, "gpu_id": 3, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None, "cluster": "clusterA"},

    "node_e": {"ip": "163.180.160.62", "agent_port": 8301, "prefix": "/nas2/data/dlwmznzl1/test/TrainingCode(DART)", "status": "idle", "current_job_id": None},
    "node_f": {"ip": "163.180.160.62", "agent_port": 8302, "prefix": "/nas2/data/dlwmznzl1/test/TrainingCode(DART)", "status": "idle", "current_job_id": None},
    "node_g": {"ip": "163.180.160.62", "agent_port": 8303, "prefix": "/data/breath12/CCGRID/test/TrainingCode(DART)", "status": "idle", "current_job_id": None},
    "node_h": {"ip": "163.180.160.62", "agent_port": 8304, "prefix": "/data/breath12/CCGRID/test/TrainingCode(DART)", "status": "idle", "current_job_id": None},
}

ACTIVE_JOBS = {}

# JSON으로 변환하면 안 되는 키 목록
NON_SERIALIZABLE_KEYS = ["stop_event"]

class PortManager:
    def __init__(self, start=29500, end=29600):
        self.available_ports = list(range(start, end))
        self.lock = threading.Lock()

    def get_port(self):
        with self.lock:
            if not self.available_ports:
                raise Exception("No free ports available")
            return self.available_ports.pop(0)

    def release_port(self, port):
        with self.lock:
            self.available_ports.append(port)

port_manager = PortManager()
job_counter = itertools.count(1)
job_locks = defaultdict(threading.Lock) 

# --- 2. API 데이터 모델 ---
class LaunchJobRequest(BaseModel):
    job_id: str
    nodes: List[str]
    model: str
    dataset: str
    epochs: int = 100
    batch_size: Optional[int] = None

    # resume
    resume_from_checkpoint: Optional[str] = None

    # idempotency / SSOT
    request_id: Optional[str] = None       # executor가 주는 request_id
    attempt: Optional[int] = None          # scheduler가 SSOT로 주는 attempt (없으면 server가 fallback)
    reason: Optional[str] = None           # launch reason (optional)

    run_id: Optional[str] = None
    allow_preempted: bool = False

class PreemptJobRequest(BaseModel):
    job_id: str
    # executor가 보내는 필드들(받아두면 디버깅에 도움)
    cluster_id: Optional[str] = None
    reason: Optional[str] = None
    run_id: Optional[str] = None
    force: bool = False

class JobRequest(BaseModel):
    nodes_with_paths: Dict[str, str]  
    epochs: int = 100
    batch_size: Optional[int] = None

class ResizeRequest(BaseModel):
    job_id: str
    new_nodes: List[str] 

class CheckpointReport(BaseModel):
    job_id: str
    current_epoch: int
    total_epochs: int
    latest_accuracy: float
    latest_eval_loss: float
    checkpoint_path: str

class JobStopReport(BaseModel):
    job_id: str
    run_id: Optional[str] = None  # 추가: 어떤 "런"이 멈췄는지 식별
    reason: Optional[str] = None  # 추가: PREEMPT/RESIZE/CANCEL 등

class JobCompleteReport(BaseModel):
    job_id: str
    exit_code: int = 0
    run_id: Optional[str] = None

# --- 3. 핵심 헬퍼 함수 ---
def _resolve_resume_path(job_id: str, requested: Optional[str], latest: Optional[str]) -> Optional[str]:
    cand = None
    if requested:
        cand = str(requested)
        if os.path.exists(cand):
            return cand
    if latest:
        cand = str(latest)
        if os.path.exists(cand):
            return cand
    return None

@app.post("/launch_job")
async def launch_job(req: LaunchJobRequest):
    job_id = str(req.job_id).strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    node_names = [str(x) for x in (req.nodes or []) if str(x)]
    model = str(req.model)
    dataset = str(req.dataset)
    epochs = int(req.epochs or 0)
    batch_size = req.batch_size

    now = time.time()
    request_id = (req.request_id or "").strip() or None
    reason = (req.reason or "LAUNCH").upper().strip()

    # 요청 run_id
    req_run_id = (getattr(req, "run_id", None) or "").strip() or None

    # ✅ attempt SSOT: scheduler가 주는 값 우선, 없으면 fallback(서버 추정)
    try:
        attempt = int(req.attempt) if req.attempt is not None else None
    except Exception:
        attempt = None
    if attempt is None or attempt <= 0:
        prev0 = ACTIVE_JOBS.get(job_id) or {}
        try:
            attempt = int(prev0.get("attempt", 0) or 0) + 1
        except Exception:
            attempt = 1
    if attempt <= 0:
        attempt = 1

    if not node_names:
        raise HTTPException(status_code=400, detail="nodes must not be empty")

    # ----------------------------
    # ✅ (LOG) request visible
    # ----------------------------
    log.info(
        "[LAUNCH_REQ] job_id=%s req_run_id=%s req_attempt=%s ws=%s nodes=%s model=%s dataset=%s request_id=%s reason=%s allow_preempted=%s",
        job_id,
        req_run_id,
        getattr(req, "attempt", None),
        len(node_names),
        node_names,
        model,
        dataset,
        request_id,
        reason,
        bool(getattr(req, "allow_preempted", False)),
    )

    nodes_with_paths: Dict[str, str] = {}
    resume_from: Optional[str] = None

    # ---------------------------------------------------------
    # 0) job-level lock
    # ---------------------------------------------------------
    with job_locks[job_id]:
        cur = ACTIVE_JOBS.get(job_id) or {}
        cur_req = (cur.get("last_launch_request_id") or "").strip() or None
        cur_status = (cur.get("status") or "").lower()
        cur_run_id = (cur.get("run_id") or "").strip() or None

        log.info(
            "[LAUNCH_STATE] job_id=%s cur_status=%s cur_run_id=%s cur_req=%s blocked_until=%.3f now=%.3f",
            job_id,
            cur_status,
            cur_run_id,
            cur_req,
            float(cur.get("blocked_until", 0.0) or 0.0),
            now,
        )

        # ✅ (G2) 이미 running/launching이면: 멱등 수렴만 허용
        if cur_status in ("launching", "running"):
            # 1) request_id 완전 동일 replay면 즉시 OK
            if request_id and cur_req and request_id == cur_req:
                log.info(
                    "[LAUNCH_IDEMP] job_id=%s reason=request_id_replay run_id=%s attempt=%s",
                    job_id, cur_run_id, cur.get("attempt"),
                )
                return {
                    "status": "already_running",
                    "job_id": job_id,
                    "run_id": cur_run_id,
                    "attempt": cur.get("attempt"),
                    "note": "idempotent_replay",
                }

            # 2) run_id 없이 두드리면 현재 run으로 수렴
            if not req_run_id:
                log.info(
                    "[LAUNCH_IDEMP] job_id=%s reason=no_run_id run_id=%s attempt=%s",
                    job_id, cur_run_id, cur.get("attempt"),
                )
                return {
                    "status": "already_running",
                    "job_id": job_id,
                    "run_id": cur_run_id,
                    "attempt": cur.get("attempt"),
                    "note": "replay_without_run_id",
                }

            # 3) 요청 run_id가 있는데, 현재 run_id와 다르면 충돌
            if cur_run_id and req_run_id != cur_run_id:
                log.warning(
                    "[LAUNCH_CONFLICT] job_id=%s reason=different_run req_run_id=%s cur_run_id=%s",
                    job_id, req_run_id, cur_run_id,
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "job_already_running_different_run",
                        "job_id": job_id,
                        "req_run_id": req_run_id,
                        "cur_run_id": cur_run_id,
                    },
                )

            # 4) 같은 run_id면 멱등 OK
            log.info(
                "[LAUNCH_IDEMP] job_id=%s reason=same_run_id run_id=%s attempt=%s",
                job_id, (cur_run_id or req_run_id), cur.get("attempt"),
            )
            return {
                "status": "already_running",
                "job_id": job_id,
                "run_id": cur_run_id or req_run_id,
                "attempt": cur.get("attempt"),
                "note": "idempotent_same_run_id",
            }

        # ✅ job block window
        blocked_until = float(cur.get("blocked_until", 0.0) or 0.0)
        if (not getattr(req, "allow_preempted", False)) and now < blocked_until:
            log.warning(
                "[LAUNCH_REJECT] job_id=%s reason=job_blocked blocked_until=%.3f now=%.3f",
                job_id, blocked_until, now,
            )
            raise HTTPException(
                status_code=409,
                detail={"reason": "job_blocked", "job_id": job_id, "blocked_until": blocked_until, "now": now},
            )

        # ---------------------------------------------------------
        # ✅ run_id SSOT sticky
        # ---------------------------------------------------------
        if cur_run_id:
            run_id_final = cur_run_id
        elif req_run_id:
            run_id_final = req_run_id
        else:
            run_id_final = uuid.uuid4().hex[:12]

        # ---------------------------------------------------------
        # 1) 노드 유효성 + draining (만료 시 자동 해제) + run_id 기반 idempotency
        # ---------------------------------------------------------
        for n in node_names:
            if n not in NODE_REGISTRY:
                log.warning("[LAUNCH_REJECT] job_id=%s reason=node_not_found node=%s", job_id, n)
                raise HTTPException(status_code=404, detail=f"Node '{n}' not found")

            nd = NODE_REGISTRY.get(n) or {}
            st = (nd.get("status") or "idle").lower()

            # ✅ draining은 "상태 문자열"이 아니라 "시간 조건"으로 판정해야 함
            if st == "draining":
                du = float(nd.get("drain_until", 0.0) or 0.0)

                # du가 없거나 이상하면 보수적으로 막되, 영원히 막지 않게 (여기선 그대로 reject)
                if du <= 0.0:
                    log.warning(
                        "[LAUNCH_REJECT] job_id=%s reason=node_draining_missing_until node=%s drain_until=%.3f now=%.3f",
                        job_id, n, du, now,
                    )
                    raise HTTPException(
                        status_code=409,
                        detail={"reason": "node_draining", "job_id": job_id, "node": n, "drain_until": du, "now": now},
                    )

                # ✅ 만료됐으면 draining 해제하고 계속 진행
                if now >= du:
                    try:
                        nd["status"] = "idle"
                        nd["drain_until"] = 0.0
                        NODE_REGISTRY[n] = nd
                        st = "idle"
                    except Exception:
                        pass
                else:
                    log.warning(
                        "[LAUNCH_REJECT] job_id=%s reason=node_draining node=%s drain_until=%.3f now=%.3f",
                        job_id, n, du, now,
                    )
                    raise HTTPException(
                        status_code=409,
                        detail={"reason": "node_draining", "job_id": job_id, "node": n, "drain_until": du, "now": now},
                    )

            cur_j = str(nd.get("current_job_id") or "")
            cur_node_run = (nd.get("current_run_id") or "").strip() or None

            if st != "idle" and cur_j and cur_j != job_id:
                log.warning(
                    "[LAUNCH_REJECT] job_id=%s reason=node_busy node=%s status=%s current_job_id=%s",
                    job_id, n, nd.get("status"), cur_j,
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "node_busy",
                        "job_id": job_id,
                        "node": n,
                        "status": nd.get("status"),
                        "current_job_id": cur_j,
                    },
                )

            if st in ("launching", "running", "preempting", "busy") and cur_j == job_id:
                if cur_node_run and cur_node_run == run_id_final:
                    continue
                log.warning(
                    "[LAUNCH_REJECT] job_id=%s reason=job_already_active_on_node node=%s req_run_id=%s cur_run_id=%s st=%s",
                    job_id, n, run_id_final, cur_node_run, nd.get("status"),
                )
                raise HTTPException(
                    status_code=409,
                    detail={
                        "reason": "job_already_active_on_node",
                        "job_id": job_id,
                        "node": n,
                        "status": nd.get("status"),
                        "current_job_id": cur_j,
                        "req_run_id": run_id_final,
                        "cur_run_id": cur_node_run,
                    },
                )

        # ---------------------------------------------------------
        # 2) script path 구성 + reserve
        # ---------------------------------------------------------
        try:
            script_file = MODEL_DS_MAP[(model, dataset)]
        except KeyError:
            script_file = make_filename(model, dataset)

        master_node = node_names[0]
        master_addr = NODE_REGISTRY[master_node]["ip"]

        nodes_with_paths = {}
        for n in node_names:
            prefix = NODE_REGISTRY[n]["prefix"]
            nodes_with_paths[n] = f"{prefix}/{script_file}"

        for n in node_names:
            NODE_REGISTRY[n]["status"] = "launching"
            NODE_REGISTRY[n]["current_job_id"] = job_id
            NODE_REGISTRY[n]["current_run_id"] = run_id_final

        # ---------------------------------------------------------
        # 3) resume 결정
        # ---------------------------------------------------------
        prev_latest = (cur.get("latest_checkpoint_path") if isinstance(cur, dict) else None)
        resume_from = _resolve_resume_path(job_id, req.resume_from_checkpoint, prev_latest)

        # ---------------------------------------------------------
        # 4) ACTIVE_JOBS 업데이트
        # ---------------------------------------------------------
        prev = ACTIVE_JOBS.get(job_id, {}) if job_id in ACTIVE_JOBS else {}
        ACTIVE_JOBS[job_id] = {
            "status": "launching",
            "nodes": node_names,
            "model": model,
            "dataset": dataset,
            "epochs": epochs,
            "master_addr": master_addr,
            "checkpoint_dir": f"./checkpoints/{job_id}",
            "batch_size_per_gpu": batch_size,

            "attempt": int(attempt),
            "last_launch_request_id": request_id,

            "latest_checkpoint_path": resume_from or prev.get("latest_checkpoint_path"),
            "run_id": run_id_final,

            "blocked_until": float(prev.get("blocked_until", 0.0) or 0.0),
            "requested_stop_reason": None,
        }

        log.info(
            "[LAUNCH_RESERVED] job_id=%s run_id=%s attempt=%d nodes=%s master=%s resume=%s",
            job_id, run_id_final, int(attempt), node_names, master_addr, resume_from,
        )

    # ---------------------------------------------------------
    # 5) worker fanout launch (lock 밖)
    # ---------------------------------------------------------
    log.info(
        "[LAUNCH_TRIGGER] job_id=%s run_id=%s attempt=%d fanout_nodes=%s",
        job_id,
        (ACTIVE_JOBS.get(job_id) or {}).get("run_id"),
        int(attempt),
        list(nodes_with_paths.keys()),
    )

    ok, code, detail = _trigger_run_on_nodes(
        job_id=job_id,
        nodes_with_paths=nodes_with_paths,
        epochs=epochs,
        resume_from=resume_from,
        batch_size=batch_size,
        attempt=int(attempt),
    )

    log.info(
        "[LAUNCH_TRIGGER_RESULT] job_id=%s run_id=%s ok=%s code=%s detail=%s",
        job_id,
        (ACTIVE_JOBS.get(job_id) or {}).get("run_id"),
        ok,
        code,
        str(detail)[:300],
    )

    if (not ok) and code == "ALREADY_RUNNING":
        with job_locks[job_id]:
            st2 = ACTIVE_JOBS.get(job_id) or {}
            st2["status"] = "running"
            ACTIVE_JOBS[job_id] = st2

        log.info(
            "[LAUNCH_IDEMP] job_id=%s reason=worker_reports_already_running run_id=%s attempt=%d",
            job_id,
            (ACTIVE_JOBS.get(job_id) or {}).get("run_id"),
            int(attempt),
        )

        return {
            "status": "already_running",
            "job_id": job_id,
            "run_id": (ACTIVE_JOBS.get(job_id) or {}).get("run_id"),
            "attempt": int(attempt),
            "note": "worker_reports_already_running",
            "detail": detail,
        }

    if not ok:
        log.error(
            "[LAUNCH_ROLLBACK] job_id=%s run_id=%s attempt=%d reason=%s",
            job_id,
            (ACTIVE_JOBS.get(job_id) or {}).get("run_id"),
            int(attempt),
            code,
        )

        with job_locks[job_id]:
            try:
                st3 = ACTIVE_JOBS.get(job_id) or {}
                mp = st3.get("master_port")
                if mp is not None:
                    try:
                        port_manager.release_port(int(mp))
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                ACTIVE_JOBS.pop(job_id, None)
            except Exception:
                pass

            for n in node_names:
                try:
                    NODE_REGISTRY[n]["status"] = "idle"
                    NODE_REGISTRY[n]["current_job_id"] = None
                    NODE_REGISTRY[n]["current_run_id"] = None
                    NODE_REGISTRY[n]["drain_until"] = 0.0
                except Exception:
                    pass

        raise HTTPException(status_code=500, detail={"reason": "launch_failed", "job_id": job_id, "detail": detail})

    with job_locks[job_id]:
        st4 = ACTIVE_JOBS.get(job_id) or {}
        st4["status"] = "running"
        ACTIVE_JOBS[job_id] = st4

        log.info(
            "[LAUNCH_OK] job_id=%s run_id=%s attempt=%s nodes=%s",
            job_id,
            st4.get("run_id"),
            st4.get("attempt"),
            node_names,
        )

        return {
            "status": "job_launched",
            "job_id": job_id,
            "nodes": node_names,
            "run_id": st4.get("run_id"),
            "attempt": st4.get("attempt"),
            "master_addr": st4.get("master_addr"),
            "master_port": st4.get("master_port"),
            "resume_from": st4.get("latest_checkpoint_path"),
        }

def _is_terminal_status(s: str) -> bool:
    s = (s or "").upper().strip()
    return s in ("FINISHED", "FAILED", "CANCELLED", "COMPLETED", "PREEMPTED", "STOPPED")

def _normalize_worker_status(s: str) -> str:
    s = (s or "").upper().strip()
    # 워커가 FINISHED/FAILED/PREEMPTED/CANCELLED/STOPPED 등으로 보냄
    if s in ("DONE", "SUCCESS"):
        return "FINISHED"
    if s in ("CANCELED",):
        return "CANCELLED"
    if not s:
        return "FAILED"
    return s

_STATUS_GUARD_LOCK = threading.Lock()
_TERMINAL_GUARD_LOCK = threading.Lock()
_STATUS_SEEN: Set[Tuple[str, str, int, str]] = set()
_TERMINAL_SEEN: Set[Tuple[str, str, int, str]] = set()

# 메모리 무한증식 방지(간단 GC)
_STATUS_SEEN_MAX = 20000
_TERMINAL_SEEN_MAX = 20000

def _gc_seen_sets():
    # 아주 단순하게 크기 초과 시 초기화 (실험용으로 충분)
    # 더 정교하게 하려면 dict(ts)로 바꾸면 됨.
    global _STATUS_SEEN, _TERMINAL_SEEN
    if len(_STATUS_SEEN) > _STATUS_SEEN_MAX:
        _STATUS_SEEN = set(list(_STATUS_SEEN)[-5000:])
    if len(_TERMINAL_SEEN) > _TERMINAL_SEEN_MAX:
        _TERMINAL_SEEN = set(list(_TERMINAL_SEEN)[-5000:])

class WorkerJobStatusReport(BaseModel):
    # worker가 실제로 보내는 필드에 맞춤
    job_id: str
    status: str
    exit_code: int = 0
    stderr_tail: Optional[str] = None
    run_id: Optional[str] = None
    attempt: int = 1
    terminated_reason: Optional[str] = None
    end_ts: Optional[float] = None

    class Config:
        extra = "allow"  # worker가 더 보내도 422 안 뜸


class JobStatusReport(BaseModel):
    job_id: str
    status: str
    exit_code: int
    stderr_tail: Optional[str] = None
    final_accuracy: Optional[float] = None

    run_id: Optional[str] = None # 추가
    terminated_reason: Optional[str] = None # 추가

@app.post("/report_job_status")
async def report_job_status(req: WorkerJobStatusReport):
    job_id = (req.job_id or "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    rep_status = _normalize_worker_status((req.status or "").strip())
    rep_run_id = (str(req.run_id or "").strip() or None)
    rep_attempt = int(req.attempt or 1)
    if rep_attempt <= 0:
        rep_attempt = 1

    exit_code = int(req.exit_code or 0)
    stderr_tail = (req.stderr_tail or "")
    terminated_reason = (str(req.terminated_reason or "").upper().strip() or None)

    now = time.time()
    end_ts = float(getattr(req, "end_ts", None) or req.end_ts or now) if hasattr(req, "end_ts") else float(now)
    drain_until = float(now + float(DRAIN_SECONDS))

    # ---- 0) 멱등 가드: (job_id, run_id, attempt, status) ----
    status_key = (job_id, str(rep_run_id or "NO_RUN"), int(rep_attempt), str(rep_status))
    with _STATUS_GUARD_LOCK:
        if status_key in _STATUS_SEEN:
            return {
                "ok": True,
                "status": "acked_duplicate",
                "job_id": job_id,
                "run_id": rep_run_id,
                "attempt": rep_attempt,
                "state": rep_status,
            }
        _STATUS_SEEN.add(status_key)
        _gc_seen_sets()

    nodes = []
    master_port = None
    cluster_id = None

    # ✅ 정확도 후보들(우선순위: ACTIVE_JOBS.final_accuracy -> ACTIVE_JOBS.current_accuracy/latest_accuracy -> req.final_accuracy)
    final_accuracy = None
    req_final_accuracy = None
    try:
        req_final_accuracy = getattr(req, "final_accuracy", None)
    except Exception:
        req_final_accuracy = None

    # ---- 1) ACTIVE_JOBS 반영 + stale run_id 방지 ----
    with job_locks[job_id]:
        info = ACTIVE_JOBS.get(job_id) or {}

        if not info:
            return {
                "ok": True,
                "status": "acked_no_job",
                "job_id": job_id,
                "run_id": rep_run_id,
                "attempt": rep_attempt,
                "state": rep_status,
            }

        cur_run_id = (str(info.get("run_id") or "").strip() or None)

        # stale run_id 보호
        if rep_run_id and cur_run_id and rep_run_id != cur_run_id:
            return {
                "ok": True,
                "status": "acked_ignored_stale",
                "job_id": job_id,
                "rep_run_id": rep_run_id,
                "cur_run_id": cur_run_id,
                "attempt": rep_attempt,
                "state": rep_status,
            }

        # run_id 동기화
        if rep_run_id and (not cur_run_id or cur_run_id == rep_run_id):
            info["run_id"] = rep_run_id
            cur_run_id = rep_run_id

        # attempt 저장(큰 값 우선)
        cur_attempt = int(info.get("attempt") or 0)
        if cur_attempt <= 0 or rep_attempt > cur_attempt:
            info["attempt"] = int(rep_attempt)

        info["last_status_ts"] = float(now)
        info["last_status"] = str(rep_status)
        info["last_exit_code"] = int(exit_code)
        info["last_stderr_tail"] = stderr_tail[-2000:] if stderr_tail else ""
        info["last_terminated_reason"] = terminated_reason
        info["status"] = rep_status.lower()

        cluster_id = (str(info.get("cluster_id") or "").strip() or None)
        nodes = list(info.get("nodes", []) or [])
        master_port = info.get("master_port", None)

        # ✅ final_accuracy 결정(스냅샷)
        # 1) info["final_accuracy"]
        fa = info.get("final_accuracy", None)

        # 2) fallback: current_accuracy / latest_accuracy 등 (체크포인트가 여기 중 하나로 넣고 있을 수 있음)
        if fa is None:
            for k in ("current_accuracy", "latest_accuracy", "last_accuracy", "acc", "accuracy"):
                v = info.get(k, None)
                if v is not None:
                    fa = v
                    break

        # 3) 그래도 없으면 req.final_accuracy (worker가 실어줄 수도 있음)
        if fa is None and req_final_accuracy is not None:
            fa = req_final_accuracy

        # ✅ float 정규화 (문자/Decimal/None 대비)
        try:
            final_accuracy = float(fa) if fa is not None else None
        except Exception:
            final_accuracy = None

        # terminal이면 drain 강화
        if _is_terminal_status(rep_status):
            info["blocked_until"] = float(max(float(info.get("blocked_until", 0.0) or 0.0), drain_until))

        ACTIVE_JOBS[job_id] = info

        # node draining 유지(launch 폭주 방지)
        if nodes:
            for n in nodes:
                try:
                    nd = NODE_REGISTRY.get(n) or {}
                    if _is_terminal_status(rep_status):
                        nd["status"] = "draining"
                        nd["drain_until"] = float(max(float(nd.get("drain_until", 0.0) or 0.0), drain_until))
                        nd["current_job_id"] = job_id
                        if cur_run_id:
                            nd["current_run_id"] = cur_run_id
                    NODE_REGISTRY[n] = nd
                except Exception:
                    pass

    # ---- 2) terminal이면 port 반환 (1회) ----
    do_terminal_actions = False
    if _is_terminal_status(rep_status):
        term_key = (job_id, str(rep_run_id or "NO_RUN"), int(rep_attempt), str(rep_status))
        with _TERMINAL_GUARD_LOCK:
            if term_key not in _TERMINAL_SEEN:
                _TERMINAL_SEEN.add(term_key)
                do_terminal_actions = True
                _gc_seen_sets()

        if do_terminal_actions:
            try:
                if master_port is not None:
                    port_manager.release_port(int(master_port))
            except Exception:
                pass

    # ---- 3) scheduler 통지: terminal이면 반드시 보냄 ----
    scheduler_notified = False
    notify_err = None

    if _is_terminal_status(rep_status) and do_terminal_actions:
        # ✅ scheduler가 “preempt vs complete”를 판단할 수 있게 status/reason 포함해서 보냄
        payload = {
            "job_id": job_id,
            "status": rep_status,
            "exit_code": int(exit_code),
            "final_accuracy": final_accuracy,
            "run_id": rep_run_id,
            "attempt": int(rep_attempt),
            "cluster_id": cluster_id,
            "reason": terminated_reason,
            "stderr_tail": stderr_tail[-2000:] if stderr_tail else "",
            "end_ts": float(end_ts),
        }

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(f"{SCHEDULER__ADDRESS}/report_job_completed", json=payload)
            if 200 <= int(r.status_code) < 300:
                scheduler_notified = True
            else:
                try:
                    notify_err = f"status={r.status_code} body={r.text[:2000]}"
                except Exception:
                    notify_err = f"status={r.status_code} body=<no-text>"
        except Exception as e:
            notify_err = repr(e)

    return {
        "ok": True,
        "status": "acked",
        "job_id": job_id,
        "run_id": rep_run_id,
        "attempt": int(rep_attempt),
        "state": rep_status,
        "scheduler_notified": bool(scheduler_notified),
        "notify_err": notify_err,
        "drain_until": float(drain_until) if _is_terminal_status(rep_status) else None,
        # ✅ 디버깅용: 실제로 completion에 실어보낸 정확도
        "final_accuracy_sent": final_accuracy,
    }

@app.post("/report_job_completed")
async def report_job_completed(report: JobCompleteReport):
    job_id = report.job_id

    info = ACTIVE_JOBS.get(job_id) or {}
    cur_run_id = info.get("run_id")
    rep_run_id = getattr(report, "run_id", None) or cur_run_id or "NO_RUN"

    # ✅ 완료 알림 1회 보장: (job_id, run_id) 단위로
    key = (str(job_id), str(rep_run_id))

    with _COMPLETION_LOCK:
        if key in _COMPLETION_SENT:
            log.info(f"[{job_id}] Completion suppressed (already sent). run_id={rep_run_id}")
            return {"status": "completion_acked", "scheduler_notified": False, "reason": "already_sent"}
        _COMPLETION_SENT.add(key)

    # 2) local cleanup
    info = ACTIVE_JOBS.get(job_id)

    if info is None:
        log.warning(f"[{job_id}] Job not found in ACTIVE_JOBS (completion received). run_id={rep_run_id}")
    else:
        mp = info.get("master_port")
        if mp is not None:
            try:
                port_manager.release_port(int(mp))
            except Exception:
                pass

        for n in info.get("nodes", []):
            try:
                NODE_REGISTRY[n]["status"] = "idle"
                NODE_REGISTRY[n]["current_job_id"] = None
            except Exception:
                pass

        info["status"] = "COMPLETED"
        info["exit_code"] = int(getattr(report, "exit_code", 0) or 0)

    log.info(f"[{job_id}] Job completed (exit_code={int(getattr(report, 'exit_code', 0) or 0)}) run_id={rep_run_id}")

    # 3) notify scheduler
    scheduler_url = f"{SCHEDULER__ADDRESS}/report_job_completed"
    payload = {
        "job_id": job_id,
        "exit_code": int(getattr(report, "exit_code", 0) or 0),
        "final_accuracy": (info.get("final_accuracy") if isinstance(info, dict) else None),
        "run_id": rep_run_id,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(scheduler_url, json=payload)

        if r.status_code == 200:
            log.info(f"[{job_id}] Reported completion to Scheduler OK (from /report_job_completed) run_id={rep_run_id}")
            return {"status": "completion_acked", "scheduler_notified": True, "run_id": rep_run_id}
        else:
            log.warning(f"[{job_id}] Scheduler ack failed: {r.status_code} {r.text} run_id={rep_run_id}")

    except Exception as e:
        log.warning(f"[{job_id}] Failed to notify Scheduler: {e} run_id={rep_run_id}")

    # 실패하면 재시도 가능하게 풀어줌
    with _COMPLETION_LOCK:
        _COMPLETION_SENT.discard(key)

    return {"status": "completion_acked", "scheduler_notified": False, "reason": "notify_failed", "run_id": rep_run_id}

@app.post("/report_job_stopped")
def report_job_stopped(req: JobStopReport):
    """
    Worker -> Global stop ACK endpoint.

    목표(SSOT 안정화):
      - stale run_id stop ack는 무시
      - stop ack를 받으면 최소 drain window 보장(blocked_until / node drain 유지)
      - terminal 확정은 report_job_status/report_job_completed에서만 (여긴 preempting 유지)
      - report_job_status 유실/지연 대비: force_release_deadline_ts를 설정해 stuck 수렴 가능
      - resize workflow 대기 해제: stop_event.set()
    """
    job_id = str(getattr(req, "job_id", "") or "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    rep_run_id = (str(getattr(req, "run_id", "") or "").strip() or None)
    reason = (str(getattr(req, "reason", "") or "PREEMPT").upper().strip() or "PREEMPT")
    now = float(time.time())

    # --- tunables ---
    try:
        FORCE_RELEASE_SEC = float(globals().get("FORCE_RELEASE_SEC", 60.0) or 60.0)
    except Exception:
        FORCE_RELEASE_SEC = 60.0
    if FORCE_RELEASE_SEC < 15.0:
        FORCE_RELEASE_SEC = 15.0  # 너무 짧으면 위험

    try:
        drain_seconds = float(DRAIN_SECONDS)
    except Exception:
        drain_seconds = 3.0
    if drain_seconds < 1.0:
        drain_seconds = 1.0

    drain_until = float(now + drain_seconds)

    with job_locks[job_id]:
        info = ACTIVE_JOBS.get(job_id) or {}

        # job 없으면 idempotent OK
        if not info:
            log.info(f"[{job_id}] report_job_stopped: job not found -> idempotent_ok run_id={rep_run_id} reason={reason}")
            return {"ok": True, "status": "acked_no_job", "job_id": job_id, "run_id": rep_run_id, "reason": reason}

        cur_run_id = (str(info.get("run_id") or "").strip() or None)

        # stale run_id stop ack는 무시 (새 run을 끊는 사고 방지)
        if rep_run_id and cur_run_id and rep_run_id != cur_run_id:
            log.warning(f"[{job_id}] report_job_stopped ignored (stale run_id). rep={rep_run_id} cur={cur_run_id}")
            return {"ok": True, "status": "acked_ignored_stale", "job_id": job_id, "cur_run_id": cur_run_id}

        # --- run_id SSOT 동기화는 "조건부"로만 ---
        # 1) cur_run_id가 비어 있고 rep가 있으면 채움
        # 2) cur_run_id가 있는데 rep가 None/""이면 덮어쓰기 금지
        # 3) cur_run_id와 rep가 같으면 그대로
        if (not cur_run_id) and rep_run_id:
            info["run_id"] = rep_run_id
            cur_run_id = rep_run_id

        # stop ack 반영
        info["last_stop_ack_ts"] = float(now)
        info["last_stop_reason"] = reason
        info["requested_stop_reason"] = reason

        # blocked/drain 윈도우 보장
        prev_block = float(info.get("blocked_until", 0.0) or 0.0)
        info["blocked_until"] = float(max(prev_block, drain_until))

        # ✅ 강제 수렴 데드라인: terminal status/report가 유실되면 결국 이걸로 정리되게
        prev_fr = float(info.get("force_release_deadline_ts", 0.0) or 0.0)
        if prev_fr <= 0.0:
            info["force_release_deadline_ts"] = float(now + FORCE_RELEASE_SEC)
        else:
            # 무한 연장 방지: 기존 값 유지(필요하면 더 이르게만)
            info["force_release_deadline_ts"] = float(prev_fr)

        # terminal 확정 전까지 preempting 유지
        st = (str(info.get("status") or "").lower() or "")
        if st not in ("finished", "failed", "cancelled", "completed"):
            info["status"] = "preempting"

        # ✅ resize workflow unblock: stop_event 세팅
        # (주의) stop_event가 "현재 런"을 기다리는 용도라면, stale 보호가 위에서 끝났으니 안전
        try:
            ev = info.get("stop_event")
            if ev is not None and hasattr(ev, "set"):
                ev.set()
        except Exception:
            pass

        # SSOT 반영
        ACTIVE_JOBS[job_id] = info

        # 노드 draining 유지 + current_job_id 유지 (launch 폭주 방지)
        nodes = list(info.get("nodes", []) or [])
        for n in nodes:
            n = str(n)
            if not n:
                continue
            try:
                nd = NODE_REGISTRY.get(n) or {}
                nd["status"] = "draining"
                nd["drain_until"] = float(max(float(nd.get("drain_until", 0.0) or 0.0), float(info["blocked_until"])))
                nd["current_job_id"] = job_id
                # run_id는 있으면만 고정
                if cur_run_id:
                    nd["current_run_id"] = cur_run_id
                NODE_REGISTRY[n] = nd
            except Exception:
                pass

    log.info(
        f"[{job_id}] report_job_stopped acked run_id={(rep_run_id or cur_run_id)} "
        f"reason={reason} blocked_until={float(ACTIVE_JOBS.get(job_id, {}).get('blocked_until') or 0.0)} "
        f"force_release_deadline_ts={float(ACTIVE_JOBS.get(job_id, {}).get('force_release_deadline_ts') or 0.0)}"
    )

    return {
        "ok": True,
        "status": "acked",
        "job_id": job_id,
        "run_id": (rep_run_id or cur_run_id),
        "reason": reason,
        "blocked_until": float(ACTIVE_JOBS.get(job_id, {}).get("blocked_until") or 0.0),
        "force_release_deadline_ts": float(ACTIVE_JOBS.get(job_id, {}).get("force_release_deadline_ts") or 0.0),
    }

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

@app.post("/report_progress")
async def report_progress(report: ProgressReport):
    job_id = report.job_id
    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    ACTIVE_JOBS[job_id]["last_progress"] = report.model_dump()
    ACTIVE_JOBS[job_id]["last_progress_ts"] = time.time()
    return {"status": "progress_acked"}


class JobMetricsReport(BaseModel):
    job_id: str
    attained_service: float

@app.post("/report_job_metrics")
async def report_job_metrics(report: JobMetricsReport):
    job_id = report.job_id
    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    ACTIVE_JOBS[job_id]["attained_service"] = report.attained_service
    ACTIVE_JOBS[job_id]["attained_service_ts"] = time.time()
    return {"status": "metrics_acked"}

# [추가] 500 에러 방지를 위한 헬퍼 함수 정의
def _get_serializable_job_status(job_info: dict) -> dict:
    """Returns a copy of the job info dict, stripped of non-JSON-serializable objects."""
    if not job_info:
        return {}
    
    # 얕은 복사
    serializable_info = job_info.copy()
    
    # non-serializable 키 제거
    for key in NON_SERIALIZABLE_KEYS:
        serializable_info.pop(key, None) # 키가 없어도 에러 안 나게 .pop 사용
        
    return serializable_info

def _safe_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None

def _safe_json_obj(obj: Any, max_len: int = 800) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= max_len else (s[:max_len] + "...(truncated)")

def _trigger_run_on_nodes(
    job_id: str,
    nodes_with_paths: Dict[str, str],
    epochs: int,
    resume_from: Optional[str] = None,
    batch_size: Optional[int] = None,
    attempt: int = 1,
) -> Tuple[bool, str, Dict[str, Any]]:
    if not nodes_with_paths:
        log.error(f"[{job_id}] No nodes provided to run.")
        return (False, "FAILED", {"reason": "no_nodes"})

    nodes_list = list(nodes_with_paths.keys())
    master_node_name = nodes_list[0]

    if master_node_name not in NODE_REGISTRY:
        log.error(f"[{job_id}] Master node {master_node_name} not found in NODE_REGISTRY.")
        return (False, "FAILED", {"reason": "master_not_found", "node": master_node_name})

    master_addr = NODE_REGISTRY[master_node_name]["ip"]
    world_size = len(nodes_list)

    resume_from = _resolve_resume_path(job_id, resume_from, None)

    # ✅ launch 직후 preempt 방지용 guard
    PREEMPT_GUARD_SEC = 20.0

    # ---------------------------------------------------------
    # ✅ run_id / master_port는 ACTIVE_JOBS SSOT
    #    + launch_ts / preempt_guard_until SSOT 추가
    # ---------------------------------------------------------
    with job_locks[job_id]:
        if job_id not in ACTIVE_JOBS:
            ACTIVE_JOBS[job_id] = {}

        st = ACTIVE_JOBS[job_id]
        now = time.time()

        # attempt SSOT
        try:
            st_attempt = int(st.get("attempt", 0) or 0)
        except Exception:
            st_attempt = 0
        if st_attempt <= 0:
            st_attempt = int(attempt)
        st["attempt"] = int(st_attempt)

        # ✅ run_id sticky
        run_id = (st.get("run_id") or "").strip()
        if not run_id:
            run_id = uuid.uuid4().hex[:12]
            st["run_id"] = run_id

        # ✅ master_port sticky
        master_port = st.get("master_port")
        if master_port is None:
            try:
                master_port = int(port_manager.get_port())
            except Exception as e:
                log.error(f"[{job_id}] No free master port: {e}")
                return (False, "FAILED", {"reason": "no_free_port", "err": str(e)})
            st["master_port"] = int(master_port)
        else:
            try:
                master_port = int(master_port)
            except Exception:
                master_port = None
            if master_port is None:
                try:
                    master_port = int(port_manager.get_port())
                except Exception as e:
                    log.error(f"[{job_id}] No free master port: {e}")
                    return (False, "FAILED", {"reason": "no_free_port", "err": str(e)})
                st["master_port"] = int(master_port)

        # ✅ NEW: launch 시각 + preempt guard
        st["launch_requested_ts"] = float(now)
        try:
            old_gu = float(st.get("preempt_guard_until", 0.0) or 0.0)
        except Exception:
            old_gu = 0.0
        st["preempt_guard_until"] = max(float(old_gu), float(now + float(PREEMPT_GUARD_SEC)))

        st.update({
            "nodes": nodes_list,
            "nodes_with_paths": nodes_with_paths,
            "master_addr": master_addr,
            "epochs": int(epochs),
            "batch_size_per_gpu": batch_size,
            "latest_checkpoint_path": resume_from or st.get("latest_checkpoint_path"),
            "status": st.get("status") or "launching",
        })
        ACTIVE_JOBS[job_id] = st

    log.info(
        f"[{job_id}] Launching job... run_id={run_id} attempt={attempt} "
        f"WS={world_size} Master={master_addr}:{master_port} resume={resume_from} "
        f"preempt_guard={PREEMPT_GUARD_SEC}s"
    )

    launched_nodes: List[str] = []
    already_running_hits: List[Dict[str, Any]] = []

    # ---------------------------------------------------------
    # fanout
    # ---------------------------------------------------------
    for rank, node_name in enumerate(nodes_list):
        if node_name not in NODE_REGISTRY:
            log.error(f"[{job_id}] Node {node_name} not found in NODE_REGISTRY.")
            break

        node_info = NODE_REGISTRY[node_name]
        specific_script_path = nodes_with_paths[node_name]
        gpu_id = node_info.get("gpu_id", 0)

        payload = {
            "job_id": job_id,
            "run_id": run_id,
            "attempt": int(attempt),

            "script_path": specific_script_path,
            "master_addr": master_addr,
            "master_port": int(master_port),
            "world_size": int(world_size),
            "rank": int(rank),
            "local_rank": 0,
            "gpu_id": int(gpu_id),

            "epochs": int(epochs),
            "resume_from_checkpoint": resume_from,
            "checkpoint_dir": f"./checkpoints/{job_id}",
            "global_server_addr": GLOBAL_SERVER_ADDRESS,

            "batch_size_per_gpu": int(batch_size) if batch_size is not None else 64,
            "learning_rate": 1e-3,
            "grad_accum": 1,
        }

        target_url = f"http://{node_info['ip']}:{node_info['agent_port']}/launch_task"

        try:
            log.info(
                f"[{job_id}] Sending run command to {node_name} (Rank {rank}) "
                f"path={specific_script_path} run_id={run_id} attempt={attempt}"
            )

            r = requests.post(target_url, json=payload, timeout=5)
            http = int(r.status_code)

            body = None
            if r.text:
                body = _safe_json_loads(r.text)
            if body is None:
                body = {"text": (r.text or "")[:500]}

            if 200 <= http < 300:
                try:
                    NODE_REGISTRY[node_name]["status"] = "busy"
                    NODE_REGISTRY[node_name]["current_job_id"] = job_id
                except Exception:
                    pass
                launched_nodes.append(node_name)
                continue

            if http == 409:
                detail = body.get("detail") if isinstance(body, dict) else None
                reason409 = None
                if isinstance(detail, dict):
                    reason409 = (detail.get("reason") or "").strip()

                if reason409 in ("job_already_running_different_run", "job_is_already_running", "already_running"):
                    already_running_hits.append({
                        "node": node_name,
                        "http": http,
                        "body": body,
                    })
                    log.warning(
                        f"[{job_id}] launch_task blocked on {node_name}: http=409 reason={reason409} body={_safe_json_obj(body)}"
                    )
                    break

            log.error(f"[{job_id}] launch_task failed on {node_name}: http={http} body={_safe_json_obj(body)}")
            break

        except Exception as e:
            log.error(f"[{job_id}] Failed to send command to {node_name}: {e}")
            break

    # ---------------------------------------------------------
    # CASE 1) 409 ALREADY_RUNNING: rollback 금지 + SSOT 동기화
    # ---------------------------------------------------------
    if already_running_hits:
        cur_run_id = None
        try:
            hit0 = already_running_hits[0]
            b0 = hit0.get("body") or {}
            d0 = b0.get("detail") if isinstance(b0, dict) else None
            if isinstance(d0, dict):
                cur_run_id = (d0.get("cur_run_id") or "").strip() or None
        except Exception:
            cur_run_id = None

        now2 = time.time()
        with job_locks[job_id]:
            st = ACTIVE_JOBS.get(job_id) or {}
            if cur_run_id:
                st["run_id"] = cur_run_id
            st["status"] = "running"
            st["blocked_until"] = float(now2 + float(DRAIN_SECONDS))
            # ✅ 이미 running이면 guard는 너무 길게 끌 필요 없으면 줄여도 됨(선택)
            # st["preempt_guard_until"] = max(float(st.get("preempt_guard_until",0.0) or 0.0), now2 + 5.0)
            ACTIVE_JOBS[job_id] = st

        for hit in already_running_hits:
            n = hit.get("node")
            if n and n in NODE_REGISTRY:
                try:
                    NODE_REGISTRY[n]["status"] = "busy"
                    NODE_REGISTRY[n]["current_job_id"] = job_id
                except Exception:
                    pass

        return (False, "ALREADY_RUNNING", {
            "reason": "job_already_running_on_worker",
            "hits": already_running_hits,
            "cur_run_id": cur_run_id,
            "run_id_used": run_id,
            "master_port_used": master_port,
        })

    # ---------------------------------------------------------
    # CASE 2) 전체 성공
    # ---------------------------------------------------------
    if len(launched_nodes) == len(nodes_list):
        with job_locks[job_id]:
            st = ACTIVE_JOBS.get(job_id) or {}
            st["status"] = "running"
            ACTIVE_JOBS[job_id] = st

        return (True, "OK", {
            "run_id": run_id,
            "master_port": master_port,
            "launched_nodes": launched_nodes,
        })

    # ---------------------------------------------------------
    # CASE 3) 진짜 실패: rollback
    # ---------------------------------------------------------
    log.warning(f"[{job_id}] Launch partial failure. launched={launched_nodes}, total={nodes_list}. Rolling back.")

    try:
        if launched_nodes:
            _fanout_stop_to_all_nodes(job_id, launched_nodes, run_id, reason="LAUNCH_ROLLBACK")
    except Exception:
        pass

    try:
        if master_port is not None:
            port_manager.release_port(int(master_port))
    except Exception:
        pass

    du = time.time() + float(DRAIN_SECONDS)
    for n in nodes_list:
        try:
            NODE_REGISTRY[n]["status"] = "draining"
            NODE_REGISTRY[n]["drain_until"] = du
            NODE_REGISTRY[n]["current_job_id"] = None
        except Exception:
            pass

    with job_locks[job_id]:
        st = ACTIVE_JOBS.get(job_id) or {}
        st["status"] = "error"
        st["blocked_until"] = du
        st.pop("run_id", None)
        st.pop("master_port", None)
        # ✅ 실패면 guard도 제거(다음 attempt에서 다시 세팅)
        st.pop("preempt_guard_until", None)
        st.pop("launch_requested_ts", None)
        ACTIVE_JOBS[job_id] = st

    return (False, "FAILED", {
        "reason": "launch_failed",
        "launched_nodes": launched_nodes,
        "nodes": nodes_list,
    })

def _perform_resize_job(job_id: str, new_nodes: List[str]):
    with job_locks[job_id]:
        log.info(f"[{job_id}] Starting resize process for nodes: {new_nodes}")

        if job_id not in ACTIVE_JOBS:
            log.error(f"[{job_id}] Resize failed: Job not found.")
            return

        old_job_info = ACTIVE_JOBS[job_id].copy()
        old_nodes = list(old_job_info.get("nodes", []) or [])
        if not old_nodes:
            log.error(f"[{job_id}] Resize failed: old_nodes empty.")
            ACTIVE_JOBS[job_id]["status"] = "error"
            return

        # ✅ 현재 run_id 확보
        cur_run_id = (old_job_info.get("run_id") or "").strip() or None

        master_node_name = old_nodes[0]
        master_node_info = NODE_REGISTRY.get(master_node_name) or {}
        stop_url = f"http://{master_node_info.get('ip')}:{master_node_info.get('agent_port')}/stop_task"

        ACTIVE_JOBS[job_id]["status"] = "stopping"
        ACTIVE_JOBS[job_id]["stop_event"] = threading.Event()
        ACTIVE_JOBS[job_id]["requested_stop_reason"] = "RESIZE"

    # stop 요청은 lock 밖
    try:
        log.info(f"[{job_id}] Sending stop signal to Rank 0 ({master_node_name})... run_id={cur_run_id}")
        requests.post(stop_url, json={"job_id": job_id, "run_id": cur_run_id, "reason": "RESIZE"}, timeout=5)
    except Exception as e:
        with job_locks[job_id]:
            log.error(f"[{job_id}] Resize failed: stop_task send failed: {e}")
            ACTIVE_JOBS[job_id]["status"] = "error"
        return

    # stop_event wait
    log.info(f"[{job_id}] Waiting for job to confirm stop (timeout 180s)...")
    ev = ACTIVE_JOBS[job_id].get("stop_event")
    event_triggered = False
    try:
        event_triggered = bool(ev.wait(timeout=180.0)) if ev else False
    except Exception:
        event_triggered = False

    if not event_triggered:
        with job_locks[job_id]:
            log.error(f"[{job_id}] Resize failed: stop confirmation timed out.")
            ACTIVE_JOBS[job_id]["status"] = "error"
        return

    log.info(f"[{job_id}] Stop confirmed. Proceeding resize relaunch.")

    with job_locks[job_id]:
        # old 자원 정리(포트는 report_job_status에서도 풀지만 여기서도 안전하게)
        try:
            mp = old_job_info.get("master_port")
            if mp is not None:
                port_manager.release_port(int(mp))
        except Exception:
            pass

        for node_name in old_nodes:
            try:
                NODE_REGISTRY[node_name]["status"] = "idle"
                NODE_REGISTRY[node_name]["current_job_id"] = None
            except Exception:
                pass

        resume_path = _resolve_resume_path(job_id, old_job_info.get("latest_checkpoint_path"), None)
        if not resume_path:
            log.warning(f"[{job_id}] No usable checkpoint found. Relaunching from scratch.")

        model = old_job_info.get("model")
        dataset = old_job_info.get("dataset")

        try:
            script_file = MODEL_DS_MAP[(model, dataset)]
        except Exception:
            script_file = make_filename(model, dataset)

        # 새 nodes_with_paths
        nodes_with_paths: Dict[str, str] = {}
        for n in new_nodes:
            if n not in NODE_REGISTRY:
                log.error(f"[{job_id}] Resize failed: unknown node {n}")
                ACTIVE_JOBS[job_id]["status"] = "error"
                return
            prefix = NODE_REGISTRY[n]["prefix"]
            nodes_with_paths[n] = f"{prefix}/{script_file}"

        ACTIVE_JOBS[job_id]["status"] = "resizing"

        # attempt 증가(SSOT가 외부라면 여기 값은 참고용)
        try:
            next_attempt = int(old_job_info.get("attempt", 1) or 1) + 1
        except Exception:
            next_attempt = 2
        ACTIVE_JOBS[job_id]["attempt"] = next_attempt

    ok = _trigger_run_on_nodes(
        job_id=job_id,
        nodes_with_paths=nodes_with_paths,
        epochs=int(old_job_info.get("epochs", 100) or 100),
        resume_from=resume_path,
        batch_size=old_job_info.get("batch_size_per_gpu"),
        attempt=int(ACTIVE_JOBS.get(job_id, {}).get("attempt", next_attempt)),
    )

    with job_locks[job_id]:
        if ok:
            ACTIVE_JOBS[job_id]["status"] = "running"
            log.info(f"[{job_id}] Resize process complete. run_id={ACTIVE_JOBS[job_id].get('run_id')}")
        else:
            ACTIVE_JOBS[job_id]["status"] = "error"
            log.error(f"[{job_id}] Resize relaunch failed.")

# --- 4. API 엔드포인트 ---

@app.get("/status")
async def get_status():
    """(클라이언트용) 서버의 *전체* 노드 및 잡 상태를 반환합니다."""
    
    serializable_jobs = {
        job_id: _get_serializable_job_status(job_info) # 헬퍼 함수 사용
        for job_id, job_info in ACTIVE_JOBS.items()
    }
    return {"node_registry": NODE_REGISTRY, "active_jobs": serializable_jobs}


@app.post("/resize_job")
async def resize_job(req: ResizeRequest, background_tasks: BackgroundTasks):
    job_id = req.job_id
    new_nodes = req.new_nodes

    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job_locks[job_id].locked():
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' is already processing a request.")

    current_nodes = set(ACTIVE_JOBS[job_id]["nodes"])
    log.info("Current nodes: %s", current_nodes)
    log.info("Requested new nodes: %s", new_nodes)

    for node_name in new_nodes:
        if node_name not in NODE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"New node '{node_name}' not found.")
        # 상태가 idle이 아니면서, 현재 job이 쓰고 있는 노드도 아니면 → 진짜 다른 job이 쓰는 중
        if NODE_REGISTRY[node_name]["status"] != "idle" and node_name not in current_nodes:
            raise HTTPException(status_code=409, detail=f"New node '{node_name}' is busy with another job.")

    # 여기서는 노드 이름 리스트만 넘기고,
    # 실제 script path 조립은 _perform_resize_job 안에서 다시 한다.
    background_tasks.add_task(_perform_resize_job, job_id, new_nodes)

    return {"status": "resize_initiated", "job_id": job_id, "new_nodes": new_nodes}

# preempt 폭주 방지: job별 마지막 preempt 시각
PREEMPT_GUARD_LOCK = threading.Lock()
LAST_PREEMPT_TS = {}
PREEMPT_MIN_INTERVAL_SEC = 2.0
DRAIN_SECONDS = 3.0

def _fanout_stop_to_all_nodes(job_id: str, nodes: List[str], run_id: Optional[str], reason: str) -> Dict[str, Any]:
    """
    job이 점유 중인 모든 노드의 worker agent에 /stop_task 를 던지고,
    노드별 ACK/FAIL을 수집해서 반환합니다.
    - run_id/reason 반드시 포함
    - 실패해도 예외 던지지 않고 fails에 기록
    """
    acks: List[Dict[str, Any]] = []
    fails: List[Dict[str, Any]] = []

    for n in (nodes or []):
        try:
            ni = NODE_REGISTRY.get(n) or {}
            ip = ni.get("ip")
            port = ni.get("agent_port")
            if not ip or not port:
                fails.append({"node": n, "err": "missing ip/agent_port"})
                continue

            stop_url = f"http://{ip}:{port}/stop_task"
            payload = {"job_id": job_id, "run_id": run_id, "reason": reason}

            r = requests.post(stop_url, json=payload, timeout=1.5)
            http = int(r.status_code)

            body = None
            if r.text:
                body = _safe_json_loads(r.text)
            if body is None:
                body = {"text": (r.text or "")[:300]}

            if 200 <= http < 300:
                acks.append({"node": n, "http": http, "body": body})
            else:
                fails.append({"node": n, "http": http, "body": body})

        except Exception as e:
            fails.append({"node": n, "err": repr(e)})

    return {"ok": (len(fails) == 0), "acks": acks, "fails": fails}

@app.post("/preempt_job")
def preempt_job(req: PreemptJobRequest):
    job_id = str(req.job_id).strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    reason = (req.reason or "PREEMPT").upper().strip()
    now = time.time()

    # -----------------------------
    # ✅ 0) job 없으면 idempotent OK (기존 유지)
    # -----------------------------
    job_info = ACTIVE_JOBS.get(job_id)
    if not job_info:
        touched = 0
        for n, nd in (NODE_REGISTRY or {}).items():
            try:
                if str(nd.get("current_job_id") or "") == job_id:
                    nd["status"] = "draining"
                    nd["drain_until"] = now + float(DRAIN_SECONDS)
                    nd.setdefault("current_run_id", None)
                    touched += 1
            except Exception:
                pass

        log.info(f"[{job_id}] Preempt requested but job not in ACTIVE_JOBS -> idempotent_ok touched_nodes={touched}")
        return {
            "ok": True,
            "status": "ok",
            "job_id": job_id,
            "note": "idempotent_ok_job_not_found",
            "touched_nodes": touched,
        }

    # -----------------------------
    # ✅ 1) launch 직후 preempt 폭주 방지 (NEW)
    #    - 스케줄러가 막 띄운 backfill을 바로 죽이는 thrash 차단
    # -----------------------------
    # 기본 guard. (너 스케줄러에서 넣은 값이 있으면 그걸 우선 존중)
    DEFAULT_PREEMPT_GUARD_SEC = 20.0

    with job_locks[job_id]:
        st0 = ACTIVE_JOBS.get(job_id) or {}

        # ✅ HOL reclaim(HOL_GANG_PREEMPT)은 즉시 선점이 목적이므로 guard 우회
        # (이거 없으면 reclaim이 20초 동안 막혀서 gang4가 영영 못 뜸)
        is_hol_reclaim = ("HOL_GANG_PREEMPT" in str(reason))

        # (a) 스케줄러/런처가 넣어둔 guard 우선
        try:
            guard_until = float(st0.get("preempt_guard_until", 0.0) or 0.0)
        except Exception:
            guard_until = 0.0

        # (b) guard가 없다면 "launch_requested_ts" 기준으로도 방어
        try:
            launch_ts = float(st0.get("launch_requested_ts", 0.0) or 0.0)
        except Exception:
            launch_ts = 0.0

        # 이미 preempting이면(같은 run_id 멱등은 뒤에서 처리) guard로 막지 않음
        cur_status = (st0.get("status") or "").lower().strip()

        # guard_until이 설정돼 있으면 최우선 적용 (단, HOL reclaim은 우회)
        if (not is_hol_reclaim) and cur_status not in ("preempting",) and guard_until > 0.0 and now < guard_until:
            log.info(f"[{job_id}] Preempt suppressed by guard_until={guard_until:.3f} now={now:.3f} reason={reason}")
            return {
                "ok": True,
                "status": "ok",
                "job_id": job_id,
                "note": "preempt_suppressed_guard_until",
                "guard_until": guard_until,
            }

        # guard_until이 없더라도, launch 직후(DEFAULT_PREEMPT_GUARD_SEC)에는 preempt 억제 (단, HOL reclaim은 우회)
        if (not is_hol_reclaim) and cur_status not in ("preempting",) and launch_ts > 0.0 and (now - launch_ts) < float(DEFAULT_PREEMPT_GUARD_SEC):
            log.info(f"[{job_id}] Preempt suppressed by launch_guard since={now-launch_ts:.3f}s reason={reason}")
            return {
                "ok": True,
                "status": "ok",
                "job_id": job_id,
                "note": "preempt_suppressed_launch_guard",
                "launch_ts": launch_ts,
            }

    # -----------------------------
    # ✅ 2) debounce (기존 2초) — guard 뒤로 이동 (중요)
    # -----------------------------
    with PREEMPT_GUARD_LOCK:
        last = float(LAST_PREEMPT_TS.get(job_id, 0.0) or 0.0)
        if now - last < PREEMPT_MIN_INTERVAL_SEC:
            return {"ok": True, "status": "ok", "job_id": job_id, "note": "debounced"}
        LAST_PREEMPT_TS[job_id] = now

    # ---------------------------------------------------------
    # ✅ 3) SSOT 마킹: preempting + run_id 고정 + blocked_until 설정
    # ---------------------------------------------------------
    with job_locks[job_id]:
        job_info = ACTIVE_JOBS.get(job_id) or {}
        nodes = list(job_info.get("nodes", []) or [])

        if not nodes:
            job_info["status"] = "preempted"
            job_info["blocked_until"] = float(now + float(DRAIN_SECONDS))
            job_info["requested_stop_reason"] = reason
            ACTIVE_JOBS[job_id] = job_info
            return {"ok": True, "status": "ok", "job_id": job_id, "note": "no_nodes_treated_as_done"}

        run_id = (req.run_id or job_info.get("run_id") or "").strip() or None

        cur_status = (job_info.get("status") or "").lower()
        cur_preempt_run = (job_info.get("last_preempt_run_id") or "").strip() or None
        if cur_status == "preempting" and run_id and cur_preempt_run == run_id:
            return {
                "ok": True,
                "status": "ok",
                "job_id": job_id,
                "run_id": run_id,
                "note": "idempotent_preempt_same_run",
                "blocked_until": float(job_info.get("blocked_until") or 0.0),
            }

        blocked_until = float(now + float(DRAIN_SECONDS))
        job_info["blocked_until"] = blocked_until
        job_info["status"] = "preempting"
        job_info["preempt_requested_ts"] = float(now)
        job_info["requested_stop_reason"] = reason
        job_info["last_preempt_run_id"] = run_id

        # ✅ (선택) preempt 시점에 guard도 잠깐 더 늘려서 연쇄 preempt 방지
        #    (fanout 실패/지연으로 또 때리는 케이스)
        try:
            gu = float(job_info.get("preempt_guard_until", 0.0) or 0.0)
        except Exception:
            gu = 0.0
        if "HOL_GANG_PREEMPT" in reason:
            # HOL reclaim은 연쇄 선점이 필요할 수 있어 guard를 길게 늘리지 않음
            job_info["preempt_guard_until"] = float(now + 0.5)
        else:
            job_info["preempt_guard_until"] = max(gu, float(now + float(DRAIN_SECONDS)))

        ACTIVE_JOBS[job_id] = job_info

        drain_until = float(now + float(DRAIN_SECONDS))
        for n in nodes:
            try:
                nd = NODE_REGISTRY.get(n) or {}
                nd["status"] = "draining"
                nd["drain_until"] = drain_until
                nd["current_job_id"] = job_id
                if run_id:
                    nd["current_run_id"] = run_id
                NODE_REGISTRY[n] = nd
            except Exception:
                pass

    # ---------------------------------------------------------
    # ✅ 4) stop fanout (락 밖) + ACK 수집
    # ---------------------------------------------------------
    ack: Dict[str, Any] = {"ok": False, "acks": [], "fails": []}
    try:
        ack = _fanout_stop_to_all_nodes(job_id, nodes, run_id, reason=reason) or ack
    except Exception as e:
        log.warning(f"[{job_id}] fanout_stop failed: {e!r}")

    # ---------------------------------------------------------
    # ✅ 5) 결과 정리
    # ---------------------------------------------------------
    try:
        du2 = float(time.time() + float(DRAIN_SECONDS))
        for a in (ack.get("acks") or []):
            n = (a.get("node") or "").strip()
            if not n or n not in NODE_REGISTRY:
                continue
            try:
                nd = NODE_REGISTRY.get(n) or {}
                nd["status"] = "draining"
                nd["drain_until"] = max(float(nd.get("drain_until") or 0.0), du2)
                nd["current_job_id"] = job_id
                if run_id:
                    nd["current_run_id"] = run_id
                NODE_REGISTRY[n] = nd
            except Exception:
                pass
    except Exception:
        pass

    with job_locks[job_id]:
        st = ACTIVE_JOBS.get(job_id) or {}
        st["status"] = "preempting"
        st["blocked_until"] = float(now + float(DRAIN_SECONDS))
        st["last_preempt_ack"] = ack
        ACTIVE_JOBS[job_id] = st

    log.info(f"[{job_id}] Preempt requested. run_id={run_id} reason={reason} ack={_safe_json_obj(ack)}")
    return {
        "ok": True,
        "status": "ok",
        "job_id": job_id,
        "run_id": run_id,
        "blocked_until": float(now + float(DRAIN_SECONDS)),
        "drain_until": float(now + float(DRAIN_SECONDS)),
        "ack": ack,
    }

@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """(클라이언트용) 특정 잡(job)의 상세 상태를 반환합니다."""
    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    log.info(f"[ClientQuery] Returning status for {job_id}")
    
    serializable_details = _get_serializable_job_status(ACTIVE_JOBS[job_id]) # 헬퍼 함수 사용
    return {"job_id": job_id, "status_details": serializable_details}


# --- 5. 워커 노드로부터 보고를 받는 API ---
@app.post("/report_checkpoint")
async def report_checkpoint(report: CheckpointReport):
    job_id = report.job_id
    if job_id in ACTIVE_JOBS:
        ACTIVE_JOBS[job_id]["latest_checkpoint_path"] = report.checkpoint_path
        ACTIVE_JOBS[job_id]["current_epoch"] = report.current_epoch
        ACTIVE_JOBS[job_id]["total_epochs"] = report.total_epochs
        ACTIVE_JOBS[job_id]["current_accuracy"] = report.latest_accuracy
        ACTIVE_JOBS[job_id]["current_eval_loss"] = report.latest_eval_loss
        
        log.info(f"[{job_id}] Received report: Epoch {report.current_epoch}/{report.total_epochs} (Acc: {report.latest_accuracy:.2f}%)")
        return {"status": "checkpoint_acked"}
    
    raise HTTPException(status_code=404, detail="Job not found")

@app.post("/report_job_stopped")
def report_job_stopped(req: JobStopReport):
    job_id = str(getattr(req, "job_id", "") or "").strip()
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    rep_run_id = (str(getattr(req, "run_id", "") or "").strip() or None)
    reason = (str(getattr(req, "reason", "") or "PREEMPT").upper().strip() or "PREEMPT")
    now = time.time()

    FORCE_RELEASE_SEC = float(globals().get("FORCE_RELEASE_SEC", 60.0) or 60.0)
    drain_until = float(now + float(DRAIN_SECONDS))

    with job_locks[job_id]:
        info = ACTIVE_JOBS.get(job_id) or {}

        # job 없으면 idempotent OK
        if not info:
            log.info(f"[{job_id}] report_job_stopped: job not found -> idempotent_ok run_id={rep_run_id} reason={reason}")
            return {"ok": True, "status": "acked_no_job", "job_id": job_id}

        cur_run_id = (str(info.get("run_id") or "").strip() or None)

        # stale run_id면 무시 (새 run을 끊으면 안 됨)
        if rep_run_id and cur_run_id and rep_run_id != cur_run_id:
            log.warning(f"[{job_id}] report_job_stopped ignored (stale run_id). rep={rep_run_id} cur={cur_run_id}")
            return {"ok": True, "status": "acked_ignored_stale", "job_id": job_id, "cur_run_id": cur_run_id}

        # stop ack 반영
        info["last_stop_ack_ts"] = float(now)
        info["last_stop_reason"] = reason

        # blocked/drain 연장
        prev_block = float(info.get("blocked_until", 0.0) or 0.0)
        info["blocked_until"] = float(max(prev_block, drain_until))

        # ✅ 강제 수렴 데드라인 (terminal status가 안 오면 결국 이걸로 수렴)
        prev_fr = float(info.get("force_release_deadline_ts", 0.0) or 0.0)
        if prev_fr <= 0.0:
            info["force_release_deadline_ts"] = float(now + FORCE_RELEASE_SEC)
        else:
            # 이미 있으면 더 늦추지 말고 유지(무한 연장 방지)
            info["force_release_deadline_ts"] = float(prev_fr)

        # 상태: terminal은 report_job_status가 확정, 여기서는 preempting 유지
        st = (str(info.get("status") or "").lower() or "")
        if st not in ("finished", "failed", "cancelled", "completed"):
            info["status"] = "preempting"

        # 노드 draining 유지 + current_job_id 유지 (launch 폭주 방지)
        nodes = list(info.get("nodes", []) or [])
        for n in nodes:
            try:
                nd = NODE_REGISTRY.get(n) or {}
                nd["status"] = "draining"
                nd["drain_until"] = float(max(float(nd.get("drain_until", 0.0) or 0.0), info["blocked_until"]))
                nd["current_job_id"] = job_id
                nd["current_run_id"] = cur_run_id
                NODE_REGISTRY[n] = nd
            except Exception:
                pass

        ACTIVE_JOBS[job_id] = info

    log.info(f"[{job_id}] report_job_stopped acked run_id={rep_run_id or cur_run_id} reason={reason} drain_until={drain_until}")
    return {
        "ok": True,
        "status": "acked",
        "job_id": job_id,
        "run_id": (rep_run_id or cur_run_id),
        "reason": reason,
        "drain_until": float(drain_until),
        "force_release_deadline_ts": float(ACTIVE_JOBS.get(job_id, {}).get("force_release_deadline_ts") or 0.0),
    }

# argparse를 사용하여 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Server Manager")
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Port to bind the server to (default: 8000)')
    parser.add_argument('--public_ip', type=str, required=True,
                        help='Publicly reachable IP or hostname of this server (e.g., 192.168.0.100)')
    args = parser.parse_args()

    GLOBAL_SERVER_ADDRESS = f"http://{args.public_ip}:{args.port}"
    
    log.info(f"Starting Global Server Manager on {args.host}:{args.port}")
    log.info(f"Workers will report back to: {GLOBAL_SERVER_ADDRESS}")
    
    uvicorn.run(app, host=args.host, port=args.port)