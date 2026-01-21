import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import time
import threading
import itertools
import logging
from typing import Optional, Dict
from collections import defaultdict
import argparse
from typing import List


# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

app = FastAPI()

# --- 1. 글로벌 상태 관리 ---

SCHEDULER__ADDRESS = "http://163.180.117.216:8082"
GLOBAL_SERVER_ADDRESS = None
NON_SERIALIZABLE_KEYS = ["stop_event"]

MODEL_BASENAME = {
    "ResNet-50":        "Resnet50",
    "ResNet-18":        "Resnet18",
    "DenseNet-121":     "Densenet121",
    "MobileNetV2":      "MobileNetV2",
    "EfficientNetV2-S": "EfficientNetV2",  # 파일명은 -S 없이 EfficientNetV2.* 로 존재
    "EfficientNetV2":   "EfficientNetV2",
    "VGG-11":           "VGG11",
    "DistilBERT":       "DistilBert",
}

DS_PREFIX = {
    "CIFAR-100":     "DDP_CIFAR100_",
    "CIFAR-10":      "DDP_CIFAR10_",
    "TinyImageNet":  "DDP_TinyImageNet_",
    "Fashion-MNIST": "DDP_FashoinMNIST_",  # 실제 파일명 철자(Fashoin)와 맞춤
    "MNIST":         "DDP_MNIST_",
    "SST2":          "DDP_SST2_",
    "SST-2":         "DDP_SST2_",
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
}

NODE_REGISTRY = {
    # cluster A
    "node_a": {"ip": "163.180.117.216", "agent_port": 8001, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None},
    "node_b": {"ip": "163.180.117.216", "agent_port": 8002, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None},
    "node_c": {"ip": "163.180.117.216", "agent_port": 8003, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None},
    "node_d": {"ip": "163.180.117.216", "agent_port": 8004, "prefix": "/home/ubuntu216/SUN", "status": "idle", "current_job_id": None},

    # cluster B
    "node_e": {"ip": "163.180.160.61", "agent_port": 8005, "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode", "status": "idle", "current_job_id": None},
    "node_f": {"ip": "163.180.160.61", "agent_port": 8006, "prefix": "/nas2/data/dlwmznzl1/CCGRID/TrainingCode", "status": "idle", "current_job_id": None},
    "node_g": {"ip": "163.180.160.61", "agent_port": 8007, "prefix": "/data/breath12/CCGRID/TrainingCode", "status": "idle", "current_job_id": None},
    "node_h": {"ip": "163.180.160.61", "agent_port": 8008, "prefix": "/data/breath12/CCGRID/TrainingCode", "status": "idle", "current_job_id": None},
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

class JobRequest(BaseModel):
    nodes_with_paths: Dict[str, str]  
    epochs: int = 100
    batch_size: Optional[int] = None

class ResizeRequest(BaseModel):
    job_id: str
    new_nodes_with_paths: Dict[str, str]

class CheckpointReport(BaseModel):
    job_id: str
    current_epoch: int
    total_epochs: int
    latest_accuracy: float
    latest_eval_loss: float
    checkpoint_path: str

class JobStopReport(BaseModel):
    job_id: str

class JobCompleteReport(BaseModel):
    job_id: str
    exit_code: int = 0



# --- 3. 핵심 헬퍼 함수 ---
@app.post("/launch_job")
async def launch_job(req: LaunchJobRequest):
    job_id = req.job_id
    node_names = req.nodes
    model = req.model
    dataset = req.dataset
    epochs = req.epochs
    batch_size=req.batch_size

    # --- 노드 검증 ---
    if not node_names:
        raise HTTPException(status_code=400, detail="nodes must not be empty")
    for n in node_names:
        if n not in NODE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Node '{n}' not found")
        if NODE_REGISTRY[n]["status"] != "idle" and NODE_REGISTRY[n]["current_job_id"] != job_id:
            raise HTTPException(status_code=409, detail=f"Node '{n}' is busy")

    # --- 스크립트 파일명 결정 ---
    try:
        script_file = MODEL_DS_MAP[(model, dataset)]
    except KeyError:
        script_file = make_filename(model, dataset)

    # --- master 주소/포트 ---
    master_node = node_names[0]
    master_addr = NODE_REGISTRY[master_node]["ip"]
    master_port = port_manager.get_port()

    # --- 노드별 실제 경로 구성 ---
    nodes_with_paths = {}
    for n in node_names:
        prefix = NODE_REGISTRY[n]["prefix"]
        script_path = f"{prefix}/{script_file}"
        nodes_with_paths[n] = script_path

    ACTIVE_JOBS[job_id] = {
        "status": "launching",
        "nodes": node_names,
        "model": model,
        "dataset": dataset,
        "epochs": epochs,
        "master_addr": master_addr,
        "master_port": master_port,
        "checkpoint_dir": f"./checkpoints/{job_id}",
        "batch_size_per_gpu": batch_size,
    }

    # --- 실제 실행 ---
    success = _trigger_run_on_nodes(
        job_id=job_id,
        nodes_with_paths=nodes_with_paths,
        epochs=epochs,
        resume_from=None,
        batch_size=batch_size,
    )

    if not success:
        del ACTIVE_JOBS[job_id]
        for n in node_names:
            NODE_REGISTRY[n]["status"] = "idle"
        raise HTTPException(status_code=500, detail="Failed to launch job on all nodes.")

    ACTIVE_JOBS[job_id]["status"] = "running"
    log.info(f"[{job_id}] Launch successful | master={master_addr}:{master_port}")
    return {"status": "job_launched", "job_id": job_id, "nodes": node_names}

@app.post("/report_job_completed")
async def report_job_completed(report: JobCompleteReport):
    job_id = report.job_id
    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    info = ACTIVE_JOBS[job_id]

    if "master_port" in info:
        try: port_manager.release_port(info["master_port"])
        except Exception: pass
    for n in info.get("nodes", []):
        NODE_REGISTRY[n]["status"] = "idle"
        NODE_REGISTRY[n]["current_job_id"] = None

    info["status"] = "completed"
    info["exit_code"] = report.exit_code

    log.info(f"[{job_id}] Job completed")

    try:
        fifo_url = SCHEDULER__ADDRESS + "/report_job_completed"
        payload = {
            "job_id": job_id,
            "exit_code": report.exit_code
        }
        r = requests.post(fifo_url, json=payload, timeout=15)
        if r.status_code == 200:
            log.info(f"[{job_id}] Reported completion to Scheduler")
        else:
            log.warning(f"[{job_id}] Scheduler ack failed: {r.status_code} {r.text}")
    except Exception as e:
        log.warning(f"[{job_id}] Failed to notify Scheduler: {e}")

    return {"status": "completion_acked"}

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


def _trigger_run_on_nodes(job_id: str, nodes_with_paths: Dict[str, str], epochs: int, resume_from: Optional[str] = None, batch_size: Optional[int] = None,):
    if not nodes_with_paths:
        log.error(f"[{job_id}] No nodes provided to run.")
        return False
    
    nodes_list = list(nodes_with_paths.keys()) 
    master_node_name = nodes_list[0]           
    master_addr = NODE_REGISTRY[master_node_name]["ip"]
    master_port = port_manager.get_port()
    world_size = len(nodes_list)

    log.info(f"[{job_id}] Launching job... WorldSize={world_size}, Master={master_addr}:{master_port}")

    ACTIVE_JOBS[job_id].update({
        "status": "running",
        "nodes": nodes_list, 
        "nodes_with_paths": nodes_with_paths, 
        "master_addr": master_addr,
        "master_port": master_port,
        "epochs": epochs,
        "batch_size_per_gpu": batch_size,
    })

    for rank, node_name in enumerate(nodes_list):
        node_info = NODE_REGISTRY[node_name]
        specific_script_path = nodes_with_paths[node_name] 
        
        payload = {
            "job_id": job_id,
            "script_path": specific_script_path, 
            "master_addr": master_addr,
            "master_port": master_port,
            "world_size": world_size,
            "rank": rank,
            "epochs": epochs,
            "resume_from_checkpoint": resume_from,
            "checkpoint_dir": f"./checkpoints/{job_id}",
            "global_server_addr": GLOBAL_SERVER_ADDRESS,
            "batch_size": batch_size,
        }
        
        target_url = f"http://{node_info['ip']}:{node_info['agent_port']}/run_task"
        
        try:
            log.info(f"[{job_id}] Sending run command to {node_name} (Rank {rank}) with path {specific_script_path}")
            requests.post(target_url, json=payload, timeout=5)
            NODE_REGISTRY[node_name]["status"] = "busy"
            NODE_REGISTRY[node_name]["current_job_id"] = job_id
        except requests.RequestException as e:
            log.error(f"[{job_id}] Failed to send command to {node_name}: {e}")
            return False
    return True


def _perform_resize_job(job_id: str, new_nodes_with_paths: Dict[str, str]):
    with job_locks[job_id]:
        log.info(f"[{job_id}] Starting resize process for nodes: {list(new_nodes_with_paths.keys())}")
        
        if job_id not in ACTIVE_JOBS:
            log.error(f"[{job_id}] Resize failed: Job not found.")
            return

        # --- 1. 기존 잡 중지 ---
        old_job_info = ACTIVE_JOBS[job_id].copy()
        old_nodes = old_job_info["nodes"] 
        master_node_name = old_nodes[0]
        master_node_info = NODE_REGISTRY[master_node_name]
        
        stop_url = f"http://{master_node_info['ip']}:{master_node_info['agent_port']}/stop_task"
        ACTIVE_JOBS[job_id]["status"] = "stopping"
        ACTIVE_JOBS[job_id]["stop_event"] = threading.Event() 
        
        try:
            log.info(f"[{job_id}] Sending stop signal to Rank 0 ({master_node_name})...")
            requests.post(stop_url, json={"job_id": job_id}, timeout=5)
        except requests.RequestException as e:
            log.error(f"[{job_id}] Failed to send stop signal: {e}")
            ACTIVE_JOBS[job_id]["status"] = "error"
            return

        # --- 2. 워커(Rank 0)가 /report_job_stopped를 호출할 때까지 대기 ---
        log.info(f"[{job_id}] Waiting for job to confirm stop (timeout 180s)...") # 타임아웃 180초로 늘림
        event_triggered = ACTIVE_JOBS[job_id]["stop_event"].wait(timeout=180.0) 
        
        if not event_triggered:
            log.error(f"[{job_id}] Resize failed: Job stop confirmation timed out.")
            ACTIVE_JOBS[job_id]["status"] = "error" 
            return
        
        log.info(f"[{job_id}] Stop confirmed.")

        # --- 3. 기존 자원 정리 ---
        port_manager.release_port(old_job_info["master_port"])
        for node_name in old_nodes:
            NODE_REGISTRY[node_name]["status"] = "idle"
            NODE_REGISTRY[node_name]["current_job_id"] = None
        
        resume_path = old_job_info.get("latest_checkpoint_path")
        if not resume_path:
            log.warning(f"[{job_id}] No checkpoint reported. Relaunching from scratch.")

        # --- 4. 새 구성으로 잡 재시작 ---
        log.info(f"[{job_id}] Relaunching job with new nodes: {list(new_nodes_with_paths.keys())}")
        ACTIVE_JOBS[job_id]["status"] = "resizing"
        
        _trigger_run_on_nodes(
            job_id=job_id,
            nodes_with_paths=new_nodes_with_paths, 
            epochs=old_job_info["epochs"], 
            resume_from=resume_path,
            batch_size=old_job_info.get("batch_size_per_gpu"),
        )
        
        log.info(f"[{job_id}] Resize process complete.")


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
    
    if job_id not in ACTIVE_JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job_locks[job_id].locked():
        raise HTTPException(status_code=409, detail=f"Job '{job_id}' is already processing a request.")

    new_nodes_with_paths = req.new_nodes_with_paths
    new_node_names = list(new_nodes_with_paths.keys())

    current_nodes = set(ACTIVE_JOBS[job_id]["nodes"])
    print("Current nodes:", current_nodes) # 디버깅용 print
    print("Requested new nodes:", new_node_names) # 디버깅용 print
    for node_name in new_node_names:
        if node_name not in NODE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"New node '{node_name}' not found.")
        # [수정] 아래 조건문 로직 오류 수정
        if NODE_REGISTRY[node_name]["status"] != "idle" and node_name not in current_nodes:
             # 상태가 idle이 아니면서 && 현재 잡에서 사용 중인 노드도 아니라면 -> 진짜 busy한 노드
            raise HTTPException(status_code=409, detail=f"New node '{node_name}' is busy with another job.")

    background_tasks.add_task(_perform_resize_job, job_id, new_nodes_with_paths)

    return {"status": "resize_initiated", "job_id": job_id, "new_nodes": new_node_names}


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
async def report_job_stopped(report: JobStopReport):
    job_id = report.job_id
    if job_id in ACTIVE_JOBS and ACTIVE_JOBS[job_id].get("stop_event"):
        log.info(f"[{job_id}] Received stop confirmation.")
        ACTIVE_JOBS[job_id]["stop_event"].set()
        return {"status": "stop_acked"}
    raise HTTPException(status_code=404, detail="Job or stop event not found")


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
