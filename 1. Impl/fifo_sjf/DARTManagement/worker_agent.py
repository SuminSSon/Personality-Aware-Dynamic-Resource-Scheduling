import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import logging
import time
import threading
from typing import Optional
import argparse
import requests

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (WorkerAgent) %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

app = FastAPI()

# --- 1. 에이전트 상태 관리 ---
RUNNING_PROCESSES = {}
RUNNING_META = {}
RUNNING_PROCESSES_LOCK = threading.Lock()

# --- 2. API 데이터 모델 ---
class TaskConfig(BaseModel):
    job_id: str
    script_path: str
    master_addr: str
    master_port: int
    world_size: int
    rank: int
    epochs: int
    resume_from_checkpoint: Optional[str] = None
    checkpoint_dir: str
    global_server_addr: str
    # 필요 시 서버에서 명시적으로 내려보낼 수도 있음(없으면 파일명에서 추정)
    dataset: Optional[str] = None
    model_name: Optional[str] = None
    batch_size: Optional[int] = None

class StopRequest(BaseModel):
    job_id: str

# --- 3. 헬퍼 함수 ---
def _get_stop_flag_path(job_id: str) -> str:
    return f"/tmp/{job_id}_stop.flag"

def _safe_filename(s: str) -> str:
    """파일명에 안전한 문자만 남김"""
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", " ")) else "-" for ch in str(s))

def _infer_dataset_model(script_path: str) -> tuple[str, str]:
    """
    스크립트 파일명에서 {DATASET}_{MODEL}.py 형태를 추정.
    예: CIFAR100_Densenet121.py -> ("CIFAR100", "Densenet121")
    실패하면 ("unknown", "unknown") 반환
    """
    base = os.path.basename(script_path)
    name, _ = os.path.splitext(base)
    parts = [p for p in name.replace("-", "_").split("_") if p]
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return parts[0], "unknown"
    return "unknown", "unknown"

def _process_monitor_thread():
    while True:
        try:
            terminated = []
            with RUNNING_PROCESSES_LOCK:
                for job_id, proc in list(RUNNING_PROCESSES.items()):
                    status = proc.poll()
                    if status is not None:
                        log.info(f"Job {job_id} terminated (Exit Code: {status}). Cleaning up.")
                        terminated.append((job_id, status))

            for job_id, status in terminated:
                meta = RUNNING_META.pop(job_id, {}) if 'RUNNING_META' in globals() else {}
                rank = meta.get("rank")
                server = (meta.get("global_server_addr") or "").rstrip("/")

                with RUNNING_PROCESSES_LOCK:
                    RUNNING_PROCESSES.pop(job_id, None)
                flag_path = _get_stop_flag_path(job_id)
                if os.path.exists(flag_path):
                    try:
                        os.remove(flag_path)
                    except Exception:
                        pass

                if server and rank == 0:
                    try:
                        url = f"{server}/report_job_completed"
                        payload = {"job_id": job_id, "exit_code": int(status) if status is not None else 0}
                        requests.post(url, json=payload, timeout=3.0)
                        log.info(f"[{job_id}] Completion reported to {url}")
                    except Exception as e:
                        log.warning(f"[{job_id}] Failed to report completion: {e}")

        except Exception as e:
            log.error(f"Error in process monitor thread: {e}")

        time.sleep(5)

# --- 4. API 엔드포인트 ---
@app.get("/health")
async def health_check():
    return {"status": "alive", "running_jobs": list(RUNNING_PROCESSES.keys())}

# [수정] 함수 시그니처에서 'args: argparse.Namespace = None' 제거
@app.post("/run_task")
async def run_task(config: TaskConfig):
    """ (Global Server용) DDP 학습 스크립트를 실행하라는 명령 """
    
    # [수정] 전역 변수 'agent_args'를 직접 사용
    gpu_id_to_use = agent_args.gpu_id if 'agent_args' in globals() else 0

    log.info(f"[{config.job_id}] Received task request: Rank {config.rank}/{config.world_size} (Target GPU: {gpu_id_to_use})")

    with RUNNING_PROCESSES_LOCK:
        if config.job_id in RUNNING_PROCESSES:
            log.warning(f"[{config.job_id}] Job already running. Attempting to kill old one...")
            RUNNING_PROCESSES[config.job_id].kill()
            del RUNNING_PROCESSES[config.job_id]

        stop_flag_path = _get_stop_flag_path(config.job_id)
        if os.path.exists(stop_flag_path):
            log.info(f"[{config.job_id}] Removing stale stop flag: {stop_flag_path}")
            os.remove(stop_flag_path)

        cmd = [
            "python", config.script_path,
            "--nproc_per_node", "1",
            "--nnodes", str(config.world_size),
            "--node_rank", str(config.rank),
            "--master_addr", config.master_addr,
            "--master_port", str(config.master_port),
            "--epochs", str(config.epochs),
            "--checkpoint_dir", config.checkpoint_dir,
            "--job_id", config.job_id,
            "--global_server_addr", config.global_server_addr,
        ]
        if config.resume_from_checkpoint:
            cmd.extend(["--resume_from", config.resume_from_checkpoint])
        if config.batch_size is not None:
            cmd += ["--batch_size", str(int(config.batch_size))]
            
        log.info(f"[{config.job_id}] Executing command: {' '.join(cmd)}")
        
        # 환경 변수 설정
        env = os.environ.copy()
        env["LOCAL_RANK"] = "0" 
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id_to_use) 

        if config.batch_size is not None:
            env["BATCH_SIZE_PER_GPU"] = str(int(config.batch_size))
        
        
        log.info(
            f"[{config.job_id}] Setting CUDA_VISIBLE_DEVICES={gpu_id_to_use}"
            + (f", BATCH_SIZE_PER_GPU={env['BATCH_SIZE_PER_GPU']}" if "BATCH_SIZE_PER_GPU" in env else "")
        )

        RUNNING_META[config.job_id] = {
            "rank": config.rank,
            "global_server_addr": config.global_server_addr,
            "batch_size": config.batch_size,
            "dataset": (config.dataset or _infer_dataset_model(config.script_path)[0]),
            "model_name": (config.model_name or _infer_dataset_model(config.script_path)[1]),
        }

        # ----- 로그 파일명 커스텀: "rank X_job ID_총 학습노드 갯수_테스트셋_모델명_타임스탬프.log"
        log_dir = "./job_logs"
        os.makedirs(log_dir, exist_ok=True)

        # 1) 테스트셋/모델명 결정(명시 필드 우선, 없으면 파일명에서 추정)
        dataset = (config.dataset or "").strip() if hasattr(config, "dataset") else ""
        model_name = (config.model_name or "").strip() if hasattr(config, "model_name") else ""
        if not dataset or not model_name:
            ds_guess, model_guess = _infer_dataset_model(config.script_path)
            if not dataset:
                dataset = ds_guess
            if not model_name:
                model_name = model_guess

        # 2) 안전한 파일명으로 정리 + 타임스탬프
        dataset_safe = _safe_filename(dataset)
        model_safe = _safe_filename(model_name)
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())

        # 3) 최종 파일명 조립
        filename = f"rank {config.rank}_{config.job_id}_node {config.world_size}_{dataset_safe}_{model_safe}_{ts}.log"
        log_path = os.path.join(log_dir, filename)
        # ----- 커스텀 파일명 끝

        try:
            log_file = open(log_path, "w")
            process = subprocess.Popen(
                cmd, 
                env=env, 
                stdout=log_file, 
                stderr=subprocess.STDOUT
            )
            RUNNING_PROCESSES[config.job_id] = process
            log.info(f"[{config.job_id}] Task started successfully (PID: {process.pid}).")
            return {
                "status": "task_started", 
                "job_id": config.job_id,
                "pid": process.pid,
                "log_path": log_path
            }
        except Exception as e:
            log.error(f"[{config.job_id}] Failed to start process: {e}")
            raise HTTPException(status_code=500, detail=f"Process execution failed: {e}")

@app.post("/stop_task")
async def stop_task(req: StopRequest):
    job_id = req.job_id
    log.info(f"[{job_id}] Received stop request.")
    with RUNNING_PROCESSES_LOCK:
        if job_id not in RUNNING_PROCESSES:
            log.warning(f"[{job_id}] Stop request received, but job not found.")
            return {"status": "job_not_found_or_already_stopped"}
        stop_flag_path = _get_stop_flag_path(job_id)
        try:
            with open(stop_flag_path, "w") as f: 
                f.write("stop")
            log.info(f"[{job_id}] Created stop flag file at {stop_flag_path}.")
            return {"status": "stop_signal_sent"}
        except Exception as e:
            log.error(f"[{job_id}] Failed to create stop flag: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create stop flag: {e}")

# argparse에 --gpu_id 추가
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Agent")
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host to bind the agent to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8001, 
                        help='Port to bind the agent to (default: 8001)')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='GPU ID for the DDP process to use (default: 0)')
    
    # args를 전역 변수로 저장하여 API 핸들러에서 접근 가능하게 함
    global agent_args 
    agent_args = parser.parse_args()

    monitor_thread = threading.Thread(target=_process_monitor_thread, daemon=True)
    monitor_thread.start()
    
    log.info(f"Starting Worker Agent on {agent_args.host}:{agent_args.port}, assigned GPU: {agent_args.gpu_id}")
    uvicorn.run(app, host=agent_args.host, port=agent_args.port)
