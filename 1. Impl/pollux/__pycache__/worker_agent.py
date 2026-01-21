# worker_agent.py (unrecognized arguments 최종 수정)

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
import sys 

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (WorkerAgent:%(port)s) %(message)s", 
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

app = FastAPI()

# --- 1. 에이전트 상태 관리 ---
RUNNING_PROCESSES = {} 
RUNNING_PROCESSES_LOCK = threading.Lock()
agent_args = None 

# --- 2. API 데이터 모델 ---
class TaskConfig(BaseModel):
    job_id: str
    script_path: str
    master_addr: str
    master_port: int
    world_size: int
    rank: int
    local_rank: int 
    gpu_id: int     
    epochs: int
    resume_from_checkpoint: Optional[str] = None
    checkpoint_dir: str
    global_server_addr: str
    dataset: Optional[str] = None
    model_name: Optional[str] = None

class StopRequest(BaseModel):
    job_id: str

# --- 3. 헬퍼 함수 ---
def _get_stop_flag_path(job_id: str) -> str:
    return f"/tmp/adaptdl_stop_{job_id}"

# --- 4. API 엔드포인트 ---

@app.post("/launch_task")
async def launch_task(config: TaskConfig):
    job_id = config.job_id
    
    log_adapter = logging.LoggerAdapter(log, {'port': agent_args.port})
    
    log_adapter.info(f"[{job_id}] Received launch request for script: {config.script_path}")
    
    with RUNNING_PROCESSES_LOCK:
        if job_id in RUNNING_PROCESSES:
            log_adapter.warning(f"[{job_id}] Job is already running. Ignoring launch request.")
            raise HTTPException(status_code=400, detail="Job is already running")

        env = os.environ.copy()
        
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id) 
        
        log_adapter.info(f"[{job_id}] Setting CUDA_VISIBLE_DEVICES={config.gpu_id}")

        cmd = [
            sys.executable, 
            config.script_path,
            "--nproc_per_node", "1", 
            "--nnodes", str(config.world_size),
            "--node_rank", str(config.rank),
            "--master_addr", config.master_addr,
            "--master_port", str(config.master_port),
            "--epochs", str(config.epochs),
            "--job_id", config.job_id,
            "--checkpoint_dir", config.checkpoint_dir,
            "--global_server_addr", config.global_server_addr
        ]
        
        # [수정] model_name 과 dataset 인자 전달 제거
        # if config.model_name:
        #     cmd.extend(["--model_name", config.model_name])
        # if config.dataset:
        #      cmd.extend(["--dataset", config.dataset])

        if config.resume_from_checkpoint:
            cmd.extend(["--resume", config.resume_from_checkpoint])
            log_adapter.info(f"[{job_id}] Resuming from checkpoint: {config.resume_from_checkpoint}")

        try:
            log_adapter.info(f"[{job_id}] Executing command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            RUNNING_PROCESSES[job_id] = process
            log_adapter.info(f"[{job_id}] Process started (PID: {process.pid})")
            
            threading.Thread(target=_monitor_process, args=(job_id, process, agent_args.port), daemon=True).start()

            return {"status": "job_launched", "job_id": job_id, "pid": process.pid}
        
        except Exception as e:
            log_adapter.error(f"[{job_id}] Failed to launch subprocess: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to launch job: {e}")


def _monitor_process(job_id: str, process: subprocess.Popen, port: int):
    log_adapter = logging.LoggerAdapter(log, {'port': port})
    
    try:
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                log_adapter.info(f"[{job_id} PID:{process.pid}] {line.strip()}")
        if process.stderr:
             for line in iter(process.stderr.readline, ''):
                log_adapter.error(f"[{job_id} PID:{process.pid}] {line.strip()}")
    except Exception as e:
        log_adapter.warning(f"[{job_id}] Log monitoring interrupted: {e}")
        
    process.wait()
    log_adapter.info(f"[{job_id} PID:{process.pid}] Process finished with code {process.returncode}")
    
    with RUNNING_PROCESSES_LOCK:
        if job_id in RUNNING_PROCESSES:
            del RUNNING_PROCESSES[job_id]
            
@app.post("/stop_task")
async def stop_task(request: StopRequest):
    job_id = request.job_id
    log_adapter = logging.LoggerAdapter(log, {'port': agent_args.port})
    
    with RUNNING_PROCESSES_LOCK:
        if job_id not in RUNNING_PROCESSES:
            log_adapter.warning(f"[{job_id}] Received stop request, but job not found or already stopped.")
            return {"status": "job_not_found_or_already_stopped"}
        
        stop_flag_path = _get_stop_flag_path(job_id)
        try:
            with open(stop_flag_path, "w") as f: 
                f.write("stop")
            log_adapter.info(f"[{job_id}] Created stop flag file at {stop_flag_path}.")
            return {"status": "stop_signal_sent"}
        except Exception as e:
            log_adapter.error(f"[{job_id}] Failed to create stop flag: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create stop flag: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Agent")
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host to bind the agent to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8001, 
                        help='Port to bind the agent to (default: 8001)')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='GPU ID for this agent to manage (default: 0)')
    
    agent_args = parser.parse_args()
    
    log_adapter = logging.LoggerAdapter(log, {'port': agent_args.port})
    log_adapter.info(f"Starting Worker Agent on {agent_args.host}:{agent_args.port}, managing GPU_ID: {agent_args.gpu_id}")

    uvicorn.run(app, host=agent_args.host, port=agent_args.port)