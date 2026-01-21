# worker_agent.py

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
import httpx
import socket

# NVIDIA GPU 라이브러리 (없으면 에러 대신 더미 데이터 전송하도록 처리됨)
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# --- [수정] 로깅 설정 변경 ---
# 1. 포맷에서 %(port)s 제거 (httpx 등 외부 라이브러리와의 충돌 방지)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (WorkerAgent) %(message)s", 
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# 2. httpx 라이브러리의 불필요한 INFO 로그(매 요청마다 발생) 차단
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI()

# --- 전역 변수 ---
RUNNING_PROCESSES = {} 
RUNNING_PROCESSES_LOCK = threading.Lock()
AGENT_ARGS = None 
GLOBAL_SERVER_URL = None

# --- 데이터 모델 ---
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

# --- 텔레메트리 스레드 ---
def telemetry_loop(gpu_id, server_url, node_id):
    # [수정] 일반 log 사용 (Adapter 제거)
    log.info(f"Telemetry thread started for GPU {gpu_id} -> {server_url}")
    
    handle = None
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except Exception as e:
            log.error(f"NVML Init Failed: {e}")

    while True:
        try:
            if handle:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # mW -> W
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / 1024**2 # MB
                mem_total = mem_info.total / 1024**2
            else:
                # Dummy Data if NVML fails
                util, power, mem_used, mem_total = 0.0, 0.0, 0.0, 16384.0

            payload = {
                "node_id": node_id,
                "gpu_index": gpu_id,
                "gpu_util": float(util),
                "power_w": float(power),
                "mem_used_mb": float(mem_used),
                "mem_total_mb": float(mem_total)
            }

            # Fire and forget request
            try:
                httpx.post(f"{server_url}/report_telemetry", json=payload, timeout=1)
            except Exception as e:
                # 서버 죽었을 때 로그 너무 많이 남기지 않도록 warning 정도만
                pass 

        except Exception as e:
            log.error(f"Telemetry Error: {e}")
        
        time.sleep(5) # 5초 주기

# --- API 엔드포인트 ---

@app.post("/launch_task")
async def launch_task(config: TaskConfig):
    job_id = config.job_id
    # [수정] 포트 정보를 메시지에 직접 포함
    prefix = f"[{AGENT_ARGS.port}]"
    log.info(f"{prefix} [{job_id}] Launch Request received.")
    
    with RUNNING_PROCESSES_LOCK:
        if job_id in RUNNING_PROCESSES:
            raise HTTPException(status_code=400, detail="Job is already running")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        
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
        
        if config.resume_from_checkpoint:
            cmd.extend(["--resume", config.resume_from_checkpoint])
            # [수정] Model Name, Dataset 인자 관련 코드는 기존에 주석처리 되어 있었으므로 유지하거나 필요시 삭제

        try:
            process = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            RUNNING_PROCESSES[job_id] = process
            
            # 모니터링 스레드 시작
            threading.Thread(
                target=_monitor_process, 
                args=(job_id, process, AGENT_ARGS.port, config.global_server_addr), 
                daemon=True
            ).start()

            return {"status": "job_launched", "job_id": job_id, "pid": process.pid}
        
        except Exception as e:
            log.error(f"{prefix} [{job_id}] Failed to launch: {e}")
            raise HTTPException(status_code=500, detail=str(e))

def _monitor_process(job_id: str, process: subprocess.Popen, port: int, server_addr: str):
    # [수정] 일반 log 사용
    prefix = f"[{port}]"
    
    try:
        # 로그 스트리밍
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                log.info(f"{prefix} [{job_id} OUT] {line.strip()}")
        if process.stderr:
             for line in iter(process.stderr.readline, ''):
                log.error(f"{prefix} [{job_id} ERR] {line.strip()}")
    except Exception:
        pass
        
    process.wait()
    return_code = process.returncode
    log.info(f"{prefix} [{job_id}] Process finished. Return Code: {return_code}")
    
    status = "FINISHED" if return_code == 0 else "FAILED"
    
    # 서버에 종료 상태 보고
    try:
        httpx.post(f"{server_addr}/report_job_status", json={"job_id": job_id, "status": status}, timeout=5)
    except Exception as e:
        log.error(f"{prefix} Failed to report status to server: {e}")

    with RUNNING_PROCESSES_LOCK:
        if job_id in RUNNING_PROCESSES:
            del RUNNING_PROCESSES[job_id]

@app.post("/stop_task")
async def stop_task(request: StopRequest):
    job_id = request.job_id
    prefix = f"[{AGENT_ARGS.port}]"
    
    with RUNNING_PROCESSES_LOCK:
        if job_id not in RUNNING_PROCESSES:
            return {"status": "already_stopped"}
        
        # Graceful Stop을 위한 파일 플래그 생성
        stop_flag_path = f"/tmp/adaptdl_stop_{job_id}"
        try:
            with open(stop_flag_path, "w") as f: 
                f.write("stop")
            log.info(f"{prefix} [{job_id}] Created stop flag at {stop_flag_path}")
            return {"status": "stop_signal_sent"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Agent")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--server_ip', type=str, default='163.180.117.216', help="Global Server IP (e.g., 192.168.0.5)")
    
    AGENT_ARGS = parser.parse_args()
    GLOBAL_SERVER_URL = f"http://{AGENT_ARGS.server_ip}:8000"
    
    log.info(f"Worker Agent Starting. Managing GPU {AGENT_ARGS.gpu_id} on Port {AGENT_ARGS.port}")

    # 백그라운드 텔레메트리 시작
    telem_thread = threading.Thread(
        target=telemetry_loop,
        args=(AGENT_ARGS.gpu_id, GLOBAL_SERVER_URL, f"node_{AGENT_ARGS.port}"),
        daemon=True
    )
    telem_thread.start()

    uvicorn.run(app, host=AGENT_ARGS.host, port=AGENT_ARGS.port)