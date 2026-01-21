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
JOB_ATTEMPTS: dict[str, int] = {}

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

    # 선택 인자들 (global_server에서 넣어주는 값들)
    dataset: Optional[str] = None
    model_name: Optional[str] = None

    # Pollux / 튜너 쪽과 맞추기 위한 학습 설정
    batch_size_per_gpu: int = 64
    learning_rate: float = 1e-3
    grad_accum: int = 1

    attempt: int = 1


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

def _get_stop_flag_path(job_id: str, attempt: int) -> str:
    return f"/tmp/{job_id}_attempt-{attempt}.flag"

# --- API 엔드포인트 ---
@app.post("/launch_task")
async def launch_task(config: TaskConfig):
    job_id = config.job_id
    prefix = f"[{AGENT_ARGS.port}]"
    log.info(f"{prefix} [{job_id}] Launch Request received.")
    
    with RUNNING_PROCESSES_LOCK:
        # 이미 돌고 있는 job이면 거부
        if job_id in RUNNING_PROCESSES:
            raise HTTPException(status_code=400, detail="Job is already running")

        # job별 attempt 증가
        attempt = JOB_ATTEMPTS.get(job_id, 0) + 1
        JOB_ATTEMPTS[job_id] = attempt

        # attempt별 고유 stop flag 경로
        stop_flag_path = f"/tmp/{job_id}-attempt-{attempt}.flag"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

        # DDP 스크립트에서 env로 읽을 수 있게 내려줌
        env["JOB_ID"] = job_id
        env["STOP_FLAG_PATH"] = stop_flag_path
        env["ATTEMPT"] = str(attempt)

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
            "--global_server_addr", config.global_server_addr,

            # ↓↓↓ 학습/모델 관련 인자들
            "--batch_size_per_gpu", str(config.batch_size_per_gpu),
            "--learning_rate", str(config.learning_rate),
            "--grad_accum", str(config.grad_accum),
        ]

        cmd.extend(["--attempt", str(attempt)])

        # dataset / model_name은 있는 경우만 넘김
        if config.dataset:
            cmd.extend(["--dataset", config.dataset])
        if config.model_name:
            cmd.extend(["--model_name", config.model_name])

        # ============================
        #  재시작(resume) 경로 결정
        # ============================
        resume_path: Optional[str] = None

        # 1) Global server가 명시적으로 내려준 값이 있으면 그걸 우선 사용
        if config.resume_from_checkpoint:
            resume_path = config.resume_from_checkpoint
        else:
            # 2) 그게 없다면, attempt > 1 이고, checkpoint_dir에 latest_checkpoint가 있으면 자동으로 붙여줌
            if attempt > 1 and config.checkpoint_dir:
                latest_ckpt = os.path.join(config.checkpoint_dir, "latest_checkpoint.pth")
                if os.path.exists(latest_ckpt):
                    resume_path = latest_ckpt
                    log.info(
                        f"{prefix} [{job_id}] Auto-resume enabled: "
                        f"attempt={attempt}, resume_from={latest_ckpt}"
                    )

        # 최종적으로 결정된 resume_path가 있으면 인자로 추가
        if resume_path:
            cmd.extend(["--resume_from", resume_path])

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # 이제 프로세스뿐 아니라 stop_flag_path도 같이 저장
            RUNNING_PROCESSES[job_id] = {
                "process": process,
                "stop_flag_path": stop_flag_path,
            }
            
            # 모니터링 스레드 시작
            threading.Thread(
                target=_monitor_process, 
                args=(job_id, process, AGENT_ARGS.port, config.global_server_addr), 
                daemon=True
            ).start()

            log.info(
                f"{prefix} [{job_id}] Launched attempt {attempt} with "
                f"STOP_FLAG_PATH={stop_flag_path}, pid={process.pid}"
            )

            return {
                "status": "job_launched",
                "job_id": job_id,
                "pid": process.pid,
                "attempt": attempt,
            }
        
        except Exception as e:
            log.error(f"{prefix} [{job_id}] Failed to launch: {e}")
            raise HTTPException(status_code=500, detail=str(e))

def _monitor_process(job_id: str, process: subprocess.Popen, port: int, server_addr: str):
    prefix = f"[{port}]"
    
    stderr_tail_lines = []
    max_tail_lines = 20

    try:
        # stdout 스트리밍
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                line = line.rstrip("\n")
                log.info(f"{prefix} [{job_id} OUT] {line}")

        # stderr 스트리밍 + tail 저장
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                line = line.rstrip("\n")
                log.error(f"{prefix} [{job_id} ERR] {line}")
                stderr_tail_lines.append(line)
                if len(stderr_tail_lines) > max_tail_lines:
                    stderr_tail_lines.pop(0)

    except Exception as e:
        log.error(f"{prefix} [{job_id}] Log streaming error: {e}")
        
    process.wait()
    return_code = process.returncode
    log.info(f"{prefix} [{job_id}] Process finished. Return Code: {return_code}")
    
    status = "FINISHED" if return_code == 0 else "FAILED"
    
    # stderr tail은 너무 길지 않게 자름
    stderr_tail_text = "\n".join(stderr_tail_lines)
    if len(stderr_tail_text) > 2000:
        stderr_tail_text = stderr_tail_text[-2000:]
    
    payload = {
        "job_id": job_id,
        "status": status,
        "exit_code": return_code,
        "stderr_tail": stderr_tail_text,
    }

    # 더 이상 여기서 final_accuracy는 보내지 않음
    try:
        httpx.post(f"{server_addr}/report_job_status", json=payload, timeout=5)
    except Exception as e:
        log.error(f"{prefix} Failed to report status to server: {e}")

    with RUNNING_PROCESSES_LOCK:
        if job_id in RUNNING_PROCESSES:
            del RUNNING_PROCESSES[job_id]

@app.post("/stop_task")
async def stop_task(request: StopRequest):
    job_id = request.job_id
    prefix = f"[{AGENT_ARGS.port}]"
    
    # 먼저 프로세스 핸들을 잡는다
    with RUNNING_PROCESSES_LOCK:
        entry = RUNNING_PROCESSES.get(job_id)

    if entry is None:
        log.info(f"{prefix} [{job_id}] stop_task: no running process (already_stopped).")
        # 그래도 글로벌에게는 "stop 완료" 신호를 한 번 보내 주는 편이 안전함
        try:
            if GLOBAL_SERVER_URL:
                httpx.post(
                    f"{GLOBAL_SERVER_URL}/report_job_stopped",
                    json={"job_id": job_id},
                    timeout=3,
                )
        except Exception:
            pass
        return {"status": "already_stopped"}
    
    # 구조: {"process": Popen, "stop_flag_path": str}
    process = entry["process"]
    stop_flag_path = entry.get("stop_flag_path", f"/tmp/{job_id}_stop.flag")

    # Graceful Stop을 위한 파일 플래그 생성
    try:
        with open(stop_flag_path, "w") as f: 
            f.write("stop")
        log.info(f"{prefix} [{job_id}] Created stop flag at {stop_flag_path}")
    except Exception as e:
        log.error(f"{prefix} [{job_id}] Failed to create stop flag: {e}")
    
    # 우선 terminate로 부드럽게 종료 시도
    try:
        log.info(f"{prefix} [{job_id}] Sending SIGTERM to process {process.pid}")
        process.terminate()
        try:
            process.wait(timeout=30)
            log.info(f"{prefix} [{job_id}] Process terminated gracefully.")
        except subprocess.TimeoutExpired:
            log.warning(f"{prefix} [{job_id}] SIGTERM timeout; sending SIGKILL.")
            process.kill()
            process.wait()
            log.info(f"{prefix} [{job_id}] Process killed by SIGKILL.")
    except Exception as e:
        log.error(f"{prefix} [{job_id}] Error while stopping process: {e}")
    
    # 로컬 프로세스 테이블에서 정리
    with RUNNING_PROCESSES_LOCK:
        RUNNING_PROCESSES.pop(job_id, None)

    # 글로벌 서버에 stop 완료 보고 (stop_event 깨우기용)
    try:
        if GLOBAL_SERVER_URL:
            httpx.post(
                f"{GLOBAL_SERVER_URL}/report_job_stopped",
                json={"job_id": job_id},
                timeout=5,
            )
            log.info(f"{prefix} [{job_id}] Reported job_stopped to global server.")
    except Exception as e:
        log.error(f"{prefix} [{job_id}] Failed to report job_stopped: {e}")

    return {"status": "stopped"}

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