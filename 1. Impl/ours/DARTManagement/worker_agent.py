# worker_agent.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time
from typing import Optional, List, Any, Dict
import argparse, sys, httpx, threading, time, os, subprocess

SCHEDULER_URL = "http://163.180.117.216:8082"

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

AGENT_ARGS = None
GLOBAL_SERVER_URL = None

# --- 데이터 모델 ---
class TaskConfig(BaseModel):
    job_id: str
    run_id: Optional[str] = None
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
    model: Optional[str] = None
    batch_size_per_gpu: int = 64
    learning_rate: float = 1e-3
    grad_accum: int = 1
    attempt: int = 1

class StopRequest(BaseModel):
    job_id: str
    run_id: Optional[str] = None
    reason: Optional[str] = None

# --- 텔레메트리 스레드 ---
def telemetry_loop(gpu_id, server_url, node_id):
    # cluster_id는 사용자가 워커 복사해서 "손수" 박는다고 했으니 유지
    CLUSTER_ID = "clusterA"

    log.info(f"Telemetry thread started for GPU {gpu_id} -> {server_url}")

    handle = None
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        except Exception as e:
            # 초기 1회는 남겨야 디버깅 가능
            log.warning(f"NVML Init Failed: {e!r}")
            handle = None

    while True:
        try:
            if handle:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem_info.used / 1024**2  # MB
                mem_total = mem_info.total / 1024**2
            else:
                util, power, mem_used, mem_total = 0.0, 0.0, 0.0, 16384.0

            payload = {
                "cluster_id": CLUSTER_ID,
                "node_id": node_id,
                "gpu_index": gpu_id,
                "gpu_util": float(util),
                "power_w": float(power),
                "mem_used_mb": float(mem_used),
                "mem_total_mb": float(mem_total),
            }

            try:
                httpx.post(f"{server_url}/v1/telemetry", json=payload, timeout=1)
            except Exception:
                # 과도한 로그 방지
                pass

        except Exception as e:
            log.error(f"Telemetry Error: {e!r}")

        time.sleep(5)

def _get_stop_flag_path(job_id: str, attempt: int) -> str:
    return f"/tmp/{job_id}_attempt-{int(attempt)}.flag"

# --- API 엔드포인트 ---
@app.post("/launch_task")
async def launch_task(config: TaskConfig):
    job_id = str(config.job_id)
    run_id = str(getattr(config, "run_id", None) or "").strip()
    prefix = f"[{AGENT_ARGS.port}]"
    log.info(f"{prefix} [{job_id}] Launch Request received. run_id={run_id or 'NONE'}")

    # ✅ attempt SSOT
    try:
        attempt = int(getattr(config, "attempt", 1) or 1)
    except Exception:
        attempt = 1
    if attempt <= 0:
        attempt = 1

    with RUNNING_PROCESSES_LOCK:
        entry = RUNNING_PROCESSES.get(job_id)

        # -----------------------------
        # CASE 1) 이미 실행 중: 멱등 처리
        # -----------------------------
        if entry is not None:
            cur_run_id = (entry.get("run_id") or "").strip() or None
            cur_attempt = int(entry.get("attempt") or 1)
            proc = entry.get("process")
            pid = getattr(proc, "pid", None)

            # ✅ 요청 run_id가 비어 있으면 "이미 실행 중일 때만" 멱등 200
            #    (새 실행 요청인데 run_id 누락을 숨기면 SSOT가 깨집니다)
            if not run_id:
                log.info(
                    f"{prefix} [{job_id}] launch_task idempotent_ok (req run_id missing; already running). "
                    f"cur_run_id={cur_run_id or 'NONE'} attempt={cur_attempt} pid={pid}"
                )
                return {
                    "status": "already_running",
                    "job_id": job_id,
                    "pid": pid,
                    "attempt": cur_attempt,
                    "run_id": cur_run_id,
                    "note": "idempotent_replay_missing_run_id",
                }

            # ✅ 같은 run_id면 멱등 200
            if cur_run_id and run_id == cur_run_id:
                log.info(
                    f"{prefix} [{job_id}] launch_task idempotent_ok (same run_id). "
                    f"run_id={run_id} attempt={cur_attempt} pid={pid}"
                )
                return {
                    "status": "already_running",
                    "job_id": job_id,
                    "pid": pid,
                    "attempt": cur_attempt,
                    "run_id": cur_run_id,
                    "note": "idempotent_replay",
                }

            # ✅ run_id mismatch면 409
            log.warning(
                f"{prefix} [{job_id}] launch_task rejected due to run_id mismatch req={run_id} cur={cur_run_id or 'NONE'}"
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "job_already_running_different_run",
                    "job_id": job_id,
                    "req_run_id": run_id,
                    "cur_run_id": cur_run_id,
                },
            )

        # -----------------------------
        # CASE 2) 새 실행: run_id 필수 (버그 숨기지 않기)
        # -----------------------------
        if not run_id:
            log.warning(f"{prefix} [{job_id}] launch_task rejected: missing run_id for new launch.")
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "missing_run_id",
                    "job_id": job_id,
                },
            )

        stop_flag_path = _get_stop_flag_path(job_id, attempt)
        try:
            if os.path.exists(stop_flag_path):
                os.remove(stop_flag_path)
        except Exception:
            pass

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
        env["JOB_ID"] = job_id
        env["STOP_FLAG_PATH"] = stop_flag_path
        env["ATTEMPT"] = str(attempt)
        env["RUN_ID"] = run_id

        cmd = [
            sys.executable,
            config.script_path,
            "--nproc_per_node", "1",
            "--nnodes", str(config.world_size),
            "--node_rank", str(config.rank),
            "--master_addr", config.master_addr,
            "--master_port", str(config.master_port),
            "--epochs", str(config.epochs),
            "--job_id", job_id,
            "--checkpoint_dir", str(config.checkpoint_dir),
            "--global_server_addr", str(config.global_server_addr),
            "--batch_size", str(config.batch_size_per_gpu),
            "--init-lr", str(config.learning_rate),
            "--attempt", str(attempt),
        ]

        if config.dataset:
            cmd.extend(["--dataset", config.dataset])
        if config.model:
            cmd.extend(["--model", config.model])

        # ✅ resume 결정
        resume_path: Optional[str] = None
        if config.resume_from_checkpoint:
            resume_path = config.resume_from_checkpoint
        else:
            if attempt > 1 and config.checkpoint_dir:
                latest_ckpt = os.path.join(config.checkpoint_dir, "latest_checkpoint.pth")
                if os.path.exists(latest_ckpt):
                    resume_path = latest_ckpt

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

            # ✅ SSOT 저장(워커 내부)
            RUNNING_PROCESSES[job_id] = {
                "process": process,
                "stop_flag_path": stop_flag_path,
                "attempt": attempt,
                "run_id": run_id,
                "stop_requested": False,
                "stop_requested_ts": 0.0,
                "stop_ack_sent": False,
                "last_stop_reason": None,
            }

        except Exception as e:
            log.error(f"{prefix} [{job_id}] Failed to launch: {e!r}")
            RUNNING_PROCESSES.pop(job_id, None)
            raise HTTPException(status_code=500, detail=str(e))

    # ✅ monitor thread는 lock 밖에서 시작
    threading.Thread(
        target=_monitor_process,
        args=(job_id, process, AGENT_ARGS.port, config.global_server_addr, run_id, attempt),
        daemon=True
    ).start()

    return {
        "status": "job_launched",
        "job_id": job_id,
        "pid": process.pid,
        "attempt": attempt,
        "run_id": run_id,
        "resume_from": resume_path,
    }

@app.post("/stop_task")
async def stop_task(request: StopRequest):
    """
    ✅ 단 하나의 stop_task만 유지하세요(중복 정의 삭제).

    의미:
    - 여기서 보내는 ACK는 'stop을 걸었다'는 의미(terminal 확정 아님)
    - terminal(PREEMPTED/FINISHED/FAILED)은 monitor가 /report_job_status로 확정
    """
    job_id = str(getattr(request, "job_id", "") or "").strip()
    prefix = f"[{AGENT_ARGS.port}]"
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    req_run_id = (getattr(request, "run_id", None) or "").strip() or None
    reason = (getattr(request, "reason", None) or "PREEMPT").upper().strip() or "PREEMPT"

    # -----------------------------
    # 1) entry snapshot + validation (lock 짧게)
    # -----------------------------
    with RUNNING_PROCESSES_LOCK:
        entry = RUNNING_PROCESSES.get(job_id)

        if entry is None:
            log.info(f"{prefix} [{job_id}] stop_task: no running process (already_stopped). req_run_id={req_run_id}")
            # ✅ idempotent ACK: 없어도 200
            try:
                _report_job_stopped(
                    server_addr=(GLOBAL_SERVER_URL or ""),
                    job_id=job_id,
                    run_id=req_run_id,
                    reason=reason,
                    phase="STOP_ACK",
                )
            except Exception:
                pass
            return {"status": "already_stopped", "job_id": job_id, "run_id": req_run_id, "reason": reason}

        cur_run_id = (entry.get("run_id") or "").strip() or None

        # ✅ stale stop 방지: 요청 run_id가 있고 현재 run_id와 다르면 409로 거절
        if req_run_id and cur_run_id and req_run_id != cur_run_id:
            log.warning(
                f"{prefix} [{job_id}] stop_task rejected due to run_id mismatch req={req_run_id} cur={cur_run_id}"
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "run_id_mismatch",
                    "job_id": job_id,
                    "req_run_id": req_run_id,
                    "cur_run_id": cur_run_id,
                },
            )

        # ✅ 멱등: 이미 stop이 요청된 상태면 바로 ACK
        if entry.get("stop_requested", False):
            log.info(
                f"{prefix} [{job_id}] stop_task idempotent_ok (stop already requested). "
                f"cur_run_id={cur_run_id} reason={entry.get('last_stop_reason')}"
            )
            return {
                "status": "stop_already_requested",
                "job_id": job_id,
                "run_id": cur_run_id,
                "reason": entry.get("last_stop_reason") or reason,
            }

        # stop 요청 마킹 (SSOT: worker 내부)
        entry["stop_requested"] = True
        entry["stop_requested_ts"] = time.time()
        entry["last_stop_reason"] = reason
        entry.setdefault("stop_ack_sent", False)
        RUNNING_PROCESSES[job_id] = entry

        process: subprocess.Popen = entry["process"]
        stop_flag_path = entry.get("stop_flag_path") or f"/tmp/{job_id}.flag"

    # -----------------------------
    # 2) stop flag 작성 + terminate (락 밖)
    # -----------------------------
    try:
        with open(stop_flag_path, "w") as f:
            f.write("stop")
        log.info(f"{prefix} [{job_id}] Created stop flag at {stop_flag_path} reason={reason}")
    except Exception as e:
        log.error(f"{prefix} [{job_id}] Failed to create stop flag: {e!r}")

    try:
        if process and (process.poll() is None):
            log.info(f"{prefix} [{job_id}] Sending SIGTERM to pid={process.pid} reason={reason}")
            process.terminate()
        else:
            log.info(f"{prefix} [{job_id}] Process already exited before terminate. pid={getattr(process, 'pid', None)}")
    except Exception as e:
        log.error(f"{prefix} [{job_id}] Error while sending SIGTERM: {e!r}")

    # -----------------------------
    # 3) 글로벌에 stop ACK (1회만)
    # -----------------------------
    try:
        send_ack = False
        with RUNNING_PROCESSES_LOCK:
            ent2 = RUNNING_PROCESSES.get(job_id) or {}
            if not ent2.get("stop_ack_sent", False):
                ent2["stop_ack_sent"] = True
                RUNNING_PROCESSES[job_id] = ent2
                send_ack = True

        if send_ack:
            _report_job_stopped(
                server_addr=(GLOBAL_SERVER_URL or ""),
                job_id=job_id,
                run_id=cur_run_id,
                reason=reason,
                phase="STOP_ACK",
            )
            log.info(f"{prefix} [{job_id}] Reported stop_ack to global server. run_id={cur_run_id} reason={reason}")
    except Exception as e:
        log.warning(f"{prefix} [{job_id}] stop_ack report failed: {e!r}")

    # -----------------------------
    # 4) 비동기 대기 + 강제 kill + 정리
    # -----------------------------
    def _wait_and_force_kill():
        try:
            try:
                if process and (process.poll() is None):
                    process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning(f"{prefix} [{job_id}] SIGTERM grace timeout; sending SIGKILL.")
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=10)
                except Exception:
                    pass
        finally:
            # 여기서 RUNNING_PROCESSES를 지우지 마세요.
            # monitor가 최종 상태 보고 후 정리하는 게 SSOT에 더 안정적입니다.
            pass

    threading.Thread(target=_wait_and_force_kill, daemon=True).start()

    return {
        "status": "stop_issued",
        "job_id": job_id,
        "run_id": cur_run_id,
        "reason": reason,
        "note": "sigterm_sent_async_wait",
    }

def _report_job_stopped(*, server_addr: str, job_id: str, run_id: Optional[str], reason: Optional[str]):
    payload = {"job_id": job_id, "run_id": run_id, "reason": reason}
    try:
        httpx.post(f"{server_addr}/report_job_stopped", json=payload, timeout=5)
    except Exception:
        pass

def _monitor_process(
    job_id: str,
    process: subprocess.Popen,
    port: int,
    server_addr: str,
    run_id: str,
    attempt: int,
):
    prefix = f"[{port}]"
    max_tail_lines = 20

    def _read_stop_state():
        """stop_requested / last_stop_reason는 실행 중에도 바뀌므로 '끝난 직후'에 다시 읽어야 SSOT입니다."""
        stop_req = False
        reason = None
        try:
            with RUNNING_PROCESSES_LOCK:
                ent = RUNNING_PROCESSES.get(job_id) or {}
                stop_req = bool(ent.get("stop_requested", False))
                reason = (ent.get("last_stop_reason") or "").upper().strip() or None
        except Exception:
            stop_req = False
            reason = None
        return stop_req, reason

    # (참고용) 시작 시점 스냅샷
    stop_requested_0, last_stop_reason_0 = _read_stop_state()

    stdout_text = ""
    stderr_text = ""

    try:
        out, err = process.communicate()
        stdout_text = out or ""
        stderr_text = err or ""
    except Exception as e:
        log.error(f"{prefix} [{job_id}] communicate() error: {e!r}")

    return_code = int(getattr(process, "returncode", 1) or 1)

    # ✅ 종료 직후에 SSOT 다시 읽기 (이게 핵심)
    stop_requested_1, last_stop_reason_1 = _read_stop_state()

    # 둘 중 하나라도 True면 "요청에 의한 종료"로 본다(경합 상황 안전)
    stop_requested = bool(stop_requested_0 or stop_requested_1)

    # reason은 종료 직후 것을 우선
    terminated_reason = last_stop_reason_1 or last_stop_reason_0

    # stderr tail
    stderr_lines = []
    try:
        stderr_lines = list(stderr_text.splitlines())
    except Exception:
        stderr_lines = []

    tail = "\n".join(stderr_lines[-max_tail_lines:])
    if len(tail) > 2000:
        tail = tail[-2000:]

    # ✅ 확정 상태 결정 (stop_requested/terminated_reason 우선)
    status = "FAILED"

    if stop_requested:
        if terminated_reason in ("PREEMPT", "RESIZE", "LAUNCH_ROLLBACK"):
            status = "PREEMPTED"
        elif terminated_reason in ("CANCEL", "CANCELLED"):
            status = "CANCELLED"
        else:
            status = "STOPPED"
            if not terminated_reason:
                terminated_reason = "STOP_REQUESTED"
    else:
        status = "FINISHED" if return_code == 0 else "FAILED"

    payload = {
        "job_id": job_id,
        "status": status,
        "exit_code": return_code,
        "stderr_tail": tail,
        "run_id": (run_id or None),
        "attempt": int(attempt),
        "terminated_reason": (terminated_reason or None),
        # 있으면 도움이 되는 필드(서버가 dict Body면 그냥 무시/보존 가능)
        "end_ts": time.time(),
    }

    # ✅ 응답코드/바디까지 반드시 로깅 (422/500 원인 즉시 확인 가능)
    try:
        r = httpx.post(f"{server_addr}/report_job_status", json=payload, timeout=10)
        if 200 <= int(r.status_code) < 300:
            log.info(
                f"{prefix} [{job_id}] reported status={status} exit={return_code} run_id={run_id} "
                f"reason={terminated_reason} stop0={stop_requested_0} stop1={stop_requested_1}"
            )
        else:
            body = ""
            try:
                body = r.text[:2000]
            except Exception:
                body = "<no-body>"
            log.error(
                f"{prefix} [{job_id}] report_job_status failed status_code={r.status_code} "
                f"resp={body} payload={payload}"
            )
    except Exception as e:
        log.error(f"{prefix} [{job_id}] Failed to report status to server: {e!r} payload={payload}")

    # ✅ 마지막 정리: monitor가 SSOT를 끝까지 책임지고 제거
    try:
        with RUNNING_PROCESSES_LOCK:
            RUNNING_PROCESSES.pop(job_id, None)
    except Exception:
        pass

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
        args=(AGENT_ARGS.gpu_id, SCHEDULER_URL, f"node_{AGENT_ARGS.port}"),
        daemon=True
    )
    telem_thread.start()

    uvicorn.run(app, host=AGENT_ARGS.host, port=AGENT_ARGS.port)