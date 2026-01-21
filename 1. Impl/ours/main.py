from fastapi import FastAPI, Request, Response, Header, HTTPException, Query
from typing import Optional, Dict, Any
import os, time
import logging  # [수정] 로깅 필터링을 위해 추가
from app.schemas import SubmitReq, TelemetryIn, JobCompleteReport, SubmitReqFixed
import app.ours_core as ours_core
from app.scheduler_state import get_global_state, STATE_LOCK, status_snapshot_compact
from app.logger import init_run_logger


# [수정] 특정 경로(/v1/telemetry)의 로그를 걸러내는 필터 클래스 정의
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # 로그 메시지에 "/v1/telemetry"가 포함되어 있으면 False를 반환하여 출력하지 않음
        return "/v1/telemetry" not in record.getMessage()


app = FastAPI(title="Ours Scheduler (Global Queue)", version="0.8.0")

@app.middleware("http")
async def timing_header(request: Request, call_next):
    t0 = time.time()
    resp: Response = await call_next(request)
    resp.headers["X-Ours-Elapsed-ms"] = str(int((time.time()-t0)*1000))
    return resp

@app.get("/status")
def status():
    return status_snapshot_compact()

@app.get("/job_status/{job_id}")
def job_status_api(job_id: str):
    state = get_global_state()
    with STATE_LOCK:
        jr = state.jobs.get(job_id)

        if jr is None:
            return {
                "job_id": job_id,
                "status": "UNKNOWN",
                "reason": "job_not_found_in_scheduler",
            }

        return {
            "job_id": job_id,
            "status": jr.status,
            "cluster_id": jr.cluster_id,
            "g_cur": getattr(jr, "g_cur", None),
            "g_target": getattr(jr, "g_target", None),
            "nodes": getattr(jr, "nodes", []),
            "submit_ts": jr.submit_ts,
            "start_ts": jr.start_ts,
            "end_ts": jr.end_ts,
        }

@app.post("/submit")
def submit(req: SubmitReq):
    out = ours_core.submit_job_core(req)
    try:
        ours_core.request_tick()
    except Exception:
        print("exception~!")
        pass
    return out

@app.post("/report_job_completed")
def report_job_completed(rep: JobCompleteReport):
    out = ours_core.report_job_completed_core(rep)
    try:
        ours_core.request_tick()
    except Exception:
        pass
    return out

@app.post("/v1/telemetry")
def push_telemetry(payload: TelemetryIn, request: Request):
    return ours_core.handle_telemetry(payload, request)

def _make_run_log_dir() -> str:
    base = os.getenv("SCHED_LOG_ROOT", "./logs")
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = os.path.join(base, ts)
    return run_dir

@app.on_event("startup")
def _on_startup():
    # [수정] Uvicorn 접속 로그(uvicorn.access)에 위에서 만든 필터 적용
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

    run_dir = _make_run_log_dir()
    init_run_logger(run_dir)

    ours_core.ensure_tick_thread_started()
    ours_core.request_tick()


@app.get("/tick_state")
def tick_state():
    return {
        "pid": os.getpid(),
        "thread_alive": bool(ours_core._TICK_THREAD and ours_core._TICK_THREAD.is_alive()),
        "pending": bool(getattr(ours_core, "_TICK_PENDING", False)),
        "tick_event_id": id(getattr(ours_core, "_TICK_EVENT", None)),
        "tick_lock_id": id(getattr(ours_core, "_TICK_LOCK", None)),
        "module_file": getattr(ours_core, "__file__", None),
    }