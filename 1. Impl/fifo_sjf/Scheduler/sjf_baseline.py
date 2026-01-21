from __future__ import annotations
import os, time, uuid, threading, csv, json, logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Request, Response, Header
from pydantic import BaseModel, Field

from app.executor import launch_or_reuse, job_status
from app.metrics import get_cluster_csp, lambda_from_csp, ingest_telemetry as metrics_ingest_telemetry

import psycopg2
import statistics

# 설정 / 경로
DEFAULT_G           = 1 
DEFAULT_EPOCHS      = 20 
SCHED_TICK_SEC      = 1.0 
CSP_SAMPLER_SEC     = 5.0 

EXP_FINISH_TARGET = int(os.getenv("EXP_FINISH_TARGET", "50"))
EXP_SELF_SHUTDOWN_GRACE_SEC = int(os.getenv("EXP_SELF_SHUTDOWN_GRACE_SEC", "60"))

FINISHED_JOB_IDS: set[str] = set()
_SHUTDOWN_SCHEDULED = False

PROF_DB_DSN = os.getenv("PROF_DB_DSN", "postgresql://prof:profpw@163.180.117.216:5432/profdb")

CLUSTER_NODES = {
    "clusterA": ["node_a", "node_b", "node_c", "node_d"],
    "clusterB": ["node_e", "node_f", "node_g", "node_h"],
}


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_STAMP    = time.strftime("%Y%m%d-%H%M%S", time.localtime())
SJF_DIR  = os.path.join(_BASE_DIR, "SJF_log", f"SJF-{_STAMP}")
os.makedirs(SJF_DIR, exist_ok=True)

STOP_EVENT = threading.Event() 

# 로깅
logger = logging.getLogger("SJF")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(os.path.join(SJF_DIR, "scheduler.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
logger.propagate = False

def now_ts() -> float: return time.time()
def iso_ms(ts: float) -> str: return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

# CSV 초기화 & 쓰기
def _csv_init(path: str, header: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

EVENTS_PATH = os.path.join(SJF_DIR, "job_events.csv")
_csv_init(EVENTS_PATH, ["ts","event","job_id","cluster","world_size","note","metadata_json"])

def write_event(event: str, job_id: str, cluster: str, g: int, note: str = "", meta: Dict[str,Any] = None):
    with open(EVENTS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iso_ms(now_ts()), event, job_id, cluster, g, note,
            json.dumps(meta or {}, ensure_ascii=False)
        ])

QUEUE_EVENTS_PATH = os.path.join(SJF_DIR, "queue_events.csv")
_csv_init(QUEUE_EVENTS_PATH, ["ts","event","job_id","queue_len_after","note"])
def _write_queue_event(event: str, job_id: str, q_len_after: int, note: str = ""):
    with open(QUEUE_EVENTS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([iso_ms(now_ts()), event, job_id, q_len_after, note])

CSP_CSV_PATH = os.path.join(SJF_DIR, "csp_metrics.csv")
_csv_init(CSP_CSV_PATH, ["ts","cluster","queue_len","slots_total","slots_used","free",
                         "U_t","E_t","p_fair","lam_time","lam_cost","lam_fair"])

TELEM_CSV_PATH = os.path.join(SJF_DIR, "telemetry.csv")
_csv_init(TELEM_CSV_PATH, ["ts","node_id","gpu_index","gpu_util","power_w","mem_used_mb","mem_total_mb"])

JOB_METRICS_PATH = os.path.join(SJF_DIR, "job_metrics.csv")
_csv_init(JOB_METRICS_PATH, ["job_id","cluster","model","dataset","world_size",
                             "submitted_ts","started_ts","end_ts","queued_sec","jct_sec","status"])

def _append_job_metric_started(job_id: str, cluster: str, queued_sec: int):
    """잡 시작 시점 기록(큐잉 시간 포함)."""
    meta = RUNNING[cluster][job_id]
    with open(JOB_METRICS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            job_id, cluster,
            meta.get("model"), meta.get("dataset"), meta.get("world_size"),
            meta.get("submitted_ts"), meta.get("started_ts"),
            "", queued_sec, "", "running"
        ])

def _append_job_metric_finished(job_id: str, cluster: str, end_ts: float, jct_sec: Optional[int], status: str):
    """잡 종료 시점 기록(JCT 포함)."""
    meta = RUNNING.get(cluster, {}).get(job_id, {})
    with open(JOB_METRICS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            job_id, cluster,
            meta.get("model"), meta.get("dataset"), meta.get("world_size"),
            meta.get("submitted_ts"), meta.get("started_ts"),
            end_ts, meta.get("queued_sec"), jct_sec, status
        ])

# 데이터 모델
class SubmitReq(BaseModel):
    job_id: Optional[str] = None
    model: str
    dataset: str
    user_request: Optional[str] = None 
    world_size: Optional[int] = None     
    epochs: Optional[int] = None
    batch_size: Optional[int] = None 

@dataclass
class SJFJob:
    job_id: str
    model: str
    dataset: str
    world_size: int = DEFAULT_G
    epochs: int = DEFAULT_EPOCHS
    batch_size_per_gpu: Optional[int] = None
    submitted_ts: float = field(default_factory=now_ts)
    start_ts: Optional[float] = None
    est_runtime_sec: float = 0.0

class TelemetryIn(BaseModel):
    node_id: str = Field(..., description="e.g., node-e")
    gpu_index: int = 0
    ts: Optional[float] = None
    gpu_util: Optional[float] = None
    power_w: Optional[float] = None
    mem_used_mb: Optional[float] = None
    mem_total_mb: Optional[float] = None

class JobCompleteReport(BaseModel):
    job_id: str
    cluster: Optional[str] = None
    exit_code: int = 0

# 서버/상태
app = FastAPI(title="SJF Baseline Scheduler (Global Queue)", version="0.7.0")
SJF_LOCK = threading.RLock()
QUEUE : List[SJFJob] = []
RUNNING: Dict[str, Dict[str, Any]] = {"clusterA": {}, "clusterB": {}}

# 슬롯 스냅샷(로컬 상태 기반)
def _nodes_in_use(cluster: str) -> set:
    used = set()
    for meta in RUNNING.get(cluster, {}).values():
        for n in meta.get("nodes", []):
            used.add(n)
    return used

def _cluster_slots_snapshot() -> Dict[str, Dict[str,int]]:
    snap: Dict[str, Dict[str,int]] = {}
    for c, nodes in CLUSTER_NODES.items():
        total = len(nodes)
        used_nodes = _nodes_in_use(c)
        used = len(used_nodes)
        free = max(0, total - used)
        snap[c] = {"slots_total": total, "slots_used": used, "free": free}
    return snap

def _available_nodes_by_cluster() -> Dict[str, List[str]]:
    """클러스터별 '지금 비어있는' 노드 목록."""
    out: Dict[str, List[str]] = {}
    for c, nodes in CLUSTER_NODES.items():
        used = _nodes_in_use(c)
        out[c] = [n for n in nodes if n not in used]
    return out

def _make_job_id() -> str:
    return "job-" + uuid.uuid4().hex[:12]

def _queue_position(job_id: str) -> int:
    for i, j in enumerate(QUEUE, start=1):
        if j.job_id == job_id:
            return i
    return 0

def _try_start_sjf_any() -> bool:
    """
    SJF (Shortest Job First) + resource constraint + clusterA-first:
    - 대기 큐에서 est_runtime_sec가 가장 작은 job부터 보면서
      clusterA에 world_size만큼 free node가 있으면 clusterA에 배치.
      없으면 clusterB에 배치.
    - 선점 없음. 이미 실행 중인 job은 건드리지 않는다.
    """
    if not QUEUE:
        return False

    avail = _available_nodes_by_cluster()

    # 🔹 SJF: 추정 실행시간 기준 오름차순 정렬된 view
    sjf_order = sorted(QUEUE, key=lambda j: (j.est_runtime_sec or 1e30, j.submitted_ts))

    # ✅ 정책: 항상 clusterA 먼저, 안 되면 clusterB
    cluster_order = ["clusterA", "clusterB"]

    for job in sjf_order:
        placed = False

        for cluster in cluster_order:
            nodes = avail.get(cluster, []) or []
            free_cnt = len(nodes)

            # clusterA free=1인데 g=2면 탈락 → clusterB로 넘어감
            if free_cnt < job.world_size:
                continue

            chosen_nodes = nodes[: job.world_size]

            created, launch_info = launch_or_reuse(
                job_id=job.job_id,
                cluster=cluster,
                world_size=job.world_size,
                dataset=job.dataset,
                model=job.model,
                epochs=job.epochs or DEFAULT_EPOCHS,
                preferred_nodes=chosen_nodes,
                batch_size=job.batch_size_per_gpu,
            )
            status = (launch_info or {}).get("status")

            if status in ("started", "reused", "ok", "running"):
                # 🔹 QUEUE에서 해당 job 제거 (FIFO와 달리 중간에 있을 수 있음)
                idx = next(i for i, j in enumerate(QUEUE) if j.job_id == job.job_id)
                QUEUE.pop(idx)

                job.start_ts = now_ts()
                queued_sec = job.start_ts - job.submitted_ts

                RUNNING[cluster][job.job_id] = {
                    "cluster": cluster,
                    "model": job.model,
                    "dataset": job.dataset,
                    "world_size": job.world_size,
                    "batch_size_per_gpu": job.batch_size_per_gpu,
                    "nodes": (launch_info or {}).get("nodes", chosen_nodes),
                    "submitted_ts": job.submitted_ts,
                    "started_ts": job.start_ts,
                    "queued_sec": queued_sec,
                    "est_runtime_sec": job.est_runtime_sec,
                }

                _write_queue_event(
                    "dequeue_start",
                    job.job_id,
                    q_len_after=len(QUEUE),
                    note=f"SJF g={job.world_size}, cluster={cluster}, T_est={job.est_runtime_sec:.2f}s",
                )

                note = (
                    f"SJF started_on={cluster}, g={job.world_size}, "
                    f"nodes={RUNNING[cluster][job.job_id]['nodes']}, "
                    f"queued_sec={RUNNING[cluster][job.job_id]['queued_sec']:.1f}, "
                    f"T_est={job.est_runtime_sec:.2f}s"
                )
                logger.info(f"[START-SJF] {job.job_id} | {note}")

                write_event(
                    "started",
                    job.job_id,
                    cluster,
                    job.world_size,
                    note,
                    {
                        "model": job.model,
                        "dataset": job.dataset,
                        "nodes": RUNNING[cluster][job.job_id]["nodes"],
                        "queued_sec": RUNNING[cluster][job.job_id]["queued_sec"],
                        "started_ts": job.start_ts,
                        "est_runtime_sec": job.est_runtime_sec,
                    },
                )
                _append_job_metric_started(job.job_id, cluster, RUNNING[cluster][job.job_id]["queued_sec"])
                return True

            # launch 실패면 다음 클러스터로 넘어가서 시도
            logger.warning(
                f"[SJF] launch failed: job_id={job.job_id}, status={status}, info={launch_info}, cluster={cluster}"
            )

        # 이 job은 A/B 둘 다 현재 불가 → 더 짧은 다른 job을 계속 시도
        if not placed:
            continue

    # 어떤 job도 시작하지 못함
    return False

def _estimate_job_runtime_sec(model: str, dataset: str, world_size: int, epochs: int) -> float:
    """
    T_est(j) = epochs * avg(epoch_time_measured_sec(model, dataset, g=world_size))
    profiling 데이터가 없으면 큰 값으로 fallback (SJF에서 가장 뒤로 밀리게).
    """
    if not PROF_DB_DSN:
        # DB 설정이 없으면 그냥 큰 값
        return float(epochs) * 3600.0  # 1 epoch=1h 가정 (매우 보수적)

    try:
        conn = psycopg2.connect(PROF_DB_DSN)
    except Exception as e:
        # DB 연결 실패 시에도 SJF 깨지지 않도록 큰 값
        logger.error(f"[SJF] profiling DB connect failed: {e}")
        return float(epochs) * 3600.0

    try:
        with conn:
            with conn.cursor() as cur:
                # ✅ 현재 schema: minimal_profiling(model_name, dataset, g, epoch_time_measured_sec, ...)
                cur.execute(
                    """
                    SELECT epoch_time_measured_sec
                    FROM minimal_profiling
                    WHERE model_name = %s
                      AND dataset    = %s
                      AND gpu_count  = %s
                      AND epoch_time_measured_sec IS NOT NULL
                    ORDER BY ts DESC
                    LIMIT 10
                    """,
                    (model, dataset, world_size),
                )
                rows = cur.fetchall()
        if not rows:
            logger.warning(
                f"[SJF] no profiling rows for model={model}, dataset={dataset}, g={world_size}; fallback large T_est"
            )
            return float(epochs) * 3600.0

        epoch_times = [float(r[0]) for r in rows if r[0] is not None]
        if not epoch_times:
            return float(epochs) * 3600.0

        avg_epoch = statistics.fmean(epoch_times)
        T_est = float(epochs) * avg_epoch
        logger.info(
            f"[SJF] T_est={T_est:.2f}s (epochs={epochs}, avg_epoch={avg_epoch:.2f}s, "
            f"model={model}, dataset={dataset}, g={world_size})"
        )
        return T_est
    except Exception as e:
        logger.error(f"[SJF] error querying profiling DB: {e}")
        return float(epochs) * 3600.0
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _scheduler_loop():
    logger.info("SJF scheduler loop started (global queue)")
    while not STOP_EVENT.is_set():
        try:
            with SJF_LOCK:
                progressed = False
                while _try_start_sjf_any():
                    progressed = True
                if not progressed:
                    pass
        except Exception as e:
            logger.error(f"scheduler loop error: {e}")
        time.sleep(SCHED_TICK_SEC)
    logger.info("SJF scheduler loop stopping")

# CSP 샘플러
def _csp_sampler_loop():
    logger.info("CSP sampler loop started")
    while not STOP_EVENT.is_set():
        try:
            with SJF_LOCK:
                snap = _cluster_slots_snapshot()
                q_len = len(QUEUE)
                for c in ("clusterA","clusterB"):
                    slots_total = snap.get(c,{}).get("slots_total", 0)
                    slots_used  = snap.get(c,{}).get("slots_used", 0)
                    free        = snap.get(c,{}).get("free", 0)
                    csp = get_cluster_csp(c, q_len, slots_total)
                    lam = lambda_from_csp(csp.get("p_fair",0.0), csp.get("E_t",0.0))
                    with open(CSP_CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            iso_ms(now_ts()), c, q_len, slots_total, slots_used, free,
                            csp.get("U_t",0.0), csp.get("E_t",0.0), csp.get("p_fair",0.0),
                            lam.get("time",0.0), lam.get("cost",0.0), lam.get("fair",0.0)
                        ])
        except Exception as e:
            logger.error(f"csp sampler error: {e}")
        time.sleep(CSP_SAMPLER_SEC)

def _maybe_schedule_self_shutdown():
    global _SHUTDOWN_SCHEDULED
    if _SHUTDOWN_SCHEDULED:
        return
    if len(FINISHED_JOB_IDS) >= EXP_FINISH_TARGET:
        _SHUTDOWN_SCHEDULED = True
        logger.info(
            f"[EXP] {len(FINISHED_JOB_IDS)}/{EXP_FINISH_TARGET} jobs finished. "
            f"Scheduling self-shutdown in {EXP_SELF_SHUTDOWN_GRACE_SEC}s..."
        )
        def _do():
            try:
                logger.info("[EXP] Self-shutdown now.")
                for h in logger.handlers:
                    try: h.flush()
                    except: pass
            finally:
                os._exit(0) 
        threading.Timer(EXP_SELF_SHUTDOWN_GRACE_SEC, _do).start()

@app.middleware("http")
async def timing_header(request: Request, call_next):
    t0 = time.time()
    resp: Response = await call_next(request)
    resp.headers["X-SJF-Elapsed-ms"] = str(int((time.time()-t0)*1000))
    return resp

# API
@app.get("/health")
def health():
    with SJF_LOCK:
        snap = _cluster_slots_snapshot()
        return {
            "ok": True, "mode": "sjf-global", "log_dir": SJF_DIR,
            "queue_len": len(QUEUE),
            "running": {c: list(RUNNING[c].keys()) for c in RUNNING},
            "free_slots": {c: snap[c]["free"] for c in snap},
        }

@app.get("/status")
def status():
    with SJF_LOCK:
        snap = _cluster_slots_snapshot()
        return {
            "mode": "sjf-global",
            "queue_len": len(QUEUE),
            "queue_head": None if not QUEUE else {
                "job_id": QUEUE[0].job_id,
                "model":  QUEUE[0].model,
                "dataset": QUEUE[0].dataset,
                "world_size": QUEUE[0].world_size,
                "submitted_ts": QUEUE[0].submitted_ts,
            },
            "running": {
                c: [
                    {"job_id": jid, **RUNNING[c][jid]}
                    for jid in RUNNING[c].keys()
                ] for c in RUNNING
            },
            "clusters": snap,
        }

@app.get("/job_status/{job_id}")
def job_status_api(job_id: str):
    return job_status(job_id)

@app.post("/submit")
def submit(req: SubmitReq):
    api_t0 = time.time()
    print(req)
    job_id = req.job_id or _make_job_id()
    g = int(req.world_size) if (req.world_size and req.world_size > 0) else DEFAULT_G
    epochs = int(req.epochs) if (req.epochs and req.epochs > 0) else DEFAULT_EPOCHS

    bs = None
    if req.batch_size is not None:
        try:
            bs_val = int(req.batch_size)
            if bs_val <= 0:
                raise ValueError("batch_size must be positive")
            bs = bs_val
        except Exception:
            raise HTTPException(422, f"invalid batch_size: {req.batch_size}")

    # 🔹 SJF 전용 락 사용
    with SJF_LOCK:
        # 중복 job_id 방지
        if any(j.job_id == job_id for j in QUEUE) or \
           job_id in RUNNING["clusterA"] or job_id in RUNNING["clusterB"]:
            raise HTTPException(409, f"job_id {job_id} already exists")

        # 🔹 SJF용 추정 실행시간 계산
        est_T = _estimate_job_runtime_sec(
            model=req.model,
            dataset=req.dataset,
            world_size=g,
            epochs=epochs,
        )

        # 🔹 SJFJob 사용
        job = SJFJob(
            job_id=job_id,
            model=req.model,
            dataset=req.dataset,
            world_size=g,
            epochs=epochs,
            batch_size_per_gpu=bs,
            est_runtime_sec=est_T,
        )
        QUEUE.append(job)

        pos = _queue_position(job_id)

        note = f"SJF g={g}, T_est={est_T:.2f}s"
        if bs is not None:
            note += f", batch={bs}"

        _write_queue_event("enqueue", job_id, q_len_after=len(QUEUE), note=note)

        logger.info(
            f"[SUBMIT-SJF] {job_id} g={job.world_size}, model={job.model}, "
            f"dataset={job.dataset}, epochs={job.epochs}, "
            f"T_est={est_T:.2f}s" + (f", batch={bs}" if bs is not None else "")
        )

        write_event(
            "submitted",
            job_id,
            cluster="-",
            g=job.world_size,
            note=note,
            meta={
                "model": job.model,
                "dataset": job.dataset,
                "epochs": job.epochs,
                "batch_size_per_gpu": bs,
                "user_request": req.user_request or "",
                "est_runtime_sec": est_T,
            },
        )

    api_ms = int((time.time() - api_t0) * 1000)
    snap2 = _cluster_slots_snapshot()
    return {
        "mode": "sjf-global",
        "job_id": job_id,
        "queued": True,
        "position": pos,
        "default_world_size": DEFAULT_G,
        "free_slots": {c: snap2[c]["free"] for c in snap2},
        "duration_ms": {"total": api_ms},
        "batch_size_per_gpu": bs,
        "est_runtime_sec": est_T,
    }

@app.post("/report_job_completed")
def report_job_completed(rep: JobCompleteReport):
    job_id = rep.job_id
    with SJF_LOCK:
        cluster_found = rep.cluster
        if not cluster_found:
            for c in ("clusterA","clusterB"):
                if job_id in RUNNING[c]:
                    cluster_found = c
                    break
        if not cluster_found:
            return {"status": "unknown_or_already_cleared", "job_id": job_id}

        started_ts = RUNNING[cluster_found][job_id].get("started_ts")
        end_ts = now_ts()
        jct_sec = (end_ts - started_ts) if started_ts else None
        queued_sec = RUNNING[cluster_found][job_id].get("queued_sec")

        # 🔹 추가된 로그: 끝난 시간 / JCT / queued_ms를 scheduler.log에 남김
        logger.info(
            f"[DONE] {job_id} | finished_at={iso_ms(end_ts)}, cluster={cluster_found}, "
            f"exit={rep.exit_code}, jct_sec={jct_sec}, queued_sec={queued_sec}"
        )

        write_event(
            "finished", job_id, cluster_found,
            RUNNING[cluster_found][job_id].get("world_size",0),
            note=f"exit={rep.exit_code}",
            meta={"end_ts": end_ts, "jct_sec": jct_sec, "queued_sec": queued_sec}
        )
        _append_job_metric_finished(job_id, cluster_found, end_ts, jct_sec, status="completed")

        del RUNNING[cluster_found][job_id]

        FINISHED_JOB_IDS.add(job_id)
        _maybe_schedule_self_shutdown()

    return {"status": "completion_acked", "job_id": job_id, "cluster": cluster_found}

@app.post("/maint/sweep_running")
def sweep_running():
    t0 = time.time(); removed = []
    with SJF_LOCK:
        for c in ("clusterA","clusterB"):
            for jid in list(RUNNING[c].keys()):
                st = job_status(jid) or {}
                state = st.get("state") or st.get("status")
                if state in ("finished","stopped","error","unknown"):
                    end_ts = now_ts()
                    started_ts = RUNNING[c][jid].get("started_ts")
                    queued_sec = RUNNING[c][jid].get("queued_sec", None)
                    jct_sec = (end_ts - started_ts) if started_ts else None
                    meta = {
                        "job_status": st, "end_ts": end_ts, "started_ts": started_ts,
                        "jct_sec": jct_sec, "queued_sec": queued_sec
                    }

                    write_event(
                        "finished", jid, c,
                        RUNNING[c][jid].get("world_size",0),
                        meta=meta
                    )
                    _append_job_metric_finished(jid, c, end_ts, jct_sec, status=state or "finished")

                    # 🔹 추가된 로그: sweep로 정리된 job도 끝난 시각을 scheduler.log에 남김
                    logger.info(
                        f"[SWEEP-DONE] {jid} | finished_at={iso_ms(end_ts)}, cluster={c}, "
                        f"state={state}, jct_sec={jct_sec}, queued_sec={queued_sec}"
                    )

                    removed.append((c, jid, state))
                    del RUNNING[c][jid]
    ms = int((time.time()-t0)*1000)
    logger.info(f"[SWEEP] removed={removed} in {ms}ms")
    return {"removed": removed, "duration_ms": ms}

# Telemetry 수신 (운영 지표 기록용)
_TELEMETRY_TOKEN = os.getenv("TELEMETRY_TOKEN", None)

def ingest_telemetry_csv(node_id: str, gpu_index: int, ts: float,
                         gpu_util: Optional[float], power_w: Optional[float],
                         mem_used_mb: Optional[float], mem_total_mb: Optional[float]):
    ts_effective = ts or now_ts()
    with open(TELEM_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iso_ms(ts_effective), node_id, gpu_index,
            "" if gpu_util is None else f"{gpu_util:.1f}",
            "" if power_w  is None else f"{power_w:.1f}",
            "" if mem_used_mb is None else f"{mem_used_mb:.0f}",
            "" if mem_total_mb is None else f"{mem_total_mb:.0f}",
        ])

@app.post("/v1/telemetry", summary="Cluster worker telemetry push")
def push_telemetry(payload: TelemetryIn, x_telemetry_token: Optional[str] = Header(default=None)):
    t0 = time.time()
    if _TELEMETRY_TOKEN:
        if not x_telemetry_token or x_telemetry_token != _TELEMETRY_TOKEN:
            raise HTTPException(status_code=401, detail="invalid telemetry token")

    # 1) app.metrics 쪽으로 푸시 → get_cluster_csp 에서 ClusterB U_t/E_t 계산에 사용
    metrics_ingest_telemetry(
        node_id=payload.node_id,
        gpu_index=payload.gpu_index,
        ts=payload.ts,
        gpu_util=payload.gpu_util,
        power_w=payload.power_w,
        mem_used_mb=payload.mem_used_mb,
        mem_total_mb=payload.mem_total_mb,
    )

    # 2) 로컬 CSV 로깅 (오프라인 분석용)
    ingest_telemetry_csv(
        node_id=payload.node_id,
        gpu_index=payload.gpu_index,
        ts=payload.ts or time.time(),
        gpu_util=payload.gpu_util,
        power_w=payload.power_w,
        mem_used_mb=payload.mem_used_mb,
        mem_total_mb=payload.mem_total_mb,
    )

    dur = int((time.time() - t0) * 1000)
    util_s  = f"{payload.gpu_util:.1f}%" if payload.gpu_util is not None else "NA"
    power_s = f"{payload.power_w:.1f}W"   if payload.power_w  is not None else "NA"
    if payload.mem_used_mb is not None and payload.mem_total_mb is not None:
        mem_s = f"{payload.mem_used_mb:.0f}/{payload.mem_total_mb:.0f}MB"
    else:
        mem_s = "NA"

    print(f"[{time.strftime('%H:%M:%S')}] Telemetry from {payload.node_id} | GPU {payload.gpu_index} | "
          f"util={util_s} | power={power_s} | mem={mem_s} ({dur}ms)")
    return {"status": "ok", "duration_ms": dur}

# ==================================================================
@app.on_event("startup")
def _on_startup():
    STOP_EVENT.clear()
    th = threading.Thread(target=_scheduler_loop, daemon=True)
    th.start()
    th2 = threading.Thread(target=_csp_sampler_loop, daemon=True)
    th2.start()
    logger.info(f"startup complete | log_dir={SJF_DIR}")

@app.on_event("shutdown")
def _on_shutdown():
    STOP_EVENT.set()
    logger.info("shutdown signal")

