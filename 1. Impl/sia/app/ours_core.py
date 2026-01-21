from fastapi import HTTPException
import os, time, uuid, threading, csv, json, logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import psycopg2
import statistics

from app.executor import launch_or_reuse, job_status, resize_job, preempt_job
from app.metrics import get_cluster_csp, lambda_from_csp, ingest_telemetry as metrics_ingest_telemetry

from app.llm import get_policy_from_user_request, PolicyVector
from app.core import JobSpec, JobProfilingMetrics
from app.profiling import (
    get_allowed_gpus_for_job,
    load_profiling_rows_for_job,
)
from app.scheduler import ClusterState, choose_placement_for_job
from app.schemas import SubmitReq, TelemetryIn, JobCompleteReport

DEFAULT_G           = 1
DEFAULT_EPOCHS      = 20
SCHED_TICK_SEC      = 1.0
CSP_SAMPLER_SEC     = 5.0
REALLOC_TICK_SEC    = 60.0
REALLOC_COOLDOWN_SEC = 300.0  
_LAST_REALLOC_TS: float = 0.0


GPU_PRICE_PER_HOUR = 1.0  # relative cost per GPU-hour
ENERGY_PRICE_PER_KWH = 1.0  # relative energy price

PROF_DB_DSN = os.getenv(
    "PROF_DB_DSN",
    "postgresql://prof:profpw@localhost:5432/profdb"
)

EXP_FINISH_TARGET = int(os.getenv("EXP_FINISH_TARGET", "50"))
EXP_SELF_SHUTDOWN_GRACE_SEC = int(os.getenv("EXP_SELF_SHUTDOWN_GRACE_SEC", "60"))

FINISHED_JOB_IDS: set[str] = set()
_SHUTDOWN_SCHEDULED = False

CLUSTER_NODES = {
    "clusterA": ["node_a", "node_b", "node_c", "node_d"],
    "clusterB": ["node_e", "node_f", "node_g", "node_h"],
}

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_STAMP    = time.strftime("%Y%m%d-%H%M%S", time.localtime())
OURS_DIR  = os.path.join(_BASE_DIR, "..", "OURS_log", f"Ours-{_STAMP}")
os.makedirs(OURS_DIR, exist_ok=True)

STOP_EVENT = threading.Event()

logger = logging.getLogger("OURS")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(os.path.join(OURS_DIR, "scheduler.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
logger.propagate = False
REALLOC_EVENTS_PATH = os.path.join(OURS_DIR, "realloc_events.csv")
_csv_init(REALLOC_EVENTS_PATH, ["ts","cluster","job_id","role","g_before","g_after","U_before","U_after"])

HOL_PREEMPTED: Dict[str, float] = {}
PREEMPT_WINDOW_SEC = 5.0

def now_ts() -> float: return time.time()
def iso_ms(ts: float) -> str: return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def _csv_init(path: str, header: List[str]):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

EVENTS_PATH = os.path.join(OURS_DIR, "job_events.csv")
_csv_init(EVENTS_PATH, ["ts","event","job_id","cluster","world_size","note","metadata_json"])

def write_event(event: str, job_id: str, cluster: str, g: int, note: str = "", meta: Dict[str,Any] = None):
    with open(EVENTS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            iso_ms(now_ts()), event, job_id, cluster, g, note,
            json.dumps(meta or {}, ensure_ascii=False)
        ])

QUEUE_EVENTS_PATH = os.path.join(OURS_DIR, "queue_events.csv")
_csv_init(QUEUE_EVENTS_PATH, ["ts","event","job_id","queue_len_after","note"])
def _write_queue_event(event: str, job_id: str, q_len_after: int, note: str = ""):
    with open(QUEUE_EVENTS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([iso_ms(now_ts()), event, job_id, q_len_after, note])

CSP_CSV_PATH = os.path.join(OURS_DIR, "csp_metrics.csv")
_csv_init(CSP_CSV_PATH, ["ts","cluster","queue_len","slots_total","slots_used","free",
                         "U_t","E_t","p_fair","lam_time","lam_cost","lam_fair"])

TELEM_CSV_PATH = os.path.join(OURS_DIR, "telemetry.csv")
_csv_init(TELEM_CSV_PATH, ["ts","node_id","gpu_index","gpu_util","power_w","mem_used_mb","mem_total_mb"])

JOB_METRICS_PATH = os.path.join(OURS_DIR, "job_metrics.csv")
_csv_init(JOB_METRICS_PATH, ["job_id","cluster","model","dataset","world_size",
                             "submitted_ts","started_ts","end_ts","queued_sec","jct_sec","status"])

def _append_job_metric_started(job_id: str, cluster: str, queued_sec: int):
    meta = RUNNING[cluster][job_id]
    with open(JOB_METRICS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            job_id, cluster,
            meta.get("model"), meta.get("dataset"), meta.get("world_size"),
            meta.get("submitted_ts"), meta.get("started_ts"),
            "", queued_sec, "", "running"
        ])

def _append_job_metric_finished(job_id: str, cluster: str, end_ts: float, jct_sec: Optional[int], status: str):
    meta = RUNNING.get(cluster, {}).get(job_id, {})
    with open(JOB_METRICS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            job_id, cluster,
            meta.get("model"), meta.get("dataset"), meta.get("world_size"),
            meta.get("submitted_ts"), meta.get("started_ts"),
            end_ts, meta.get("queued_sec"), jct_sec, status
        ])

@dataclass
class OursJob:
    job_id: str
    model: str
    dataset: str
    policy: Optional[PolicyVector] = None
    allowed_gpus: List[int] = field(default_factory=lambda: [1, 2, 4])
    world_size: int = DEFAULT_G
    epochs: int = DEFAULT_EPOCHS
    batch_size_per_gpu: Optional[int] = None
    submitted_ts: float = field(default_factory=now_ts)
    start_ts: Optional[float] = None
    is_filler: bool = False
    target_g: int = DEFAULT_G

def _get_attr(obj, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default

OURS_LOCK = threading.RLock()
QUEUE : List[OursJob] = []
RUNNING: Dict[str, Dict[str, Any]] = {"clusterA": {}, "clusterB": {}}

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


def build_cluster_states_from_snapshot(
    snap: Dict[str, Dict[str, int]],
    queue_len: int,
) -> List[ClusterState]:
    clusters: List[ClusterState] = []
    for c, info in snap.items():
        slots_total = info.get("slots_total", 0)
        slots_used = info.get("slots_used", 0)
        free = info.get("free", 0)

        csp = get_cluster_csp(c, queue_len, slots_total)
        E_t = csp.get("E_t", 0.0)
        p_fair = csp.get("p_fair", 1.0)

        clusters.append(
            ClusterState(
                cluster_id=c,
                free_gpu=free,
                total_gpu=slots_total,
                energy_pressure=E_t,
                fairness_factor=p_fair,
            )
        )
    return clusters

def _get_epoch_time_sec(model: str, dataset: str, cluster_id: str, g: int) -> Optional[float]:
    """
    minimal_profiling에서 (model, dataset, cluster_id, g)에 대한
    epoch_time_measured_sec 평균값을 가져온다.
    """
    if not PROF_DB_DSN:
        return None

    try:
        conn = psycopg2.connect(PROF_DB_DSN)
    except Exception as e:
        logger.error(f"[ETA] profiling DB connect failed: {e}")
        return None

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT AVG(epoch_time_measured_sec)
                    FROM minimal_profiling
                    WHERE model_name = %s
                      AND dataset    = %s
                      AND cluster_id = %s
                      AND gpu_count  = %s
                      AND epoch_time_measured_sec IS NOT NULL
                    """,
                    (model, dataset, cluster_id, g),
                )
                row = cur.fetchone()
                if not row or row[0] is None:
                    return None
                return float(row[0])
    except Exception as e:
        logger.error(f"[ETA] error querying profiling DB: {e}")
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _compute_target_g_for_job(
    policy: PolicyVector,
    profiling_rows: List[JobProfilingMetrics],
    allowed_gpus: List[int],
) -> int:
    """
    Admission layer에서 사용할 target_g 계산:
    - profiling_rows에서 allowed_gpus에 해당하는 g만 모으고
    - 성능/비용/에너지를 정규화한 후
    - U(g) = λ_time * perf_norm(g) - λ_cost * cost_norm(g) - λ_energy * energy_norm(g)
      를 최대화하는 g를 선택한다.
    """
    if not allowed_gpus:
        return DEFAULT_G

    # g -> metrics 집계
    by_g: Dict[int, Dict[str, float]] = {}
    for row in profiling_rows:
        g = _get_attr(row, "gpu_count", None)
        if g is None:
            g = _get_attr(row, "g", None)
        if g is None or g not in allowed_gpus:
            continue

        sps = _get_attr(row, "throughput_sps", None)
        if sps is None:
            sps = _get_attr(row, "throughput", None)
        epoch_sec = _get_attr(row, "epoch_time_measured_sec", None)
        power_w = _get_attr(row, "avg_power_w_per_gpu", None)

        # 필수 값이 없으면 이 row는 스킵
        if sps is None or epoch_sec is None or power_w is None:
            continue

        sps = float(sps)
        epoch_sec = float(epoch_sec)
        power_w = float(power_w)

        # 여러 row가 있을 수 있으니 평균을 내도록 누적
        if g not in by_g:
            by_g[g] = {
                "count": 0,
                "sps_sum": 0.0,
                "epoch_sum": 0.0,
                "power_sum": 0.0,
            }

        by_g[g]["count"] += 1
        by_g[g]["sps_sum"] += sps
        by_g[g]["epoch_sum"] += epoch_sec
        by_g[g]["power_sum"] += power_w

    if not by_g:
        # profiling은 있는데 allowed_gpus와 교집합이 없거나 값이 비어 있는 경우
        # 보수적으로 최소 g 사용
        return min(allowed_gpus)

    # 평균값으로 환산, 그리고 비용/에너지 추정
    metrics_by_g: Dict[int, Dict[str, float]] = {}
    for g, agg in by_g.items():
        cnt = max(1, agg["count"])
        sps_avg = agg["sps_sum"] / cnt
        epoch_avg = agg["epoch_sum"] / cnt
        power_avg = agg["power_sum"] / cnt

        # 1 epoch 기준 비용 / 에너지 (절대값보다는 상대값이 중요)
        # 비용: g 개 GPU * epoch 시간 * 단위 가격
        cost_epoch = g * (epoch_avg / 3600.0) * GPU_PRICE_PER_HOUR
        # 에너지: g 개 GPU * 전력(W) * 시간(h)
        energy_epoch = g * (power_avg * epoch_avg / 3600.0)

        metrics_by_g[g] = {
            "sps": sps_avg,
            "cost_epoch": cost_epoch,
            "energy_epoch": energy_epoch,
        }

    # 정규화 상수 계산
    sps_max = max(m["sps"] for m in metrics_by_g.values() if m["sps"] > 0.0)
    cost_max = max(m["cost_epoch"] for m in metrics_by_g.values() if m["cost_epoch"] > 0.0)
    energy_max = max(m["energy_epoch"] for m in metrics_by_g.values() if m["energy_epoch"] > 0.0)

    # 혹시라도 모두 0일 때 방어
    if sps_max <= 0:
        sps_max = 1.0
    if cost_max <= 0:
        cost_max = 1.0
    if energy_max <= 0:
        energy_max = 1.0

    best_g = None
    best_u = -1e9

    for g in sorted(metrics_by_g.keys()):
        m = metrics_by_g[g]
        perf_norm = m["sps"] / sps_max
        cost_norm = m["cost_epoch"] / cost_max
        energy_norm = m["energy_epoch"] / energy_max

        # U(g) 계산
        u = (
            policy.lambda_time * perf_norm
            - policy.lambda_cost * cost_norm
            - policy.lambda_energy * energy_norm
        )

        # 동점이면 더 작은 g 선택 (보수적)
        if best_g is None or u > best_u + 1e-9 or (abs(u - best_u) <= 1e-9 and g < best_g):
            best_g = g
            best_u = u

    if best_g is None:
        return min(allowed_gpus)

    return best_g

def _build_metrics_by_g_for_cluster(
    profiling_rows: List[JobProfilingMetrics],
    allowed_gpus: List[int],
    cluster_id: str,
) -> Dict[int, Dict[str, float]]:
    """
    (model, dataset, cluster_id, g)에 대한 profiling rows에서
    g별 sps / cost_epoch / energy_epoch 평균값을 만든다.
    """
    by_g: Dict[int, Dict[str, float]] = {}

    for row in profiling_rows:
        c = _get_attr(row, "cluster_id", None)
        if c is None:
            c = _get_attr(row, "cluster", None)
        if c != cluster_id:
            continue

        g = _get_attr(row, "gpu_count", None)
        if g is None:
            g = _get_attr(row, "g", None)
        if g is None or g not in allowed_gpus:
            continue

        sps = _get_attr(row, "throughput_sps", None)
        if sps is None:
            sps = _get_attr(row, "throughput", None)
        epoch_sec = _get_attr(row, "epoch_time_measured_sec", None)
        power_w = _get_attr(row, "avg_power_w_per_gpu", None)

        if sps is None or epoch_sec is None or power_w is None:
            continue

        sps = float(sps)
        epoch_sec = float(epoch_sec)
        power_w = float(power_w)

        if g not in by_g:
            by_g[g] = {
                "count": 0,
                "sps_sum": 0.0,
                "epoch_sum": 0.0,
                "power_sum": 0.0,
            }

        by_g[g]["count"] += 1
        by_g[g]["sps_sum"] += sps
        by_g[g]["epoch_sum"] += epoch_sec
        by_g[g]["power_sum"] += power_w

    metrics_by_g: Dict[int, Dict[str, float]] = {}
    for g, agg in by_g.items():
        cnt = max(1, agg["count"])
        sps_avg = agg["sps_sum"] / cnt
        epoch_avg = agg["epoch_sum"] / cnt
        power_avg = agg["power_sum"] / cnt

        cost_epoch = g * (epoch_avg / 3600.0) * GPU_PRICE_PER_HOUR
        energy_epoch = g * (power_avg * epoch_avg / 3600.0)

        metrics_by_g[g] = {
            "sps": sps_avg,
            "cost_epoch": cost_epoch,
            "energy_epoch": energy_epoch,
        }

    return metrics_by_g


def _build_utility_table_for_job_cluster(
    job: OursJob,
    policy: PolicyVector,
    cluster_id: str,
    profiling_rows: List[JobProfilingMetrics],
    allowed_gpus: List[int],
) -> Dict[int, Dict[str, float]]:
    """
    주어진 job / cluster에 대해 g별 정규화된 metric과 U(g)를 계산해 테이블로 반환.
    return: { g: {"U": ..., "perf_norm": ..., "cost_norm": ..., "energy_norm": ...}, ... }
    """
    metrics_by_g = _build_metrics_by_g_for_cluster(
        profiling_rows=profiling_rows,
        allowed_gpus=allowed_gpus,
        cluster_id=cluster_id,
    )
    if not metrics_by_g:
        return {}

    sps_max = max(m["sps"] for m in metrics_by_g.values() if m["sps"] > 0.0)
    cost_max = max(m["cost_epoch"] for m in metrics_by_g.values() if m["cost_epoch"] > 0.0)
    energy_max = max(m["energy_epoch"] for m in metrics_by_g.values() if m["energy_epoch"] > 0.0)

    if sps_max <= 0:
        sps_max = 1.0
    if cost_max <= 0:
        cost_max = 1.0
    if energy_max <= 0:
        energy_max = 1.0

    table: Dict[int, Dict[str, float]] = {}
    for g, m in metrics_by_g.items():
        perf_norm = m["sps"] / sps_max
        cost_norm = m["cost_epoch"] / cost_max
        energy_norm = m["energy_epoch"] / energy_max

        u = (
            policy.lambda_time * perf_norm
            - policy.lambda_cost * cost_norm
            - policy.lambda_energy * energy_norm
        )

        table[g] = {
            "U": u,
            "perf_norm": perf_norm,
            "cost_norm": cost_norm,
            "energy_norm": energy_norm,
        }

    return table

def _estimate_job_runtime_sec(job: OursJob, cluster_id: str, g: int) -> Optional[float]:
    """
    주어진 job이 (cluster_id, g)에서 수행될 때 예상 실행시간(초)을 추정.
    epochs * epoch_time_measured_sec 를 사용.
    """
    epoch_sec = _get_epoch_time_sec(job.model, job.dataset, cluster_id, g)
    if epoch_sec is None:
        return None
    epochs = job.epochs or DEFAULT_EPOCHS
    return epoch_sec * float(epochs)

def _estimate_eta_for_cluster_for_hol(
    cluster_id: str,
    g_target: int,
    snap: Dict[str, Dict[str, int]],
) -> Optional[float]:
    """
    cluster_id에서 현재 running job들이 끝나면서
    g_target 개의 GPU 슬롯이 확보될 것으로 예상되는 시각(절대 시각, ts)을 추정.

    - ETA 계산에는 backfill filler job(is_filler=True)을 포함하지 않는다.
      (main job만으로 HoL 예약 시각을 결정)
    """
    info = snap.get(cluster_id, {})
    free_now = info.get("free", 0)
    total = info.get("slots_total", 0)

    if total <= 0 or g_target <= 0:
        return None

    g_target_capped = min(g_target, total)
    if g_target_capped <= free_now:
        return now_ts()  # 이미 충분한 슬롯 존재

    need = g_target_capped - free_now
    running_jobs = RUNNING.get(cluster_id, {})
    eta_slots: List[float] = []
    t_now = now_ts()

    for jid, meta in running_jobs.items():
        # 🔹 filler job은 ETA 계산에서 제외
        if meta.get("is_filler", False):
            continue

        g = int(meta.get("world_size", 1))
        if g <= 0:
            continue

        model = meta.get("model")
        dataset = meta.get("dataset")
        epochs = meta.get("epochs") or DEFAULT_EPOCHS
        started_ts = meta.get("started_ts")
        if not model or not dataset or not started_ts:
            continue

        epoch_sec = _get_epoch_time_sec(model, dataset, cluster_id, g)
        if epoch_sec is None:
            continue

        total_runtime = float(epochs) * epoch_sec
        elapsed = max(0.0, t_now - float(started_ts))
        remaining = max(0.0, total_runtime - elapsed)
        finish_ts = t_now + remaining

        eta_slots.extend([finish_ts] * g)

    if len(eta_slots) < need:
        return None

    eta_slots.sort()
    kth = eta_slots[need - 1]
    return kth

def _pop_job_from_queue_by_id(job_id: str) -> Optional[OursJob]:
    """
    QUEUE에서 job_id에 해당하는 OursJob을 제거하고 리턴.
    못 찾으면 None.
    """
    for idx, j in enumerate(QUEUE):
        if j.job_id == job_id:
            return QUEUE.pop(idx)
    return None

def _try_backfill_eta() -> bool:
    """
    HOL이 자원 부족으로 시작되지 못할 때,
    ETA 안에 끝날 수 있는 짧은 g=1 job을 뒤에서 골라 backfill로 실행한다.
    - 1단계: ETA-safe + SJF
    - 2단계: ETA-safe 후보가 없으면, g=1에 한해 FIFO fallback
    """
    if len(QUEUE) <= 1:
        return False

    hol = QUEUE[0]
    if not hol.allowed_gpus:
        return False

    snap = _cluster_slots_snapshot()

    if getattr(hol, "target_g", None):
        g_target = int(hol.target_g)
    else:
        g_target = max(hol.allowed_gpus)

    # HOL이 지금도 바로 들어갈 수 있으면 backfill 필요 없음
    any_cluster_can_start_now = any(info.get("free", 0) >= g_target for _, info in snap.items())
    if any_cluster_can_start_now:
        return False

    # ETA 계산
    eta_by_cluster: Dict[str, float] = {}
    for c in CLUSTER_NODES.keys():
        eta = _estimate_eta_for_cluster_for_hol(c, g_target, snap)
        if eta is not None:
            eta_by_cluster[c] = eta

    if not eta_by_cluster:
        return False

    t_now = now_ts()

    # ---------- (1) ETA-safe + SJF ----------
    best_eta_safe: Optional[Dict[str, Any]] = None

    for j in QUEUE[1:]:
        # backfill은 g=1 job만
        if 1 not in j.allowed_gpus:
            continue

        for c, eta_c in eta_by_cluster.items():
            free = snap.get(c, {}).get("free", 0)
            if free < 1:
                continue

            runtime = _estimate_job_runtime_sec(j, c, 1)
            if runtime is None:
                continue

            finish_ts = t_now + runtime
            if finish_ts > eta_c:
                # ETA-safe 아님
                continue

            cand = {"job": j, "cluster": c, "g": 1, "runtime": runtime}
            if best_eta_safe is None or runtime < best_eta_safe["runtime"]:
                best_eta_safe = cand

    chosen: Optional[Dict[str, Any]] = best_eta_safe

    # ---------- (2) ETA-safe 없으면 FIFO fallback (g=1만) ----------
    if chosen is None:
        for j in QUEUE[1:]:
            if 1 not in j.allowed_gpus:
                continue
            # FIFO 순서대로, 가장 먼저 만나는 g=1 job
            for c, info in snap.items():
                if info.get("free", 0) >= 1:
                    chosen = {"job": j, "cluster": c, "g": 1, "runtime": None}
                    break
            if chosen is not None:
                break

    if not chosen:
        return False

    j = chosen["job"]
    c = chosen["cluster"]
    g = chosen["g"]

    # free node 재계산
    avail = _available_nodes_by_cluster()
    free_nodes = avail.get(c, [])
    if len(free_nodes) < g:
        logger.info(
            f"[BACKFILL] Candidate {j.job_id} on {c} g={g}, but only {len(free_nodes)} free nodes now; skip"
        )
        return False

    chosen_nodes = free_nodes[:g]

    popped = _pop_job_from_queue_by_id(j.job_id)
    if not popped:
        return False
    backfill_job = popped

    created, launch_info = launch_or_reuse(
        job_id=backfill_job.job_id,
        cluster=c,
        world_size=g,
        dataset=backfill_job.dataset,
        model=backfill_job.model,
        epochs=backfill_job.epochs or DEFAULT_EPOCHS,
        preferred_nodes=chosen_nodes,
        batch_size=backfill_job.batch_size_per_gpu,
    )
    status = (launch_info or {}).get("status")

    if status not in ("started", "reused", "ok", "running"):
        logger.warning(
            f"[BACKFILL] launch failed: job={backfill_job.job_id}, "
            f"cluster={c}, g={g}, status={status}, info={launch_info}"
        )
        return False

    backfill_job.start_ts = now_ts()
    queued_sec = backfill_job.start_ts - backfill_job.submitted_ts
    backfill_job.world_size = g

    RUNNING[c][backfill_job.job_id] = {
        "cluster": c,
        "model": backfill_job.model,
        "dataset": backfill_job.dataset,
        "world_size": g,
        "epochs": backfill_job.epochs,
        "batch_size_per_gpu": backfill_job.batch_size_per_gpu,
        "nodes": launch_info.get("nodes", chosen_nodes),
        "submitted_ts": backfill_job.submitted_ts,
        "started_ts": backfill_job.start_ts,
        "queued_sec": queued_sec,
        "lambda_time": backfill_job.policy.lambda_time if backfill_job.policy else None,
        "lambda_cost": backfill_job.policy.lambda_cost if backfill_job.policy else None,
        "lambda_energy": backfill_job.policy.lambda_energy if backfill_job.policy else None,
        "is_filler": True,
    }

    _write_queue_event(
        "dequeue_backfill",
        backfill_job.job_id,
        q_len_after=len(QUEUE),
        note=f"g={g}, cluster={c}",
    )

    note = (
        f"BACKFILL started_on={c}, g={g}, "
        f"nodes={RUNNING[c][backfill_job.job_id]['nodes']}, "
        f"queued_sec={queued_sec}"
    )
    logger.info(f"[BACKFILL-START] {backfill_job.job_id} | {note}")

    write_event(
        "backfill_started",
        backfill_job.job_id,
        c,
        g,
        note,
        {
            "model": backfill_job.model,
            "dataset": backfill_job.dataset,
            "nodes": RUNNING[c][backfill_job.job_id]["nodes"],
            "queued_sec": queued_sec,
            "started_ts": backfill_job.start_ts,
        },
    )
    _append_job_metric_started(backfill_job.job_id, c, queued_sec)

    return True

def _preempt_fillers_for_hol() -> bool:
    """
    HOL이 placement는 나왔는데 filler 때문에 슬롯이 막혀 있는 경우:
    - HOL에 대해 다시 한 번 placement를 돌려서 (cluster, g_target)을 구하고
    - 해당 클러스터의 ETA가 가까운 경우(ETA - now <= PREEMPT_WINDOW_SEC)에만
      is_filler=True job들을 선점해서 g_target 만큼 free slot 확보를 시도.
    - 각 HOL job에 대해서는 평생 딱 한 번만 preemption을 수행한다.
    """
    if not QUEUE:
        return False

    hol = QUEUE[0]
    if hol.policy is None:
        return False

    # 이미 이 HOL에 대해 preemption을 했으면 두 번 다시 안 한다.
    if hol.job_id in HOL_PREEMPTED:
        return False

    snap = _cluster_slots_snapshot()
    queue_len = len(QUEUE)

    profiling_rows: List[JobProfilingMetrics] = load_profiling_rows_for_job(
        hol.model, hol.dataset
    )
    if not profiling_rows:
        return False

    clusters: List[ClusterState] = build_cluster_states_from_snapshot(
        snap, queue_len
    )
    job_spec = JobSpec(
        job_id=hol.job_id,
        model_name=hol.model,
        dataset=hol.dataset,
        allowed_gpus=hol.allowed_gpus,
    )

    decision = choose_placement_for_job(
        job=job_spec,
        policy=hol.policy,
        profiling_rows=profiling_rows,
        clusters=clusters,
        mu_fair=0.1,
    )

    if decision.chosen_cluster is None or decision.chosen_gpus is None:
        return False

    candidate_cluster = decision.chosen_cluster
    g_target = decision.chosen_gpus

    # ETA 계산 (filler 제외)
    eta_c = _estimate_eta_for_cluster_for_hol(candidate_cluster, g_target, snap)
    if eta_c is None:
        return False

    now = now_ts()
    if eta_c - now > PREEMPT_WINDOW_SEC:
        # 아직 ETA가 멀면 preempt 안 하고 나중 tick을 기다린다.
        return False

    # 해당 클러스터의 현재 free + filler로 확보 가능한 g 수 확인
    free_now = snap.get(candidate_cluster, {}).get("free", 0)
    running_jobs = RUNNING.get(candidate_cluster, {})
    fillers = []
    for jid, meta in running_jobs.items():
        if not meta.get("is_filler", False):
            continue
        g = int(meta.get("world_size", 1))
        if g <= 0:
            continue
        fillers.append((jid, g, meta))

    if not fillers:
        return False

    total_filler_g = sum(g for _, g, _ in fillers)
    if free_now + total_filler_g < g_target:
        return False

    needed = max(0, g_target - free_now)
    if needed <= 0:
        return False

    # filler job 우선순위: g 큰 놈부터
    fillers.sort(key=lambda x: -x[1])

    preempted_any = False

    for jid, g, meta in fillers:
        if needed <= 0:
            break

        ok, resp = preempt_job(jid)
        if not ok:
            logger.warning(f"[BACKFILL-PREEMPT] preempt_job failed for {jid}: {resp}")
            continue

        if jid in RUNNING[candidate_cluster]:
            running_meta = RUNNING[candidate_cluster].pop(jid)
        else:
            running_meta = meta

        model = running_meta.get("model")
        dataset = running_meta.get("dataset")
        if not model or not dataset:
            logger.warning(f"[BACKFILL-PREEMPT] missing model/dataset for {jid}, skip requeue")
            preempted_any = True
            continue

        lam_time = float(running_meta.get("lambda_time", 0.5))
        lam_cost = float(running_meta.get("lambda_cost", 0.25))
        lam_energy = float(running_meta.get("lambda_energy", 0.25))

        policy = PolicyVector(
            lambda_time=lam_time,
            lambda_cost=lam_cost,
            lambda_energy=lam_energy,
        )
        allowed = get_allowed_gpus_for_job(model, dataset)
        epochs = running_meta.get("epochs") or DEFAULT_EPOCHS
        submitted_ts = running_meta.get("submitted_ts") or now_ts()
        batch = running_meta.get("batch_size_per_gpu")

        target_g = _compute_target_g_for_job(
            policy=policy,
            profiling_rows=load_profiling_rows_for_job(model, dataset),
            allowed_gpus=allowed,
        )

        requeued = OursJob(
            job_id=jid,
            model=model,
            dataset=dataset,
            policy=policy,
            allowed_gpus=allowed,
            world_size=DEFAULT_G,
            epochs=epochs,
            batch_size_per_gpu=batch,
            submitted_ts=submitted_ts,
            target_g=target_g,
        )

        if len(QUEUE) >= 1:
            QUEUE.insert(1, requeued)
        else:
            QUEUE.append(requeued)

        needed -= g
        preempted_any = True

        write_event(
            "backfill_preempted",
            jid,
            candidate_cluster,
            g,
            note=f"preempted for HOL {hol.job_id}",
            meta={
                "cluster": candidate_cluster,
                "freed_g": g,
                "requeued_pos": _queue_position(jid),
            },
        )
        logger.info(
            f"[BACKFILL-PREEMPT] preempted filler {jid} on {candidate_cluster} "
            f"(g={g}) for HOL {hol.job_id}; needed_remain={needed}"
        )

    if preempted_any:
        HOL_PREEMPTED[hol.job_id] = now

    return preempted_any

def _should_trigger_reallocation(now_ts: float) -> bool:
    """
    이론에서 말한 Rescheduling Event 조건을 근사 구현:

    - TIME-heavy job(lambda_time >= 0.7)이 target보다 g_cur < g_target 이면
      '시간 SLA 위반 가능성'으로 간주.
    - COST/ENERGY-heavy job(lambda_cost >= 0.7 or lambda_energy >= 0.7)이
      g_cur > g_target이면 '과다 자원 사용'으로 간주.
    - 위 둘 중 하나라도 존재하고, 마지막 재배치 이후 REALLOC_COOLDOWN_SEC 이상 지났을 때만 True.
    """
    global _LAST_REALLOC_TS

    if now_ts - _LAST_REALLOC_TS < REALLOC_COOLDOWN_SEC:
        return False

    any_violation = False

    for c in ("clusterA", "clusterB"):
        running = RUNNING.get(c, {})
        if not running:
            continue

        for jid, meta in running.items():
            model = meta.get("model")
            dataset = meta.get("dataset")
            if not model or not dataset:
                continue

            lam_time = float(meta.get("lambda_time", 0.5))
            lam_cost = float(meta.get("lambda_cost", 0.25))
            lam_energy = float(meta.get("lambda_energy", 0.25))

            # g_target 이 이미 메타에 있으면 쓰고, 없으면 on-the-fly로 계산
            if "target_g" in meta:
                g_target = int(meta["target_g"])
            else:
                allowed = get_allowed_gpus_for_job(model, dataset)
                prof_rows: List[JobProfilingMetrics] = load_profiling_rows_for_job(
                    model, dataset
                )
                if not allowed or not prof_rows:
                    continue
                policy = PolicyVector(
                    lambda_time=lam_time,
                    lambda_cost=lam_cost,
                    lambda_energy=lam_energy,
                )
                g_target = _compute_target_g_for_job(policy, prof_rows, allowed)
                RUNNING[c][jid]["target_g"] = g_target

            g_cur = int(meta.get("world_size", 1))

            # TIME-heavy SLA violation: g_cur < g_target
            if lam_time >= 0.7 and g_cur < g_target:
                any_violation = True
                break

            # COST/ENERGY-heavy over-provision: g_cur > g_target
            if (lam_cost >= 0.7 or lam_energy >= 0.7) and g_cur > g_target:
                any_violation = True
                break

        if any_violation:
            break

    return any_violation

# ---- submit 코어 ----
def submit_job_core(req: SubmitReq) -> Dict[str, Any]:
    api_t0 = time.time()
    job_id = req.job_id or _make_job_id()

    epochs = int(req.epochs) if (req.epochs and req.epochs > 0) else DEFAULT_EPOCHS

    bs = None
    if req.batch_size is not None:
        bs_val = int(req.batch_size)
        if bs_val <= 0:
            raise HTTPException(422, f"invalid batch_size: {req.batch_size}")
        bs = bs_val

    # LLM policy
    policy: PolicyVector = get_policy_from_user_request(req.user_request or "")

    # allowed gpus
    allowed_gpus = get_allowed_gpus_for_job(req.model, req.dataset)

    # profiling 기반 target_g 계산 (없으면 최소 g 사용)
    profiling_rows: List[JobProfilingMetrics] = load_profiling_rows_for_job(
        req.model, req.dataset
    )
    if profiling_rows:
        target_g = _compute_target_g_for_job(policy, profiling_rows, allowed_gpus)
    else:
        target_g = min(allowed_gpus) if allowed_gpus else DEFAULT_G

    with OURS_LOCK:
        if any(j.job_id == job_id for j in QUEUE) or \
           job_id in RUNNING["clusterA"] or job_id in RUNNING["clusterB"]:
            raise HTTPException(409, f"job_id {job_id} already exists")

        job = OursJob(
            job_id=job_id,
            model=req.model,
            dataset=req.dataset,
            policy=policy,
            allowed_gpus=allowed_gpus,
            world_size=DEFAULT_G,
            epochs=epochs,
            batch_size_per_gpu=bs,
            target_g=target_g,   # ★ 추가
        )

        QUEUE.append(job)
        pos = _queue_position(job_id)

        note = f"allowed_gpus={allowed_gpus}"
        if bs is not None:
            note += f", batch={bs}"
        _write_queue_event("enqueue", job_id, q_len_after=len(QUEUE), note=note)

        logger.info(
            f"[SUBMIT] {job_id} model={job.model}, dataset={job.dataset}, "
            f"epochs={job.epochs}, allowed_gpus={allowed_gpus}, "
            f"lambda={policy.as_dict()}"
            + (f", batch={bs}" if bs is not None else "")
        )

        write_event(
            "submitted",
            job_id,
            cluster="-",
            g=job.world_size,
            meta={
                "model": job.model,
                "dataset": job.dataset,
                "epochs": job.epochs,
                "batch_size_per_gpu": bs,
                "user_request": req.user_request or "",
                "lambda_time": policy.lambda_time,
                "lambda_cost": policy.lambda_cost,
                "lambda_energy": policy.lambda_energy,
                "target_g": target_g,
            },
        )

    api_ms = int((time.time() - api_t0) * 1000)
    snap2 = _cluster_slots_snapshot()
    return {
        "mode": "ours-global",
        "job_id": job_id,
        "queued": True,
        "position": pos,
        "default_world_size": DEFAULT_G,
        "free_slots": {c: snap2[c]["free"] for c in snap2},
        "duration_ms": {"total": api_ms},
        "batch_size_per_gpu": bs,
        "lambda": policy.as_dict(),
    }

def report_job_completed_core(rep: JobCompleteReport) -> Dict[str, Any]:
    job_id = rep.job_id
    with OURS_LOCK:
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

def _run_reallocation_tick():
    """
    HOL 중심 donor–receiver shrink:

    - HOL이 REALLOC_MIN_WAIT_SEC 이상 대기 중일 때만 동작.
    - HOL에 profiling row가 있어야 함.
    - HOL의 target_g (policy 기반)를 우선 사용하되, 없으면 allowed_gpus 중 최소값 사용.
    - donor cluster 선택: (free + shrinkable) * p_fair * (1 - E_t) 최대인 곳.
    - donor 선택: filler 먼저, 그 다음 lambda_time 낮은 job, 그 다음 world_size 큰 job.
    """
    if not QUEUE:
        _elastic_scaleup_tick()
        return

    hol = QUEUE[0]
    wait_sec = now_ts() - hol.submitted_ts
    if wait_sec < REALLOC_MIN_WAIT_SEC:
        return

    # HOL에 profiling이 없으면 리스크 크니까 스킵
    profiling_rows: List[JobProfilingMetrics] = load_profiling_rows_for_job(
        hol.model, hol.dataset
    )
    if not profiling_rows:
        logger.info(
            f"[REALLOC] HOL {hol.job_id} has no profiling rows ({hol.model}, {hol.dataset}); skip"
        )
        return

    if not hol.allowed_gpus:
        return

    # policy 기반 target_g 우선 사용, 없으면 conservative하게 최소 g
    if getattr(hol, "target_g", None):
        try:
            target_g = int(hol.target_g)
        except Exception:
            target_g = min(hol.allowed_gpus)
    else:
        target_g = min(hol.allowed_gpus)

    if target_g <= 1:
        # 1GPU만 있으면 donor shrink까지 할 필요는 낮음
        return

    snap = _cluster_slots_snapshot()
    queue_len = len(QUEUE)

    best_cluster = None
    best_score = 0.0
    donors_by_cluster: Dict[str, List[Dict[str, Any]]] = {}

    # 1) 클러스터별 후보 / 점수 계산
    for c in ("clusterA", "clusterB"):
        running = RUNNING.get(c, {})
        if not running:
            continue

        donors: List[Dict[str, Any]] = []
        for jid, meta in running.items():
            g_cur = int(meta.get("world_size", 1))
            if g_cur <= 1:
                continue

            donors.append({
                "job_id": jid,
                "world_size": g_cur,
                "nodes": list(meta.get("nodes", [])),
                "lambda_time": float(meta.get("lambda_time", 0.5)),
                "is_filler": bool(meta.get("is_filler", False)),
            })

        if not donors:
            continue

        donors_by_cluster[c] = donors

        free_now = snap.get(c, {}).get("free", 0)
        shrinkable = sum(max(d["world_size"] - 1, 0) for d in donors)

        csp = get_cluster_csp(c, queue_len, snap.get(c, {}).get("slots_total", 0))
        p_fair = float(csp.get("p_fair", 1.0))
        E_t = float(csp.get("E_t", 0.0))

        potential = free_now + shrinkable
        if potential <= 0:
            continue

        # CSP 기반 스코어: (free+shrinkable) * p_fair * (1 - E_t)
        score = potential * p_fair * max(0.0, 1.0 - E_t)

        logger.info(
            f"[REALLOC] cluster={c} | free={free_now}, shrinkable={shrinkable}, "
            f"p_fair={p_fair:.3f}, E_t={E_t:.3f}, potential={potential}, score={score:.3f}"
        )

        # 아예 target_g조차 못 맞추는 클러스터는 후보에서 제외
        if potential < target_g:
            continue

        if score > best_score:
            best_score = score
            best_cluster = c

    if best_cluster is None:
        logger.info(
            f"[REALLOC] No cluster can support HOL {hol.job_id} (target_g={target_g}) "
            f"with donors; skip this tick."
        )
        return

    # 2) 선택된 클러스터에서 실제 donor shrink
    donors = donors_by_cluster[best_cluster]
    free_now = snap[best_cluster]["free"]
    shrinkable_total = sum(max(d["world_size"] - 1, 0) for d in donors)

    if free_now + shrinkable_total < target_g:
        logger.info(
            f"[REALLOC] cluster {best_cluster} cannot support HOL {hol.job_id}: "
            f"target_g={target_g}, free={free_now}, shrinkable={shrinkable_total}"
        )
        return

    # donor 정렬:
    #   1) filler 먼저 줄이고 (is_filler=True 우선)
    #   2) lambda_time 낮은 job (time-first가 아닌 job) 먼저
    #   3) world_size 큰 job 먼저
    donors.sort(
        key=lambda d: (
            d["is_filler"] is False,   # filler=True -> False -> 앞쪽
            d["lambda_time"],          # 작은 lambda_time 먼저
            -d["world_size"],          # 큰 g 먼저
        )
    )

    needed = max(0, target_g - free_now)
    to_resize: List[tuple[str, List[str]]] = []

    for d in donors:
        if needed <= 0:
            break

        g_cur = d["world_size"]
        nodes_cur = d["nodes"]

        max_give = g_cur - 1  # 최소 1GPU는 남겨야 함
        if max_give <= 0:
            continue

        give = min(max_give, needed)
        if give <= 0:
            continue

        # 뒤에서 give개를 떼어내고, 앞부분만 유지
        keep_nodes = nodes_cur[:-give]
        if not keep_nodes:
            # 방어로직: 최소 1개는 남겨야 하므로
            keep_nodes = [nodes_cur[0]]
            give = g_cur - 1  # 나머지 전부 반환
            if give <= 0:
                continue

        needed -= give
        to_resize.append((d["job_id"], keep_nodes))

    if not to_resize:
        return

    # 3) 실제 resize 호출
    for donor_job_id, new_nodes in to_resize:
        ok, info = resize_job(donor_job_id, new_nodes)
        if not ok:
            logger.warning(
                f"[REALLOC] resize donor {donor_job_id} -> {new_nodes} failed: {info}"
            )
            continue

        if donor_job_id in RUNNING[best_cluster]:
            RUNNING[best_cluster][donor_job_id]["world_size"] = len(new_nodes)
            RUNNING[best_cluster][donor_job_id]["nodes"] = list(new_nodes)

        logger.info(
            f"[REALLOC] donor {donor_job_id} shrunk to g={len(new_nodes)} "
            f"for HOL {hol.job_id} on {best_cluster}"
        )
        write_event(
            "resize_donor",
            donor_job_id,
            best_cluster,
            len(new_nodes),
            note=f"donor for HOL {hol.job_id} (target_g={target_g})",
            meta={"new_nodes": new_nodes},
        )

def _elastic_scaleup_tick():
    """
    Elastic layer: free slot이 있을 때 marginal utility 기반으로 scale-up.
    - 현재 구현에서는 공정성을 위해 queue가 비어 있을 때만 동작한다.
      (대기중인 HOL을 밀어내지 않도록)
    - 각 클러스터에서:
        * free > 0 이면
        * running job 중에서 g를 늘릴 수 있는 후보들에 대해
          ΔU_per_gpu = (U(g_new) - U(g_cur)) / (g_new - g_cur) 를 계산
        * ΔU_per_gpu가 최대인 job 하나를 선택해 resize_job 수행
    """
    # 큐에 잡이 있으면, scale-up으로 HOL을 지연시키지 않는다.
    if QUEUE:
        return

    snap = _cluster_slots_snapshot()
    avail_by_cluster = _available_nodes_by_cluster()

    for c, info in snap.items():
        free = info.get("free", 0)
        if free <= 0:
            continue

        running_jobs = RUNNING.get(c, {})
        if not running_jobs:
            continue

        best_jid = None
        best_meta = None
        best_delta_per_gpu = 0.0

        for jid, meta in running_jobs.items():
            model = meta.get("model")
            dataset = meta.get("dataset")
            if not model or not dataset:
                continue

            lam_time = float(meta.get("lambda_time", 0.5))
            lam_cost = float(meta.get("lambda_cost", 0.25))
            lam_energy = float(meta.get("lambda_energy", 0.25))
            policy = PolicyVector(
                lambda_time=lam_time,
                lambda_cost=lam_cost,
                lambda_energy=lam_energy,
            )

            allowed = get_allowed_gpus_for_job(model, dataset)
            if not allowed:
                continue

            g_cur = int(meta.get("world_size", 1))
            g_max = max(allowed)
            if g_cur >= g_max:
                continue

            profiling_rows: List[JobProfilingMetrics] = load_profiling_rows_for_job(
                model, dataset
            )
            if not profiling_rows:
                continue

            util_table = _build_utility_table_for_job_cluster(
                job=None,  # job 자체는 여기서 안 쓰니 None
                policy=policy,
                cluster_id=c,
                profiling_rows=profiling_rows,
                allowed_gpus=allowed,
            )
            if not util_table:
                continue

            if g_cur not in util_table:
                # 현재 g에 대한 profiling이 없으면 스킵
                continue

            # 이번 tick에서 붙일 수 있는 최대 g_new
            g_new = min(g_cur + free, g_max)
            if g_new <= g_cur:
                continue
            if g_new not in util_table:
                # g_new에 대한 profiling 없으면, 한 단계 낮춰본다
                candidates = [g for g in util_table.keys() if g_cur < g <= g_max]
                if not candidates:
                    continue
                g_new = min(max(candidates), g_cur + free)  # g_cur보다 크면서 profiling 있는 최대 g
                if g_new <= g_cur or g_new not in util_table:
                    continue

            U_cur = util_table[g_cur]["U"]
            U_new = util_table[g_new]["U"]
            delta_u = U_new - U_cur
            delta_g = g_new - g_cur
            if delta_g <= 0:
                continue

            delta_per_gpu = delta_u / float(delta_g)

            # marginal utility가 양수인 것만 의미 있음
            if delta_per_gpu <= 0:
                continue

            if best_jid is None or delta_per_gpu > best_delta_per_gpu:
                best_jid = jid
                best_meta = meta
                best_delta_per_gpu = delta_per_gpu

        if not best_jid or not best_meta:
            continue

        # 실제로 resize 실행
        g_cur = int(best_meta.get("world_size", 1))
        model = best_meta.get("model")
        dataset = best_meta.get("dataset")
        allowed = get_allowed_gpus_for_job(model, dataset)
        g_max = max(allowed)

        # free / allowed / profiling-consistent g_new 다시 계산 (방어용)
        free_nodes = avail_by_cluster.get(c, [])
        free_slots = len(free_nodes)
        if free_slots <= 0:
            continue

        profiling_rows = load_profiling_rows_for_job(model, dataset)
        util_table = _build_utility_table_for_job_cluster(
            job=None,
            policy=PolicyVector(
                lambda_time=float(best_meta.get("lambda_time", 0.5)),
                lambda_cost=float(best_meta.get("lambda_cost", 0.25)),
                lambda_energy=float(best_meta.get("lambda_energy", 0.25)),
            ),
            cluster_id=c,
            profiling_rows=profiling_rows,
            allowed_gpus=allowed,
        )
        if not util_table or g_cur not in util_table:
            continue

        # 후보 g_new들은 g_cur < g <= g_max, free_slots 이내
        candidates = [
            g for g in util_table.keys()
            if g_cur < g <= g_max and (g - g_cur) <= free_slots
        ]
        if not candidates:
            continue

        # 다시 한 번 ΔU_per_gpu 기준으로 가장 좋은 g_new 선택
        best_g_new = None
        best_delta_per_gpu_2 = 0.0
        U_cur = util_table[g_cur]["U"]

        for g_new in candidates:
            U_new = util_table[g_new]["U"]
            delta_u = U_new - U_cur
            delta_g = g_new - g_cur
            if delta_g <= 0:
                continue
            delta_per_gpu = delta_u / float(delta_g)
            if delta_per_gpu <= 0:
                continue
            if best_g_new is None or delta_per_gpu > best_delta_per_gpu_2:
                best_g_new = g_new
                best_delta_per_gpu_2 = delta_per_gpu

        if best_g_new is None:
            continue

        extra = best_g_new - g_cur
        if extra <= 0 or extra > free_slots:
            continue

        used_nodes: List[str] = list(best_meta.get("nodes", []))
        new_nodes = used_nodes + free_nodes[:extra]
        if len(new_nodes) != best_g_new:
            continue

        ok, info_resp = resize_job(best_jid, new_nodes)
        if not ok:
            logger.warning(
                f"[ELASTIC] resize_job failed for {best_jid} on {c}: {info_resp}"
            )
            continue

        best_meta["world_size"] = best_g_new
        best_meta["nodes"] = new_nodes

        logger.info(
            f"[ELASTIC] scaled up {best_jid} on {c}: g {g_cur} -> {best_g_new}, "
            f"ΔU_per_gpu={best_delta_per_gpu_2:.4f}, nodes={new_nodes}"
        )

        write_event(
            "resized_elastic",
            best_jid,
            c,
            best_g_new,
            note=f"elastic scale-up from g={g_cur}, free_slots={free_slots}",
            meta={
                "old_g": g_cur,
                "new_g": best_g_new,
                "delta_u_per_gpu": best_delta_per_gpu_2,
                "nodes": new_nodes,
            },
        )

def start_background_loops():
    STOP_EVENT.clear()

    th = threading.Thread(target=_scheduler_loop, daemon=True)
    th.start()

    th2 = threading.Thread(target=_csp_sampler_loop, daemon=True)
    th2.start()

    # HOL 중심 donor–receiver shrink 루프
    th3 = threading.Thread(target=_reallocation_loop, daemon=True)
    th3.start()

    logger.info(f"startup complete | log_dir={OURS_DIR}")

def stop_background_loops():
    STOP_EVENT.set()
    logger.info("shutdown signal")

def health_snapshot() -> Dict[str, Any]:
    with OURS_LOCK:
        snap = _cluster_slots_snapshot()
        return {
            "ok": True,
            "mode": "ours-global",
            "log_dir": OURS_DIR,
            "queue_len": len(QUEUE),
            "running": {c: list(RUNNING[c].keys()) for c in RUNNING},
            "free_slots": {c: snap[c]["free"] for c in snap},
        }

def status_snapshot() -> Dict[str, Any]:
    with OURS_LOCK:
        snap = _cluster_slots_snapshot()
        return {
            "mode": "ours-global",
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

def _try_start_hol_any() -> bool:
    """
    HOL job을 대상으로 글로벌 스케줄러를 실행해
    (cluster, g)를 결정하고, 가능하면 즉시 시작한다.
    """
    if not QUEUE:
        return False

    hol = QUEUE[0]

    if hol.policy is None:
        logger.error(f"[OURS] HOL job {hol.job_id} has no policy; skipping")
        return False

    snap = _cluster_slots_snapshot()
    queue_len = len(QUEUE)

    # ClusterState 리스트 구성 (free_gpu, total_gpu, E_t^c, f_fair(c))
    clusters: List[ClusterState] = build_cluster_states_from_snapshot(snap, queue_len)

    # profiling 로딩
    profiling_rows: List[JobProfilingMetrics] = load_profiling_rows_for_job(
        hol.model, hol.dataset
    )

    if not profiling_rows:
        logger.warning(f"[OURS] No profiling rows for job {hol.job_id} ({hol.model}, {hol.dataset})")
        return False

    # JobSpec 구성
    job_spec = JobSpec(
        job_id=hol.job_id,
        model_name=hol.model,
        dataset=hol.dataset,
        allowed_gpus=hol.allowed_gpus,
    )
    
    ideal_target_g = hol.target_g if hasattr(hol, 'target_g') else max(hol.allowed_gpus)

    # Global Scheduler로 placement 결정
    decision = choose_placement_for_job(
        job=job_spec,
        policy=hol.policy,
        profiling_rows=profiling_rows,
        clusters=clusters,
        mu_fair=0.1,
    )

    if decision.chosen_cluster is None or decision.chosen_gpus is None:
        logger.info(f"[OURS] No valid placement for HOL={hol.job_id} (reason={decision.reason})")
        return False

    cluster = decision.chosen_cluster
    g = decision.chosen_gpus

    # free node 목록 계산
    avail = _available_nodes_by_cluster()
    free_nodes = avail.get(cluster, [])
    if len(free_nodes) < g:
        logger.info(
            f"[OURS] Placement says cluster={cluster}, g={g}, "
            f"but only {len(free_nodes)} free nodes; skipping this tick"
        )
        return False

    chosen_nodes = free_nodes[:g]

    created, launch_info = launch_or_reuse(
        job_id=hol.job_id,
        cluster=cluster,
        world_size=g,
        dataset=hol.dataset,
        model=hol.model,
        epochs=hol.epochs or DEFAULT_EPOCHS,
        preferred_nodes=chosen_nodes,
        batch_size=hol.batch_size_per_gpu,
    )
    status = (launch_info or {}).get("status")

    if status in ("started", "reused", "ok", "running"):
        QUEUE.pop(0)
        hol.start_ts = now_ts()
        queued_sec = hol.start_ts - hol.submitted_ts

        # 실제 world_size 업데이트
        hol.world_size = g

        RUNNING[cluster][hol.job_id] = {
            "cluster": cluster,
            "model": hol.model,
            "dataset": hol.dataset,
            "world_size": g,
            "target_g": ideal_target_g,
            "epochs": hol.epochs,  # ✅ ETA 계산용
            "batch_size_per_gpu": hol.batch_size_per_gpu,
            "nodes": launch_info.get("nodes", chosen_nodes),
            "submitted_ts": hol.submitted_ts,
            "started_ts": hol.start_ts,
            "queued_sec": queued_sec,
            "lambda_time": hol.policy.lambda_time,
            "lambda_cost": hol.policy.lambda_cost,
            "lambda_energy": hol.policy.lambda_energy,
        }

        _write_queue_event(
            "dequeue_start",
            hol.job_id,
            q_len_after=len(QUEUE),
            note=f"g={g}, cluster={cluster}",
        )

        note = (
            f"started_on={cluster}, g={g}, "
            f"nodes={RUNNING[cluster][hol.job_id]['nodes']}, "
            f"queued_sec={RUNNING[cluster][hol.job_id]['queued_sec']}, "
            f"lambda={hol.policy.as_dict()}"
        )
        logger.info(f"[START] {hol.job_id} | {note}")

        write_event(
            "started",
            hol.job_id,
            cluster,
            g,
            note,
            {
                "model": hol.model,
                "dataset": hol.dataset,
                "nodes": RUNNING[cluster][hol.job_id]["nodes"],
                "queued_sec": RUNNING[cluster][hol.job_id]["queued_sec"],
                "started_ts": hol.start_ts,
                "lambda_time": hol.policy.lambda_time,
                "lambda_cost": hol.policy.lambda_cost,
                "lambda_energy": hol.policy.lambda_energy,
            },
        )
        _append_job_metric_started(hol.job_id, cluster, RUNNING[cluster][hol.job_id]["queued_sec"])
        return True

    else:
        logger.warning(
            f"[OURS] launch failed: status={status}, info={launch_info} "
            f"(cluster={cluster}, g={g})"
        )
        return False

def _reallocation_loop():
    """
    60초마다 한 번씩 재배치 트리거 조건을 체크하고,
    조건을 만족할 때만 Elastic Reallocation을 수행하는 루프.
    """
    global _LAST_REALLOC_TS

    logger.info("Reallocation loop started")
    while not STOP_EVENT.is_set():
        try:
            now = now_ts()
            with OURS_LOCK:
                if _should_trigger_reallocation(now):
                    _run_reallocation_tick()
                    _LAST_REALLOC_TS = now
        except Exception as e:
            logger.error(f"reallocation loop error: {e}")
        time.sleep(REALLOC_TICK_SEC)
    logger.info("Reallocation loop stopping")

def _scheduler_loop():
    logger.info("Ours scheduler loop started (global queue)")
    while not STOP_EVENT.is_set():
        try:
            with OURS_LOCK:
                progressed = False

                # 1) HOL 기준으로 가능한 만큼 job 시작
                while _try_start_hol_any():
                    progressed = True

                if QUEUE:
                    # 2) HOL이 막혀 있는 상황에서, filler preempt로 HOL join 시도
                    if _preempt_fillers_for_hol():
                        progressed = True
                        # preemption으로 슬롯이 열렸을 수 있으니 HOL 다시 시도
                        while _try_start_hol_any():
                            progressed = True

                    # 3) 여전히 HOL이 못 뜨면, ETA 기반 비선점 backfill
                    if _try_backfill_eta():
                        progressed = True

        except Exception as e:
            logger.error(f"scheduler loop error: {e}")
        time.sleep(SCHED_TICK_SEC)
    logger.info("Ours scheduler loop stopping")

def _csp_sampler_loop():
    logger.info("CSP sampler loop started")
    while not STOP_EVENT.is_set():
        try:
            with OURS_LOCK:
                snap = _cluster_slots_snapshot()
                q_len = len(QUEUE)
                for c in ("clusterA", "clusterB"):
                    slots_total = snap.get(c, {}).get("slots_total", 0)
                    slots_used  = snap.get(c, {}).get("slots_used", 0)
                    free        = snap.get(c, {}).get("free", 0)
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
    logger.info("CSP sampler loop stopping")

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

def sweep_running_core() -> Dict[str, Any]:
    t0 = time.time()
    removed = []
    with OURS_LOCK:
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

                    logger.info(
                        f"[SWEEP-DONE] {jid} | finished_at={iso_ms(end_ts)}, cluster={c}, "
                        f"state={state}, jct_sec={jct_sec}, queued_sec={queued_sec}"
                    )

                    removed.append((c, jid, state))
                    del RUNNING[c][jid]

                    FINISHED_JOB_IDS.add(jid)
                    _maybe_schedule_self_shutdown()
    ms = int((time.time()-t0)*1000)
    logger.info(f"[SWEEP] removed={removed} in {ms}ms")
    return {"removed": removed, "duration_ms": ms}

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

def handle_telemetry(payload: TelemetryIn, x_telemetry_token: Optional[str]) -> Dict[str, Any]:
    t0 = time.time()
    if _TELEMETRY_TOKEN:
        if not x_telemetry_token or x_telemetry_token != _TELEMETRY_TOKEN:
            raise HTTPException(status_code=401, detail="invalid telemetry token")

    # 1) app.metrics 쪽으로 푸시
    metrics_ingest_telemetry(
        node_id=payload.node_id,
        gpu_index=payload.gpu_index,
        ts=payload.ts,
        gpu_util=payload.gpu_util,
        power_w=payload.power_w,
        mem_used_mb=payload.mem_used_mb,
        mem_total_mb=payload.mem_total_mb,
    )

    # 2) 로컬 CSV 로깅
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