from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import os
import logging
import math
import threading
import time

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:
    psycopg2 = None
    DictCursor = None

logger = logging.getLogger(__name__)

# DB 설정
PROF_DB_DSN = os.getenv(
    "PROF_DB_DSN",
    "postgresql://prof:profpw@163.180.117.216:5432/profdb",
)

PROF_DB_TABLE = os.getenv("PROF_DB_TABLE", "minimal_profiling")

# Model alias
MODEL_ALIASES: Dict[str, List[str]] = {
    "DistilBERT": ["DistilBERT-base", "distilbert-base-uncased", "DistilBERT-base-uncased"],
    "DistilBERT-base": ["DistilBERT", "distilbert-base-uncased"],
}

_PROF_CACHE_LOCK = threading.Lock()
_PROF_CACHE: Dict[Tuple[str, str, str, int], Tuple[float, Optional[Dict[str, Any]]]] = {}
_PROF_TTL_SEC = float(os.getenv("OURS_PROF_TTL_SEC", "600"))  # 기본 10분

_INFLIGHT: Dict[Tuple[str, str, str, int], threading.Event] = {}
_INFLIGHT_LOCK = threading.Lock()

# debug 로그 너무 많이 찍히는 것 방지(기본 0=끔)
_PROF_DEBUG_SAMPLE_N = int(os.getenv("OURS_PROF_DEBUG_SAMPLE_N", "0"))
_PROF_DEBUG_CNT = 0

def _norm(x: Optional[str]) -> str:
    return (x or "").strip()

def _norm_l(x: Optional[str]) -> str:
    return _norm(x).lower()

# Dataclass
@dataclass
class ProfilingEntry:
    model_name: str
    dataset: str
    cluster_id: str
    gpu_count: int
    batch_size: int
    throughput_sps: float
    util_avg: float
    avg_power_w_per_gpu: float
    epoch_time_measured_sec: float
    ts: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> "ProfilingEntry":
        # row의 키는 DictCursor 기준. 여기서는 model_name을 “정답”으로 둠.
        return cls(
            model_name=str(row.get("model_name") or ""),
            dataset=str(row.get("dataset") or ""),
            cluster_id=str(row.get("cluster_id") or ""),
            gpu_count=int(row.get("gpu_count") or 0),
            batch_size=int(row.get("batch_size") or 0),
            throughput_sps=float(row.get("throughput_sps") or 0.0),
            util_avg=float(row.get("util_avg") or 0.0),
            avg_power_w_per_gpu=float(row.get("avg_power_w_per_gpu") or 0.0),
            epoch_time_measured_sec=float(row.get("epoch_time_measured_sec") or 0.0),
            ts=str(row.get("ts")) if row.get("ts") is not None else None,
        )

# DB connection (lazy singleton)
_CONN = None

def _ensure_conn():
    global _CONN
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Install psycopg2-binary or psycopg2.")

    if _CONN is None or _CONN.closed:
        logger.info("[profiling] Connecting to DB: %s", PROF_DB_DSN)

        try:
            _CONN = psycopg2.connect(
                PROF_DB_DSN,
                connect_timeout=2,  # seconds
                options="-c statement_timeout=2000",  # ms
            )
            _CONN.autocommit = True
        except Exception:
            logger.exception("[profiling] DB connect failed (timeout or network).")
            raise

    return _CONN

# Core lookup (THE function)
def get_profiling_entry(
    model_name: str,
    dataset: str,
    cluster_id: str,
    gpu_count: int,
) -> Optional[Dict[str, Any]]:

    def _should_log() -> bool:
        # 0이면 아예 디버그 로그 끔
        nonlocal_debug = False
        try:
            global _PROF_DEBUG_CNT
            if _PROF_DEBUG_SAMPLE_N <= 0:
                return False
            _PROF_DEBUG_CNT += 1
            return (_PROF_DEBUG_CNT % _PROF_DEBUG_SAMPLE_N) == 0
        except Exception:
            return False

    m0 = _norm(model_name)
    d0 = _norm(dataset)
    c0 = _norm(cluster_id)
    g0 = int(gpu_count or 0)

    # ---- args check ----
    if not m0 or not d0 or not c0 or g0 <= 0:
        if _should_log():
            logger.warning("[prof-debug] EARLY-RETURN: invalid args m=%r d=%r c=%r g=%r", m0, d0, c0, g0)
        return None

    # cluster_id canonicalize
    c_lower = c0.lower().strip()
    if c_lower in ("clustera", "cluster_a", "cluster-a", "a"):
        c0 = "clusterA"
    elif c_lower in ("clusterb", "cluster_b", "cluster-b", "b"):
        c0 = "clusterB"

    # model 후보(본명 + alias)
    candidates = [m0]
    for alt in MODEL_ALIASES.get(m0, []):
        if alt not in candidates:
            candidates.append(alt)

    # ✅ 캐시 key는 "요청 원형"이 아니라 canonicalized 값을 써야 cache miss가 안 남
    # candidates 전체를 key에 넣으면 키 폭발하니, "대표 모델 m0"만 사용하고
    # 내부 조회에서 alias를 시도한다.
    cache_key = (m0, d0, c0, g0)

    now = time.time()

    # ---- 1) TTL cache hit ----
    with _PROF_CACHE_LOCK:
        hit = _PROF_CACHE.get(cache_key)
        if hit is not None:
            exp_ts, val = hit
            if now <= exp_ts:
                return val
            else:
                _PROF_CACHE.pop(cache_key, None)

    # ---- 2) single-flight (동일 key 동시 호출 합치기) ----
    with _INFLIGHT_LOCK:
        ev = _INFLIGHT.get(cache_key)
        if ev is None:
            ev = threading.Event()
            _INFLIGHT[cache_key] = ev
            is_leader = True
        else:
            is_leader = False

    if not is_leader:
        # 다른 스레드가 DB 조회 중 → 끝날 때까지 대기 후 캐시에서 가져옴
        ev.wait(timeout=2.5)
        with _PROF_CACHE_LOCK:
            hit = _PROF_CACHE.get(cache_key)
            if hit is not None:
                exp_ts, val = hit
                if time.time() <= exp_ts:
                    return val
        # 그래도 없으면(리더 실패) 그냥 직접 조회로 fallback

    try:
        # ---- (기존 DB 조회 로직 그대로) ----
        conn = _ensure_conn()

        select_cols = """
            model_name,
            dataset,
            cluster_id,
            gpu_count,
            batch_size,
            throughput_sps,
            util_avg,
            avg_power_w_per_gpu,
            epoch_time_measured_sec,
            ts
        """

        def _fetch_one(query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
            try:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(query, params)
                    row = cur.fetchone()
                    if not row:
                        return None
                    return dict(row)
            except Exception:
                logger.exception("[prof-debug] QUERY FAILED params=%r", params)
                return None

        q_exact = f"""
            SELECT {select_cols}
            FROM {PROF_DB_TABLE}
            WHERE LOWER(TRIM(model_name)) = LOWER(TRIM(%s))
              AND LOWER(TRIM(dataset))    = LOWER(TRIM(%s))
              AND LOWER(TRIM(cluster_id)) = LOWER(TRIM(%s))
              AND gpu_count               = %s
            ORDER BY ts DESC
            LIMIT 1
        """

        if _should_log():
            logger.warning(
                "[prof-debug] ENTER get_profiling_entry(m=%r d=%r c=%r g=%r) table=%s dsn=%s",
                m0, d0, c0, g0, PROF_DB_TABLE, PROF_DB_DSN
            )

        for m in candidates:
            row = _fetch_one(q_exact, (m, d0, c0, g0))
            if _should_log():
                logger.warning("[prof-debug] exact try: m=%r d=%r c=%r g=%r -> %s",
                               m, d0, c0, g0, "HIT" if row else "MISS")
            if row:
                # ✅ 캐시 저장
                with _PROF_CACHE_LOCK:
                    _PROF_CACHE[cache_key] = (time.time() + _PROF_TTL_SEC, row)
                return row

        q_closest = f"""
            SELECT {select_cols}
            FROM {PROF_DB_TABLE}
            WHERE LOWER(TRIM(model_name)) = LOWER(TRIM(%s))
              AND LOWER(TRIM(dataset))    = LOWER(TRIM(%s))
              AND LOWER(TRIM(cluster_id)) = LOWER(TRIM(%s))
            ORDER BY ABS(gpu_count - %s) ASC, ts DESC
            LIMIT 1
        """

        for m in candidates:
            row = _fetch_one(q_closest, (m, d0, c0, g0))
            if _should_log():
                logger.warning("[prof-debug] closest try: m=%r d=%r c=%r g=%r -> %s",
                               m, d0, c0, g0, "HIT" if row else "MISS")
            if row:
                with _PROF_CACHE_LOCK:
                    _PROF_CACHE[cache_key] = (time.time() + _PROF_TTL_SEC, row)
                return row

        if _should_log():
            logger.warning("[prof-debug] FINAL MISS")

        # ✅ MISS도 캐시(짧게) 해두면 “없는 키”를 계속 DB로 안 감
        miss_ttl = min(30.0, _PROF_TTL_SEC)  # MISS는 30초만 캐시
        with _PROF_CACHE_LOCK:
            _PROF_CACHE[cache_key] = (time.time() + miss_ttl, None)

        return None

    finally:
        # single-flight release
        if is_leader:
            with _INFLIGHT_LOCK:
                ev = _INFLIGHT.pop(cache_key, None)
                if ev is not None:
                    ev.set()

# =========================
# Convenience helpers
# =========================

def get_latest_profiling_for_job(job_id: str) -> Optional[ProfilingEntry]:
    conn = _ensure_conn()
    q = f"""
        SELECT
            model_name,
            dataset,
            cluster_id,
            gpu_count,
            batch_size,
            throughput_sps,
            util_avg,
            avg_power_w_per_gpu,
            epoch_time_measured_sec,
            ts
        FROM {PROF_DB_TABLE}
        WHERE job_id = %s
        ORDER BY ts DESC
        LIMIT 1
    """
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(q, (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return ProfilingEntry.from_row(row)
    except Exception:
        logger.exception("[profiling] get_latest_profiling_for_job failed job_id=%s", job_id)
        return None

def estimate_epoch_time_from_profiling(entry: Optional[ProfilingEntry]) -> Optional[float]:
    if entry is None:
        return None

    t = float(entry.epoch_time_measured_sec or 0.0)
    if t > 0 and math.isfinite(t):
        return t

    sps = float(entry.throughput_sps or 0.0)
    if sps > 0 and math.isfinite(sps):
        return 1.0 / sps

    return None

_SPEED_FACTOR_CACHE: Dict[str, float] = {}

def get_cluster_speed_factor(cluster_id: str) -> float:
    cid = _norm(cluster_id)
    if not cid:
        return 1.0
    if cid in _SPEED_FACTOR_CACHE:
        return _SPEED_FACTOR_CACHE[cid]

    conn = _ensure_conn()
    q = f"""
        SELECT cluster_id, AVG(throughput_sps) AS avg_sps
        FROM {PROF_DB_TABLE}
        WHERE gpu_count = 1
        GROUP BY cluster_id
    """
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute(q)
            rows = cur.fetchall() or []
            if not rows:
                _SPEED_FACTOR_CACHE[cid] = 1.0
                return 1.0

            sps_map: Dict[str, float] = {}
            for r in rows:
                c = str(r.get("cluster_id") or "").strip()
                sps_map[c] = float(r.get("avg_sps") or 0.0)

            max_sps = max(sps_map.values()) if sps_map else 0.0
            if max_sps <= 0:
                _SPEED_FACTOR_CACHE[cid] = 1.0
                return 1.0

            for c, sps in sps_map.items():
                _SPEED_FACTOR_CACHE[c] = max(0.1, float(sps) / float(max_sps))

            if cid not in _SPEED_FACTOR_CACHE:
                _SPEED_FACTOR_CACHE[cid] = 1.0
            return _SPEED_FACTOR_CACHE[cid]
    except Exception:
        logger.exception("[profiling] get_cluster_speed_factor failed cluster_id=%s", cluster_id)
        _SPEED_FACTOR_CACHE[cid] = 1.0
        return 1.0