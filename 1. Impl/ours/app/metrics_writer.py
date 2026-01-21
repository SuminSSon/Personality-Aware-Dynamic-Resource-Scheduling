from __future__ import annotations

import os
import csv
import time
from typing import Any, Dict, Optional

# job_metrics.csv 스키마를 "고정"합니다. 절대 바꾸지 말고, 바꿔야 하면 버전업하세요.
JOB_METRICS_FIELDS = [
    "job_id",
    "cluster",
    "model",
    "dataset",
    "world_size",
    "submitted_ts",
    "started_ts",
    "end_ts",
    "queued_sec",
    "jct_sec",
    "status",
]

def _log_dir() -> str:
    return os.getenv("LOG_DIR", "./logs")

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)

def _atomic_append_dict_row(path: str, fields: list[str], row: Dict[str, Any]) -> None:
    _ensure_dir(path)
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)

    # newline="" 중요: csv가 OS별로 깨지는 것 방지
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",  # 스키마 밖 키는 버림(밀림 방지)
        )
        if need_header:
            w.writeheader()
        w.writerow(row)

def compute_queued_sec(submitted_ts: Any, started_ts: Any) -> float:
    """
    queued_sec = started_ts - submitted_ts
    - started_ts가 없으면 0.0 (아직 시작 안 했으니까)
    - submitted_ts가 없으면 0.0 (데이터가 없으면 안전값)
    """
    sub = _safe_float(submitted_ts, 0.0)
    sta = _safe_float(started_ts, 0.0)
    if sub <= 0.0 or sta <= 0.0:
        return 0.0
    return max(0.0, sta - sub)

def compute_jct_sec(submitted_ts: Any, end_ts: Any) -> float:
    """
    jct_sec = end_ts - submitted_ts
    - end_ts가 없으면 0.0 (아직 종료 안 했으니까)
    """
    sub = _safe_float(submitted_ts, 0.0)
    end = _safe_float(end_ts, 0.0)
    if sub <= 0.0 or end <= 0.0:
        return 0.0
    return max(0.0, end - sub)

def upsert_job_metrics_row(
    *,
    job_id: str,
    cluster: Optional[str],
    model: Optional[str],
    dataset: Optional[str],
    world_size: Any,
    submitted_ts: Any,
    started_ts: Any,
    end_ts: Any,
    status: str,
    queued_sec: Optional[Any] = None,
    jct_sec: Optional[Any] = None,
) -> None:
    """
    "upsert"라고 했지만 실제로는 append만 합니다.
    (정말 upsert가 필요하면 sqlite/pg로 가야 합니다.)
    중요한 건: row 스키마가 절대 밀리지 않게 "항상 동일 컬럼/타입"으로 찍는 것.
    """
    path = os.path.join(_log_dir(), "job_metrics.csv")

    sub_f = _safe_float(submitted_ts, 0.0)
    sta_f = _safe_float(started_ts, 0.0) if started_ts is not None else 0.0
    end_f = _safe_float(end_ts, 0.0) if end_ts is not None else 0.0

    # queued/jct는 외부에서 주면 그걸 쓰되, 숫자 보장
    if queued_sec is None:
        queued = compute_queued_sec(sub_f, sta_f)
    else:
        queued = _safe_float(queued_sec, 0.0)

    if jct_sec is None:
        jct = compute_jct_sec(sub_f, end_f)
    else:
        jct = _safe_float(jct_sec, 0.0)

    row = {
        "job_id": str(job_id),
        "cluster": str(cluster) if cluster is not None else "",
        "model": str(model) if model is not None else "",
        "dataset": str(dataset) if dataset is not None else "",
        "world_size": _safe_int(world_size, 0),

        # 숫자는 항상 float로 고정. 0이어도 0.000000으로 찍힘.
        "submitted_ts": f"{sub_f:.6f}",
        "started_ts": f"{sta_f:.6f}" if sta_f > 0.0 else "0.000000",
        "end_ts": f"{end_f:.6f}" if end_f > 0.0 else "0.000000",

        "queued_sec": f"{float(queued):.6f}",
        "jct_sec": f"{float(jct):.6f}",

        "status": (status or "").lower().strip(),
    }

    _atomic_append_dict_row(path, JOB_METRICS_FIELDS, row)