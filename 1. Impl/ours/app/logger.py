from __future__ import annotations

import os
import csv
import json
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

_RUNLOG = None

def _ts_str(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"

def _now() -> float:
    return float(time.time())

def init_run_logger(log_dir: str) -> None:
    global _RUNLOG
    if _RUNLOG is not None:
        try:
            if getattr(_RUNLOG, "paths", None) and _RUNLOG.paths.root == log_dir:
                return
        except Exception:
            pass
        raise RuntimeError(f"RunLogger already initialized (current={getattr(_RUNLOG.paths,'root',None)}, new={log_dir})")
    _RUNLOG = RunLogger(log_dir)

def get_run_logger() -> "RunLogger":
    global _RUNLOG
    if _RUNLOG is None:
        raise RuntimeError("RunLogger not initialized. Call init_run_logger(run_dir) at startup.")
    return _RUNLOG

@dataclass
class LogPaths:
    root: str
    scheduler_log: str
    job_events_csv: str
    queue_events_csv: str
    http_events_csv: str
    job_metrics_csv: str
    telemetry_csv: str
    csp_metrics_csv: str
    rebalance_log: str

class RunLogger:
    """
    ✅ 변경점 요약
    - job_events.csv: seq 컬럼 추가 (job별 event_seq 기록)
    - queue_events.csv: qlen_eligible/global/home/clusterq로 분리
    - http_events.csv: HTTP 요청/응답 분리 스트림
    - 기존 호출 호환: queue_event(queue_len=...) 호출해도 동작
    """

    def __init__(self, log_dir: str):
        _ensure_dir(log_dir)
        self.paths = LogPaths(
            root=log_dir,
            scheduler_log=os.path.join(log_dir, "scheduler.log"),
            job_events_csv=os.path.join(log_dir, "job_events.csv"),
            queue_events_csv=os.path.join(log_dir, "queue_events.csv"),
            http_events_csv=os.path.join(log_dir, "http_events.csv"),
            job_metrics_csv=os.path.join(log_dir, "job_metrics.csv"),
            telemetry_csv=os.path.join(log_dir, "telemetry.csv"),
            csp_metrics_csv=os.path.join(log_dir, "csp_metrics.csv"),
            rebalance_log=os.path.join(log_dir, "rebalance.log"),
        )
        self._lock = threading.Lock()
        self._seq_local: Dict[str, int] = {}  # seq 안 들어오면 로컬 fallback

        # 상태전이
        self._init_csv(self.paths.job_events_csv, [
            "ts", "seq", "event", "job_id", "cluster", "world_size", "note", "metadata_json"
        ])

        # 큐 조작
        # ✅ 헤더 순서: ts, seq, event, job_id, ...
        self._init_csv(self.paths.queue_events_csv, [
            "ts", "seq", "event", "job_id",
            "queue_len",
            "qlen_eligible", "qlen_global", "qlen_home", "qlen_clusterq",
            "note"
        ])

        # HTTP boundary
        self._init_csv(self.paths.http_events_csv, [
            "ts", "event", "job_id", "cluster",
            "op", "method", "url",
            "http_status", "ok", "req_id", "latency_ms",
            "req_json", "resp_json"
        ])

        # metrics
        self._init_csv(self.paths.job_metrics_csv, [
            "job_id", "cluster", "model", "dataset", "world_size",
            "submitted_ts", "started_ts", "end_ts", "queued_sec", "jct_sec",
            "final_accuracy", "status"
        ])

        self._init_csv(self.paths.telemetry_csv, [
            "ts", "node_id", "gpu_index", "gpu_util", "power_w", "mem_used_mb", "mem_total_mb"
        ])

        # ✅ csp_metrics: queue_len 의미를 global로 명시
        self._init_csv(self.paths.csp_metrics_csv, [
            "ts", "cluster", "qlen_global",
            "total_gpus", "used_gpus", "free_gpus",
            "util", "price_per_gpu_hour", "speed_factor",
            "energy_pressure", "cost_norm", "speed_norm"
        ])

        self._init_csv(self.paths.rebalance_log, [
            "ts", "action", "job_id", "cluster",
            "g_before", "g_after",
            "donor_job_id", "receiver_job_id",
            "reason", "delta_score", "meta_json"
        ])

    def _init_csv(self, path: str, header: List[str]) -> None:
        try:
            if os.path.exists(path):
                if os.path.getsize(path) == 0:
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(header)
                return
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)
        except Exception as e:
            raise RuntimeError(f"failed to init csv {path}: {e}") from e

    # -------------------------
    # scheduler.log
    # -------------------------
    def log_info(self, msg: str) -> None:
        now = time.time()
        ms = int((now - int(now)) * 1000)
        line = f"{_ts_str(now)},{ms:03d} [INFO] {msg}\n"
        with self._lock:
            with open(self.paths.scheduler_log, "a", encoding="utf-8") as f:
                f.write(line)

    def log_warn(self, msg: str) -> None:
        now = time.time()
        ms = int((now - int(now)) * 1000)
        line = f"{_ts_str(now)},{ms:03d} [WARN] {msg}\n"
        with self._lock:
            with open(self.paths.scheduler_log, "a", encoding="utf-8") as f:
                f.write(line)

    # -------------------------
    # seq fallback
    # -------------------------
    def _next_seq_local(self, job_id: str) -> int:
        job_id = str(job_id)
        cur = int(self._seq_local.get(job_id, 0))
        cur += 1
        self._seq_local[job_id] = cur
        return cur

    # -------------------------
    # job_events.csv
    # -------------------------
    def job_event(
        self,
        event: str,
        job_id: str,
        cluster: str,
        world_size: int,
        note: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
        seq: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        md: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            md.update(metadata)
        elif metadata is not None:
            md["metadata_raw"] = metadata

        for k, v in (kwargs or {}).items():
            if v is None:
                continue
            md[k] = v

        if seq is None:
            seq = self._next_seq_local(str(job_id))

        row = [
            _ts_str(ts),
            int(seq),
            str(event),
            str(job_id),
            str(cluster),
            int(world_size),
            str(note or ""),
            _safe_json(md),
        ]
        with self._lock:
            with open(self.paths.job_events_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    # -------------------------
    # queue_events.csv  ✅ 헤더/row 순서 정합성 고침
    # -------------------------
    def queue_event(
        self,
        event: str,
        job_id: str,
        queue_len: Optional[int] = None,  # 기존 호출 호환
        note: str = "",
        ts: Optional[float] = None,
        *,
        qlen_eligible: Optional[int] = None,
        qlen_global: Optional[int] = None,
        qlen_home: Optional[int] = None,
        qlen_clusterq: Optional[int] = None,
        seq: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        # 기본 queue_len은 eligible로 통일
        if qlen_eligible is None and queue_len is not None:
            qlen_eligible = int(queue_len)
        if queue_len is None and qlen_eligible is not None:
            queue_len = int(qlen_eligible)
        if queue_len is None:
            queue_len = -1

        # note extra (기존 호환)
        if kwargs:
            extra = " ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
            if extra:
                note = f"{note} {extra}".strip()

        # ✅ 헤더 순서: ts, seq, event, job_id, queue_len, qlen_eligible, qlen_global, qlen_home, qlen_clusterq, note
        row = [
            _ts_str(ts),
            int(seq) if seq is not None else -1,
            str(event),
            str(job_id),
            int(queue_len),
            int(qlen_eligible) if qlen_eligible is not None else -1,
            int(qlen_global) if qlen_global is not None else -1,
            int(qlen_home) if qlen_home is not None else -1,
            int(qlen_clusterq) if qlen_clusterq is not None else -1,
            str(note or ""),
        ]
        with self._lock:
            with open(self.paths.queue_events_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    # -------------------------
    # http_events.csv
    # -------------------------
    def http_event(
        self,
        event: str,
        job_id: str,
        cluster: str,
        *,
        op: str,
        method: str,
        url: str,
        http_status: int,
        ok: bool,
        req_id: str = "",
        latency_ms: float = 0.0,
        req: Optional[Dict[str, Any]] = None,
        resp: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> None:
        row = [
            _ts_str(ts),
            str(event),
            str(job_id),
            str(cluster),
            str(op),
            str(method),
            str(url),
            int(http_status),
            bool(ok),
            str(req_id or ""),
            float(latency_ms),
            _safe_json(req or {}),
            _safe_json(resp or {}),
        ]
        with self._lock:
            with open(self.paths.http_events_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    # -------------------------
    # metrics/telemetry
    # -------------------------
    def job_metrics_upsert_row(
        self,
        job_id: str,
        cluster: str,
        model: str,
        dataset: str,
        world_size: int,
        submitted_ts: float,
        started_ts: Optional[float],
        end_ts: Optional[float],
        queued_sec: Optional[float],
        jct_sec: Optional[float],
        final_accuracy: Optional[float],
        status: str,
    ) -> None:
        row = [
            job_id,
            cluster,
            model,
            dataset,
            int(world_size),
            f"{submitted_ts:.6f}" if submitted_ts is not None else "",
            f"{started_ts:.6f}" if started_ts is not None else "",
            f"{end_ts:.6f}" if end_ts is not None else "",
            f"{queued_sec:.6f}" if queued_sec is not None else "",
            f"{jct_sec:.6f}" if jct_sec is not None else "",
            f"{final_accuracy:.6f}" if final_accuracy is not None else "",
            status,
        ]
        with self._lock:
            with open(self.paths.job_metrics_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def telemetry_sample(
        self,
        node_id: str,
        gpu_index: int,
        gpu_util: float,
        power_w: float,
        mem_used_mb: float,
        mem_total_mb: float,
        ts: Optional[float] = None,
    ) -> None:
        row = [
            _ts_str(ts),
            node_id,
            int(gpu_index),
            float(gpu_util),
            float(power_w),
            float(mem_used_mb),
            float(mem_total_mb),
        ]
        with self._lock:
            with open(self.paths.telemetry_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def csp_metrics(
        self,
        cluster: str,
        qlen_global: int,
        total_gpus: int,
        used_gpus: int,
        free_gpus: int,
        util: float,
        price_per_gpu_hour: float,
        speed_factor: float,
        energy_pressure: float,
        cost_norm: float,
        speed_norm: float,
        ts: Optional[float] = None,
    ) -> None:
        row = [
            _ts_str(ts),
            cluster,
            int(qlen_global),
            int(total_gpus),
            int(used_gpus),
            int(free_gpus),
            float(util),
            float(price_per_gpu_hour),
            float(speed_factor),
            float(energy_pressure),
            float(cost_norm),
            float(speed_norm),
        ]
        with self._lock:
            with open(self.paths.csp_metrics_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

    def rebalance_event(
        self,
        action: str,
        job_id: str,
        cluster: str,
        g_before: int,
        g_after: int,
        donor_job_id: str = "",
        receiver_job_id: str = "",
        reason: str = "",
        delta_score: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> None:
        row = [
            _ts_str(ts),
            action,
            job_id,
            cluster,
            int(g_before),
            int(g_after),
            donor_job_id,
            receiver_job_id,
            reason,
            float(delta_score),
            _safe_json(meta or {}),
        ]
        with self._lock:
            with open(self.paths.rebalance_log, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)