# app/queue.py
from __future__ import annotations

import time
import logging
from typing import Optional, List, Dict, Any

log = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


class QueueJob:
    """
    ClusterQueue 내부에서만 쓰는 lightweight wrapper.
    (중요) global_queue에는 절대 들어가지 않습니다. global_queue는 job_id(str)만 유지하세요.
    """

    def __init__(
        self,
        job_id: str,
        g_req: Optional[int] = None,
        user_id: Optional[str] = None,
        submit_ts: Optional[float] = None,
        enqueue_seq: int = 0,
        seq: Optional[int] = None,
        **kwargs: Any,
    ):
        self.job_id = str(job_id)

        # ---- g_req infer ----
        inferred_g = None
        for k in ("world_size", "g", "gpu", "gpu_count", "g_target", "min_g", "g_min"):
            if k in kwargs and kwargs[k] is not None:
                inferred_g = kwargs[k]
                break
        if g_req is None and inferred_g is not None:
            g_req = inferred_g

        try:
            self.g_req = max(1, int(g_req)) if g_req is not None else 1
        except Exception:
            self.g_req = 1
            log.warning("[QueueJob] g_req invalid for job_id=%s -> default=1", self.job_id)

        # optional meta (있으면 채우고, 없어도 동작해야 함)
        self.user_id = user_id
        self.submit_ts = float(submit_ts if submit_ts is not None else _now())

        # seq 호환
        if seq is not None:
            try:
                self.enqueue_seq = int(seq)
            except Exception:
                self.enqueue_seq = int(enqueue_seq)
        else:
            self.enqueue_seq = int(enqueue_seq)

        # 아래는 “있으면 기록” 정도 (없어도 됨)
        self.model = str(kwargs.get("model") or kwargs.get("model_name") or "")
        self.dataset = str(kwargs.get("dataset") or "")
        try:
            self.g_target = int(kwargs.get("g_target") or self.g_req)
        except Exception:
            self.g_target = int(self.g_req)

        self.policy = kwargs.get("policy")
        self.pinned_cluster = kwargs.get("pinned_cluster") or kwargs.get("pinned_cluster_id")
        self.admitted_cluster_id = kwargs.get("admitted_cluster_id")
        self.home_cluster_id = kwargs.get("home_cluster_id")

        self.is_gang = bool(kwargs.get("is_gang") or (self.g_req == 4))

    @property
    def seq(self) -> int:
        return int(self.enqueue_seq)

    @seq.setter
    def seq(self, v: int) -> None:
        try:
            self.enqueue_seq = int(v)
        except Exception:
            self.enqueue_seq = 0


class ClusterQueue:
    def __init__(self, cluster_id: str):
        self.cluster_id = str(cluster_id)
        self._jobs: List[QueueJob] = []
        self.eta_dirty: bool = True
        self._hol_eta_ts: Optional[float] = None

    @property
    def jobs(self) -> List[QueueJob]:
        return self._jobs

    def __len__(self) -> int:
        return len(self._jobs)

    def __iter__(self):
        return iter(self._jobs)

    def __contains__(self, job_id: str) -> bool:
        jid = str(job_id)
        for x in self._jobs:
            if str(getattr(x, "job_id", "")) == jid:
                return True
        return False

    def enqueue(self, qj: QueueJob) -> None:
        if qj is None:
            return
        jid = str(getattr(qj, "job_id", "") or "")
        if not jid:
            return

        # 중복 방지
        for x in self._jobs:
            if str(getattr(x, "job_id", "")) == jid:
                return

        self._jobs.append(qj)

        def _key(x: QueueJob):
            try:
                seq = int(getattr(x, "enqueue_seq", 0) or 0)
            except Exception:
                seq = 0
            try:
                ts = float(getattr(x, "submit_ts", 0.0) or 0.0)
            except Exception:
                ts = 0.0
            return (seq, ts)

        self._jobs.sort(key=_key)
        self.eta_dirty = True

    def remove(self, job_id: str) -> None:
        jid = str(job_id)
        before = len(self._jobs)
        self._jobs = [x for x in self._jobs if str(getattr(x, "job_id", "")) != jid]
        if len(self._jobs) != before:
            self.eta_dirty = True

    def reinsert_by_seq(self, qj: QueueJob) -> None:
        if qj is None:
            return
        jid = str(getattr(qj, "job_id", "") or "")
        if not jid:
            return

        for x in self._jobs:
            if str(getattr(x, "job_id", "")) == jid:
                return

        try:
            seq_new = int(getattr(qj, "enqueue_seq", 0) or 0)
        except Exception:
            seq_new = 0

        insert_pos = len(self._jobs)
        for i, x in enumerate(self._jobs):
            try:
                seq_x = int(getattr(x, "enqueue_seq", 0) or 0)
            except Exception:
                seq_x = 0
            if seq_new < seq_x:
                insert_pos = i
                break

        self._jobs.insert(insert_pos, qj)
        self.eta_dirty = True

    def pop_at(self, idx: int):
        return self._jobs.pop(idx)


def make_status_queue_view(state: Any) -> dict:
    """
    /status에서 큐 상태를 안전하게 보여주기 위한 뷰.
    원칙: state.global_queue는 job_id(str) 리스트.
    """
    raw = getattr(state, "global_queue", None)
    q = list(raw or [])

    jobs_map = getattr(state, "jobs", {}) or {}

    jobs: List[str] = []
    detail: List[Dict[str, Any]] = []

    for it in q:
        # global_queue는 원칙적으로 str이지만, 레거시가 섞여도 깨지지 않게 방어
        if isinstance(it, str):
            jid = it
        else:
            jid = getattr(it, "job_id", None)
            if jid is None:
                jid = str(it)

        jid = str(jid or "")
        if not jid:
            continue

        jobs.append(jid)

        jr = jobs_map.get(jid)
        if jr is None:
            detail.append({"job_id": jid})
            continue

        def _gi(v, default=0) -> int:
            try:
                return int(v)
            except Exception:
                return int(default)

        def _gf(v, default=0.0) -> float:
            try:
                return float(v)
            except Exception:
                return float(default)

        g_target = _gi(getattr(jr, "g_target", None), 1)
        g_req = _gi(getattr(jr, "g_req", None), 0)
        if g_req <= 0:
            g_req = g_target if g_target > 0 else 1

        detail.append(
            {
                "job_id": jid,
                "model": getattr(jr, "model", None),
                "dataset": getattr(jr, "dataset", None),
                "status": getattr(jr, "status", None),
                "g_req": int(g_req),
                "g_target": int(g_target),
                "submit_ts": _gf(getattr(jr, "submit_ts", None), 0.0),
                "queue_seq": _gi(getattr(jr, "queue_seq", None), 0),
                "queue_kind": getattr(jr, "queue_kind", None),
                "queue_cluster_id": getattr(jr, "queue_cluster_id", None),
                "admitted_cluster_id": getattr(jr, "admitted_cluster_id", None),
                "pinned_cluster": getattr(jr, "pinned_cluster", None),
                "preempt_count": _gi(getattr(jr, "preempt_count", None), 0),
            }
        )

    return {"len": len(jobs), "jobs": jobs, "jobs_detail": detail}