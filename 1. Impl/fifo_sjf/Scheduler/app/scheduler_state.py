from __future__ import annotations
from typing import Dict, Deque, List, Optional, Any, Tuple
from collections import deque
import threading
import time

class SchedulerState:
    """
    전역 스케줄러 상태:
      - clusters: {name: {"slots_total": int, "slots_used": int}}
      - assign: {cluster: {job_id: g}}   # 각 잡에 할당된 GPU 수
      - queue: Deque[str]                # 대기 잡 (job_id)
      - jobs:  {job_id: {"policy": {...}, "profiles": {cluster: {"g":[..],"sps":[..],"cost":[..]}},
                         "eta": float, "arrival_ts": float, "cluster": Optional[str]}}
    thread-safe 보장을 위해 간단한 Lock 사용.
    """
    def __init__(self, cluster_slots: Dict[str, int]):
        self._lock = threading.Lock()
        self.clusters: Dict[str, Dict[str, int]] = {
            c: {"slots_total": n, "slots_used": 0} for c, n in cluster_slots.items()
        }
        self.assign: Dict[str, Dict[str, int]] = {c: {} for c in cluster_slots}
        self.queue: Deque[str] = deque()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.policy_cache: Dict[str, Dict[str, Any]] = {}  # job_id -> policy card (최근)
        self.last_summary: Dict[str, Any] = {}

    # ---------- 기본 유틸 ----------
    def lock(self):
        return self._lock

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "clusters": self.clusters.copy(),
                "assign": {c: a.copy() for c, a in self.assign.items()},
                "queue": list(self.queue),
                "jobs": {jid: {"eta": j["eta"], "cluster": j.get("cluster")} for jid, j in self.jobs.items()},
            }

    def capacity_left(self, cluster: str) -> int:
        c = self.clusters[cluster]
        return max(0, c["slots_total"] - c["slots_used"])

    def slots_used(self, cluster: str) -> int:
        return self.clusters[cluster]["slots_used"]

    def set_assign(self, cluster: str, job_id: str, g: int):
        """클러스터 내 잡의 g를 설정(증감에 따라 slots_used 갱신)."""
        prev = self.assign[cluster].get(job_id, 0)
        if prev == g:
            return
        delta = g - prev
        self.assign[cluster][job_id] = g
        self.clusters[cluster]["slots_used"] += delta
        if self.clusters[cluster]["slots_used"] < 0:
            self.clusters[cluster]["slots_used"] = 0

    # ---------- 잡 수명주기 ----------
    def admit_job(self, job_id: str, profiles_by_cluster: Dict[str, Dict[str, List[float]]],
                  policy_card: Dict[str, Any], eta: float):
        with self._lock:
            self.jobs[job_id] = {
                "policy": policy_card,
                "profiles": profiles_by_cluster,
                "eta": float(eta),
                "arrival_ts": time.time(),
                "cluster": None,
            }
            self.queue.append(job_id)

    def finish_job(self, job_id: str) -> Dict[str, Any]:
        """잡 종료: 자원 회수 및 상태 정리."""
        with self._lock:
            # 어떤 클러스터에서 돌고 있었는지 찾음
            for c in self.assign:
                g = self.assign[c].pop(job_id, None)
                if g is not None:
                    self.clusters[c]["slots_used"] -= g
                    if self.clusters[c]["slots_used"] < 0:
                        self.clusters[c]["slots_used"] = 0
            # 큐에서 제거
            try:
                self.queue.remove(job_id)
            except ValueError:
                pass
            meta = self.jobs.pop(job_id, None)
            return {"removed": job_id, "meta": meta}

    # ---------- 헬퍼 ----------
    def set_job_cluster(self, job_id: str, cluster: str):
        if job_id in self.jobs:
            self.jobs[job_id]["cluster"] = cluster

    def get_job_profiles(self, job_id: str, cluster: str) -> Optional[Dict[str, List[float]]]:
        j = self.jobs.get(job_id)
        if not j:
            return None
        prof = j["profiles"].get(cluster)
        return prof

    def list_jobs_on_cluster(self, cluster: str) -> List[str]:
        return [jid for jid, g in self.assign[cluster].items() if g > 0]

    def all_jobs(self) -> List[str]:
        return list(self.jobs.keys())

