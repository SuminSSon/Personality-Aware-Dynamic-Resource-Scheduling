from __future__ import annotations
from typing import Dict, Any, List, Optional
from .scheduler_state import SchedulerState
from .metrics import get_cluster_csp, compute_p_fair
from .profiling import build_compact_profiles
from .llm import generate_policy_card
from .core import zero_idle_fill, exchange_improve

class AdaptiveScheduler:
    """
    트리거:
      - admission(job) : 대기 등록 → 정책 생성 → 배치 시도
      - completion(job): 자원 회수 → zero-idle 채우기
      - rebalance()    : 주기(2분) → CSP 업데이트 → 만석 교환
    """
    def __init__(self, state: SchedulerState):
        self.state = state

    # ----- 내부 유틸 -----
    def _profiles_by_cluster(self, compact_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
        out = {}
        for row in compact_list:
            out[row["cluster"]] = {"g": row["g"], "sps": row["sps"], "cost": row["cost"]}
        return out

    def _select_cluster_for_job(self, job_id: str) -> str:
        """
        간단 정책: 남은 슬롯 여유가 큰 클러스터를 선호.
        같으면 clusterA 우선.
        """
        left = {c: self.state.capacity_left(c) for c in self.state.clusters}
        if not left:
            return "clusterA"
        # 최대 여유
        c_best = max(left.items(), key=lambda kv: (kv[1], kv[0]))[0]
        self.state.set_job_cluster(job_id, c_best)
        return c_best

    # ----- 트리거 -----
    def on_admission(self, job_id: str, model: str, dataset: str, user_request: str,
                     qA: int, nA: int, qB: int, nB: int) -> Dict[str, Any]:
        """
        1) 프로파일 생성(2줄)
        2) CSP 생성
        3) LLM 정책 카드
        4) 상태 등록 + Zero-idle → 교환
        """
        compact = build_compact_profiles(model=model, dataset=dataset)
        cspA = get_cluster_csp("clusterA", qA, nA)
        cspB = get_cluster_csp("clusterB", qB, nB)
        payload = {
            "job_id": job_id,
            "user_request": user_request,
            "profiling_compact": compact,
            "csp": [cspA, cspB],
        }
        card = generate_policy_card(payload)
        eta = float(card["η_perf"])
        prof_by_cluster = self._profiles_by_cluster(compact)
        self.state.admit_job(job_id, prof_by_cluster, card, eta)

        # 초기 라우팅(여유 큰 클러스터)
        c_sel = self._select_cluster_for_job(job_id)

        # Zero-idle fill
        pf = compute_p_fair(sum(self.state.assign[c_sel].values()) + len(self.state.queue), self.state.clusters[c_sel]["slots_total"])
        fill = zero_idle_fill(self.state, c_sel, p_fair=pf)

        # 만석 교환
        pf2 = compute_p_fair(len(self.state.queue), self.state.clusters[c_sel]["slots_total"])
        exch = exchange_improve(self.state, c_sel, p_fair=pf2)

        return {
            "event": "admission",
            "job_id": job_id,
            "cluster": c_sel,
            "policy_card": card,
            "zero_idle": fill,
            "exchange": exch,
            "state": self.state.summary(),
        }

    def on_completion(self, job_id: str, qA: int, nA: int, qB: int, nB: int) -> Dict[str, Any]:
        removed = self.state.finish_job(job_id)

        # 빈 슬롯 생겼을 가능성 → 각 클러스터 zero-idle
        results = {}
        for c in self.state.clusters:
            pf = compute_p_fair(len(self.state.queue), self.state.clusters[c]["slots_total"])
            results[c] = zero_idle_fill(self.state, c, p_fair=pf)

        return {
            "event": "completion",
            "job_id": job_id,
            "removed": removed,
            "zero_idle_all": results,
            "state": self.state.summary(),
        }

    def on_rebalance(self, qA: int, nA: int, qB: int, nB: int) -> Dict[str, Any]:
        # 리밸런스 시: 각 클러스터에 대해 ΔU>0 교환만
        results = {}
        for c in self.state.clusters:
            pf = compute_p_fair(len(self.state.queue), self.state.clusters[c]["slots_total"])
            results[c] = exchange_improve(self.state, c, p_fair=pf)
        return {
            "event": "rebalance",
            "exchange_all": results,
            "state": self.state.summary(),
        }

