from __future__ import annotations
from typing import Dict, List, Any
import math

# -----------------------------
# λ 계산 및 효용/한계효용
# -----------------------------
def compute_lambda(eta_perf: float, p_fair: float) -> Dict[str, float]:
    eta = max(0.0, min(1.0, float(eta_perf)))
    pf  = max(0.0, min(1.0, float(p_fair)))
    return {"time": eta * (1.0 - pf), "cost": (1.0 - eta) * (1.0 - pf), "fair": pf}

def _norm_list(xs: List[float]) -> List[float]:
    m = max(xs) if xs else 0.0
    return [x / m if m > 0 else 0.0 for x in xs]

def utility_for_g(lam: Dict[str, float], sps: List[float], cost: List[float], g: int) -> float:
    """U(0)=0, g>=1부터 정규화 점수 사용."""
    if g <= 0:
        return 0.0
    idx = min(g, len(sps)) - 1
    sps_n  = _norm_list(sps)
    cost_n = _norm_list(cost)
    return lam["time"] * sps_n[idx] - lam["cost"] * cost_n[idx] + lam["fair"]

def mu_add(lam: Dict[str, float], sps: List[float], cost: List[float], g: int) -> float:
    if g >= len(sps):
        return -1e9
    return utility_for_g(lam, sps, cost, g + 1) - utility_for_g(lam, sps, cost, g)

def mu_drop(lam: Dict[str, float], sps: List[float], cost: List[float], g: int) -> float:
    if g <= 1:
        return 1e9
    return utility_for_g(lam, sps, cost, g) - utility_for_g(lam, sps, cost, g - 1)

# -----------------------------
# Zero-idle 채우기 / 교환 (ΔU>0)
# -----------------------------
def zero_idle_fill(state, cluster: str, p_fair: float) -> Dict[str, Any]:
    """
    빈 슬롯이 존재하면 MU_add 최대인 잡에 +1씩 할당(ΔU>0에서만).
    state.assign[cluster][job] = g 를 직접 갱신.
    """
    changes = []
    with state.lock():
        slots_left = state.capacity_left(cluster)
        if slots_left <= 0:
            return {"filled": 0, "changes": []}

        # 대표 η: 해당 클러스터에 라우팅된 잡의 평균
        etas = [state.jobs[j]["eta"] for j in state.all_jobs()
                if state.jobs[j].get("cluster") in (cluster, None)]
        eta = sum(etas) / len(etas) if etas else 0.5
        lam = compute_lambda(eta, p_fair)

        candidates = list(state.queue) + state.list_jobs_on_cluster(cluster)
        for jid in candidates:
            if state.jobs.get(jid) and state.jobs[jid].get("cluster") is None:
                state.set_job_cluster(jid, cluster)

        while slots_left > 0:
            best_j, best_gain = None, -1e9
            for jid in candidates:
                prof = state.get_job_profiles(jid, cluster)
                if not prof:
                    continue
                g_now = state.assign[cluster].get(jid, 0)
                gain = mu_add(lam, prof["sps"], prof["cost"], g_now)
                if gain > best_gain:
                    best_gain, best_j = gain, jid

            if best_j is None or best_gain <= 0:
                break

            g_now = state.assign[cluster].get(best_j, 0)
            state.set_assign(cluster, best_j, g_now + 1)
            try:
                state.queue.remove(best_j)
            except ValueError:
                pass
            slots_left -= 1
            changes.append({"job": best_j, "cluster": cluster, "delta": +1, "gain": round(best_gain, 4)})

    return {"filled": len(changes), "changes": changes}

def exchange_improve(state, cluster: str, p_fair: float, max_iters: int = 32) -> Dict[str, Any]:
    """
    만석일 때 (수혜자 j, 기부자 k) 페어 탐색. ΔU = MU_add(j) - MU_drop(k) > 0 인 경우 교환.
    """
    steps = []
    with state.lock():
        if state.capacity_left(cluster) > 0:
            return {"moves": steps}

        etas = [state.jobs[j]["eta"] for j in state.list_jobs_on_cluster(cluster)]
        eta = sum(etas) / len(etas) if etas else 0.5
        lam = compute_lambda(eta, p_fair)

        for _ in range(max_iters):
            jobs_on = state.list_jobs_on_cluster(cluster)
            if not jobs_on and not state.queue:
                break
            best = (None, None, -1e9)  # (j,k,ΔU)

            # 수혜자 후보: 기존 잡 + 큐
            for j in jobs_on + list(state.queue):
                prof_j = state.get_job_profiles(j, cluster)
                if not prof_j:
                    continue
                g_j = state.assign[cluster].get(j, 0)
                gain = mu_add(lam, prof_j["sps"], prof_j["cost"], g_j)
                if gain <= 0:
                    continue
                # 기부자 후보: g_k > 1
                for k in jobs_on:
                    if k == j:
                        continue
                    g_k = state.assign[cluster].get(k, 0)
                    if g_k <= 1:
                        continue
                    prof_k = state.get_job_profiles(k, cluster)
                    if not prof_k:
                        continue
                    loss = mu_drop(lam, prof_k["sps"], prof_k["cost"], g_k)
                    dU = gain - loss
                    if dU > best[2]:
                        best = (j, k, dU)

            j, k, dU = best
            if dU <= 0 or j is None or k is None:
                break

            g_k = state.assign[cluster].get(k, 0)
            g_j = state.assign[cluster].get(j, 0)
            state.set_assign(cluster, k, g_k - 1)
            state.set_assign(cluster, j, g_j + 1)
            try:
                state.queue.remove(j)
            except ValueError:
                pass
            steps.append({"to": j, "from": k, "cluster": cluster, "deltaU": round(dU, 4)})

    return {"moves": steps}

