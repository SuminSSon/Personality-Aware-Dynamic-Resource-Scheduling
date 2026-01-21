# app/rebalance.py
from __future__ import annotations

import time, math, inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.executor import resize_job, launch_or_reuse
from app.scheduler_state import STATE_LOCK
from app.backfill_policy import get_global_state, _score_S_jcg, _recompute_cluster_counters_locked, _compute_f_fair_all_clusters, MU_FAIR, TIME_REF, COST_REF, ENERGY_REF
from app.profiling import get_profiling_entry


logger = logging.getLogger(__name__)

# Config
REB_PERIOD_SEC = float(__import__("os").getenv("OURS_REBALANCE_PERIOD_SEC", "180.0"))

# Module-level tick guard
_REB_LOCK = None
_REB_LAST_TS = 0.0


def _ensure_globals():
    global _REB_LOCK, _REB_LAST_TS
    if _REB_LOCK is None:
        import threading
        _REB_LOCK = threading.Lock()
        _REB_LAST_TS = 0.0

def rebalance_on_free_gpus_event(
    state: Any,
    cid: str,
    free_nodes: List[str],
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    _ensure_globals()
    now = float(now_ts or time.time())

    # 이벤트는 lock이 잡혀있으면 스킵(중복 이벤트 폭주 방지)
    if not _REB_LOCK.acquire(blocking=False):
        return {"ok": True, "skipped": True, "reason": "lock_busy", "cid": str(cid), "free_n": len(free_nodes)}

    try:
        return _event_driven_use_free_nodes_once(state=state, cid=str(cid), free_nodes=list(free_nodes), now_ts=now)
    except Exception:
        logger.exception("[rebalance] event failed cid=%s", cid)
        return {"ok": False, "error": "exception_in_event", "cid": str(cid)}
    finally:
        try:
            _REB_LOCK.release()
        except Exception:
            pass

def _periodic_reallocation_once(state: Any, now_ts: float) -> Dict[str, Any]:
    # 0) snapshot: clusters, running jobs
    clusters = getattr(state, "clusters", {}) or {}
    jobs_map = getattr(state, "jobs", {}) or {}

    # 1) cluster별로 한 번씩만 시도(너무 공격적으로 하지 않기)
    actions: List[Dict[str, Any]] = []
    examined = 0

    for cid0, cr in (clusters or {}).items():
        cid = str(cid0)
        # running jobs detail이 있으면 그걸 우선 사용 (없으면 jobs_map에서 RUNNING 필터)
        running = _get_running_jobs_in_cluster(state, cid, cr, jobs_map)

        # donor/receiver 후보 만들기
        donors = _candidate_donors(running)
        receivers = _candidate_receivers(running)

        examined += (len(donors) + len(receivers))
        if not donors or not receivers:
            continue

        best = _pick_best_transfer_pair(state, cid, donors, receivers, cr)
        if not best:
            continue

        # apply: donor -1, receiver +1 (가능하면 같은 클러스터 내에서만)
        ok, info = _apply_transfer_pair(state, cid, best, cr, now_ts)
        actions.append({"cid": cid, "ok": bool(ok), "info": info})

    return {
        "ok": True,
        "mode": "periodic",
        "ts": float(now_ts),
        "clusters_n": int(len(clusters or {})),
        "examined": int(examined),
        "actions": actions,
    }

def _event_driven_use_free_nodes_once(state: Any, cid: str, free_nodes: List[str], now_ts: float) -> Dict[str, Any]:
    if not free_nodes:
        return {"ok": True, "mode": "event", "cid": cid, "skipped": True, "reason": "no_free_nodes", "ts": now_ts}

    clusters = getattr(state, "clusters", {}) or {}
    jobs_map = getattr(state, "jobs", {}) or {}
    cr = (clusters or {}).get(cid)

    running = _get_running_jobs_in_cluster(state, cid, cr, jobs_map)
    receivers = _candidate_receivers(running)
    if not receivers:
        return {"ok": True, "mode": "event", "cid": cid, "ts": now_ts, "free_n": len(free_nodes), "action": "no_receivers"}

    # 가장 ΔU_gain 큰 receiver부터 1GPU씩 늘리는 방향(단, hook에서 실패할 수 있음)
    receivers_sorted = sorted(
        receivers,
        key=lambda jr: _delta_u_gain(state, cid, jr, cr),
        reverse=True,
    )

    used = 0
    acts: List[Dict[str, Any]] = []
    for r in receivers_sorted:
        if used >= len(free_nodes):
            break
        # 1 GPU 확장
        ok, info = _apply_resize_plus1(state, cid, r, cr, now_ts, free_nodes[used])
        acts.append({"job_id": str(r.get("job_id")), "ok": bool(ok), "info": info})
        if ok:
            used += 1

    return {
        "ok": True,
        "mode": "event",
        "cid": cid,
        "ts": float(now_ts),
        "free_n": int(len(free_nodes)),
        "used": int(used),
        "actions": acts,
        "note": "queue/backfill decision hook will be added later",
    }

def _candidate_donors(running: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for jr in running:
        g_cur = int(jr.get("g_cur") or 0)
        g_tgt = int(jr.get("g_target") or 0)
        g_min = int(jr.get("g_min") or 1)
        if g_cur > max(g_tgt, g_min) and g_cur > 1:
            out.append(jr)
    return out

def _candidate_receivers(running: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for jr in running:
        g_cur = int(jr.get("g_cur") or 0)
        g_tgt = int(jr.get("g_target") or 0)
        g_max = int(jr.get("g_max") or 4)  # 시스템 상한/잡 상한 중 작은 값을 나중에 넣으면 됨
        if g_tgt > 0 and g_cur < min(g_tgt, g_max):
            out.append(jr)
    return out

def _pick_best_transfer_pair(
    state: Any,
    cid: str,
    donors: List[Dict[str, Any]],
    receivers: List[Dict[str, Any]],
    cr: Any,
) -> Optional[Dict[str, Any]]:
    best = None
    best_net = 0.0

    for d in donors:
        loss = _delta_u_loss(state, cid, d, cr)
        for r in receivers:
            gain = _delta_u_gain(state, cid, r, cr)
            net = float(gain) - float(loss)
            if net > best_net:
                best_net = net
                best = {"donor": d, "receiver": r, "net": best_net, "gain": gain, "loss": loss}

    if best is None:
        return None
    # strictly positive만 허용
    if float(best.get("net") or 0.0) <= 0.0:
        return None
    return best

# Utility ΔU (marginal)
def _delta_u_loss(state: Any, cid: str, jobr: Dict[str, Any], cr: Any) -> float:
    g_cur = int(jobr.get("g_cur") or 1)
    if g_cur <= 1:
        return 1e18
    # U(j, g_cur) - U(j, g_cur-1)
    u1 = _utility_U(state, cid, jobr, cr, g_cur)
    u0 = _utility_U(state, cid, jobr, cr, g_cur - 1)
    return float(u1 - u0)

def _delta_u_gain(state: Any, cid: str, jobr: Dict[str, Any], cr: Any) -> float:
    g_cur = int(jobr.get("g_cur") or 1)
    # U(j, g_cur+1) - U(j, g_cur)
    u1 = _utility_U(state, cid, jobr, cr, g_cur + 1)
    u0 = _utility_U(state, cid, jobr, cr, g_cur)
    return float(u1 - u0)

def _utility_U(state: Any, cid: str, jobr: Dict[str, Any], cr: Any, g: int) -> float:
    # ---- required fields ----
    policy = jobr.get("policy") or {}
    price = float(_get_attr(cr, "price_per_gpu_hour", 1.0) or 1.0)
    f_fair = float(jobr.get("f_fair_c", 1.0) or 1.0)

    # profiling lookup (hook)
    prof = _hook_get_profiling_for(jobr, cid=cid, g=int(g))
    sps = float((prof or {}).get("throughput_sps") or 0.0)
    if sps <= 0.0:
        # profiling 없으면 매우 보수적으로: 0점 취급
        return -1e18

    # energy: avg_power_w_per_gpu * g 가 "예측 pcur" (cluster telemetry 말고 profiling 기반)
    pavg = float((prof or {}).get("avg_power_w_per_gpu") or 0.0)
    pcur_job = float(pavg) * float(max(1, int(g)))

    # scorer hook (너의 _score_S_jcg에 연결)
    return float(_hook_score(policy=policy, sps=sps, g=int(g), price_per_gpu_hour=price, pcur_w=pcur_job, f_fair_c=f_fair))

def _apply_transfer_pair(state: Any, cid: str, best: Dict[str, Any], cr: Any, now_ts: float) -> Tuple[bool, Dict[str, Any]]:
    d = best["donor"]
    r = best["receiver"]

    ok1, info1 = _apply_resize_minus1(state, cid, d, cr, now_ts)
    if not ok1:
        return False, {"stage": "donor_minus1_failed", "donor": d.get("job_id"), "info": info1, "best": best}

    ok2, info2 = _apply_resize_plus1(state, cid, r, cr, now_ts, node_hint=None)
    if not ok2:
        # 롤백을 하려면 여기서 donor +1 복구가 필요(지금 뼈대에서는 생략)
        return False, {"stage": "receiver_plus1_failed", "receiver": r.get("job_id"), "info": info2, "best": best}

    return True, {"stage": "ok", "donor": d.get("job_id"), "receiver": r.get("job_id"), "best": best}

def _apply_resize_minus1(state: Any, cid: str, jobr: Dict[str, Any], cr: Any, now_ts: float) -> Tuple[bool, Dict[str, Any]]:
    job_id = str(jobr.get("job_id") or "")
    g_cur = int(jobr.get("g_cur") or 1)
    g_new = max(1, g_cur - 1)
    if g_new == g_cur:
        return False, {"reason": "cannot_shrink", "job_id": job_id, "g_cur": g_cur}

    return _hook_resize_job(
        state=state,
        jobr=jobr,
        job_id=job_id,
        cid=cid,
        new_g=g_new,
        reason="REBALANCE_SHRINK",
        now_ts=now_ts,
        node_hint=None,
    )

def _apply_resize_plus1(
    state: Any,
    cid: str,
    jobr: Dict[str, Any],
    cr: Any,
    now_ts: float,
    node_hint: Optional[str],
) -> Tuple[bool, Dict[str, Any]]:
    job_id = str(jobr.get("job_id") or "")
    g_cur = int(jobr.get("g_cur") or 1)
    g_max = int(jobr.get("g_max") or 4)
    g_new = min(g_max, g_cur + 1)
    if g_new == g_cur:
        return False, {"reason": "cannot_grow", "job_id": job_id, "g_cur": g_cur, "g_max": g_max}

    return _hook_resize_job(
        state=state,
        jobr=jobr,
        job_id=job_id,
        cid=cid,
        new_g=g_new,
        reason="REBALANCE_GROW",
        now_ts=now_ts,
        node_hint=node_hint,
    )

# Running jobs snapshot helpers
def _get_running_jobs_in_cluster(state: Any, cid: str, cr: Any, jobs_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # 1) cluster.running_jobs_detail
    try:
        rjd = _get_attr(cr, "running_jobs_detail", None)
        if isinstance(rjd, list) and rjd:
            for x in rjd:
                if not isinstance(x, dict):
                    continue
                out.append(_normalize_job_rec(x))
            return out
    except Exception:
        pass

    # 2) jobs_map fallback
    for jid, jr in (jobs_map or {}).items():
        try:
            st = str(_get_attr(jr, "status", "") or "").upper()
            if st != "RUNNING":
                continue
            jcid = str(_get_attr(jr, "cluster", "") or _get_attr(jr, "assigned_cluster", "") or "")
            if jcid != str(cid):
                continue
            rec = {
                "job_id": str(jid),
                "g_cur": int(_get_attr(jr, "g_cur", 0) or _get_attr(jr, "world_size", 0) or 1),
                "g_target": int(_get_attr(jr, "g_target", 0) or 1),
                "g_min": int(_get_attr(jr, "g_min", 1) or 1),
                "g_max": int(_get_attr(jr, "g_max", 4) or 4),
                "policy": _get_attr(jr, "policy", {}) or {},
                "model": _get_attr(jr, "model", None) or _get_attr(jr, "model_name", None),
                "dataset": _get_attr(jr, "dataset", None),
            }
            out.append(rec)
        except Exception:
            continue

    return out

def _normalize_job_rec(x: Dict[str, Any]) -> Dict[str, Any]:
    # running_jobs_detail에 뭐가 있든 최소 키로 normalize
    job_id = str(x.get("job_id") or x.get("id") or "")
    return {
        "job_id": job_id,
        "g_cur": int(x.get("g_cur") or x.get("world_size") or 1),
        "g_target": int(x.get("g_target") or 1),
        "g_min": int(x.get("g_min") or 1),
        "g_max": int(x.get("g_max") or 4),
        "policy": x.get("policy") or {},
        "model": x.get("model") or x.get("model_name"),
        "dataset": x.get("dataset"),
        # fairness는 cluster별로 계산한 값을 job에 붙여둘 수도 있음(없으면 1.0)
        "f_fair_c": float(x.get("f_fair_c") or 1.0),
    }

def _get_attr(obj: Any, k: str, default=None):
    try:
        if isinstance(obj, dict):
            return obj.get(k, default)
        return getattr(obj, k, default)
    except Exception:
        return default

def _get_current_nodes_for_job(state: Any, jobr: Dict[str, Any], job_id: str) -> List[str]:
    # 1) jobr에 nodes가 있으면 우선
    nodes = jobr.get("nodes")
    if isinstance(nodes, list) and nodes:
        return [str(n) for n in nodes if n]

    # 2) state.jobs[job_id].nodes fallback
    try:
        jobs = getattr(state, "jobs", {}) or {}
        jr = jobs.get(job_id)
        if jr is not None:
            ns = getattr(jr, "nodes", None)
            if isinstance(ns, list) and ns:
                return [str(n) for n in ns if n]
            if isinstance(jr, dict):
                ns2 = jr.get("nodes")
                if isinstance(ns2, list) and ns2:
                    return [str(n) for n in ns2 if n]
    except Exception:
        pass

    return []

def _free_nodes_in_cluster_ssot(state: Any, cid: str, cr: Any) -> List[str]:
    # SSOT helper가 있으면 그걸 사용
    try:
        fn = globals().get("_free_nodes_in_cluster", None)
        if callable(fn):
            out = fn(state, str(cid)) or []
            return [str(n) for n in out if n]
    except Exception:
        pass

    # fallback: cluster.nodes + node_owner
    nodes: List[str] = []
    try:
        if isinstance(cr, dict):
            nodes = [str(n) for n in (cr.get("nodes") or []) if n]
        else:
            nodes = [str(n) for n in (getattr(cr, "nodes", None) or []) if n]
    except Exception:
        nodes = []

    owner = getattr(state, "node_owner", {}) or {}
    out: List[str] = []
    for n in nodes:
        if not owner.get(n):
            out.append(n)
    return out

def _hook_resize_job(
    *,
    state: Any,
    jobr: Dict[str, Any],
    job_id: str,
    cid: str,
    new_g: int,
    reason: str,
    now_ts: float,
    node_hint: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    # (A) 현재 노드 목록 확보
    cur_nodes = _get_current_nodes_for_job(state, jobr, job_id)
    cur_nodes = [str(n) for n in cur_nodes if n]

    if not cur_nodes:
        return False, {"reason": "missing_current_nodes", "job_id": job_id, "cid": cid}

    cur_g = int(len(cur_nodes))
    new_g = int(new_g)

    if new_g <= 0:
        return False, {"reason": "invalid_new_g", "job_id": job_id, "cid": cid, "new_g": new_g}

    if new_g == cur_g:
        return True, {"ok": True, "reason": "no_change", "job_id": job_id, "cid": cid, "g": cur_g}

    # (B) 최종 new_nodes 만들기
    if new_g < cur_g:
        # shrink: 뒤에서부터 줄이기(단순)
        new_nodes = cur_nodes[:new_g]

    else:
        # grow: free node 하나(또는 여러 개) 추가해야 함
        clusters = getattr(state, "clusters", {}) or {}
        cr = (clusters or {}).get(str(cid))

        free = _free_nodes_in_cluster_ssot(state, str(cid), cr)
        free_set = set(free) - set(cur_nodes)

        # node_hint 있으면 우선 사용
        add_list: List[str] = []
        if node_hint and node_hint in free_set:
            add_list.append(str(node_hint))

        need = new_g - cur_g
        if len(add_list) < need:
            for n in free:
                if n in free_set and n not in add_list:
                    add_list.append(n)
                if len(add_list) >= need:
                    break

        if len(add_list) < need:
            return False, {
                "reason": "not_enough_free_nodes_for_grow",
                "job_id": job_id,
                "cid": cid,
                "cur_g": cur_g,
                "new_g": new_g,
                "free_n": len(free_set),
                "cur_nodes": cur_nodes,
            }

        new_nodes = cur_nodes + add_list[:need]

    # (C) executor 호출
    # NOTE: executor.resize_job이 nodes 길이==new_world_size 강제하므로 여기서 맞춰야 함
    resp = resize_job(
        cluster_id=str(cid),
        job_id=str(job_id),
        new_world_size=int(new_g),
        nodes=list(new_nodes),
        reason=str(reason),
        extra={"ts": float(now_ts), "via": "rebalance"},
    )

    ok = bool(resp.get("ok"))
    info = dict(resp or {})
    info.update({"cur_nodes": cur_nodes, "new_nodes": new_nodes, "cur_g": cur_g, "new_g": new_g})
    return ok, info

def _apply_resize_now_optimistic(
    *,
    state: Any,
    cid: str,
    job_id: str,
    new_nodes: List[str],
    reason: str,
) -> Dict[str, Any]:
    now_ts = float(__import__("time").time())

    # 0) spam 방지 (SSOT lock)
    with STATE_LOCK:
        st = get_global_state()
        if not can_resize_locked(st, job_id, now_ts):
            return {"ok": False, "reason": "resize_not_allowed_now", "job_id": str(job_id), "cid": str(cid)}
        mark_resize_inflight_locked(st, job_id, now_ts)
        # 짧은 쿨다운: 같은 tick/연속 호출 방지
        set_resize_cooldown_locked(st, job_id, now_ts, cooldown_sec=float(getattr(st, "RESIZE_COOLDOWN_SEC", 20.0) or 20.0))

    # 1) global server 호출 (LOCK 밖)
    out = resize_job(
        cluster_id=str(cid),
        job_id=str(job_id),
        new_world_size=int(len(new_nodes)),
        nodes=list(new_nodes),
        reason=str(reason or "elastic_resize"),
        extra={"src": "rebalance"},
    )

    ok = bool(out.get("ok"))

    # 2) 결과 반영 (SSOT lock)
    with STATE_LOCK:
        st = get_global_state()
        if ok:
            # ✅ “됐다고 신뢰”니까 즉시 SSOT 반영
            apply_resize_ssot_locked(st, job_id=str(job_id), cid=str(cid), new_nodes=list(new_nodes), now_ts=now_ts)
            clear_resize_inflight_locked(st, job_id)
            # 성공 후에도 너무 잦은 resize 방지
            set_resize_cooldown_locked(st, job_id, now_ts, cooldown_sec=float(getattr(st, "RESIZE_OK_COOLDOWN_SEC", 30.0) or 30.0))
        else:
            clear_resize_inflight_locked(st, job_id)
            set_resize_cooldown_locked(st, job_id, now_ts, cooldown_sec=float(getattr(st, "RESIZE_FAIL_COOLDOWN_SEC", 60.0) or 60.0))

    out.setdefault("job_id", str(job_id))
    out.setdefault("cid", str(cid))
    out.setdefault("new_nodes", list(new_nodes))
    out.setdefault("reason", str(reason))
    return out

REB_PERIOD_SEC = float(__import__("os").getenv("OURS_REB_PERIOD_SEC", "180.0"))

def rebalance_tick_global(
    *,
    state: Optional[Any] = None,
    now_ts: Optional[float] = None,
    reason: str = "periodic",
) -> None:
    now = float(now_ts if now_ts is not None else time.time())

    # ---- 180초 gate (periodic만) ----
    # state에 last ts를 저장해도 되고, module global로 저장해도 됨. 여기서는 module global.
    global _REB_LAST_TS
    try:
        _REB_LAST_TS
    except Exception:
        _REB_LAST_TS = 0.0

    if reason == "periodic":
        if (now - float(_REB_LAST_TS or 0.0)) < float(REB_PERIOD_SEC):
            return
        _REB_LAST_TS = float(now)

    st = state if state is not None else get_global_state()

    # ---- snapshot ----
    with STATE_LOCK:
        clusters = getattr(st, "clusters", {}) or {}
        jobs = getattr(st, "jobs", {}) or {}
        node_owner = getattr(st, "node_owner", {}) or {}
        cq = getattr(st, "cluster_queues", {}) or {}
        policy_map = getattr(st, "job_policy", {}) or {}  # 있으면 사용(없으면 job.policy fallback)

        f_fair_map = _compute_f_fair_all_clusters(st) or {}

    # ---- helpers ----
    def _i(x, d=0) -> int:
        try: return int(x)
        except Exception: return int(d)

    def _f(x, d=0.0) -> float:
        try: return float(x)
        except Exception: return float(d)

    def _cluster_nodes(cr: Any) -> List[str]:
        if isinstance(cr, dict):
            return [str(n) for n in (cr.get("nodes") or []) if n]
        return [str(n) for n in (getattr(cr, "nodes", []) or []) if n]

    def _free_nodes(cid: str, cr: Any) -> List[str]:
        nodes = _cluster_nodes(cr)
        out = []
        for n in nodes:
            if not node_owner.get(n):
                out.append(n)
        return out

    def _job_obj(jid: str) -> Any:
        return jobs.get(str(jid))

    def _job_policy(jid: str, job: Any) -> Dict[str, Any]:
        p = None
        try:
            p = policy_map.get(str(jid))
        except Exception:
            p = None
        if p is None:
            p = getattr(job, "policy", None)
        return p if isinstance(p, dict) else (dict(getattr(p, "__dict__", {}) or {}) if p is not None else {})

    def _job_g_cur(job: Any) -> int:
        # job.g_cur 있으면 사용, 없으면 nodes 길이로
        g = _i(getattr(job, "g_cur", 0) or 0, 0)
        if g > 0:
            return g
        nds = getattr(job, "nodes", None)
        if isinstance(nds, (list, tuple)):
            return max(1, len(nds))
        return 1

    def _job_g_target(job: Any) -> int:
        # intent target
        g = _i(getattr(job, "g_target", 0) or 0, 0)
        if g > 0:
            return g
        g = _i(getattr(job, "g_req", 0) or 0, 0)
        return max(1, g) if g > 0 else 1

    def _job_batch(job: Any) -> int:
        bs = getattr(job, "batch_size", None) or getattr(job, "batch_size_per_gpu", None) or getattr(job, "local_batch", None)
        return _i(bs, 0)

    def _cluster_price(cr: Any) -> float:
        if isinstance(cr, dict):
            return _f(cr.get("price_per_gpu_hour", 1.0), 1.0)
        return _f(getattr(cr, "price_per_gpu_hour", 1.0), 1.0)

    def _is_running(job: Any) -> bool:
        s = str(getattr(job, "status", "") or "").upper()
        return s in ("RUNNING", "STARTING", "RESIZING")

    def _is_queued(job: Any) -> bool:
        return str(getattr(job, "status", "") or "").upper() == "QUEUED"

    # ---- marginal utilities ----
    def _U(cid: str, job: Any, g: int) -> Optional[Dict[str, Any]]:
        """
        utility 계산을 위한 프로파일 조회 + score.
        """
        model = getattr(job, "model", None) or getattr(job, "model_name", None)
        dataset = getattr(job, "dataset", None)
        if not model or not dataset:
            return None

        prof = _hook_get_profiling_for(
            model=str(model),
            dataset=str(dataset),
            cid=str(cid),
            g=int(g),
            batch_size=int(_job_batch(job)),
            clusters=dict(clusters),
        )
        if not prof:
            return None

        cr = clusters.get(str(cid))
        price = _cluster_price(cr)

        policy = _job_policy(str(getattr(job, "job_id", "") or ""), job)
        ff = float(f_fair_map.get(str(cid), 1.0) or 1.0)

        sc = _hook_score(
            policy=policy,
            prof=dict(prof),
            g=int(g),
            price_per_gpu_hour=float(price),
            f_fair_c=float(ff),
        )
        if not sc.get("ok"):
            return None
        out = dict(sc)
        out["profiling"] = dict(prof)
        return out

    def _delta_gain_resize(cid: str, job: Any, g_cur: int) -> Optional[float]:
        u0 = _U(cid, job, int(g_cur))
        u1 = _U(cid, job, int(g_cur + 1))
        if not u0 or not u1:
            return None
        return float(u1["S"] - u0["S"])

    # ---- choose action per cluster when free exists ----
    for cid0, cr in (clusters or {}).items():
        cid = str(cid0)
        free = _free_nodes(cid, cr)
        if not free:
            continue

        # 1) candidate A: launch HoL job in this cluster queue (있으면)
        launch_jid = None
        try:
            q = (cq or {}).get(cid)
            # ClusterQueue면 내부 _jobs 사용
            items = list(getattr(q, "_jobs", []) or [])
            if items:
                # QueueJob / dict / str 대응
                x = items[0]
                if isinstance(x, str):
                    launch_jid = x
                elif isinstance(x, dict):
                    launch_jid = x.get("job_id") or x.get("id")
                elif hasattr(x, "job_id"):
                    launch_jid = getattr(x, "job_id")
                elif hasattr(x, "id"):
                    launch_jid = getattr(x, "id")
        except Exception:
            launch_jid = None

        launch_gain = None
        launch_g = None
        launch_nodes = None

        if launch_jid:
            j = _job_obj(str(launch_jid))
            if j is not None and _is_queued(j):
                # 일단 1장 launch 기준이 아니라, free 범위에서 "이득 최대 g"를 고르자 (g=1만 되는 문제 완화)
                g_cur = 0
                g_tgt = max(1, _job_g_target(j))
                gmax = min(len(free), max(1, g_tgt))
                best = None
                for gg in range(1, gmax + 1):
                    u = _U(cid, j, gg)
                    if not u:
                        continue
                    if best is None or float(u["S"]) > float(best["S"]):
                        best = {"g": gg, "S": float(u["S"]), "u": u}
                if best is not None:
                    launch_g = int(best["g"])
                    launch_gain = float(best["S"])  # absolute utility (비교용)
                    launch_nodes = free[: int(launch_g)]

        # 2) candidate B: resize receiver (running job 중 g_cur < g_target)
        resize_best = None
        for jid, job in (jobs or {}).items():
            if not job:
                continue
            if str(getattr(job, "cluster", "") or getattr(job, "cluster_id", "") or "") != cid:
                continue
            if not _is_running(job):
                continue

            gcur = _job_g_cur(job)
            gtgt = _job_g_target(job)
            if gcur >= gtgt:
                continue
            if len(free) < 1:
                continue

            dg = _delta_gain_resize(cid, job, gcur)
            if dg is None:
                continue

            if resize_best is None or float(dg) > float(resize_best["dg"]):
                resize_best = {"jid": str(jid), "job": job, "gcur": int(gcur), "gtgt": int(gtgt), "dg": float(dg)}

        # ---- decision ----
        # 주의: launch_gain은 절대 utility, resize는 delta utility.
        # 비교를 공정하게 하려면 launch도 "지금 아무것도 안 하는 상태 대비 delta"가 필요하지만,
        # 여기선 단순 정책: free가 생기면 (a) HoL launch 우선, 단 (b) resize delta가 충분히 크면 resize 선택.
        #
        # threshold는 실험으로 조절해야 함.
        RESIZE_WIN_THRESHOLD = float(getattr(st, "REB_RESIZE_WIN_THRESHOLD", 0.15) or 0.15)

        do_resize = False
        if resize_best is not None and launch_gain is not None:
            # launch는 queue job을 새로 시작시키는 큰 변화라, resize가 이기려면 delta가 꽤 커야 함
            if float(resize_best["dg"]) > float(RESIZE_WIN_THRESHOLD):
                do_resize = True
        elif resize_best is not None and launch_gain is None:
            do_resize = True
        else:
            do_resize = False

        # periodic에서는 더 보수적으로
        if reason == "periodic" and do_resize and resize_best is not None:
            if float(resize_best["dg"]) < float(RESIZE_WIN_THRESHOLD) * 2.0:
                do_resize = False

        # ---- apply ----
        if do_resize and resize_best is not None:
            jid = resize_best["jid"]
            job = resize_best["job"]
            gcur = int(resize_best["gcur"])
            new_g = int(gcur + 1)
            # new_nodes = current_nodes + one_free
            cur_nodes = list(getattr(job, "nodes", []) or [])
            if not cur_nodes:
                # nodes가 job에 없으면 여기서 resize하면 꼬임 → skip
                continue
            add = free[:1]
            new_nodes = cur_nodes + add

            # executor.resize_job 호출 (네가 준 함수)
            try:
                resize_job(
                    cluster_id=str(cid),
                    job_id=str(jid),
                    new_world_size=int(new_g),
                    nodes=list(new_nodes),
                    reason=f"elastic_resize:{reason}",
                    extra={"dg": float(resize_best["dg"]), "from": int(gcur), "to": int(new_g)},
                )
            except Exception:
                pass

            # 1장만 처리하고 다음 tick로 넘기는 게 안전
            continue

        # launch 적용
        if launch_jid and launch_nodes and launch_g and int(launch_g) > 0:
            try:
                # 기존 launch 경로 재사용 (너 코드베이스에 맞게)
                # launch_or_reuse(...) 또는 reserve_nodes + launch_or_reuse(...)
                #
                # 여기서는 "결정만" 전달. 실제 reserve/SSOT는 기존 로직에 맞춰 붙여.
                launch_or_reuse(
                    cluster_id=str(cid),
                    job_id=str(launch_jid),
                    nodes=list(launch_nodes),
                    reason=f"rebalance_launch:{reason}",
                )
            except Exception:
                pass

            continue

def _marginal_gain_for_job_in_cluster(
    *,
    job: Any,
    cid: str,
    g_cur: int,
    g_new: int,
    clusters: Dict[str, Any],
    telemetry: Any,
) -> float:
    # job spec
    model = getattr(job, "model", None) or getattr(job, "model_name", None)
    dataset = getattr(job, "dataset", None)

    if not model or not dataset:
        return 0.0

    # batch (없으면 0)
    try:
        batch = int(
            getattr(job, "batch_size", None)
            or getattr(job, "batch_size_per_gpu", None)
            or getattr(job, "local_batch", None)
            or 0
        )
    except Exception:
        batch = 0

    # cluster params
    cr = (clusters or {}).get(str(cid))
    def _f(x, d=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(d)

    price = 1.0
    p_budget = 1.0
    if isinstance(cr, dict):
        price = _f(cr.get("price_per_gpu_hour", 1.0), 1.0)
        p_budget = _f(cr.get("power_budget_w", 1.0), 1.0)
    elif cr is not None:
        price = _f(getattr(cr, "price_per_gpu_hour", 1.0), 1.0)
        p_budget = _f(getattr(cr, "power_budget_w", 1.0), 1.0)

    # profiling: g_cur/g_new 각각 가져와야 Δ가 의미 있음
    prof_cur = get_profiling_entry(str(model), str(dataset), str(cid), int(g_cur)) or {}
    prof_new = get_profiling_entry(str(model), str(dataset), str(cid), int(g_new)) or {}

    sps_cur = _f((prof_cur or {}).get("throughput_sps", 0.0), 0.0)
    sps_new = _f((prof_new or {}).get("throughput_sps", 0.0), 0.0)
    if not (sps_cur > 0.0 and math.isfinite(sps_cur)):
        return 0.0
    if not (sps_new > 0.0 and math.isfinite(sps_new)):
        return 0.0

    # ✅ energy: profiling avg_power_w_per_gpu 우선
    ppg_cur = _f((prof_cur or {}).get("avg_power_w_per_gpu", 0.0), 0.0)
    ppg_new = _f((prof_new or {}).get("avg_power_w_per_gpu", 0.0), 0.0)

    # fallback: telemetry에서 cluster power를 쪼개는 건 왜곡이 커서 "최후"에만
    # (job별 power가 없으면 energy term이 약해짐)
    if ppg_cur <= 0.0:
        ppg_cur = 0.0
    if ppg_new <= 0.0:
        ppg_new = 0.0

    pcur_job_cur = float(ppg_cur) * float(g_cur) if ppg_cur > 0.0 else 0.0
    pcur_job_new = float(ppg_new) * float(g_new) if ppg_new > 0.0 else 0.0

    # policy: job에 저장돼있거나 state에서 끌어오는 구조면 여기서 받아오면 됨
    policy = {}
    try:
        if isinstance(job, dict):
            policy = dict(job.get("policy") or {})
        else:
            policy = dict(getattr(job, "policy", {}) or {})
    except Exception:
        policy = {}

    # fairness는 cluster-level이라 여기선 1로 둠 (pair-swap 단계에서 도입)
    f_fair_c = 1.0

    U_cur = _score_S_jcg(
        policy=policy,
        sps=float(sps_cur),
        g=int(g_cur),
        price_per_gpu_hour=float(price),
        p_current_w_cluster=float(pcur_job_cur),  # ✅ job power sum
        p_budget_w_cluster=float(p_budget),
        f_fair_c=float(f_fair_c),
    )
    U_new = _score_S_jcg(
        policy=policy,
        sps=float(sps_new),
        g=int(g_new),
        price_per_gpu_hour=float(price),
        p_current_w_cluster=float(pcur_job_new),  # ✅ job power sum
        p_budget_w_cluster=float(p_budget),
        f_fair_c=float(f_fair_c),
    )

    if not (math.isfinite(float(U_cur)) and math.isfinite(float(U_new))):
        return 0.0

    return float(U_new - U_cur)

def scheduler_tick_attach_rebalance(*, free_event: bool = False) -> None:
    try:
        rebalance_tick_global(force=bool(free_event), reason=("free_event" if free_event else "periodic_180s"))
    except Exception:
        pass

def apply_resize_ssot_locked(
    st: Any,
    *,
    job_id: str,
    cid: str,
    new_nodes: List[str],
    now_ts: Optional[float] = None,
) -> None:
    import time

    job_id = str(job_id)
    cid = str(cid)
    now_ts = float(now_ts or time.time())
    nodes = [str(n) for n in (new_nodes or []) if n]

    # 1) node_owner: 기존 job 소유 노드 제거 후 새 노드 할당
    node_owner = getattr(st, "node_owner", None)
    if not isinstance(node_owner, dict):
        node_owner = {}
        setattr(st, "node_owner", node_owner)

    for n, owner in list(node_owner.items()):
        if str(owner) == job_id:
            node_owner.pop(str(n), None)

    for n in nodes:
        node_owner[str(n)] = job_id

    # 2) jobs[job_id] 업데이트
    jobs = getattr(st, "jobs", None)
    if not isinstance(jobs, dict):
        jobs = {}
        setattr(st, "jobs", jobs)

    jr = jobs.get(job_id)
    if jr is not None:
        try:
            if isinstance(jr, dict):
                jr["cluster_id"] = cid
                jr["nodes"] = list(nodes)
                jr["g_cur"] = int(len(nodes))
                jr["last_resize_ts"] = float(now_ts)
            else:
                setattr(jr, "cluster_id", cid)
                setattr(jr, "nodes", list(nodes))
                setattr(jr, "g_cur", int(len(nodes)))
                setattr(jr, "last_resize_ts", float(now_ts))
        except Exception:
            pass

    # 3) cluster counters 재계산(있으면)
    try:
        _recompute_cluster_counters_locked(st)
    except Exception:
        pass

def can_resize_locked(st: Any, job_id: str, now_ts: float) -> bool:
    job_id = str(job_id)
    now_ts = float(now_ts)

    infl = getattr(st, "resize_inflight", None)
    if isinstance(infl, dict) and infl.get(job_id):
        return False

    cd = getattr(st, "resize_cooldown_until", None)
    if isinstance(cd, dict):
        try:
            until = float(cd.get(job_id) or 0.0)
            if until > now_ts:
                return False
        except Exception:
            pass

    return True

def mark_resize_inflight_locked(st: Any, job_id: str, now_ts: float) -> None:
    job_id = str(job_id)
    now_ts = float(now_ts)

    infl = getattr(st, "resize_inflight", None)
    if not isinstance(infl, dict):
        infl = {}
        setattr(st, "resize_inflight", infl)

    infl[job_id] = {"ts": now_ts}

def clear_resize_inflight_locked(st: Any, job_id: str) -> None:
    job_id = str(job_id)
    infl = getattr(st, "resize_inflight", None)
    if isinstance(infl, dict):
        infl.pop(job_id, None)

def set_resize_cooldown_locked(st: Any, job_id: str, now_ts: float, cooldown_sec: float) -> None:
    job_id = str(job_id)
    now_ts = float(now_ts)
    cooldown_sec = float(cooldown_sec)

    cd = getattr(st, "resize_cooldown_until", None)
    if not isinstance(cd, dict):
        cd = {}
        setattr(st, "resize_cooldown_until", cd)

    cd[job_id] = float(now_ts + cooldown_sec)

def _hook_get_profiling_for(
    *,
    model: str,
    dataset: str,
    cid: str,
    g: int,
    batch_size: int,
    clusters: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    def _f(x: Any, d=0.0) -> float:
        try:
            v = float(x)
            if not math.isfinite(v):
                return float(d)
            return v
        except Exception:
            return float(d)

    def _i(x: Any, d=0) -> int:
        try:
            return int(x)
        except Exception:
            return int(d)

    def _cluster_speed_factor(cr: Any) -> float:
        if isinstance(cr, dict):
            return _f(cr.get("speed_factor", 1.0), 1.0)
        return _f(getattr(cr, "speed_factor", 1.0), 1.0)

    def _call_get_prof(m: str, dset: str, c: str, gg: int, bs: int) -> Optional[Dict[str, Any]]:
        try:
            sig = inspect.signature(get_profiling_entry)
            params = list(sig.parameters.keys())
        except Exception:
            params = []

        try:
            if params:
                kwargs: Dict[str, Any] = {}
                if "model_name" in params:
                    kwargs["model_name"] = m
                elif "model" in params:
                    kwargs["model"] = m

                if "dataset" in params:
                    kwargs["dataset"] = dset

                if "cluster_id" in params:
                    kwargs["cluster_id"] = c
                elif "cid" in params:
                    kwargs["cid"] = c

                if "gpu_count" in params:
                    kwargs["gpu_count"] = int(gg)
                elif "g" in params:
                    kwargs["g"] = int(gg)

                if bs > 0:
                    if "batch_size" in params:
                        kwargs["batch_size"] = int(bs)
                    elif "batch" in params:
                        kwargs["batch"] = int(bs)

                prof = get_profiling_entry(**kwargs) if kwargs else None
                if isinstance(prof, dict) and prof:
                    return prof

            # legacy positional fallback
            prof2 = get_profiling_entry(m, dset, c, int(gg))
            if isinstance(prof2, dict) and prof2:
                if bs > 0 and ("batch_size" not in prof2):
                    prof2["batch_size"] = int(bs)
                return prof2
        except Exception:
            return None

        return None

    def _normalize_prof(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not (isinstance(p, dict) and p):
            return None

        # epoch_time: DB column epoch_time_measured_sec
        et = _f(p.get("epoch_time_measured_sec", p.get("epoch_time_sec", 0.0)), 0.0)
        if et <= 0.0:
            # throughput_sps로 역산(있으면)
            sps = _f(p.get("throughput_sps", 0.0), 0.0)
            if sps > 0:
                # 1 epoch = steps_per_epoch 필요하지만 없으니 et는 못 만듦 → score에 쓰기 어려움
                # 여기서는 et가 없으면 그대로 둠
                pass

        pw = _f(p.get("avg_power_w_per_gpu", p.get("avg_power_w", 0.0)), 0.0)

        out = dict(p)
        out["epoch_time_sec"] = float(et)
        out["avg_power_w_per_gpu"] = float(pw)
        if "throughput_sps" in out:
            out["throughput_sps"] = float(_f(out.get("throughput_sps"), 0.0))
        return out

    model = str(model)
    dataset = str(dataset)
    cid = str(cid)
    g = max(1, _i(g, 1))
    batch_size = _i(batch_size, 0)

    # 1) exact
    p0 = _normalize_prof(_call_get_prof(model, dataset, cid, g, batch_size) or {})
    if p0:
        return p0

    # 2) other cluster fallback (scale by speed_factor)
    try:
        cr_t = (clusters or {}).get(cid)
        sf_t = _cluster_speed_factor(cr_t) if cr_t is not None else 1.0
        for cid2, cr2 in (clusters or {}).items():
            cid2 = str(cid2)
            if cid2 == cid:
                continue
            p2_raw = _call_get_prof(model, dataset, cid2, g, batch_size)
            p2 = _normalize_prof(p2_raw or {})
            if not p2:
                continue

            sf_2 = _cluster_speed_factor(cr2)
            ratio = (sf_t / sf_2) if (sf_2 and sf_2 > 0.0) else 1.0

            out = dict(p2)
            # speed_factor가 "빠를수록 큼"이면:
            # - throughput_sps는 ratio 곱
            # - epoch_time은 ratio로 나눔
            if out.get("throughput_sps"):
                out["throughput_sps"] = float(_f(out["throughput_sps"], 0.0)) * float(ratio)
            if out.get("epoch_time_sec"):
                out["epoch_time_sec"] = float(_f(out["epoch_time_sec"], 0.0)) / float(max(1e-9, ratio))

            out["fallback_from_cluster"] = str(cid2)
            out["fallback_speed_ratio"] = float(ratio)
            return out
    except Exception:
        pass

    # 3) scaled_from_g (same cluster)
    # g' 프로파일 있으면 epoch_time을 선형으로 "대충" 스케일(최후수단)
    # 현실 goodput은 선형이 아니지만, 최소한 비교는 가능해짐.
    try:
        # 가까운 g부터 찾자
        candidates = [1, 2, 4, 8]
        candidates = [x for x in candidates if x != g and x > 0]
        for gh in candidates:
            p_hi = _normalize_prof(_call_get_prof(model, dataset, cid, gh, batch_size) or {})
            if not p_hi:
                continue

            et_hi = _f(p_hi.get("epoch_time_sec"), 0.0)
            if et_hi <= 0.0:
                continue

            # naive scaling: time ~ 1/g (대충)
            et = float(et_hi) * (float(gh) / float(g))
            out = dict(p_hi)
            out["epoch_time_sec"] = float(et)
            out["scaled_from_g"] = int(gh)

            # power/gpu는 그대로(가정)
            return out
    except Exception:
        pass

    return None

def _hook_score(
    *,
    policy: Dict[str, Any] | Any,
    prof: Dict[str, Any],
    g: int,
    price_per_gpu_hour: float,
    f_fair_c: float,
    mu_fair: float = MU_FAIR,
) -> Dict[str, Any]:
    def _f(x: Any, d=0.0) -> float:
        try:
            v = float(x)
            if not math.isfinite(v):
                return float(d)
            return v
        except Exception:
            return float(d)

    def _policy_norm(p: Any) -> Dict[str, float]:
        if p is None:
            d = {}
        elif isinstance(p, dict):
            d = p
        else:
            if hasattr(p, "model_dump") and callable(getattr(p, "model_dump")):
                try:
                    d = dict(p.model_dump())
                except Exception:
                    d = {}
            elif hasattr(p, "dict") and callable(getattr(p, "dict")):
                try:
                    d = dict(p.dict())
                except Exception:
                    d = {}
            else:
                d = dict(getattr(p, "__dict__", {}) or {})

        lam_t = _f(d.get("lambda_time", 0.34), 0.34)
        lam_c = _f(d.get("lambda_cost", 0.33), 0.33)
        lam_e = _f(d.get("lambda_energy", 0.33), 0.33)

        lam_t = max(0.0, min(1.0, lam_t))
        lam_c = max(0.0, min(1.0, lam_c))
        lam_e = max(0.0, min(1.0, lam_e))
        s = lam_t + lam_c + lam_e
        if s <= 1e-9:
            lam_t, lam_c, lam_e = 0.34, 0.33, 0.33
        else:
            lam_t, lam_c, lam_e = lam_t / s, lam_c / s, lam_e / s
        return {"lam_t": lam_t, "lam_c": lam_c, "lam_e": lam_e}

    g = int(g) if int(g or 0) > 0 else 1
    price_per_gpu_hour = _f(price_per_gpu_hour, 1.0)
    f_fair_c = _f(f_fair_c, 1.0)

    pn = _policy_norm(policy)
    lam_t, lam_c, lam_e = pn["lam_t"], pn["lam_c"], pn["lam_e"]

    epoch_time = _f(prof.get("epoch_time_sec", prof.get("epoch_time_measured_sec", 0.0)), 0.0)
    if epoch_time <= 0.0:
        return {"ok": False, "S": -1e18, "terms": {"reason": "missing_epoch_time"}}

    avg_power_w_per_gpu = _f(prof.get("avg_power_w_per_gpu", 0.0), 0.0)
    if avg_power_w_per_gpu < 0.0:
        avg_power_w_per_gpu = 0.0

    # time gain (bigger is better)
    time_gain = (float(TIME_REF) / float(epoch_time)) if float(epoch_time) > 1e-12 else 0.0

    # cost penalty per epoch
    cost_per_epoch = (float(g) * float(price_per_gpu_hour)) * (float(epoch_time) / 3600.0)  # currency/epoch
    cost_pen = (cost_per_epoch / float(COST_REF)) if float(COST_REF) > 0 else 0.0

    # energy penalty per epoch
    energy_kwh_per_epoch = (float(g) * float(avg_power_w_per_gpu) * float(epoch_time)) / 3_600_000.0
    energy_pen = (energy_kwh_per_epoch / float(ENERGY_REF)) if float(ENERGY_REF) > 0 else 0.0

    # fairness penalty
    fair_excess = max(0.0, float(f_fair_c) - 1.0)
    fair_pen = float(mu_fair) * float(fair_excess)

    S = (lam_t * time_gain) - (lam_c * cost_pen) - (lam_e * energy_pen) - float(fair_pen)

    if not math.isfinite(S):
        return {"ok": False, "S": -1e18, "terms": {"reason": "non_finite"}}

    return {
        "ok": True,
        "S": float(S),
        "terms": {
            "lam_t": float(lam_t),
            "lam_c": float(lam_c),
            "lam_e": float(lam_e),
            "epoch_time_sec": float(epoch_time),
            "time_gain": float(time_gain),
            "price_per_gpu_hour": float(price_per_gpu_hour),
            "cost_per_epoch": float(cost_per_epoch),
            "cost_pen": float(cost_pen),
            "avg_power_w_per_gpu": float(avg_power_w_per_gpu),
            "energy_kwh_per_epoch": float(energy_kwh_per_epoch),
            "energy_pen": float(energy_pen),
            "f_fair_c": float(f_fair_c),
            "fair_pen": float(fair_pen),
            "g": int(g),
        },
    }