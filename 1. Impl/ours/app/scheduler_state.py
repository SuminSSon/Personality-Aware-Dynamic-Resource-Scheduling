from __future__ import annotations

import logging
import time, uuid, math
from typing import Dict, Any, Optional, List, Tuple, Set
import math

from app.logger import get_run_logger
from app.backfill_policy import _job_sort_key_for_global_order, get_global_state, STATE_LOCK, _is_queued, _recompute_cluster_counters_locked, release_nodes_for_job_locked, _set_free_changed_locked


logger = logging.getLogger(__name__)

_ENQUEUE_SEQ = 0

# ---- status constants (single source) ----
_TERMINAL: Set[str] = {"FINISHED", "FAILED", "CANCELLED", "COMPLETED"}
_RUNNING: str = "RUNNING"
_QUEUED: str = "QUEUED"

CLUSTERS: Dict[str, List[str]] = {
    "clusterA": ["node_a", "node_b", "node_c", "node_d"],
    "clusterB": ["node_e", "node_f", "node_g", "node_h"],
}

def ensure_job_single_queue_locked(state: Any, job_id: str, prefer_cluster_id: Optional[str] = None) -> None:
    jid = str(job_id).strip()
    if not jid:
        return

    prefer = str(prefer_cluster_id).strip() if prefer_cluster_id else None

    def _jid_of(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            v = x.get("job_id") or x.get("id")
            return str(v) if v else ""
        v = getattr(x, "job_id", None) or getattr(x, "id", None)
        if v:
            return str(v)
        try:
            return str(x)  # QueueJob는 __str__이 job_id일 수 있음
        except Exception:
            return ""

    found_in_home = False
    found_in_cluster = False

    # -------------------------------------------------
    # 1) HOME 큐(home_cluster_queues)에서 jid 제거
    # -------------------------------------------------
    hq = getattr(state, "home_cluster_queues", None)
    if isinstance(hq, dict):
        for cid, lst in list(hq.items()):
            if not isinstance(lst, list) or not lst:
                continue
            before = len(lst)
            new_lst = [x for x in lst if str(x) != jid]
            if len(new_lst) != before:
                found_in_home = True
                hq[cid] = new_lst

    # -------------------------------------------------
    # 2) cluster_queues에서 jid 제거
    # -------------------------------------------------
    qs = getattr(state, "cluster_queues", None)
    if isinstance(qs, dict):
        for cid, q in list(qs.items()):
            removed_here = False

            # (a) 정식 API remove가 있으면 우선 사용
            try:
                before_len = len(q)  # ClusterQueue.__len__
            except Exception:
                before_len = None

            try:
                if hasattr(q, "remove"):
                    q.remove(jid)
                    if before_len is not None:
                        try:
                            if len(q) != before_len:
                                removed_here = True
                        except Exception:
                            pass
            except Exception:
                pass

            # (b) fallback: _jobs 선형 제거
            try:
                jobs = list(getattr(q, "_jobs", []) or [])
                if jobs:
                    new_jobs = []
                    for x in jobs:
                        xjid = _jid_of(x)
                        if str(xjid) == jid:
                            removed_here = True
                            continue
                        new_jobs.append(x)
                    if removed_here:
                        setattr(q, "_jobs", new_jobs)
                        try:
                            q.eta_dirty = True
                        except Exception:
                            pass
            except Exception:
                pass

            if removed_here:
                found_in_cluster = True

    # -------------------------------------------------
    # 3) global_queue 중복 제거(순서 유지, 첫 1개만 남김)
    # -------------------------------------------------
    gq = getattr(state, "global_queue", None)
    if isinstance(gq, list) and gq:
        seen = False
        new_gq = []
        for it in gq:
            it_jid = _jid_of(it).strip()
            if it_jid == jid:
                if not seen:
                    new_gq.append(it)
                    seen = True
                else:
                    # duplicate drop
                    continue
            else:
                new_gq.append(it)
        if len(new_gq) != len(gq):
            state.global_queue = new_gq

    # prefer가 없으면 "제거만" 하고 종료
    if not prefer:
        return

    # -------------------------------------------------
    # 4) prefer로 재삽입(안전할 때만)
    #    - 새로 제출된 job처럼 어디에도 없던 경우(found_in_home=found_in_cluster=False)는 여기서 삽입하지 않음
    #      (caller enqueue가 SSOT)
    # -------------------------------------------------
    jobs_map = getattr(state, "jobs", {}) or {}
    jr = jobs_map.get(jid)

    # 우선 jr.queue_kind 힌트가 있으면 사용, 없으면 "기존에 어디에 있었는지"로 추정
    qk = ""
    try:
        qk = str(getattr(jr, "queue_kind", "") or "").upper() if jr is not None else ""
    except Exception:
        qk = ""

    target = None  # "HOME" | "CLUSTER" | None
    if qk == "HOME":
        target = "HOME"
    elif qk in ("CLUSTER", "MOBILE"):
        target = "CLUSTER"
    else:
        # 힌트가 없으면 과거 소속으로만 결정
        if found_in_home and not found_in_cluster:
            target = "HOME"
        elif found_in_cluster and not found_in_home:
            target = "CLUSTER"
        else:
            # 둘 다였거나(깨진 상태), 둘 다 아니면(신규 제출 등) -> 여기서 삽입하지 않음
            target = None

    if target == "HOME":
        # prefer HOME 큐만 유지
        if not hasattr(state, "home_cluster_queues") or not isinstance(getattr(state, "home_cluster_queues", None), dict):
            state.home_cluster_queues = {}
        qmap = state.home_cluster_queues
        if prefer not in qmap or not isinstance(qmap.get(prefer), list):
            qmap[prefer] = []
        if str(jid) not in qmap[prefer]:
            qmap[prefer].append(str(jid))
        return

    if target == "CLUSTER":
        # prefer cluster_queues에만 유지
        qs2 = getattr(state, "cluster_queues", None)
        if not isinstance(qs2, dict) or not qs2:
            return
        q2 = qs2.get(prefer)
        if q2 is None:
            return

        # QueueJob를 가능한 한 메타 포함해서 구성
        try:
            from app.queue import QueueJob

            model = ""
            dataset = ""
            g_target = 1
            g_req = None
            user_id = None
            submit_ts = None
            enqueue_seq = 0
            pinned_cluster = None
            admitted_cluster_id = None
            home_cluster_id = None
            is_gang = False

            if jr is not None:
                model = str(getattr(jr, "model", "") or getattr(jr, "model_name", "") or "")
                dataset = str(getattr(jr, "dataset", "") or "")
                try:
                    g_target = int(getattr(jr, "g_target", 1) or 1)
                except Exception:
                    g_target = 1
                # g_req 후보
                for k in ("g_req", "world_size", "g_target"):
                    v = getattr(jr, k, None)
                    if v is not None:
                        try:
                            g_req = int(v)
                            break
                        except Exception:
                            pass
                user_id = getattr(jr, "user_id", None)
                submit_ts = getattr(jr, "submit_ts", None)
                enqueue_seq = int(getattr(jr, "enqueue_seq", 0) or getattr(jr, "seq", 0) or 0)
                pinned_cluster = getattr(jr, "pinned_cluster", None) or getattr(jr, "user_pinned_cluster_id", None)
                admitted_cluster_id = getattr(jr, "admitted_cluster_id", None)
                home_cluster_id = getattr(jr, "home_cluster_id", None)
                is_gang = bool(getattr(jr, "is_gang", False) or (int(g_req or g_target) == 4))

            qjob = QueueJob(
                job_id=jid,
                model=model,
                dataset=dataset,
                g_req=g_req,
                g_target=g_target,
                pinned_cluster=str(pinned_cluster) if pinned_cluster else None,
                admitted_cluster_id=str(admitted_cluster_id) if admitted_cluster_id else None,
                home_cluster_id=str(home_cluster_id) if home_cluster_id else None,
                is_gang=bool(is_gang),
                user_id=user_id,
                submit_ts=float(submit_ts) if submit_ts is not None else None,
                enqueue_seq=int(enqueue_seq),
                queue_kind=(getattr(jr, "queue_kind", None) if jr is not None else None),
                queue_cluster_id=str(prefer),
            )
        except Exception:
            qjob = jid  # fallback

        # 삽입(가능하면 정식 메소드 사용)
        try:
            if not isinstance(qjob, str) and hasattr(q2, "enqueue"):
                q2.enqueue(qjob)  # 내부에서 seq 정렬
                return
        except Exception:
            pass

        # fallback: _jobs append
        try:
            jobs_list = list(getattr(q2, "_jobs", []) or [])
            for x in jobs_list:
                if str(_jid_of(x)) == jid:
                    return
            jobs_list.append(qjob)
            setattr(q2, "_jobs", jobs_list)
            try:
                q2.eta_dirty = True
            except Exception:
                pass
        except Exception:
            pass

        return

    # target None이면 재삽입하지 않음 (caller가 enqueue하는 것이 SSOT)
    return

def update_cluster_telemetry(
    cluster_id: str,
    used_gpus: int,
    total_gpus: int,
    util: float,
    power_current_w: float,
) -> None:
    cid = str(cluster_id)
    now = time.time()

    def _clamp01(x: Any) -> float:
        try:
            v = float(x or 0.0)
        except Exception:
            return 0.0
        if not math.isfinite(v) or v < 0:
            return 0.0
        if v > 1.5:
            v = v / 100.0
        if v > 1.0:
            v = 1.0
        return v

    with STATE_LOCK:
        state = get_global_state()
        cr = state.clusters.get(cid)
        if cr is None:
            return

        # ✅ telemetry util은 별도 필드로
        try:
            cr.util_telemetry = float(_clamp01(util))
        except Exception:
            try:
                cr.util_telemetry = 0.0
            except Exception:
                pass

        try:
            pw = float(power_current_w or 0.0)
            if not math.isfinite(pw) or pw < 0:
                pw = 0.0
            cr.power_current_w = pw
        except Exception:
            cr.power_current_w = 0.0

        try:
            cr.last_telemetry_ts = float(now)
        except Exception:
            pass

        try:
            if int(getattr(cr, "total_gpus", 0) or 0) <= 0 and int(total_gpus or 0) > 0:
                cr.total_gpus = int(total_gpus)
        except Exception:
            pass

        try:
            cr.used_gpus_telemetry = int(max(0, int(used_gpus or 0)))
        except Exception:
            pass

def update_job_progress_and_eta_locked(
    state: Any,
    *,
    job_id: str,
    epoch_done: Optional[float] = None,
    epochs_total: Optional[float] = None,
    now_ts: Optional[float] = None,
) -> None:
    jid = str(job_id)
    now = float(now_ts if now_ts is not None else time.time())

    jr = getattr(state, "jobs", {}).get(jid)
    if jr is None:
        return

    # RUNNING만 ETA를 "의미있게" 갱신 (QUEUED는 별도 정책)
    st = str(getattr(jr, "status", "") or "")
    if st != "RUNNING":
        return

    # start_ts 없으면 ETA 산정 불가
    start_ts = getattr(jr, "start_ts", None)
    if start_ts is None:
        return

    # epoch_done/total 확보
    if epoch_done is None:
        epoch_done = getattr(jr, "epoch_done", None)
        if epoch_done is None:
            epoch_done = getattr(jr, "epochs_done", None)

    if epochs_total is None:
        epochs_total = getattr(jr, "epochs", None)

    try:
        done = float(epoch_done) if epoch_done is not None else None
    except Exception:
        done = None

    try:
        total = float(epochs_total) if epochs_total is not None else None
    except Exception:
        total = None

    # progress 비율(0~1) 추정
    ratio = None
    if done is not None and total is not None and total > 0:
        ratio = max(0.0, min(1.0, done / total))

    # elapsed 기반 ETA
    elapsed = max(0.0, now - float(start_ts))

    # 1) profiling 기반 ETA (가능하면)
    # profiling 예시: {"expected_total_sec": ..., "expected_epoch_sec": ...}
    prof = getattr(jr, "profiling", None) or {}
    exp_total = None
    try:
        # expected_total_sec가 있으면 그걸 우선 사용
        exp_total = float(prof.get("expected_total_sec")) if isinstance(prof, dict) and prof.get("expected_total_sec") else None
    except Exception:
        exp_total = None

    remaining = None

    # profiling total이 있으면:
    if exp_total is not None and exp_total > 0:
        # 이미 경과한 시간이 exp_total을 넘으면 0으로 clamp
        remaining = max(0.0, exp_total - elapsed)

    # 2) profiling이 없으면 ratio 기반으로
    if remaining is None:
        if ratio is None:
            return  # 정보가 너무 없음 -> ETA 갱신 포기
        # ratio가 0에 너무 가까우면 폭주하므로 최소값
        if ratio < 1e-6:
            return
        # total_time ≈ elapsed / ratio
        est_total = elapsed / ratio
        remaining = max(0.0, est_total - elapsed)

    # 마지막 방어: 비정상 값 제거
    if remaining is None or not math.isfinite(remaining) or remaining < 0:
        return

    # 기록
    try:
        jr.expected_end_ts = float(now + remaining)
    except Exception:
        # dict 타입 호환
        if isinstance(jr, dict):
            jr["expected_end_ts"] = float(now + remaining)

    try:
        jr.last_eta_update_ts = float(now)
    except Exception:
        pass

    # done 기록(들어온 값이 있으면 state에도 반영)
    if done is not None:
        try:
            jr.epoch_done = done
        except Exception:
            if isinstance(jr, dict):
                jr["epoch_done"] = done
    if total is not None:
        try:
            jr.epochs = int(total)
        except Exception:
            if isinstance(jr, dict):
                jr["epochs"] = int(total)

def remove_from_global_queue_locked(state: Any, job_id: str) -> bool:
    jid = str(job_id)
    gq = getattr(state, "global_queue", None)
    if not isinstance(gq, list) or not gq:
        return False

    for i, x in enumerate(list(gq)):
        if str(x) == jid:
            gq.pop(i)
            return True
    return False

def requeue_preempted_job_locked(
    state: Any,
    *,
    job_id: str,
    home_cluster_id: str,
) -> None:
    from app.queue import QueueJob  # 로컬 import로 순환 방지

    jid = str(job_id)
    home = str(home_cluster_id)

    jr = getattr(state, "jobs", {}).get(jid)
    if jr is None:
        return

    now = float(time.time())

    # 상태 전이(중요): PREEMPTED -> QUEUED
    try:
        jr.status = "QUEUED"
    except Exception:
        if isinstance(jr, dict):
            jr["status"] = "QUEUED"

    # 실행 메타 정리
    try:
        jr.cluster_id = None
    except Exception:
        if isinstance(jr, dict):
            jr["cluster_id"] = None

    try:
        jr.nodes = []
    except Exception:
        if isinstance(jr, dict):
            jr["nodes"] = []

    try:
        jr.g_cur = 0
    except Exception:
        if isinstance(jr, dict):
            jr["g_cur"] = 0

    # 큐 타이밍 갱신
    try:
        jr.last_queue_enter_ts = float(now)
    except Exception:
        if isinstance(jr, dict):
            jr["last_queue_enter_ts"] = float(now)

    # home을 확정(당신 정책: "원래 클러스터에서 다시 돌 수 있게")
    try:
        jr.home_cluster_id = home
    except Exception:
        if isinstance(jr, dict):
            jr["home_cluster_id"] = home

    # 큐 일관성 확보: home 아닌 큐에서 제거 + home에서 중복 제거
    ensure_job_single_queue_locked(state, jid, home)

    # home 큐에 삽입(queue_seq 오름차순)
    q = getattr(state, "cluster_queues", {}).get(home)
    if q is None:
        return

    qjob = QueueJob(job_id=jid)

    fn = globals().get("insert_queuejob_sorted_by_seq_locked")
    if fn is None:
        # fallback: 그냥 append
        try:
            q._jobs = list(getattr(q, "_jobs", []) or []) + [qjob]
            q.eta_dirty = True
        except Exception:
            pass
        return

    fn(q, state, qjob)

def plan_preemption_for_hol_locked(
    state: Any,
    *,
    cluster_id: str,
    hol_job_id: str,
    g_need: int,
    saved_running_job_ids: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    cid = str(cluster_id)
    hol = str(hol_job_id)
    g_need = int(max(1, g_need))

    cr = getattr(state, "clusters", {}).get(cid)
    if cr is None:
        return ([], [])

    nodes_pool = list(getattr(cr, "nodes", []) or [])
    if not nodes_pool:
        return ([], [])

    owner = getattr(state, "node_owner", {}) or {}

    # 현재 free
    free_now = [n for n in nodes_pool if n not in owner]
    if len(free_now) >= g_need:
        return ([], free_now[:g_need])

    # saved는 "프리엠션 금지 목록"
    forbidden: Set[str] = set(str(x) for x in (saved_running_job_ids or []) if x)

    # 후보 수집
    victims: List[str] = []
    for n in nodes_pool:
        o = owner.get(n)
        if not o:
            continue
        o = str(o)
        if o == hol:
            continue
        if o in forbidden:
            # saved job은 절대 희생하지 않음
            continue

        jr = getattr(state, "jobs", {}).get(o)
        if jr is None:
            continue
        if str(getattr(jr, "status", "") or "") != "RUNNING":
            continue

        pb = int(getattr(jr, "preempt_budget", 4) or 4)
        pc = int(getattr(jr, "preempt_count", 0) or 0)
        if pc >= pb:
            continue

        if o not in victims:
            victims.append(o)

    if not victims:
        return ([], [])

    # 정렬: backfill 먼저 -> preempt_count 낮은 것 -> 최근 시작한 것
    def _key(jid: str):
        jr = getattr(state, "jobs", {}).get(str(jid))
        if jr is None:
            return (1, 10**9, 0.0)
        is_bf = bool(getattr(jr, "is_backfill", False))
        pc = int(getattr(jr, "preempt_count", 0) or 0)
        st = float(getattr(jr, "start_ts", 0.0) or 0.0)
        return (0 if is_bf else 1, pc, -st)

    victims.sort(key=_key)

    # victims가 점유한 노드 계산
    victim_set = set(victims)
    preemptable_nodes: List[str] = []
    for n in nodes_pool:
        o = owner.get(n)
        if o and str(o) in victim_set:
            preemptable_nodes.append(n)

    # saved는 건드리지 않으므로, free_now + preemptable_nodes로 g_need 충족 가능해야만 함
    if len(free_now) + len(preemptable_nodes) < g_need:
        return ([], [])

    expected = free_now + preemptable_nodes
    return (victims, expected[:g_need])

def execute_preemption_and_requeue(
    *,
    victim_job_ids: List[str],
    cluster_id: str,
    reason: str = "hol_preempt",
    stop_timeout_sec: float = 200.0,
) -> Dict[str, Any]:
    from app.executor import stop_job as executor_stop_job

    cid = str(cluster_id)
    victims = [str(x) for x in (victim_job_ids or []) if x]

    # 1) PLAN SNAPSHOT (LOCK)
    plan: List[Dict[str, Any]] = []
    with STATE_LOCK:
        state = get_global_state()
        for jid in victims:
            jr = (getattr(state, "jobs", {}) or {}).get(jid)
            if jr is None:
                plan.append({"job_id": jid, "planned": False, "why": "job_missing"})
                continue

            st = str(getattr(jr, "status", "") or "")
            cur_c = getattr(jr, "cluster_id", None)
            cur_c = str(cur_c) if cur_c is not None else None

            # RUNNING만 stop 대상
            if st != "RUNNING":
                plan.append({"job_id": jid, "planned": False, "why": f"not_running status={st}"})
                continue
            # cluster mismatch면 위험(다른 클러스터 죽이기)
            if cur_c is not None and cur_c != cid:
                plan.append({"job_id": jid, "planned": False, "why": f"cluster_mismatch job_cluster={cur_c} hol_cluster={cid}"})
                continue

            home = getattr(jr, "home_cluster_id", None)
            home = str(home) if home is not None else cid

            plan.append(
                {
                    "job_id": jid,
                    "planned": True,
                    "cluster_id": cid,
                    "home_cluster_id": home,
                    "status": st,
                }
            )

    # 2) EXECUTE STOP (NO LOCK)
    stop_results: Dict[str, Dict[str, Any]] = {}
    for rec in plan:
        jid = rec["job_id"]
        if not rec.get("planned", False):
            stop_results[jid] = {"ok": False, "detail": rec.get("why", "not_planned")}
            continue

        resp = executor_stop_job(
            job_id=jid,
            cluster_id=cid,
            reason=f"{reason}:{cid}",
            checkpoint=True,
            timeout_sec=120.0,
        )
        ok = bool(resp.get("ok", False))
        stop_results[jid] = resp

    # 3) COMMIT (LOCK)
    committed: List[str] = []
    skipped: List[Dict[str, Any]] = []

    with STATE_LOCK:
        state = get_global_state()

        for rec in plan:
            jid = rec["job_id"]
            if not rec.get("planned", False):
                skipped.append({"job_id": jid, "why": rec.get("why", "not_planned")})
                continue

            sres = stop_results.get(jid) or {}
            if not bool(sres.get("ok", False)):
                skipped.append({"job_id": jid, "why": "stop_failed", "stop_info": sres})
                continue

            jr = (getattr(state, "jobs", {}) or {}).get(jid)
            if jr is None:
                skipped.append({"job_id": jid, "why": "job_missing_after_stop"})
                continue

            # 3-1) 상태/카운터/노드 해제는 기존 mark_job_preempted가 처리
            try:
                mark_job_preempted(jid, reason=f"{reason}:cluster={cid}")
            except Exception as e:
                skipped.append({"job_id": jid, "why": f"mark_job_preempted_failed: {e}"})
                continue

            committed.append(jid)

        # cluster counters 갱신
        try:
            _recompute_cluster_counters_locked(state, cid)
        except Exception:
            pass

    return {
        "ok": True,
        "cluster_id": cid,
        "committed": committed,
        "skipped": skipped,
        "stop_results": stop_results,
    }

def execute_preemption_and_requeue_locked(
    state: Any,
    *,
    victim_job_ids: List[str],
    cluster_id: str,
) -> None:
    cid = str(cluster_id)
    victims = [str(x) for x in (victim_job_ids or []) if x]
    if not victims:
        return

    # stop_job이 네트워크/RPC라 lock 잡고 있으면 위험 → 완전 해제 후 수행
    with _temporarily_release_state_lock_if_owned():
        execute_preemption_and_requeue(
            victim_job_ids=victims,
            cluster_id=cid,
            reason="hol_preempt",
            stop_timeout_sec=200.0,
        )

def status_snapshot_compact() -> Dict[str, Any]:
    def _job_victim_tag(jr: Any) -> Dict[str, Any]:
        """
        victim 표시 규칙:
        - is_hol_backfill(또는 queue_kind==HOL_BACKFILL) 이어야 함
        - hol_backfill_pin == (현재 배치 cluster_id) 이어야 함  (pin에서 돌고 있는지)
        - 그리고 "찜하지 않은 cluster 배치"면 victim 표시 금지:
            allowed = pinned_cluster or admitted_cluster_id or home_cluster_id
            victim은 (cluster_id == allowed)일 때만 True
        """
        try:
            qk = _as_str(_get_attr_or_key(jr, "queue_kind", "") or "").upper().strip()
        except Exception:
            qk = ""

        try:
            is_hbf = bool(_get_attr_or_key(jr, "is_hol_backfill", False)) or (qk == "HOL_BACKFILL")
        except Exception:
            is_hbf = False

        hol_pin = _as_str(_get_attr_or_key(jr, "hol_backfill_pin", "") or "").strip()
        hol_for = _as_str(_get_attr_or_key(jr, "hol_backfill_for", "") or "").strip()

        cur_cid = _as_str(_get_attr_or_key(jr, "cluster_id", "") or "").strip()
        pinned = _as_str(_get_attr_or_key(jr, "pinned_cluster", "") or "").strip()
        admitted = _as_str(_get_attr_or_key(jr, "admitted_cluster_id", "") or "").strip()
        home = _as_str(_get_attr_or_key(jr, "home_cluster_id", "") or "").strip()

        allowed = pinned or admitted or home  # "찜/허용" 클러스터
        on_pin = bool(hol_pin and cur_cid and hol_pin == cur_cid)
        allowed_ok = bool(cur_cid and allowed and (cur_cid == allowed))

        victim = bool(is_hbf and on_pin and allowed_ok)

        # 표시용(디버깅)
        return {
            "is_backfill": bool(_get_attr_or_key(jr, "is_backfill", False)),
            "is_hol_backfill": bool(is_hbf),
            "hol_pin": hol_pin,
            "hol_for": (hol_for or None),
            "victim": bool(victim),
            "victim_reason": (
                "OK"
                if victim
                else (
                    "not_hol_backfill"
                    if not is_hbf
                    else ("not_on_hol_pin" if not on_pin else "not_allowed_cluster")
                )
            ),
            "allowed_cluster": (allowed or None),
            "cur_cluster": (cur_cid or None),
            "preempt_ok_only_by_hol": bool(_get_attr_or_key(jr, "preempt_ok_only_by_hol", False)),
        }

    def _as_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _get_attr_or_key(obj: Any, k: str, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(k, default)
            return getattr(obj, k, default)
        except Exception:
            return default

    def _extract_job_id(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, dict):
            return _as_str(x.get("job_id") or x.get("id") or "").strip()
        if hasattr(x, "job_id"):
            try:
                return _as_str(getattr(x, "job_id") or "").strip()
            except Exception:
                return ""
        if hasattr(x, "id"):
            try:
                return _as_str(getattr(x, "id") or "").strip()
            except Exception:
                return ""
        return _as_str(x).strip()

    def _queue_items(qobj: Any) -> List[Any]:
        if qobj is None:
            return []
        try:
            return list(getattr(qobj, "_jobs", []) or [])
        except Exception:
            if isinstance(qobj, dict):
                return list(qobj.get("_jobs") or qobj.get("jobs") or [])
        return []

    def _queue_len_and_top2(qobj: Any, jobs: Dict[str, Any]) -> Dict[str, Any]:
        items = _queue_items(qobj)
        ids: List[str] = []
        for x in items:
            jid = _extract_job_id(x)
            if jid:
                ids.append(jid)

        top2 = []
        for jid in ids[:5]:
            jr = jobs.get(str(jid))
            if jr is None:
                top2.append({"job_id": str(jid)})
            else:
                info = {
                    "job_id": str(jid),
                    "model": _as_str(_get_attr_or_key(jr, "model", "") or ""),
                    "dataset": _as_str(_get_attr_or_key(jr, "dataset", "") or ""),
                    "status": _as_str(_get_attr_or_key(jr, "status", "") or ""),
                }
                # ✅ victim 태그(단, 찜하지 않은 클러스터면 victim=False로 찍힘)
                info.update(_job_victim_tag(jr))
                top2.append(info)

        return {"len": int(len(ids)), "top2": top2}

    st = get_global_state()
    out: Dict[str, Any] = {"ts": float(time.time()), "queue": {}, "queues_debug": {}, "clusters": {}}

    with STATE_LOCK:
        clusters = getattr(st, "clusters", {}) or {}
        node_owner = getattr(st, "node_owner", {}) or {}
        jobs = getattr(st, "jobs", {}) or {}

        # (1) global_queue
        gq = getattr(st, "global_queue", None)
        gq_ids: List[str] = []
        if isinstance(gq, list):
            for x in gq:
                jid = _extract_job_id(x)
                if jid:
                    gq_ids.append(jid)

        gq_top2 = []
        for jid in gq_ids[:5]:
            jr = jobs.get(str(jid))
            if jr is None:
                gq_top2.append({"job_id": str(jid)})
            else:
                info = {
                    "job_id": str(jid),
                    "model": _as_str(_get_attr_or_key(jr, "model", "") or ""),
                    "dataset": _as_str(_get_attr_or_key(jr, "dataset", "") or ""),
                    "status": _as_str(_get_attr_or_key(jr, "status", "") or ""),
                }
                info.update(_job_victim_tag(jr))
                gq_top2.append(info)

        out["queue"] = {"len": int(len(gq_ids)), "top2": gq_top2}

        # (2) queues_debug
        cluster_queues = getattr(st, "cluster_queues", {}) or {}
        home_queues = getattr(st, "home_cluster_queues", {}) or {}

        cq_dbg, cq_total = {}, 0
        for cid, q in (cluster_queues or {}).items():
            cid = str(cid)
            snap = _queue_len_and_top2(q, jobs)
            cq_dbg[cid] = snap
            cq_total += int(snap["len"])

        hq_dbg, hq_total = {}, 0
        for cid, q in (home_queues or {}).items():
            cid = str(cid)
            snap = _queue_len_and_top2(q, jobs)
            hq_dbg[cid] = snap
            hq_total += int(snap["len"])

        queued_cnt = 0
        for _jid, jr in (jobs or {}).items():
            if _as_str(_get_attr_or_key(jr, "status", "") or "").upper() == "QUEUED":
                queued_cnt += 1

        out["queues_debug"] = {
            "jobs_queued_count": int(queued_cnt),
            "cluster_queues_total_len": int(cq_total),
            "cluster_queues": cq_dbg,
            "home_queues_total_len": int(hq_total),
            "home_cluster_queues": hq_dbg,
        }

        # (3) clusters (SSOT = node_owner 기준)
        for cid, cr in (clusters or {}).items():
            cid = str(cid)

            nodes: List[str] = []
            try:
                nodes = [_as_str(n) for n in (getattr(cr, "nodes", []) or [])]
            except Exception:
                if isinstance(cr, dict):
                    nodes = [_as_str(n) for n in (cr.get("nodes") or [])]
            nodes = [n for n in nodes if n]

            owners: Dict[str, str] = {}
            for n in nodes:
                jid = node_owner.get(str(n))
                if jid is not None:
                    owners[str(n)] = str(jid)

            total = int(len(nodes))
            used = int(len(owners))
            free = int(max(0, total - used))

            out["clusters"][cid] = {
                "nodes": nodes,
                "node_owner": owners,
                "used_gpus": used,
                "free_gpus": free,
                "total_gpus": total,
            }

    return out

def hol_prepare_nodes_strict_locked(
    state: Any,
    *,
    cluster_id: str,
    hol_job_id: str,
    g_need: int,
    saved_running_job_ids: Optional[List[str]] = None,
    max_preempt_rounds: int = 2,
) -> Optional[List[str]]:
    cid = str(cluster_id)
    hol = str(hol_job_id)
    g_need = int(max(1, g_need))

    cr = getattr(state, "clusters", {}).get(cid)
    if cr is None:
        return None

    nodes_pool = list(getattr(cr, "nodes", []) or [])
    if not nodes_pool:
        return None

    def _free_nodes() -> List[str]:
        owner = getattr(state, "node_owner", {}) or {}
        return [n for n in nodes_pool if n not in owner]

    for r in range(int(max_preempt_rounds)):
        free_now = _free_nodes()
        if len(free_now) >= g_need:
            owner = getattr(state, "node_owner", {}) or {}
            real = [n for n in free_now if n not in owner]
            if len(real) >= g_need:
                return real[:g_need]
            return None

        victims, expected = plan_preemption_for_hol_locked(
            state,
            cluster_id=cid,
            hol_job_id=hol,
            g_need=g_need,
            saved_running_job_ids=saved_running_job_ids,
        )
        if not victims:
            return None

        # stop_job은 lock 밖에서 돌아야 함
        execute_preemption_and_requeue_locked(state, victim_job_ids=victims, cluster_id=cid)

        try:
            _recompute_cluster_counters_locked(state, cid)
        except Exception:
            pass

        # expected vs actual sanity (로그용)
        try:
            free_after = _free_nodes()
            if len(free_after) < min(g_need, len(expected)):
                try:
                    rl = get_run_logger()
                    rl.job_event(
                        event="hol_free_mismatch",
                        job_id=str(hol),
                        cluster=str(cid),
                        world_size=int(g_need),
                        note="expected_free_not_met",
                        metadata={"round": r, "expected": list(expected), "free_after": list(free_after)},
                    )
                except Exception:
                    pass
        except Exception:
            pass

    return None

def _gq_extract_job_id(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        v = x.get("job_id") or x.get("id")
        return str(v) if v else ""
    v = getattr(x, "job_id", None) or getattr(x, "id", None)
    if v:
        return str(v)
    try:
        return str(x)  # QueueJob가 __str__로 job_id 반환하는 경우 호환
    except Exception:
        return ""

def _global_queue_compact_locked(st: Any) -> Dict[str, int]:
    stats = {
        "removed_terminal": 0,
        "removed_duplicates": 0,
        "removed_unknown": 0,
        "scanned": 0,
        "touched_queues": 0,
    }

    jobs = getattr(st, "jobs", {}) or {}
    cluster_queues = getattr(st, "cluster_queues", {}) or {}

    try:
        terminal_set: Set[str] = set(_TERMINAL)
    except Exception:
        terminal_set = {"FINISHED", "FAILED", "CANCELLED", "COMPLETED", "PREEMPTED"}

    def _item_jid(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return str(x.get("job_id") or x.get("id") or "")
        if hasattr(x, "job_id"):
            try:
                return str(getattr(x, "job_id") or "")
            except Exception:
                return ""
        return str(x)

    def _item_g_req(x: Any, default: int = 1) -> int:
        if x is None:
            return int(default)
        if isinstance(x, dict):
            try:
                return max(1, int(x.get("g_req") or default))
            except Exception:
                return int(default)
        if hasattr(x, "g_req"):
            try:
                return max(1, int(getattr(x, "g_req") or default))
            except Exception:
                return int(default)
        return int(default)

    def _job_status(job_id: str) -> str:
        jr = jobs.get(job_id)
        if jr is None:
            return ""
        try:
            if isinstance(jr, dict):
                return str(jr.get("status", "") or "").upper()
            return str(getattr(jr, "status", "") or "").upper()
        except Exception:
            return ""

    def _is_terminal(job_id: str) -> bool:
        return _job_status(job_id) in terminal_set

    def _exists(job_id: str) -> bool:
        return job_id in jobs

    # ✅ NEW: queue kind별로 out에 넣는 타입을 강제한다.
    # kind:
    # - "GLOBAL_STR": out에 jid(str)만
    # - "HOME_STR": out에 jid(str)만
    # - "CLUSTER_DICT": out에 {"job_id": jid, "g_req": int}만
    def _compact_list(lst: List[Any], kind: str) -> List[Any]:
        if not lst:
            return lst
        stats["scanned"] += len(lst)

        seen: Set[str] = set()
        out: List[Any] = []

        for x in lst:
            jid = _item_jid(x)
            if not jid:
                continue

            if not _exists(jid):
                stats["removed_unknown"] += 1
                continue

            if _is_terminal(jid):
                stats["removed_terminal"] += 1
                continue

            if jid in seen:
                stats["removed_duplicates"] += 1
                continue

            seen.add(jid)

            if kind == "CLUSTER_DICT":
                # clusterQ는 dict-only 유지
                g_req = _item_g_req(x, 1)
                out.append({"job_id": str(jid), "g_req": int(g_req)})
            else:
                # global/home은 str-only 유지
                out.append(str(jid))

        return out

    # (1) cluster_queues[*]._jobs  -> dict-only
    if isinstance(cluster_queues, dict):
        for _, q in cluster_queues.items():
            if q is None:
                continue
            try:
                lst = list(getattr(q, "_jobs", []) or [])
            except Exception:
                lst = []
            if not lst:
                continue

            new_lst = _compact_list(lst, kind="CLUSTER_DICT")
            # 타입 강제 때문에 길이가 같아도 내용이 바뀔 수 있으니 "!=" 비교 대신 항상 set 해도 되지만,
            # 최소 수정 유지: 길이 다르거나, dict-only가 아닐 때만 touched
            touched = False
            try:
                if len(new_lst) != len(lst):
                    touched = True
                else:
                    # dict-only 확인
                    for it in lst:
                        if not isinstance(it, dict):
                            touched = True
                            break
            except Exception:
                touched = True

            if touched:
                try:
                    setattr(q, "_jobs", new_lst)
                    stats["touched_queues"] += 1
                except Exception:
                    pass

    # (2) global queue -> str-only
    gq = None
    for cand in ("global_queue", "job_queue", "queue"):
        v = getattr(st, cand, None)
        if v is not None:
            gq = v
            break

    if isinstance(gq, list):
        old = list(gq)
        new = _compact_list(old, kind="GLOBAL_STR")
        # 타입 강제라 길이 같아도 변환될 수 있음 -> 항상 반영
        try:
            gq[:] = new
            stats["touched_queues"] += 1
        except Exception:
            pass

    # (3) home_cluster_queues[*] -> str-only
    hq = getattr(st, "home_cluster_queues", None) or {}
    if isinstance(hq, dict):
        for cid, arr in hq.items():
            if arr is None:
                continue
            if isinstance(arr, list):
                old = list(arr)
                new = _compact_list(old, kind="HOME_STR")
                try:
                    arr[:] = new
                    stats["touched_queues"] += 1
                except Exception:
                    try:
                        hq[str(cid)] = new
                        stats["touched_queues"] += 1
                    except Exception:
                        pass
            else:
                try:
                    old = list(arr)
                except Exception:
                    old = []
                if old:
                    new = _compact_list(old, kind="HOME_STR")
                    try:
                        hq[str(cid)] = new
                        stats["touched_queues"] += 1
                    except Exception:
                        pass

    # (4) mobile_queue -> str-only (있다면)
    mq = getattr(st, "mobile_queue", None)
    if isinstance(mq, list):
        old = list(mq)
        new = _compact_list(old, kind="GLOBAL_STR")
        try:
            mq[:] = new
            stats["touched_queues"] += 1
        except Exception:
            pass

    # (5) home_mobile_queue -> (list면 str-only, dict면 key job_id 정리)
    hm = getattr(st, "home_mobile_queue", None)
    if isinstance(hm, list):
        old = list(hm)
        new = _compact_list(old, kind="GLOBAL_STR")
        try:
            hm[:] = new
            stats["touched_queues"] += 1
        except Exception:
            pass
    elif isinstance(hm, dict):
        keys = list(hm.keys())
        if keys:
            stats["scanned"] += len(keys)
            for k in keys:
                sjid = str(k)
                if (not _exists(sjid)) or _is_terminal(sjid):
                    try:
                        hm.pop(k, None)
                        stats["touched_queues"] += 1
                        if not _exists(sjid):
                            stats["removed_unknown"] += 1
                        else:
                            stats["removed_terminal"] += 1
                    except Exception:
                        pass

    return stats

def repair_global_queue_locked(st: Any) -> None:
    jobs: Dict[str, Any] = getattr(st, "jobs", {}) or {}

    # 기존 global_queue 읽기
    old: List[str] = []
    try:
        old = [str(x) for x in (getattr(st, "global_queue", []) or []) if str(x)]
    except Exception:
        old = []

    newq: List[str] = []
    seen = set()

    # 1) 기존 순서 유지: "존재 + QUEUED"만 남김
    for jid in old:
        if jid in seen:
            continue
        jr = jobs.get(str(jid))
        if jr is None:
            continue
        if not _is_queued(jr):
            continue
        newq.append(str(jid))
        seen.add(str(jid))

    # 2) 누락된 QUEUED 복구: key 기준으로 뒤에 추가
    missing: List[Tuple[float, str]] = []
    for jid0, jr in jobs.items():
        jid = str(jid0)
        if jid in seen:
            continue
        if not _is_queued(jr):
            continue
        missing.append((_job_sort_key_for_global_order(jr), jid))

    missing.sort(key=lambda x: (x[0], x[1]))
    for _k, jid in missing:
        newq.append(jid)
        seen.add(jid)

    try:
        st.global_queue = list(newq)
    except Exception:
        pass

def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]

def register_job_on_submit(
    *,
    job_id: str,
    model: str,
    dataset: str,
    user_id: Optional[str],
    policy: Dict[str, Any],
    g_target: int,
    profiling: Optional[Dict[str, Any]],
    admitted_cluster_id: str,
    user_pinned_cluster_id: Optional[str] = None,
    epochs: int = 20,
    batch_size_per_gpu: int = 32,
):
    from types import SimpleNamespace

    now_ts = time.time()
    jid = str(job_id).strip()
    if not jid:
        raise ValueError("job_id is empty")

    # -----------------------------
    # 0) SSOT job object 생성/갱신 (LOCK)
    # -----------------------------
    with STATE_LOCK:
        state = get_global_state()

        if (not hasattr(state, "jobs")) or (getattr(state, "jobs") is None) or (not isinstance(getattr(state, "jobs"), dict)):
            state.jobs = {}
        jobs = state.jobs

        jr = jobs.get(jid)
        if jr is None:
            RuntimeCls = None
            for cand in ("RuntimeJobState", "JobRuntimeState", "RuntimeJob"):
                RuntimeCls = globals().get(cand)
                if RuntimeCls is not None:
                    break
            if RuntimeCls is not None:
                try:
                    jr = RuntimeCls(job_id=jid)
                except Exception:
                    jr = SimpleNamespace(job_id=jid)
            else:
                jr = SimpleNamespace(job_id=jid)
            jobs[jid] = jr

        # core fields
        jr.job_id = jid
        jr.model = str(model)
        jr.dataset = str(dataset)
        jr.user_id = user_id
        jr.policy = policy or {}
        jr.profiling = profiling or {}

        # status
        try:
            jr.status = "QUEUED"
        except Exception:
            pass

        # gang 판정
        jr.is_gang = bool(((str(model) == "DenseNet-121") and (str(dataset) == "TinyImageNet")) or bool(getattr(jr, "is_gang", False)))
        jr.gang_size = 4 if jr.is_gang else 0

        # timestamps
        jr.submit_ts = float(getattr(jr, "submit_ts", None) or now_ts)
        if getattr(jr, "enqueue_ts", None) is None:
            try:
                jr.enqueue_ts = float(now_ts)
            except Exception:
                pass

        # run_id
        if not getattr(jr, "run_id", None):
            try:
                jr.run_id = _new_run_id()
            except Exception:
                pass

        # runtime reset
        jr.start_ts = None
        jr.end_ts = None
        jr.g_cur = 0
        jr.nodes = []

        # queued accounting init
        if getattr(jr, "queued_total_sec", None) is None:
            jr.queued_total_sec = 0.0
        if getattr(jr, "queued_accum_sec", None) is None:
            jr.queued_accum_sec = 0.0
        if not hasattr(jr, "queue_enter_ts"):
            jr.queue_enter_ts = None
        if not hasattr(jr, "last_queue_enter_ts"):
            jr.last_queue_enter_ts = None
        if getattr(jr, "last_queue_enter_ts", None) is None:
            jr.last_queue_enter_ts = float(now_ts)
            jr.queue_enter_ts = float(now_ts)

        jr.epochs = int(epochs)
        jr.batch_size_per_gpu = int(batch_size_per_gpu)

        if getattr(jr, "is_backfill", None) is None:
            jr.is_backfill = False
        if getattr(jr, "preempt_count", None) is None:
            jr.preempt_count = 0
        if getattr(jr, "preempt_budget", None) is None:
            jr.preempt_budget = 4
        if getattr(jr, "progress", None) is None:
            jr.progress = 0.0

        # ---- pin ----
        if (not hasattr(state, "pinned_cluster")) or (getattr(state, "pinned_cluster") is None) or (not isinstance(getattr(state, "pinned_cluster"), dict)):
            state.pinned_cluster = {}

        pin = str(user_pinned_cluster_id).strip() if user_pinned_cluster_id else None
        try:
            jr.user_pinned_cluster_id = pin
        except Exception:
            pass

        if pin:
            jr.pinned_cluster = pin
            state.pinned_cluster[jid] = pin
        else:
            jr.pinned_cluster = None
            state.pinned_cluster.pop(jid, None)

        # queue seq
        _ensure_queue_seq_locked(state, jid)

        # submit 단계에서 기존 큐 잔존 제거
        try:
            purge_job_from_all_queues_locked(state, jid, purge_global=True, purge_cluster=True, purge_home=True)
        except Exception:
            pass

        # global_queue 보장
        if (not hasattr(state, "global_queue")) or (getattr(state, "global_queue") is None) or (not isinstance(getattr(state, "global_queue"), list)):
            state.global_queue = []

        # -----------------------------
        # ✅ 핵심: submit에서 "결정" 금지, "힌트"만 저장
        # -----------------------------
        # home_cluster_id: 기본 복귀 위치(로컬리티/리큐잉용)로만 유지
        base_home = str(admitted_cluster_id).strip() if admitted_cluster_id else ""
        if getattr(jr, "home_cluster_id", None) is None and base_home:
            jr.home_cluster_id = base_home

        # preferred_cluster_id: 선호(배치 시점에 우선 시도)
        try:
            jr.preferred_cluster_id = (pin or base_home or None)
        except Exception:
            pass

        # admitted_cluster_id: feeder가 따라가며 고정되는 주범 → submit에서 확정하지 않음
        # (하위호환용으로 필드가 필요하다면 None/""로 둬라)
        try:
            jr.admitted_cluster_id = None
        except Exception:
            pass

        # g_target은 "최종"이 아니라 default hint
        # gang이면 항상 4
        g_hint_default = 4 if bool(getattr(jr, "is_gang", False)) else int(g_target or 1)
        try:
            jr.g_target_hint = int(max(1, g_hint_default))
        except Exception:
            pass

        # cluster별 g_hint_by_cluster:
        # - 이미 profiling/policy 기반으로 계산해 넣었다면 그대로 사용
        # - 없으면 최소한 home/preferred에 대해서라도 default hint를 넣어둠
        try:
            if not isinstance(getattr(jr, "g_hint_by_cluster", None), dict):
                jr.g_hint_by_cluster = {}
        except Exception:
            pass

        try:
            mp = jr.g_hint_by_cluster
            if isinstance(mp, dict):
                # pin이 있으면 pin만 강제
                if pin:
                    mp[str(pin)] = 4 if bool(getattr(jr, "is_gang", False)) else int(max(1, g_hint_default))
                else:
                    # 기본적으로 preferred/home에는 힌트를 넣어둠
                    pref = str(getattr(jr, "preferred_cluster_id", "") or "").strip()
                    home = str(getattr(jr, "home_cluster_id", "") or "").strip()
                    if pref:
                        mp[pref] = int(max(1, g_hint_default))
                    if home and home not in mp:
                        mp[home] = int(max(1, g_hint_default))
        except Exception:
            pass

        # global_queue 삽입 (정렬 insert)
        inserted = False
        try:
            _global_queue_insert_sorted_locked(state, jid)
            inserted = True
        except Exception:
            inserted = False

        if not inserted:
            try:
                if jid not in [str(x) for x in state.global_queue]:
                    state.global_queue.append(jid)
            except Exception:
                pass

        # DEBUG 로그 (핵심 필드 포함)
        try:
            glen = len(state.global_queue) if isinstance(getattr(state, "global_queue", None), list) else -1
            logger.error(
                "[SUBMIT_ENQ] jid=%s status=%s gq_len=%s pin=%s preferred=%s home=%s g_hint_default=%s g_hint_by_cluster=%s run_id=%s enqueue_ts=%s",
                jid,
                getattr(jr, "status", None),
                glen,
                getattr(jr, "pinned_cluster", None),
                getattr(jr, "preferred_cluster_id", None),
                getattr(jr, "home_cluster_id", None),
                int(getattr(jr, "g_target_hint", 0) or 0),
                getattr(jr, "g_hint_by_cluster", None),
                getattr(jr, "run_id", None),
                getattr(jr, "enqueue_ts", None),
            )
        except Exception:
            pass

        return jr

from contextlib import contextmanager

@contextmanager
def _temporarily_release_state_lock_if_owned():
    depth = 0
    is_owned = getattr(STATE_LOCK, "_is_owned", None)

    if callable(is_owned) and is_owned():
        while callable(is_owned) and is_owned():
            try:
                STATE_LOCK.release()
                depth += 1
            except RuntimeError:
                break

    try:
        yield
    finally:
        for _ in range(depth):
            try:
                STATE_LOCK.acquire()
            except Exception:
                break

def list_running_backfills_in_cluster_locked(state: Any, cluster_id: str) -> List[str]:
    out: List[str] = []
    cid = str(cluster_id)
    for jr in list(getattr(state, "jobs", {}).values()):
        if getattr(jr, "status", None) != "RUNNING":
            continue
        if str(getattr(jr, "cluster_id", "")) != cid:
            continue
        if bool(getattr(jr, "is_backfill", False)):
            out.append(str(getattr(jr, "job_id", "")))
    return [x for x in out if x]

def insert_queuejob_front_locked(q: Any, qjob: Any) -> None:
    jid = str(getattr(qjob, "job_id", "") or "")
    if not jid:
        return

    jobs = list(getattr(q, "_jobs", []) or [])

    # 기존 중복 제거
    new_jobs = []
    for x in jobs:
        xjid = getattr(x, "job_id", None)
        if xjid is None and isinstance(x, str):
            xjid = x
        if str(xjid) == jid:
            continue
        new_jobs.append(x)

    new_jobs.insert(0, qjob)
    setattr(q, "_jobs", new_jobs)

    try:
        q.eta_dirty = True
    except Exception:
        pass

def ensure_job_single_queue_front_locked(state: Any, job_id: str, prefer_cluster_id: str) -> None:
    jid = str(job_id)
    cid = str(prefer_cluster_id)

    # 모든 큐에서 제거
    ensure_job_single_queue_locked(state, jid, None)

    qs = getattr(state, "cluster_queues", {}) or {}
    q = qs.get(cid)
    if q is None:
        return

    try:
        from app.queue import QueueJob
        jr = (getattr(state, "jobs", {}) or {}).get(jid)
        g_req = int(getattr(jr, "g_target", 1) or 1) if jr is not None else 1
        seq = int(getattr(jr, "queue_seq", 10**18) or 10**18) if jr is not None else 10**18
        qjob = QueueJob(job_id=jid, g_req=g_req, enqueue_seq=seq)
    except Exception:
        qjob = jid

    if isinstance(qjob, str):
        # str만 있으면 그냥 앞에
        jobs = list(getattr(q, "_jobs", []) or [])
        jobs = [x for x in jobs if str(getattr(x, "job_id", x)) != jid]
        jobs.insert(0, jid)
        setattr(q, "_jobs", jobs)
        try:
            q.eta_dirty = True
        except Exception:
            pass
        return

    insert_queuejob_front_locked(q, qjob)

def insert_queuejob_sorted_by_seq_locked(q: Any, state: Any, qjob: Any) -> None:
    jobs = list(getattr(q, "_jobs", []) or [])
    jid = str(getattr(qjob, "job_id", "") or "")
    if not jid:
        return

    # 중복 방지
    for x in jobs:
        if str(getattr(x, "job_id", "") or "") == jid:
            return

    jr = (getattr(state, "jobs", {}) or {}).get(jid)
    seq_new = int(getattr(jr, "queue_seq", 10**18) or 10**18)

    def seq_of(x: Any) -> int:
        jx = str(getattr(x, "job_id", "") or "")
        rx = (getattr(state, "jobs", {}) or {}).get(jx)
        return int(getattr(rx, "queue_seq", 10**18) or 10**18)

    idx = len(jobs)
    for i, x in enumerate(jobs):
        if seq_new < seq_of(x):
            idx = i
            break

    jobs.insert(idx, qjob)
    setattr(q, "_jobs", jobs)

    # ETA 캐시가 있다면 dirty로
    try:
        q.eta_dirty = True
    except Exception:
        pass

def mark_job_completed(*, job_id: str, end_ts: float, status: str):
    st = get_global_state()
    jid = str(job_id)
    end_ts_f = float(end_ts)

    with STATE_LOCK:
        jr = st.jobs.get(jid)
        if jr is None:
            return None

        cur_status = str(getattr(jr, "status", "") or "").upper()
        if cur_status in _TERMINAL:
            return jr

        # 종료 시각/상태 기록
        try:
            jr.end_ts = end_ts_f
        except Exception:
            if isinstance(jr, dict):
                jr["end_ts"] = end_ts_f

        final_status = str(status or "").upper()
        if final_status not in _TERMINAL:
            final_status = "FINISHED"

        try:
            jr.status = final_status
        except Exception:
            if isinstance(jr, dict):
                jr["status"] = final_status

        cap_cluster = getattr(jr, "cluster_id", None)

        # 큐에서 제거(terminal job이 큐에 남으면 SSOT 붕괴)
        purge_job_from_all_queues_locked(st, jid)

        # 노드 해제(SSOT)
        try:
            release_nodes_for_job_locked(job_id=jid, cluster_id=cap_cluster)
        except Exception:
            pass

        # running_jobs 캐시 제거 + 카운터
        try:
            cid = str(cap_cluster) if cap_cluster is not None else None
            if cid:
                cr = getattr(st, "clusters", {}).get(cid)
                if cr is not None and isinstance(getattr(cr, "running_jobs", None), dict):
                    cr.running_jobs.pop(jid, None)
                _recompute_cluster_counters_locked(st, cid)
        except Exception:
            pass

        # runtime 정리
        try:
            jr.g_cur = 0
            jr.world_size = 0
            jr.nodes = []
            jr.is_backfill = False
        except Exception:
            pass

        return jr

def _push_front_unique(lst: list, jid: str) -> None:
    jid = str(jid)
    if not isinstance(lst, list):
        return
    # 기존 jid 제거 후 맨 앞 삽입
    lst[:] = [x for x in lst if str(x) != jid]
    lst.insert(0, jid)

def mark_job_preempted(
    *,
    job_id: str,
    reason: str = "preempted",
    requeue_front: bool = True,
    requeue_kind: str = "HOME",   # "HOME"만 쓰는 걸 추천 (지금 구조 기준)
    cluster_id_hint: Optional[str] = None,
) -> None:
    from app.scheduler_state import STATE_LOCK, get_global_state, _recompute_cluster_counters_locked
    # ↑ 파일 내부라면 import 필요 없고 직접 참조하면 됩니다.

    jid = str(job_id)

    with STATE_LOCK:
        st = get_global_state()
        jobs = getattr(st, "jobs", {}) or {}
        jr = jobs.get(jid)
        if jr is None:
            return

        now_ts = float(time.time())

        # home cluster 결정
        home = getattr(jr, "home_cluster_id", None) or getattr(jr, "admitted_cluster_id", None) or cluster_id_hint
        home = str(home) if home else str(cluster_id_hint or "")

        # 메타 업데이트
        try:
            jr.preempt_count = int(getattr(jr, "preempt_count", 0) or 0) + 1
        except Exception:
            pass
        try:
            jr.last_preempt_ts = float(now_ts)
            jr.last_preempt_reason = str(reason)
        except Exception:
            pass

        # ✅ (추가) requeue_front 우선순위를 feeder 정렬에서 보존하기 위한 플래그
        # - 핵심: HOME 큐에서 enqueue_ts 정렬에 의해 "front" 의도가 사라지는 문제 해결
        try:
            jr.requeue_front = bool(requeue_front)
            jr.requeue_priority_ts = float(now_ts)  # 디버깅/추적용
        except Exception:
            pass

        # status는 재실행 가능하게 QUEUED로
        try:
            jr.status = "QUEUED"
            jr.is_backfill = False
            jr.is_hol_backfill = False
            jr.hol_blocking_job_id = None
            jr.hol_backfill_since_ts = None
            jr.cluster_id = None
            jr.g_cur = 0
            jr.nodes = []
            jr.last_queue_enter_ts = float(now_ts)
        except Exception:
            pass

        # single-queue invariant
        if home:
            try:
                ensure_job_single_queue_locked(st, jid, home)
            except Exception:
                pass

        # global_queue 맨 앞
        gq = getattr(st, "global_queue", None)
        if isinstance(gq, list):
            if requeue_front:
                _push_front_unique(gq, jid)
            else:
                # 뒤에 넣고 중복 제거
                gq[:] = [x for x in gq if str(x) != jid]
                gq.append(jid)

        # home_cluster_queues 맨 앞
        hq = getattr(st, "home_cluster_queues", None)
        if isinstance(hq, dict) and home:
            lst = hq.get(home)
            if not isinstance(lst, list):
                lst = []
                hq[home] = lst
            if requeue_front:
                _push_front_unique(lst, jid)
            else:
                lst[:] = [x for x in lst if str(x) != jid]
                lst.append(jid)

        # counters (optional)
        try:
            if home:
                _recompute_cluster_counters_locked(st, str(home))
        except Exception:
            pass

def pick_free_nodes_ssot_locked(state: Any, *, cluster_id: str, k: int) -> List[str]:
    cid = str(cluster_id)
    kk = int(max(0, k))
    if kk <= 0:
        return []

    cr = (getattr(state, "clusters", {}) or {}).get(cid)
    if cr is None:
        return []

    try:
        nodes = list(getattr(cr, "nodes", []) or [])
    except Exception:
        nodes = list(cr.get("nodes") or []) if isinstance(cr, dict) else []

    owner = getattr(state, "node_owner", {}) or {}
    free = [str(n) for n in nodes if str(n) not in owner]
    return free[:kk]

def status_snapshot(recompute: bool = False) -> Dict[str, Any]:
    import time

    st = get_global_state()

    snap: Dict[str, Any] = {
        "ts": float(time.time()),
        "clusters": {},
        "jobs": {},
        "node_owner": {},
        "queues": {
            "global_queue": [],
            "cluster_queues": {},
        },
        "simple": {"clusters": {}},
    }

    with STATE_LOCK:
        try:
            snap["node_owner"] = dict(getattr(st, "node_owner", {}) or {})
        except Exception:
            snap["node_owner"] = {}

        try:
            gq = getattr(st, "global_queue", None)
            if isinstance(gq, list):
                snap["queues"]["global_queue"] = [str(x) for x in gq]
        except Exception:
            pass

        try:
            qs = getattr(st, "cluster_queues", {}) or {}
            for cid, q in qs.items():
                items = []
                try:
                    items = list(getattr(q, "_jobs", []) or [])
                except Exception:
                    items = []
                def _jid(x: Any) -> Optional[str]:
                    if x is None:
                        return None
                    if isinstance(x, str):
                        return str(x)
                    return getattr(x, "job_id", None) or getattr(x, "id", None)
                snap["queues"]["cluster_queues"][str(cid)] = [str(_jid(x)) for x in items if _jid(x)]
        except Exception:
            pass

        # ---- clusters ----
        clusters = getattr(st, "clusters", {}) or {}
        for cid, cr in clusters.items():
            cid = str(cid)
            try:
                nodes = list(getattr(cr, "nodes", []) or [])
            except Exception:
                nodes = []

            per_owner = {}
            try:
                for n in nodes:
                    owner = snap["node_owner"].get(str(n))
                    if owner is not None:
                        per_owner[str(n)] = owner
            except Exception:
                per_owner = {}

            if recompute:
                try:
                    _recompute_cluster_counters_locked(st, cid)
                except Exception:
                    pass

            snap["clusters"][cid] = {
                "cluster_id": cid,
                "nodes": nodes,
                "node_owner": per_owner,
                "total_gpus": int(getattr(cr, "total_gpus", 0) or 0),
                "used_gpus": int(getattr(cr, "used_gpus", 0) or 0),
                "free_gpus": int(getattr(cr, "free_gpus", 0) or 0),
                "util": float(getattr(cr, "util", 0.0) or 0.0),
                "power_current_w": float(getattr(cr, "power_current_w", 0.0) or 0.0),
                "speed_factor": float(getattr(cr, "speed_factor", 1.0) or 1.0),
                "price_per_gpu_hour": float(getattr(cr, "price_per_gpu_hour", 1.0) or 1.0),
            }

        # ---- jobs ----
        jobs = getattr(st, "jobs", {}) or {}
        for jid, jr in jobs.items():
            jid = str(jid)

            def _fget(name, default=None):
                try:
                    return getattr(jr, name, default)
                except Exception:
                    if isinstance(jr, dict):
                        return jr.get(name, default)
                    return default

            status = str(_fget("status", "") or "")
            status_u = status.upper()

            cluster_id = _fget("cluster_id", None)
            cluster_id = str(cluster_id) if cluster_id is not None else None

            try:
                nodes = list(_fget("nodes", []) or [])
            except Exception:
                nodes = []

            epoch_done = _fget("epoch_done", None)
            if epoch_done is None:
                epoch_done = _fget("epochs_done", None)
            if epoch_done is None:
                epoch_done = _fget("epoch", None)

            g_cur = int(_fget("g_cur", 0) or 0)
            g_target = int(_fget("g_target", 0) or 0)

            # ✅ 핵심: world_size를 의미 있게
            world_size = g_cur if status_u == "RUNNING" else (g_target if g_target > 0 else g_cur)

            snap["jobs"][jid] = {
                "job_id": jid,
                "status": status,
                "cluster_id": cluster_id,
                "nodes": nodes,
                "is_backfill": bool(_fget("is_backfill", False)),
                "model": _fget("model", None) or _fget("model_name", None),
                "model_name": _fget("model_name", None) or _fget("model", None),
                "dataset": _fget("dataset", None),
                "user_id": _fget("user_id", None),
                "g_cur": g_cur,
                "g_target": g_target,
                "world_size": int(world_size),
                "submit_ts": float(_fget("submit_ts", 0.0) or 0.0),
                "start_ts": _fget("start_ts", None),
                "end_ts": _fget("end_ts", None),
                "last_queue_enter_ts": _fget("last_queue_enter_ts", None),
                "queued_accum_sec": float(_fget("queued_accum_sec", 0.0) or 0.0),
                "pinned_cluster": _fget("pinned_cluster", None),
                "home_cluster_id": _fget("home_cluster_id", None),
                "queue_seq": int(_fget("queue_seq", 0) or 0),
                "epoch_done": epoch_done,
                "epochs": _fget("epochs", None),
            }

        # SIMPLE VIEW 구성
        try:
            qs = getattr(st, "cluster_queues", {}) or {}
            for cid, cr in (getattr(st, "clusters", {}) or {}).items():
                cid = str(cid)

                # 노드 -> 잡
                nodes = []
                try:
                    nodes = list(getattr(cr, "nodes", []) or [])
                except Exception:
                    pass

                node_map = {}
                owner = snap["node_owner"]
                for n in nodes:
                    n = str(n)
                    node_map[n] = owner.get(n, "-")

                # 큐 길이 + top2(model,dataset, g_target)
                qlen = 0
                top2 = []
                q = qs.get(cid)
                if q is not None:
                    try:
                        items = list(getattr(q, "_jobs", []) or [])
                    except Exception:
                        items = []
                    qlen = len(items)

                    def _jid(x: Any) -> Optional[str]:
                        if x is None:
                            return None
                        if isinstance(x, str):
                            return x
                        return getattr(x, "job_id", None) or getattr(x, "id", None)

                    for x in items[:5]:
                        qjid = _jid(x)
                        if not qjid:
                            continue
                        rj = (getattr(st, "jobs", {}) or {}).get(str(qjid))
                        if rj is None:
                            top2.append({"job_id": str(qjid), "model": None, "dataset": None, "g_target": None})
                        else:
                            top2.append({
                                "job_id": str(qjid),
                                "model": getattr(rj, "model", None) or getattr(rj, "model_name", None),
                                "dataset": getattr(rj, "dataset", None),
                                "g_target": int(getattr(rj, "g_target", 0) or 0),
                            })

                snap["simple"]["clusters"][cid] = {
                    "nodes": node_map,
                    "queue_len": int(qlen),
                    "queue_top2": top2,
                }
        except Exception:
            pass

    return snap

def health_snapshot() -> Dict[str, Any]:
    try:
        state = get_global_state()
        return {"ok": True, "clusters": list(state.clusters.keys())}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def set_job_elastic_need(job_id: str, need: bool, reasons: Optional[List[str]] = None) -> None:
    state = get_global_state()
    with STATE_LOCK:
        jr = state.jobs.get(job_id)
        if jr is None:
            return
        try:
            jr.elastic_need = bool(need)
            jr.elastic_reasons = list(reasons or [])
        except Exception:
            # dict 형태 대응
            if isinstance(jr, dict):
                jr["elastic_need"] = bool(need)
                jr["elastic_reasons"] = list(reasons or [])

def reserve_nodes(*, job_id: str, cluster_id: str, nodes: List[str]) -> None:
    """
    SSOT: state.node_owner 만이 "점유"에 대한 진실.
    node_cluster는 'node -> cluster 소속' 정적(reference) 맵으로 사용(가능하면 덮어쓰기/삭제 금지).

    보장:
    - 같은 job이 과거에 잡고 있던 노드는 먼저 전부 해제(유령 점유 방지)
      (단, node_cluster는 건드리지 않음)
    - 타 job 점유 노드와 충돌하면 예외
    - 예약 반영 후 affected cluster들만 카운터/플래그 갱신(가능하면 정확히)
    - (호환) state.clusters[cid].node_owner 뷰가 있으면 SSOT에서 재구성하여 동기화
    """
    jid = str(job_id).strip()
    cid = str(cluster_id).strip()
    nodes = [str(n).strip() for n in (nodes or []) if n is not None and str(n).strip()]

    if not jid:
        raise RuntimeError("reserve_nodes: missing job_id")
    if not cid:
        raise RuntimeError("reserve_nodes: missing cluster_id")
    if not nodes:
        raise RuntimeError("reserve_nodes: empty nodes")

    with STATE_LOCK:
        state = get_global_state()

        # --- SSOT owner map ---
        owners = getattr(state, "node_owner", None)
        if not isinstance(owners, dict):
            owners = {}
            try:
                state.node_owner = owners
            except Exception:
                pass

        # --- node -> cluster reference map (do NOT treat as occupancy SSOT) ---
        n2c = getattr(state, "node_cluster", None)
        if not isinstance(n2c, dict):
            n2c = {}
            try:
                state.node_cluster = n2c
            except Exception:
                pass

        # --- clusters container (optional view) ---
        clusters = getattr(state, "clusters", None)
        if not isinstance(clusters, dict):
            clusters = {}
            try:
                state.clusters = clusters
            except Exception:
                pass

        # -------- helpers --------
        def _get_cluster_nodes(cobj: Any) -> List[str]:
            try:
                ns = cobj.get("nodes") if isinstance(cobj, dict) else getattr(cobj, "nodes", None)
            except Exception:
                ns = None
            if not ns:
                return []
            out = []
            for x in ns:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    out.append(s)
            return out

        def _infer_cluster_of_node(node: str) -> Optional[str]:
            # 1) reference map
            try:
                c0 = n2c.get(node)
                if c0 is not None:
                    cs = str(c0).strip()
                    if cs:
                        return cs
            except Exception:
                pass
            # 2) fallback scan clusters
            try:
                for ccid, cobj in (clusters or {}).items():
                    ccid_s = str(ccid).strip()
                    if not ccid_s:
                        continue
                    cnodes = _get_cluster_nodes(cobj)
                    if node in cnodes:
                        # backfill reference map (best-effort)
                        try:
                            n2c[node] = ccid_s
                        except Exception:
                            pass
                        return ccid_s
            except Exception:
                pass
            return None

        def _sync_cluster_view(ccid: str) -> None:
            cobj = clusters.get(ccid) if isinstance(clusters, dict) else None
            if cobj is None:
                return
            cnodes = _get_cluster_nodes(cobj)
            if not cnodes:
                # nodes 목록이 없으면 정확한 재구성이 어려우므로, 최소한 job 잔존만 제거는 reserve에서 다루지 않음
                return
            view = {}
            for nn in cnodes:
                oo = owners.get(nn)
                if oo is not None:
                    view[nn] = oo
            try:
                if isinstance(cobj, dict):
                    cobj["node_owner"] = view
                else:
                    setattr(cobj, "node_owner", view)
            except Exception:
                pass

        # ----------------------------
        # 0) reclaim ALL previous nodes owned by this job (any cluster)
        #    IMPORTANT: do NOT mutate n2c here (it's reference)
        # ----------------------------
        owned_prev: List[str] = []
        affected: set = set()

        for n0, o0 in list(owners.items()):
            if o0 is None or str(o0) != jid:
                continue
            n0s = str(n0).strip()
            if not n0s:
                continue
            owned_prev.append(n0s)
            c_prev = _infer_cluster_of_node(n0s)
            if c_prev:
                affected.add(str(c_prev))

        if owned_prev:
            try:
                logger.warning(
                    "[SSOT_RESERVE_RECLAIM_PREV] job_id=%s new_cid=%s releasing_prev_nodes=%s",
                    jid, cid, owned_prev
                )
            except Exception:
                pass

            for n0s in owned_prev:
                owners.pop(n0s, None)

        # ----------------------------
        # 1) conflict check (block only other job owners)
        # ----------------------------
        for n in nodes:
            cur = owners.get(n)
            if cur is not None and str(cur) != jid:
                try:
                    logger.warning(
                        "[SSOT_RESERVE_CONFLICT] node=%s cur_owner=%s new_owner=%s cid=%s",
                        n, cur, jid, cid
                    )
                except Exception:
                    pass
                raise RuntimeError(f"reserve_conflict: node={n} owner={cur} new_owner={jid}")

        # ----------------------------
        # 2) apply reservation (SSOT owner)
        # ----------------------------
        for n in nodes:
            owners[n] = jid
            # reference map 보강(없을 때만)
            if n not in n2c or not str(n2c.get(n) or "").strip():
                try:
                    n2c[n] = cid
                except Exception:
                    pass

        affected.add(cid)

        # nodes의 실제 소속을 정확히 반영(가능하면)
        try:
            for n in nodes:
                c1 = _infer_cluster_of_node(n)
                if c1:
                    affected.add(str(c1))
        except Exception:
            pass

        # ----------------------------
        # 3) recompute counters/flags (affected only; fallback to all)
        # ----------------------------
        try:
            if affected:
                for c in sorted({str(x).strip() for x in affected if str(x).strip()}):
                    try:
                        _recompute_cluster_counters_locked(state, c)
                    except Exception:
                        pass
                    try:
                        _set_free_changed_locked(state, c)
                    except Exception:
                        pass
            else:
                for c in list((getattr(state, "clusters", {}) or {}).keys()):
                    c_str = str(c).strip()
                    if not c_str:
                        continue
                    try:
                        _recompute_cluster_counters_locked(state, c_str)
                    except Exception:
                        pass
                    try:
                        _set_free_changed_locked(state, c_str)
                    except Exception:
                        pass
        except Exception:
            # 최후 안전망: 현재 cid만이라도 갱신
            try:
                _recompute_cluster_counters_locked(state, cid)
            except Exception:
                pass
            try:
                _set_free_changed_locked(state, cid)
            except Exception:
                pass

        # ----------------------------
        # 4) (optional) sync cluster view node_owner from SSOT (for affected clusters)
        # ----------------------------
        try:
            if isinstance(clusters, dict) and clusters:
                for c in sorted({str(x).strip() for x in affected if str(x).strip()}):
                    _sync_cluster_view(c)
        except Exception:
            pass

        try:
            logger.info("[SSOT_RESERVED] job_id=%s cluster_id=%s nodes=%s", jid, cid, nodes)
        except Exception:
            pass

def transfer_node_locked(
    state: Any,
    *,
    node: str,
    from_job: str,
    to_job: str,
    cluster_id: str,
) -> bool:
    node = str(node)
    from_job = str(from_job)
    to_job = str(to_job)
    cluster_id = str(cluster_id)

    # from_job이 실제 소유 중이 아니면 실패
    if (state.node_owner or {}).get(node) != from_job:
        return False

    state.node_owner[node] = to_job
    state.node_cluster[node] = cluster_id
    return True

def compute_preferred_nodes_for_resize_locked(
    state: Any,
    job_id: str,
    cluster_id: str,
    nodes_pool: List[str],
    new_g: int,
) -> Optional[List[str]]:
    if new_g <= 0:
        return None

    # 1) nodes_pool 순서 맵(고정 순서가 곧 rank/order)
    order = {n: i for i, n in enumerate(nodes_pool)}

    # 2) 현재 job이 예약한 노드(=스케줄러 진실) 수집
    cur = []
    for n, owner in state.node_owner.items():
        if owner == job_id and state.node_cluster.get(n) == cluster_id and n in order:
            cur.append(n)

    # 3) "현재 노드 순서"를 pool order로 정규화 (중요: prefix 정의를 일관되게)
    cur.sort(key=lambda n: order[n])

    # 4) downscale: prefix 자르기
    if len(cur) >= new_g:
        return cur[:new_g]

    # 5) upscale: prefix 유지 + 뒤를 pool 순서로 채우기
    chosen = list(cur)
    chosen_set = set(chosen)

    # free 노드 = node_owner에 없는 노드
    free = [n for n in nodes_pool if n not in state.node_owner]

    for n in free:
        if n in chosen_set:
            continue
        chosen.append(n)
        chosen_set.add(n)
        if len(chosen) >= new_g:
            break

    if len(chosen) != new_g:
        return None

    chosen.sort(key=lambda n: order[n])
    return chosen

def _ensure_queue_seq_locked(state: Any, job_id: str) -> int:
    global _ENQUEUE_SEQ
    jid = str(job_id)

    jr = (getattr(state, "jobs", {}) or {}).get(jid)
    if jr is None:
        return 10**18

    seq = getattr(jr, "queue_seq", None)
    try:
        if seq is not None and int(seq) >= 0:
            return int(seq)
    except Exception:
        pass

    _ENQUEUE_SEQ += 1
    try:
        jr.queue_seq = int(_ENQUEUE_SEQ)
    except Exception:
        # dict 타입 방어
        if isinstance(jr, dict):
            jr["queue_seq"] = int(_ENQUEUE_SEQ)
    return int(_ENQUEUE_SEQ)

def _global_queue_insert_sorted_locked(st: Any, job_id: str) -> None:
    jid = str(job_id)

    jobs = getattr(st, "jobs", {}) or {}
    jr = jobs.get(jid)
    if jr is None:
        return

    # ✅ 핵심: QUEUED만 global_queue에 존재
    if not _is_queued(jr):
        return

    q = getattr(st, "global_queue", None)
    if not isinstance(q, list):
        try:
            st.global_queue = []
            q = st.global_queue
        except Exception:
            return

    def _as_float(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("inf")

    def _job_global_key(jr0: Any) -> float:
        # 1) enqueue_ts
        try:
            v = getattr(jr0, "enqueue_ts", None)
            if v is not None:
                return _as_float(v)
        except Exception:
            pass
        # 2) submit_ts
        try:
            v = getattr(jr0, "submit_ts", None)
            if v is not None:
                return _as_float(v)
        except Exception:
            pass
        # 3) queue_seq (있으면 안정 tie-break에 도움)
        try:
            v = getattr(jr0, "queue_seq", None)
            if v is not None:
                # queue_seq는 "시간"이 아니므로, 큰 상수로 스케일링해서 뒤에 배치되게 하지 않고
                # 그냥 미세 tie-break에만 쓰려면 별도 튜플 정렬이 필요하지만
                # 여기선 최소 수정: enqueue/submit이 없을 때만 사용.
                return float(v)
        except Exception:
            pass
        # 4) fallback
        return float(time.time())

    # 중복 제거 (타입은 str list로 관리)
    try:
        q = [str(x) for x in (q or []) if str(x) and str(x) != jid]
    except Exception:
        q = []

    k_new = _job_global_key(jr)

    inserted = False
    out: List[str] = []
    for x in q:
        xj = str(x)
        xjr = jobs.get(xj)

        # 깨진 항목 정리(QUEUED만 유지)
        if xjr is None or (not _is_queued(xjr)):
            continue

        k_x = _job_global_key(xjr)

        # ✅ enqueue_ts 기준 오름차순(작을수록 먼저)
        if (not inserted) and (k_new < k_x):
            out.append(jid)
            inserted = True
        out.append(xj)

    if not inserted:
        out.append(jid)

    try:
        st.global_queue = list(out)
    except Exception:
        pass

def purge_job_from_all_queues_locked(
    state: Any,
    job_id: str,
    *,
    purge_global: bool = True,
    purge_cluster: bool = True,
    purge_home: bool = True,
) -> None:
    jid = str(job_id)

    def _item_jid(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return str(x.get("job_id") or x.get("id") or "")
        if hasattr(x, "job_id"):
            try:
                return str(getattr(x, "job_id") or "")
            except Exception:
                return ""
        try:
            return str(x)
        except Exception:
            return ""

    def _purge_from_queue_obj(q: Any) -> None:
        if q is None:
            return

        # list 형태: 타입 보존 + item_jid로 필터
        if isinstance(q, list):
            try:
                q[:] = [x for x in q if (_item_jid(x) != jid and _item_jid(x))]
            except Exception:
                pass
            # ETA dirty
            try:
                setattr(q, "eta_dirty", True)
            except Exception:
                pass
            return

        # remove() 제공: 실패해도 괜찮음
        removed = False
        try:
            if hasattr(q, "remove"):
                q.remove(jid)
                removed = True
        except Exception:
            removed = False

        # _jobs 내부 제거 (QueueJob/str/dict 혼재 대응)
        if not removed:
            try:
                jobs0 = list(getattr(q, "_jobs", []) or [])
                new_jobs = [it for it in jobs0 if (_item_jid(it) != jid and _item_jid(it))]
                setattr(q, "_jobs", new_jobs)
            except Exception:
                pass

        try:
            setattr(q, "eta_dirty", True)
        except Exception:
            pass

    # 1) global queue
    if purge_global:
        try:
            gq = getattr(state, "global_queue", None)
            if isinstance(gq, list):
                gq[:] = [x for x in gq if (_item_jid(x) != jid and _item_jid(x))]
        except Exception:
            pass

    # 2) cluster_queues
    if purge_cluster:
        try:
            qs = getattr(state, "cluster_queues", None) or {}
            if isinstance(qs, dict):
                for _cid, q in qs.items():
                    _purge_from_queue_obj(q)
        except Exception:
            pass

    # 3) home_cluster_queues
    if purge_home:
        try:
            hqs = getattr(state, "home_cluster_queues", None) or {}
            if isinstance(hqs, dict):
                for _cid, q in hqs.items():
                    _purge_from_queue_obj(q)
        except Exception:
            pass

# def pick_next_gang_cluster_locked(state: Any) -> str:
#     now = float(time.time())

#     clusters_map = getattr(state, "clusters", {}) or {}
#     cluster_ids = sorted([str(c) for c in list(clusters_map.keys()) if str(c)])
#     if not cluster_ids:
#         return "clusterA"

#     # --- RR pointer ---
#     cur = getattr(state, "gang_rr_next", None)
#     if cur is None or str(cur) not in cluster_ids:
#         cur = cluster_ids[0]
#     cur = str(cur)

#     # 다음 포인터로 토글(SSOT)
#     try:
#         idx = cluster_ids.index(cur)
#         nxt = cluster_ids[(idx + 1) % len(cluster_ids)]
#         state.gang_rr_next = str(nxt)
#     except Exception:
#         # 포인터 갱신 실패해도 cur 반환은 가능
#         pass

#     # --- helpers: free nodes count with drain/blocked ---
#     node_owner = getattr(state, "node_owner", {}) or {}
#     drains = getattr(state, "drain_until_by_cluster", {}) or {}
#     cb = getattr(state, "cluster_launch_blocked_until", {}) or {}
#     if not isinstance(drains, dict):
#         drains = {}
#     if not isinstance(cb, dict):
#         cb = {}

#     def _cluster_can_launch(cid: str) -> bool:
#         try:
#             if now < float(drains.get(cid, 0.0) or 0.0):
#                 return False
#         except Exception:
#             pass
#         try:
#             if now < float(cb.get(cid, 0.0) or 0.0):
#                 return False
#         except Exception:
#             pass
#         return True

#     def _free_count(cid: str) -> int:
#         if not _cluster_can_launch(cid):
#             return 0
#         cr = clusters_map.get(cid)
#         if cr is None:
#             return 0
#         try:
#             nodes = [str(n) for n in (getattr(cr, "nodes", []) or []) if str(n)]
#         except Exception:
#             nodes = []
#         if not nodes:
#             return 0
#         free = 0
#         for n in nodes:
#             try:
#                 if node_owner.get(n) is None:
#                     free += 1
#             except Exception:
#                 continue
#         return int(free)

#     # 1) RR로 찜한 클러스터가 당장 4 free면 그대로
#     if _free_count(cur) >= 4:
#         return cur

#     # 2) RR 찜이 막혔으면: 다른 클러스터 중 free>=4인 곳으로 즉시 이동
#     best = None  # (free, cid)
#     for cid in cluster_ids:
#         if cid == cur:
#             continue
#         fc = _free_count(cid)
#         if fc >= 4:
#             if best is None or fc > best[0]:
#                 best = (fc, cid)

#     if best is not None:
#         return str(best[1])

#     # 3) 어디도 4 free가 없으면: 일단 RR 찜 유지 (대기)
#     return cur


def pick_next_gang_cluster_locked(state: Any) -> str:
    """
    목적: gang(4GPU) 기준 클러스터 pin 선택.
    - 기본은 clusterB (최대한 B에 배치)
    - 단, B 과밀/프리엠션 폭주/큐 적체 시에만 A로 완화 분산
    - 분산이 한번 발동되면 짧게 유지(hysteresis)해서 흔들림 방지
    """
    now = float(time.time())

    clusters_map = getattr(state, "clusters", {}) or {}
    cluster_ids = sorted([str(c) for c in list(clusters_map.keys()) if str(c)])
    if not cluster_ids:
        return "clusterA"

    node_owner = getattr(state, "node_owner", {}) or {}
    drains = getattr(state, "drain_until_by_cluster", {}) or {}
    cb = getattr(state, "cluster_launch_blocked_until", {}) or {}
    if not isinstance(drains, dict):
        drains = {}
    if not isinstance(cb, dict):
        cb = {}

    # ---- launch 가능 여부 ----
    def _cluster_can_launch(cid: str) -> bool:
        try:
            if now < float(drains.get(cid, 0.0) or 0.0):
                return False
        except Exception:
            pass
        try:
            if now < float(cb.get(cid, 0.0) or 0.0):
                return False
        except Exception:
            pass
        return True

    # ---- free gpu count ----
    def _free_count(cid: str) -> int:
        if not _cluster_can_launch(cid):
            return 0
        cr = clusters_map.get(cid)
        if cr is None:
            return 0
        try:
            nodes = [str(n) for n in (getattr(cr, "nodes", []) or []) if str(n)]
        except Exception:
            nodes = []
        free = 0
        for n in nodes:
            try:
                if node_owner.get(n) is None:
                    free += 1
            except Exception:
                continue
        return int(free)

    # ---- (best-effort) HOME 큐 길이 ----
    def _home_queue_len(cid: str) -> int:
        qs = getattr(state, "home_cluster_queues", None) or getattr(state, "cluster_queues", None) or {}
        try:
            q = qs.get(cid)
        except Exception:
            q = None
        if q is None:
            return 0
        # Queue 구현에 따라 _jobs/list 등을 가질 수 있음
        for attr in ("_jobs", "jobs", "items"):
            try:
                v = getattr(q, attr, None)
                if isinstance(v, list):
                    return len(v)
            except Exception:
                pass
        try:
            # q가 list 자체일 수도
            if isinstance(q, list):
                return len(q)
        except Exception:
            pass
        return 0

    # ---- (best-effort) 최근 requeue 카운트 ----
    # state에 누적 카운터가 있으면 그걸 쓰고, 없으면 0
    def _recent_requeues(cid: str) -> int:
        # 예: state.requeue_burst_by_cluster[cid] 같은 걸 운영하면 제일 좋음
        m = getattr(state, "requeue_burst_by_cluster", None)
        if isinstance(m, dict):
            try:
                return int(m.get(cid, 0) or 0)
            except Exception:
                return 0
        return 0

    preferred = "clusterB"
    fallback = "clusterA" if "clusterA" in cluster_ids else cluster_ids[0]

    # ---- hysteresis: 이전에 정한 pin을 잠깐 유지 ----
    hold_until = float(getattr(state, "gang_pin_hold_until", 0.0) or 0.0)
    hold_cid = str(getattr(state, "gang_pin_hold_cluster", "") or "")
    if now < hold_until and hold_cid in cluster_ids and _cluster_can_launch(hold_cid):
        return hold_cid

    # ---- 기본: B가 있으면 B ----
    if preferred in cluster_ids and _cluster_can_launch(preferred):
        B_free = _free_count(preferred)
        A_free = _free_count(fallback)
        B_q = _home_queue_len(preferred)
        B_rq = _recent_requeues(preferred)

        # ===== 완화 분산 트리거(“예외적으로만 A”) =====
        # 1) B가 너무 밀림: HOME 큐가 길다
        # 2) B에서 프리엠션 재큐가 최근에 연속 발생(폭주)
        # 3) A가 훨씬 여유 있는데 B는 꽉 참(자원 낭비 방지)
        # 임계치는 너 workload에 맞게 조정하면 됨.
        TRIG_Q = int(getattr(state, "GANG_PIN_B_Q_WATERMARK", 5) or 5)   # B 큐 5 이상이면 과밀로 간주 / 4까지 가능
        TRIG_RQ = int(getattr(state, "GANG_PIN_B_REQUEUE_BURST", 2) or 2) # 최근 requeue 2 이상이면 폭주
        TRIG_FREE_GAP = int(getattr(state, "GANG_PIN_FREE_GAP", 2) or 2)  # A_free - B_free >= 2면 분산 고려

        divert_to_A = False
        if B_q >= TRIG_Q:
            divert_to_A = True
        if B_rq >= TRIG_RQ:
            divert_to_A = True
        if (A_free - B_free) >= TRIG_FREE_GAP and A_free > 0:
            divert_to_A = True

        if divert_to_A and fallback != preferred and _cluster_can_launch(fallback):
            # 관성 부여: 1~2초(또는 tick 몇 번) A 유지
            try:
                state.gang_pin_hold_cluster = fallback
                state.gang_pin_hold_until = now + float(getattr(state, "GANG_PIN_HOLD_SEC", 1.5) or 1.5)
            except Exception:
                pass
            return fallback

        # 아니면 B 유지 (B가 4 free가 없어도 pin은 B로 둠: B 기준 preempt 목적)
        return preferred

    # ---- B가 없거나 막혀있으면: 4 free 있는 곳 우선, 없으면 RR ----
    best = None
    for cid in cluster_ids:
        fc = _free_count(cid)
        if fc >= 4:
            if best is None or fc > best[0]:
                best = (fc, cid)
    if best is not None:
        return str(best[1])

    # RR fallback
    cur = getattr(state, "gang_rr_next", None)
    if cur is None or str(cur) not in cluster_ids:
        cur = cluster_ids[0]
    cur = str(cur)
    try:
        idx = cluster_ids.index(cur)
        nxt = cluster_ids[(idx + 1) % len(cluster_ids)]
        state.gang_rr_next = str(nxt)
    except Exception:
        pass
    return cur

def enforce_launch_invariants(job, *, g_alloc: int) -> None:
    elastic_enabled = bool(getattr(job, "elastic_enabled", False))
    g_req = int(getattr(job, "g_req", 0) or 0)
    if g_req <= 0:
        # 기존 코드에서 g_req를 다른 함수로 계산한다면 여기 맞춰서 바꾸세요.
        g_req = int(getattr(job, "g_target", 1) or 1)

    if not elastic_enabled:
        if int(g_alloc) != int(g_req):
            raise ValueError(f"invariant violated: elastic_enabled=False but g_alloc({g_alloc}) != g_req({g_req})")

def _next_event_seq_locked(st: Any, job_id: str) -> int:
    m = getattr(st, "job_event_seq", None)
    if m is None or not isinstance(m, dict):
        m = {}
        try:
            st.job_event_seq = m
        except Exception:
            pass
    cur = int(m.get(str(job_id), 0) or 0) + 1
    m[str(job_id)] = cur
    return cur

def transition_enqueue_locked(
    st: Any,
    job_id: str,
    *,
    new_kind: str,
    new_qcid: str,
    reason: str,
    meta: Optional[Dict[str, Any]] = None,
    now_ts: Optional[float] = None,
) -> int:
    jid = str(job_id)
    nk = str(new_kind or "").upper().strip()
    qcid = str(new_qcid or "").strip()
    if not nk:
        nk = "HOME"
    if not qcid:
        qcid = "UNKNOWN"

    now = float(now_ts) if now_ts is not None else float(time.time())

    jobs = getattr(st, "jobs", {}) or {}
    jr = jobs.get(jid)
    if jr is None:
        return 0

    # prev_* 절대 null 금지
    try:
        pk = getattr(jr, "queue_kind", None)
        pc = getattr(jr, "queue_cluster_id", None)
        if not pk:
            pk = "INIT"
        if not pc:
            pc = getattr(jr, "home_cluster_id", None) or getattr(jr, "cluster_id", None) or "INIT"
        jr.prev_queue_kind = str(pk)
        jr.prev_queue_cluster_id = str(pc)
    except Exception:
        pass

    # 기존 큐에서 제거(중복 방지)
    try:
        purge_job_from_all_queues_locked(st, jid, purge_global=False, purge_cluster=False, purge_home=True)
    except Exception:
        pass

    # 새 큐 메타
    try:
        jr.queue_kind = nk
        jr.queue_cluster_id = qcid
        jr.last_queue_enter_ts = now
        jr.queue_enter_ts = now
        jr.requeue_reason = str(reason or "")
        jr.requeue_meta = dict(meta or {})
    except Exception:
        pass

    # 큐 삽입
    if nk == "HOME":
        try:
            hq = (getattr(st, "home_cluster_queues", {}) or {})
            if not isinstance(hq, dict):
                hq = {}
                st.home_cluster_queues = hq
            hq.setdefault(qcid, [])
            if jid not in hq[qcid]:
                hq[qcid].append(jid)
        except Exception:
            pass

    # global queue는 SSOT로 유지
    try:
        _global_queue_insert_sorted_locked(st, jid)
    except Exception:
        try:
            gq = getattr(st, "global_queue", None)
            if gq is None or not isinstance(gq, list):
                st.global_queue = []
                gq = st.global_queue
            if jid not in gq:
                gq.append(jid)
        except Exception:
            pass

    # 단일 큐 보정(있으면 호출)
    try:
        ensure_job_single_queue_locked(st, jid, prefer_cluster_id=None)
    except Exception:
        pass

    return _next_event_seq_locked(st, jid)

def mark_resize_inflight_locked(st: Any, job_id: str, cid: str, new_nodes: List[str], new_g: int, reason: str, request_id: Optional[str] = None, now_ts: Optional[float] = None) -> None:
    now_ts = float(now_ts or __import__("time").time())
    m = getattr(st, "resize_inflight", None)
    if not isinstance(m, dict):
        m = {}
        setattr(st, "resize_inflight", m)

    m[str(job_id)] = {
        "ts": float(now_ts),
        "job_id": str(job_id),
        "cid": str(cid),
        "new_nodes": [str(x) for x in (new_nodes or []) if x],
        "new_g": int(new_g),
        "reason": str(reason or "resize"),
        "request_id": str(request_id or ""),
    }

def clear_resize_inflight_locked(st: Any, job_id: str) -> None:
    m = getattr(st, "resize_inflight", None)
    if isinstance(m, dict):
        m.pop(str(job_id), None)

def resize_cooldown_ok_locked(st: Any, job_id: str, now_ts: Optional[float] = None) -> bool:
    now_ts = float(now_ts or __import__("time").time())
    m = getattr(st, "resize_cooldown_until", None)
    if not isinstance(m, dict):
        return True
    until = m.get(str(job_id))
    try:
        return float(until or 0.0) <= now_ts
    except Exception:
        return True

def set_resize_cooldown_locked(st: Any, job_id: str, cooldown_sec: float, now_ts: Optional[float] = None) -> None:
    now_ts = float(now_ts or __import__("time").time())
    m = getattr(st, "resize_cooldown_until", None)
    if not isinstance(m, dict):
        m = {}
        setattr(st, "resize_cooldown_until", m)
    m[str(job_id)] = float(now_ts + float(cooldown_sec))