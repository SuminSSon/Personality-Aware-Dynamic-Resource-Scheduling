from __future__ import annotations

import logging
import os, time, math, threading
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
import time, json, requests
from fastapi import HTTPException, Request
import uuid
import inspect
from types import SimpleNamespace
import re

from app.schemas import JobCompleteReport, TelemetryIn
from app.llm import get_policy_from_user_request

from app.executor import launch_or_reuse, stop_job
from app.scheduler_state import (
    get_global_state,
    register_job_on_submit,
    _global_queue_insert_sorted_locked,
    _recompute_cluster_counters_locked,
    ensure_job_single_queue_locked,
    pick_next_gang_cluster_locked,
    purge_job_from_all_queues_locked,
    _global_queue_compact_locked,
    reserve_nodes,
    _new_run_id,
)
import app.scheduler_state as ss
from app.logger import get_run_logger
from app.rebalance import rebalance_tick_global
from app.backfill_policy import _ensure_dict_attr, _job_gang_required_g, _free_nodes_in_cluster, STATE_LOCK, _is_queued, _choose_best_cluster_and_g_theory, _ss_release_nodes_for_job_locked, release_nodes_for_job, _tag_as_hol_backfill_locked, release_nodes_for_job_locked
from app.profiling import get_profiling_entry


logger = logging.getLogger(__name__)

DEFAULT_LAUNCH_TIMEOUT_SEC = float(60.0)
PREEMPT_INFLIGHT_TIMEOUT_SEC = 25.0

_TERMINAL = {"FINISHED", "FAILED", "CANCELLED", "COMPLETED"}

_TICK_THREAD: threading.Thread | None = None

# tick 실행을 한 번에 1개만 보장
_TICK_RUN_LOCK = threading.Lock()

# tick 요청을 깨우는 이벤트 (periodic + on-demand)
_TICK_WAKE_EVENT = threading.Event()

_TICK_PENDING = False

# tick loop 종료 플래그
_TICK_STOP: threading.Event = threading.Event()

# 기본 주기
TICK_PERIOD_SEC = 1.0

LaunchFn = Callable[[str, str, List[str], bool, int], bool]
StopFn = Callable[[str, str], None]
RecordQueueEventFn = Callable[[float, str, str, str], None]
RecordJobEventFn = Callable[[float, str, str, str, int, str], None]
GetEpochTimeFn = Callable[[Dict[str, Any], str, int], Optional[float]]


# --------------------------------------------
def _now() -> float:
    return time.time()

DEFAULT_EPOCHS = int(__import__("os").getenv("OURS_DEFAULT_EPOCHS", "20"))

GANG_JOBS: Dict[Tuple[str, str], int] = {
    ("DenseNet-121", "TinyImageNet"): 4,
}

# Policy representation
def _job_eligible_for_cluster_locked(state: Any, jr: Any, cid: str) -> bool:
    cid = str(cid)

    # queued only
    if jr is None or not _is_queued(jr):
        return False

    # pinned cluster
    try:
        pin = getattr(jr, "pinned_cluster", None)
        if pin and str(pin) != cid:
            return False
    except Exception:
        pass

    # queue_kind / queue_cluster_id must be consistent
    try:
        kind = str(getattr(jr, "queue_kind", "") or "").upper()
    except Exception:
        kind = ""

    try:
        qcid = str(getattr(jr, "queue_cluster_id", "") or "")
    except Exception:
        qcid = ""

    if kind == "HOME":
        if not qcid:
            qcid = str(getattr(jr, "home_cluster_id", "") or getattr(jr, "admitted_cluster_id", "") or "")
        return bool(qcid and qcid == cid)

    if kind == "ADMITTED":
        if not qcid:
            qcid = str(getattr(jr, "admitted_cluster_id", "") or getattr(jr, "home_cluster_id", "") or "")
        return bool(qcid and qcid == cid)

    if kind == "MOBILE":
        return True

    # kind empty or unknown -> disallow (prevents wrong-cluster execution)
    return False

LAUNCH_INFLIGHT_TIMEOUT_SEC = float(os.getenv("LAUNCH_INFLIGHT_TIMEOUT_SEC", "120"))

def _cleanup_launch_inflight_locked(state: Any, now_ts: float) -> None:
    jobs = getattr(state, "jobs", {}) or {}
    if not isinstance(jobs, dict) or not jobs:
        return

    # draining map 후보들(있으면 pop해서 "유령 drain" 제거)
    draining_maps: List[Dict[str, Any]] = []
    for name in (
        "node_draining_until",
        "node_drain_until",
        "node_drain_until_ts",
        "node_blocked_until",
        "node_unavailable_until",
        "nodes_draining_until",
    ):
        try:
            m = getattr(state, name, None)
            if isinstance(m, dict):
                draining_maps.append(m)
        except Exception:
            pass

    owners = getattr(state, "node_owner", None)
    if not isinstance(owners, dict):
        owners = {}

    def _owned_nodes(jid: str) -> List[str]:
        out: List[str] = []
        for n, owner in list(owners.items()):
            try:
                if owner is not None and str(owner) == str(jid):
                    out.append(str(n))
            except Exception:
                continue
        return out

    for jid, jr in list(jobs.items()):
        try:
            st = str(getattr(jr, "status", "") or "").upper()
            if st != "STARTING":
                continue
            if not bool(getattr(jr, "launch_inflight", False)):
                continue

            # since: launching_since_ts 우선, 없으면 last_launch_try_ts fallback
            since = 0.0
            try:
                since = float(getattr(jr, "launching_since_ts", 0.0) or 0.0)
            except Exception:
                since = 0.0
            if since <= 0.0:
                try:
                    since = float(getattr(jr, "last_launch_try_ts", 0.0) or 0.0)
                except Exception:
                    since = 0.0
            if since <= 0.0:
                # 타임아웃 판단 자체가 불가능하면 그냥 스킵(=원인: since 세팅 누락)
                continue

            if (float(now_ts) - float(since)) < float(LAUNCH_INFLIGHT_TIMEOUT_SEC):
                continue

            jid_s = str(jid)
            cid = getattr(jr, "cluster_id", None)

            # 1) SSOT release (노드 리스트를 먼저 뽑아서 draining도 같이 정리)
            owned = _owned_nodes(jid_s)
            try:
                _ss_release_nodes_for_job_locked(state, job_id=jid_s, nodes=owned if owned else None)
            except Exception:
                pass

            # 2) draining 기록 제거(best-effort)
            if owned and draining_maps:
                for dm in draining_maps:
                    for n in owned:
                        try:
                            dm.pop(str(n), None)
                        except Exception:
                            continue

            # 3) job 상태 초기화 + 재시도 가능 상태로 복귀
            try:
                jr.status = "QUEUED"
                _clear_requeue_victim_tags_locked(jr)
            except Exception:
                pass
            try:
                jr.launch_inflight = False
            except Exception:
                pass
            try:
                jr.launching_since_ts = None
            except Exception:
                pass

            # launch 실패 누적/쿨다운(스팸 방지)
            try:
                jr.launch_fail_count = int(getattr(jr, "launch_fail_count", 0) or 0) + 1
                fc = int(jr.launch_fail_count or 1)
                cd = min(30.0, 1.0 * (2 ** max(0, fc - 1)))
                jr.launch_cooldown_until_ts = float(time.time() + cd)
                jr.last_launch_error = "launch_inflight_timeout"
            except Exception:
                pass

            # 실행 흔적 제거
            try:
                jr.cluster_id = None
                jr.nodes = []
                jr.g_cur = 0
            except Exception:
                pass
            try:
                jr.is_backfill = False
                jr.is_hol_backfill = False
                jr.hol_blocking_job_id = None
            except Exception:
                pass

            # 4) 큐 복구: single-queue invariant + global_queue 유지
            try:
                purge_job_from_all_queues_locked(state, jid_s, purge_global=False, purge_cluster=False, purge_home=True)
            except Exception:
                pass
            try:
                _global_queue_insert_sorted_locked(state, jid_s)
            except Exception:
                pass
            try:
                pin = getattr(jr, "pinned_cluster", None)
                if pin:
                    ensure_job_single_queue_locked(state, jid_s, str(pin))
                else:
                    adm = getattr(jr, "admitted_cluster_id", None)
                    if adm:
                        ensure_job_single_queue_locked(state, jid_s, str(adm))
            except Exception:
                pass

            # 5) 로그(최소한의 증거 남기기)
            try:
                logger.warning(
                    "[cleanup_launch_inflight] timeout job_id=%s since=%.3f now=%.3f owned=%s cid=%s",
                    jid_s, float(since), float(now_ts), owned, cid
                )
            except Exception:
                pass

        except Exception:
            continue

def ensure_tick_thread_started() -> None:
    global _TICK_THREAD
    t = _TICK_THREAD
    if t is not None and t.is_alive():
        return

    _TICK_STOP.clear()
    _TICK_EVENT.clear()

    _TICK_THREAD = threading.Thread(
        target=_tick_thread_main,
        name="ours-scheduler-tick",
        daemon=True,
    )
    _TICK_THREAD.start()
    logger.info("[tick] thread started")

def ensure_job_single_queue(job_id: str, home_cluster_id: str) -> None:
    state = get_global_state()
    with STATE_LOCK:
        ensure_job_single_queue_locked(state, job_id, home_cluster_id)

def is_gang_model(job_or_model: Any) -> bool:
    if job_or_model is None:
        return False

    # --- helper: model normalize ---
    def _norm_model(x: Any) -> str:
        s = str(x or "").strip()
        if not s:
            return ""
        s2 = re.sub(r"[\s_]+", "-", s)
        return s2

    if isinstance(job_or_model, str):
        model = _norm_model(job_or_model)
        raw = str(os.getenv("OURS_GANG_MODELS", "") or "").strip()
        if not raw:
            return False
        gang_set = {_norm_model(x) for x in raw.split(",") if str(x).strip()}
        return bool(model and model in gang_set)

    for k in ("is_gang", "gang", "gang_required"):
        try:
            if bool(getattr(job_or_model, k, False)):
                return True
        except Exception:
            pass

    model = None
    policy = None
    g_hint = None

    try:
        if isinstance(job_or_model, dict):
            model = job_or_model.get("model") or job_or_model.get("model_name") or job_or_model.get("name")
            policy = job_or_model.get("policy") or {}
            g_hint = (
                job_or_model.get("g_target")
                or job_or_model.get("world_size")
                or job_or_model.get("num_gpus")
                or job_or_model.get("g_req")
            )
            # dict flag
            for k in ("is_gang", "gang", "gang_required"):
                try:
                    if bool(job_or_model.get(k, False)):
                        return True
                except Exception:
                    pass
        else:
            model = getattr(job_or_model, "model", None) or getattr(job_or_model, "model_name", None) or getattr(job_or_model, "name", None)
            policy = getattr(job_or_model, "policy", None) or {}
            g_hint = (
                getattr(job_or_model, "g_target", None)
                or getattr(job_or_model, "world_size", None)
                or getattr(job_or_model, "num_gpus", None)
                or getattr(job_or_model, "g_req", None)
            )
    except Exception:
        policy = policy or {}

    # 2-1) g 힌트로 gang 판정 (네 환경에서 gang=4 고정이면 이게 가장 확실)
    try:
        if g_hint is not None and int(g_hint) == 4:
            return True
    except Exception:
        pass

    # 3) policy dict 기반
    try:
        if isinstance(policy, dict):
            if bool(policy.get("gang", False) or policy.get("is_gang", False) or policy.get("gang_required", False)):
                return True
            # policy 내부에 g_target 같은 값이 있을 수도 있음
            pg = policy.get("g_target") or policy.get("world_size") or policy.get("num_gpus")
            try:
                if pg is not None and int(pg) == 4:
                    return True
            except Exception:
                pass
    except Exception:
        pass

    # 4) environment-configured gang models
    try:
        raw = str(os.getenv("OURS_GANG_MODELS", "") or "").strip()
        if raw:
            gang_set = {_norm_model(x) for x in raw.split(",") if str(x).strip()}
            mnorm = _norm_model(model)
            if mnorm and mnorm in gang_set:
                return True
    except Exception:
        pass

    return False

def _cluster_eligible_queue_len_locked(state: Any, cluster_id: str) -> int:
    cid = str(cluster_id)
    cr = (getattr(state, "clusters", {}) or {}).get(cid)
    if cr is None:
        return 0

    jobs = getattr(state, "jobs", {}) or {}

    # global_queue 원소가 QueueJob/str/dict 섞여 있어도 job_id 뽑기
    def _jid_of(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return str(x)
        if isinstance(x, dict):
            v = x.get("job_id") or x.get("id")
            return str(v) if v else ""
        v = getattr(x, "job_id", None) or getattr(x, "id", None)
        if v:
            return str(v)
        try:
            return str(x)  # QueueJob __str__ 호환
        except Exception:
            return ""

    raw = list(getattr(state, "global_queue", []) or [])
    cnt = 0
    for it in raw:
        jid = _jid_of(it).strip()
        if not jid:
            continue
        jr = jobs.get(jid)
        if jr is None:
            continue

        # QUEUED만 집계 (STARTING/RUNNING은 dequeue된 것으로 간주)
        if not _is_queued(jr):
            continue

        # HOME 큐는 소속 강제 (불일치 방지)
        qk = str(getattr(jr, "queue_kind", "") or "").upper()
        qcid = str(getattr(jr, "queue_cluster_id", "") or "")
        if qk == "HOME" and qcid and qcid != cid:
            continue

        # 최종 eligibility
        if _job_eligible_for_cluster_locked(jr, cid, cr):
            cnt += 1

    return int(cnt)

PREEMPT_GRACE_SEC = 10.0          # 방금 띄운 backfill 보호
HOL_PREEMPT_COOLDOWN_SEC = 8.0    # 동일 HoL에 대해 preempt 스팸 방지
DRAIN_SECONDS = 5.0

def _clear_requeue_victim_tags_locked(jr: Any) -> None:
    # backfill/victim 표식 제거 (requeue는 '정상 job'으로 복귀해야 함)
    for k, v in (
        ("is_backfill", False),
        ("is_hol_backfill", False),
        ("queue_kind", ""),
        ("hol_backfill_pin", None),
        ("hol_backfill_for", None),
        ("preempt_ok_only_by_hol", False),
    ):
        try:
            setattr(jr, k, v)
        except Exception:
            pass

    # inflight/dispatch 흔적도 같이 정리 (재런치 루프 방지)
    for k, v in (
        ("launch_inflight", False),
        ("launching_since_ts", 0.0),
        ("launch_dispatched", False),
        ("launch_dispatched_ts", 0.0),
        ("preempt_inflight", False),
        ("preempting_since_ts", 0.0),
    ):
        try:
            setattr(jr, k, v)
        except Exception:
            pass

def _cleanup_launch_inflight_locked(state: Any, now_ts: float) -> None:
    jobs = getattr(state, "jobs", {}) or {}
    owner = getattr(state, "node_owner", {}) or {}

    # launch inflight timeout 기본값 (너무 짧으면 실제 launch 중에도 풀려서 문제, 너무 길면 스팸 지속)
    timeout_sec = float(getattr(state, "launch_inflight_timeout_sec", 15.0) or 15.0)

    for jid, jr in list(jobs.items()):
        try:
            if not bool(getattr(jr, "launch_inflight", False)):
                continue

            t0 = float(getattr(jr, "launching_since_ts", 0.0) or 0.0)
            if t0 <= 0.0:
                # launch_inflight인데 시간이 없으면 즉시 정리
                t0 = now_ts - (timeout_sec + 1.0)

            if now_ts - t0 < timeout_sec:
                continue

            # timeout 발생 → inflight 해제 + backoff
            try:
                jr.launch_inflight = False
                jr.launching_since_ts = 0.0
                jr.launch_cooldown_until_ts = max(float(getattr(jr, "launch_cooldown_until_ts", 0.0) or 0.0), now_ts + 5.0)
                # 상태는 QUEUED로 수렴(재시도는 하되 스팸은 cooldown이 막는다)
                if str(getattr(jr, "status", "") or "").upper() == "LAUNCHING":
                    jr.status = "QUEUED"
                    _clear_requeue_victim_tags_locked(jr)
            except Exception:
                pass

            # reserve가 남아있는 경우가 있음 → node_owner가 이 job이면 정리
            try:
                nodes = list(getattr(jr, "nodes", []) or [])
            except Exception:
                nodes = []
            if nodes:
                owned = True
                for n in nodes:
                    cur = owner.get(str(n))
                    if cur is not None and str(cur) != str(jid):
                        owned = False
                        break
                if owned:
                    try:
                        _ss_release_nodes_for_job_locked(state, job_id=str(jid), nodes=list(nodes))
                        # placement 정보도 초기화
                        jr.nodes = []
                        jr.g_cur = 0
                        jr.cluster_id = None
                    except Exception:
                        pass

        except Exception:
            continue

PREEMPT_DRAIN_SEC_DEFAULT = 10.0

def _mark_nodes_draining_locked(state: Any, nodes: list, until_ts: float) -> None:
    d = _ensure_dict_attr(state, "node_drain_until")
    for n in nodes or []:
        nn = str(n)
        prev = float(d.get(nn, 0.0) or 0.0)
        d[nn] = max(prev, float(until_ts))

def _mark_cluster_blocked_locked(state: Any, cluster_id: str, until_ts: float) -> None:
    m = _ensure_dict_attr(state, "cluster_launch_blocked_until")
    cid = str(cluster_id)
    prev = float(m.get(cid, 0.0) or 0.0)
    m[cid] = max(prev, float(until_ts))

def _safe_json(x: Any, *, limit: int = 2000) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        try:
            s = str(x)
        except Exception:
            s = "<unserializable>"
    if limit and len(s) > limit:
        return s[:limit] + "...(truncated)"
    return s

def _stop_job_and_release_nodes(
    *,
    job_id: str,
    cluster_id: str,
    reason: str,
    checkpoint: bool,
    requeue_kind: str,
) -> Dict[str, Any]:
    import time
    from typing import Any, Dict, List, Optional, Tuple

    cid = str(cluster_id)
    jid = str(job_id)
    now_ts = float(time.time())

    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    drain_until = now_ts + float(PREEMPT_DRAIN_SEC_DEFAULT)

    try:
        FORCE_RELEASE_SEC = float(getattr(globals(), "FORCE_RELEASE_SEC", 60.0) or 60.0)
    except Exception:
        FORCE_RELEASE_SEC = 60.0

    def _bump_seq_locked(jr: Any) -> int:
        try:
            s = int(getattr(jr, "event_seq", 0) or 0) + 1
            jr.event_seq = int(s)
            return int(s)
        except Exception:
            return -1

    def _snap_queue_prev_locked(jr: Any) -> Tuple[str, str]:
        pk = str(getattr(jr, "queue_kind", None) or "UNKNOWN")
        pc = str(
            getattr(jr, "queue_cluster_id", None)
            or getattr(jr, "home_cluster_id", None)
            or getattr(jr, "admitted_cluster_id", None)
            or "UNKNOWN"
        )
        return pk, pc

    def _is_preemptible_locked(jr: Any) -> bool:
        # ✅ preempt 흐름이 이미 시작된 job은 플래그가 나중에 내려가도 stop/cleanup이 막히면 안 됨
        try:
            if bool(getattr(jr, "preempt_inflight", False)):
                return True
            if str(getattr(jr, "status", "") or "").upper() == "PREEMPTING":
                return True
        except Exception:
            pass

        try:
            if bool(getattr(jr, "is_hol_backfill", False)):
                return True
        except Exception:
            pass
        try:
            if bool(getattr(jr, "is_backfill", False)):
                return True
        except Exception:
            pass
        try:
            if str(getattr(jr, "queue_kind", "") or "").upper() in ("BACKFILL", "HOL_BACKFILL"):
                return True
        except Exception:
            pass
        return False

    nodes: List[str] = []
    prev_kind = "UNKNOWN"
    prev_qcid = "UNKNOWN"
    run_id: Optional[str] = None

    already_inflight = False
    already_issued = False

    # =========================================================
    # ✅ LOCK: state transition + (NEW) STARTING HOL_BACKFILL rollback fast-path
    # =========================================================
    with STATE_LOCK:
        st = get_global_state()
        jr = (getattr(st, "jobs", {}) or {}).get(jid)
        if jr is None:
            return {"ok": False, "reason": "job_not_found", "job_id": jid}

        # ---- 현재 상태/플래그 스냅샷 ----
        try:
            st_u = str(getattr(jr, "status", "") or "").upper()
        except Exception:
            st_u = ""

        try:
            is_hbf = bool(getattr(jr, "is_hol_backfill", False)) or str(getattr(jr, "queue_kind", "") or "").upper() == "HOL_BACKFILL"
        except Exception:
            is_hbf = False

        try:
            ok_only = bool(getattr(jr, "preempt_ok_only_by_hol", False))
        except Exception:
            ok_only = False

        try:
            is_hol_reason = ("HOL_GANG_PREEMPT" in str(reason).upper())
        except Exception:
            is_hol_reason = False

        # =========================================================
        # ✅ NEW: STARTING/LAUNCHING 상태의 HOL_BACKFILL은 stop RPC가 아니라
        # "예약(SSOT) 롤백"으로 즉시 GPU를 풀어야 함
        # - 지금 네 증상(STARTING hol_backfill이 1GPU를 잡고 HoL이 못 뜨는 문제)을 정면 해결
        # =========================================================
        if is_hbf and ok_only and st_u in ("STARTING", "LAUNCHING"):
            try:
                nodes0 = [str(n) for n in (list(getattr(jr, "nodes", []) or [])) if str(n)]
            except Exception:
                nodes0 = []

            # SSOT 예약 해제 (가능하면 job이 들고 있던 cluster_id를 우선)
            try:
                cur_cid0 = (str(getattr(jr, "cluster_id", "") or "").strip() or cid)
            except Exception:
                cur_cid0 = cid

            try:
                release_nodes_for_job_locked(job_id=str(jid), cluster_id=(str(cur_cid0).strip() or None))
            except Exception:
                pass

            _bump_seq_locked(jr)

            # 멱등: launch/preempt 플래그/점유 정리
            try:
                jr.launch_inflight = False
                jr.launching_since_ts = 0.0
            except Exception:
                pass
            try:
                jr.preempt_inflight = False
                jr.preempt_issued = False
                jr.preempting_since_ts = 0.0
            except Exception:
                pass
            try:
                jr.status = "QUEUED"
                jr.cluster_id = None
                jr.nodes = []
                jr.g_cur = 0
                jr.g_alloc = 0
                jr.world_size = 0
                _clear_requeue_victim_tags_locked(jr)
            except Exception:
                pass

            # backfill 오염 제거(유령 victim 방지)
            try:
                jr.is_backfill = False
                jr.is_hol_backfill = False
                jr.hol_for = None
                jr.hol_pin = ""
                jr.preempt_ok_only_by_hol = False
                # queue_kind는 너 구조에 따라 다르지만, 최소한 HOL_BACKFILL은 벗어나야 함
                if str(getattr(jr, "queue_kind", "") or "").upper() == "HOL_BACKFILL":
                    jr.queue_kind = "GLOBAL"
            except Exception:
                pass

            try:
                globals()["_TICK_PENDING"] = True
            except Exception:
                pass

            if rl:
                try:
                    rl.queue_event(
                        event="hol_backfill_launch_rollback",
                        job_id=str(jid),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"rollback_starting_hol_backfill cid={cur_cid0} reason={reason} nodes={nodes0}",
                        ts=float(now_ts),
                    )
                except Exception:
                    pass

            return {"ok": True, "job_id": jid, "cluster_id": str(cur_cid0), "nodes": nodes0, "note": "rolled_back_starting_hol_backfill"}

        # ✅ 안전 가드: "마지막 1개"를 자르는 걸 막지 말아야 하는 케이스가 있음 (HOL reclaim)
        try:
            # HOL reclaim / hol_backfill victim이면 solo여도 허용
            reason_u = str(reason or "").upper()
            is_hbf = False
            ok_only = False
            try:
                is_hbf = bool(getattr(jr, "is_hol_backfill", False))
            except Exception:
                is_hbf = False
            try:
                ok_only = bool(getattr(jr, "preempt_ok_only_by_hol", False))
            except Exception:
                ok_only = False

            allow_solo = ("HOL_GANG_PREEMPT" in reason_u) or (is_hbf and ok_only)

            any_other_running = False
            for _jid2, _jr2 in (getattr(st, "jobs", {}) or {}).items():
                if str(_jid2) == jid:
                    continue
                try:
                    if str(getattr(_jr2, "cluster_id", "") or "") != cid:
                        continue
                    st2 = str(getattr(_jr2, "status", "") or "").upper()
                    if st2 in ("RUNNING", "STARTING", "LAUNCHING"):
                        any_other_running = True
                        break
                except Exception:
                    continue

            my_st = str(getattr(jr, "status", "") or "").upper()
            if (not allow_solo) and (my_st in ("RUNNING", "STARTING", "LAUNCHING")) and (not any_other_running):
                try:
                    jr.last_preempt_error = f"DENY_PREEMPT_SOLO_RUNNING reason={reason}"
                except Exception:
                    pass
                if rl:
                    try:
                        rl.queue_event(
                            event="preempt_denied",
                            job_id=str(jid),
                            queue_len=-1,
                            qlen_clusterq=-1,
                            note=f"deny_solo_running jid={jid} cid={cid} reason={reason}",
                            ts=float(now_ts),
                        )
                    except Exception:
                        pass
                return {"ok": False, "reason": "deny_preempt_solo_running", "job_id": jid, "cluster_id": cid}
        except Exception:
            pass


        if not _is_preemptible_locked(jr):
            try:
                jr.last_preempt_error = f"DENY_PREEMPT_NON_BACKFILL reason={reason}"
            except Exception:
                pass
            if rl:
                try:
                    rl.queue_event(
                        event="preempt_denied",
                        job_id=str(jid),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"deny_non_backfill jid={jid} cid={cid} reason={reason}",
                        ts=float(now_ts),
                    )
                except Exception:
                    pass
            return {"ok": False, "reason": "deny_preempt_non_backfill", "job_id": jid, "cluster_id": cid}

        try:
            nodes = [str(n) for n in (list(getattr(jr, "nodes", []) or [])) if str(n)]
        except Exception:
            nodes = []

        prev_kind, prev_qcid = _snap_queue_prev_locked(jr)

        try:
            run_id = (getattr(jr, "run_id", None) or "").strip() or None
        except Exception:
            run_id = None

        try:
            already_inflight = bool(getattr(jr, "preempt_inflight", False))
        except Exception:
            already_inflight = False
        try:
            already_issued = bool(getattr(jr, "preempt_issued", False))
        except Exception:
            already_issued = False

        if not nodes:
            _bump_seq_locked(jr)
            try:
                jr.preempt_inflight = True
                jr.preempt_issued = False
                jr.preempt_reason = str(reason)
                jr.pending_requeue_kind = str(requeue_kind)
                jr.pending_prev_queue_kind = str(prev_kind)
                jr.pending_prev_queue_cluster_id = str(prev_qcid)
                jr.pending_prev_nodes = []
                jr.pending_checkpoint = bool(checkpoint)
                jr.pending_drain_until = float(drain_until)

                # ✅ requeue(QUEUED 수렴) 시 backfill/hol_backfill 플래그를 반드시 지우도록 마킹
                jr.pending_clear_backfill_flags = True

                jr.status = "PREEMPTING"
                jr.force_release_deadline_ts = float(now_ts + FORCE_RELEASE_SEC)
                jr.last_preempt_error = "no_nodes_snapshot_wait_cleanup"
                # ✅ 스래시 방지 가드
                jr.preempt_guard_until_ts = float(now_ts + 1.5)
            except Exception:
                pass

            if rl:
                try:
                    rl.queue_event(
                        event="preempt_skip_stop_no_nodes",
                        job_id=str(jid),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"jid={jid} cid={cid} reason={reason} note=no_nodes_snapshot",
                        ts=float(now_ts),
                    )
                except Exception:
                    pass

            return {"ok": True, "job_id": jid, "cluster_id": cid, "nodes": [], "note": "no_nodes_snapshot_preempting", "run_id": run_id}

        if already_inflight and already_issued:
            try:
                jr.pending_drain_until = max(float(getattr(jr, "pending_drain_until", 0.0) or 0.0), float(drain_until))
                jr.preempt_guard_until_ts = max(float(getattr(jr, "preempt_guard_until_ts", 0.0) or 0.0), float(now_ts + 1.0))
            except Exception:
                pass
            return {"ok": True, "job_id": jid, "cluster_id": cid, "nodes": list(nodes), "note": "preempt_already_issued_skip_duplicate_stop_call", "drain_until": float(drain_until), "run_id": run_id}

        _bump_seq_locked(jr)

        try:
            _mark_nodes_draining_locked(st, nodes, drain_until)
        except Exception:
            pass
        try:
            _mark_cluster_blocked_locked(st, cid, drain_until)
        except Exception:
            pass

        try:
            jr.preempt_inflight = True
            jr.preempt_issued = False
            jr.preempting_since_ts = float(getattr(jr, "preempting_since_ts", 0.0) or 0.0) or float(now_ts)
            jr.last_preempt_try_ts = float(now_ts)
            jr.preempt_reason = str(reason)

            jr.pending_requeue_kind = str(requeue_kind)
            jr.pending_prev_queue_kind = str(prev_kind)
            jr.pending_prev_queue_cluster_id = str(prev_qcid)
            jr.pending_prev_nodes = list(nodes)
            jr.pending_drain_until = float(drain_until)
            jr.pending_checkpoint = bool(checkpoint)

            # ✅ requeue(QUEUED 수렴) 시 backfill/hol_backfill 플래그를 반드시 지우도록 마킹
            jr.pending_clear_backfill_flags = True

            jr.status = "PREEMPTING"
            jr.force_release_deadline_ts = float(now_ts + FORCE_RELEASE_SEC)

            # ✅ 스래시 방지 가드 (pick 단계가 존중)
            jr.preempt_guard_until_ts = float(now_ts + 1.5)
        except Exception:
            pass

    # =========================================================
    # LOCK 밖: stop RPC
    # =========================================================
    stop_attempted = False
    try:
        stop_attempted = True
        resp = stop_job(
            cluster_id=cid,
            job_id=jid,
            reason=str(reason),
            force=True,
            run_id=run_id,
            extra={"checkpoint": bool(checkpoint), "requeue_kind": str(requeue_kind), "run_id": run_id},
        )
    except TypeError:
        try:
            stop_attempted = True
            resp = stop_job(
                cluster_id=cid,
                job_id=jid,
                reason=str(reason),
                force=True,
                extra={"checkpoint": bool(checkpoint), "requeue_kind": str(requeue_kind), "run_id": run_id},
            )
        except Exception as e:
            resp = {"ok": False, "reason": "stop_call_failed", "detail": str(e)}
    except Exception as e:
        resp = {"ok": False, "reason": "stop_call_failed", "detail": str(e)}

    resp = resp if isinstance(resp, dict) else {"resp_raw": str(resp)}
    ok_stop = bool(resp.get("ok"))
    http_status = int(resp.get("http_status", 0) or resp.get("status_code", 0) or 0)
    r_reason = str(resp.get("reason", "") or "")
    r_detail = str(resp.get("detail", "") or "")

    touched = resp.get("touched_nodes", None)
    try:
        touched_i = int(touched) if touched is not None else None
    except Exception:
        touched_i = None

    not_active = (
        (http_status == 404)
        or (touched_i == 0)
        or ("job_not_found" in r_reason.lower())
        or ("not_found" in r_reason.lower())
        or ("not in active_jobs" in r_detail.lower())
        or ("not in active_jobs" in r_reason.lower())
    )

    if rl:
        try:
            rl.queue_event(
                event="preempt_issued",
                job_id=str(jid),
                queue_len=-1,
                qlen_clusterq=-1,
                note=f"jid={jid} cid={cid} ok_stop={ok_stop} not_active={not_active} reason={reason} resp={_safe_json(resp)}",
                ts=float(time.time()),
            )
        except Exception:
            pass

    # =========================================================
    # LOCK: 후처리 플래그 업데이트
    # =========================================================
    with STATE_LOCK:
        st2 = get_global_state()
        jr2 = (getattr(st2, "jobs", {}) or {}).get(jid)
        if jr2 is not None:
            try:
                _bump_seq_locked(jr2)
            except Exception:
                pass
            try:
                if stop_attempted:
                    jr2.preempt_issued = True
            except Exception:
                pass
            try:
                jr2.pending_drain_until = float(max(float(getattr(jr2, "pending_drain_until", 0.0) or 0.0), float(drain_until)))
                jr2.launch_cooldown_until_ts = float(max(float(getattr(jr2, "launch_cooldown_until_ts", 0.0) or 0.0), float(drain_until)))
                jr2.preempt_guard_until_ts = float(max(float(getattr(jr2, "preempt_guard_until_ts", 0.0) or 0.0), float(now_ts + 1.5)))
            except Exception:
                pass

            if not_active:
                try:
                    jr2.last_preempt_error = f"executor_not_active_converge_fast reason={reason}"
                except Exception:
                    pass

                # ✅ not_active면 stop이 필요 없는 상태 -> drain을 길게 유지할 이유가 없음
                #    cleanup이 즉시 SSOT release하도록 더 공격적으로 당김
                try:
                    jr2.force_release_deadline_ts = float(min(
                        float(getattr(jr2, "force_release_deadline_ts", now_ts) or now_ts),
                        now_ts + 0.1,
                    ))
                except Exception:
                    pass
                try:
                    # drain도 짧게(다음 tick에서 바로 free로 보이게)
                    jr2.pending_drain_until = float(min(
                        float(getattr(jr2, "pending_drain_until", drain_until) or drain_until),
                        now_ts + 0.1,
                    ))
                except Exception:
                    pass

                try:
                    jr2.preempt_inflight = True
                    jr2.status = "PREEMPTING"
                    jr2.last_preempt_try_ts = float(now_ts)
                except Exception:
                    pass

                try:
                    jr2.preempt_inflight = True
                    jr2.status = "PREEMPTING"
                    jr2.last_preempt_try_ts = float(now_ts)
                except Exception:
                    pass
            elif not ok_stop:
                try:
                    jr2.last_preempt_error = str(resp.get("reason", "stop_failed"))
                    jr2.preempt_inflight = True
                    jr2.status = "PREEMPTING"
                    jr2.last_preempt_try_ts = float(now_ts)
                except Exception:
                    pass

    if not_active:
        return {
            "ok": True,
            "job_id": jid,
            "cluster_id": cid,
            "nodes": list(nodes),
            "resp": resp,
            "note": "not_active_keep_preempting_wait_cleanup",
            "drain_until": float(drain_until),
            "run_id": run_id,
        }
    if not ok_stop:
        return {
            "ok": False,
            "job_id": jid,
            "cluster_id": cid,
            "nodes": list(nodes),
            "resp": resp,
            "note": "stop_failed_keep_preempting",
            "drain_until": float(drain_until),
            "run_id": run_id,
        }
    return {
        "ok": True,
        "job_id": jid,
        "cluster_id": cid,
        "nodes": list(nodes),
        "resp": resp,
        "note": "preempt_requested_pending_ack",
        "drain_until": float(drain_until),
        "run_id": run_id,
    }

def _emit_csp_metrics_tick() -> None:
    try:
        rl = get_run_logger()
    except Exception:
        return

    now_ts = time.time()

    with STATE_LOCK:
        st = get_global_state()

        clusters = getattr(st, "clusters", {}) or {}
        ct = getattr(st, "cluster_telemetry", {}) or {}
        cq_map = getattr(st, "cluster_queues", {}) or {}

        # global queue length (원하면 여기를 진짜 global_queue 길이로 쓰세요)
        gq = getattr(st, "global_queue", None)
        qlen_global = int(len(gq)) if isinstance(gq, list) else 0

        for cid, cr in (clusters or {}).items():
            cid = str(cid)

            total = int(getattr(cr, "total_gpus", 0) or 0)
            used = int(getattr(cr, "used_gpus", 0) or 0)
            free = int(getattr(cr, "free_gpus", max(0, total - used)) or 0)

            # cluster queue length
            qlen_cluster = 0
            q = cq_map.get(cid)
            if q is not None:
                try:
                    qlen_cluster = int(len(getattr(q, "_jobs", []) or []))
                except Exception:
                    try:
                        qlen_cluster = int(len(q))  # 마지막 fallback
                    except Exception:
                        qlen_cluster = 0

            price = float(getattr(cr, "price_per_gpu_hour", 0.0) or 0.0)
            sf = float(getattr(cr, "speed_factor", 1.0) or 1.0)

            # telemetry 우선
            util_out = 0.0  # 0..100
            p_cur = 0.0     # total W

            tel = ct.get(cid)
            if isinstance(tel, dict):
                util_out = float(tel.get("util_avg", 0.0) or 0.0)
                p_cur = float(tel.get("power_sum", 0.0) or 0.0)
            else:
                u = float(getattr(cr, "util", 0.0) or 0.0)
                util_out = (u * 100.0) if u <= 1.5 else u
                p_cur = float(getattr(cr, "power_current_w", 0.0) or 0.0)

            # energy_pressure
            p_bud = float(getattr(cr, "power_budget_w", 0.0) or 0.0)
            if p_bud <= 0 and total > 0:
                p_bud = float(total) * 300.0  # 임시 budget
            energy_pressure = 0.0 if p_bud <= 0 else max(0.0, min(1.0, p_cur / p_bud))

            cost_norm = 0.0
            speed_norm = 0.0

            try:
                rl.csp_metrics(
                    cluster=cid,
                    qlen_global=int(qlen_global),
                    total_gpus=int(total),
                    used_gpus=int(used),
                    free_gpus=int(free),
                    util=float(util_out),
                    price_per_gpu_hour=float(price),
                    speed_factor=float(sf),
                    energy_pressure=float(energy_pressure),
                    cost_norm=float(cost_norm),
                    speed_norm=float(speed_norm),
                    ts=now_ts,
                )
            except Exception as e:
                logger.warning("[csp_metrics] write failed cid=%s err=%r", cid, e)

def _cleanup_drains_locked(state: Any, now_ts: float) -> None:
    # node drain
    d = getattr(state, "node_drain_until", None)
    if isinstance(d, dict):
        for k in list(d.keys()):
            try:
                v = float(d.get(k, 0.0) or 0.0)
            except Exception:
                v = 0.0
            if v <= now_ts:
                d.pop(k, None)

    # cluster blocked
    m = getattr(state, "cluster_launch_blocked_until", None)
    if isinstance(m, dict):
        for cid in list(m.keys()):
            try:
                v = float(m.get(cid, 0.0) or 0.0)
            except Exception:
                v = 0.0
            if v <= now_ts:
                m.pop(cid, None)

def _start_job_on_cluster(
    job: Any,
    cluster_id: str,
    node_names: List[str],
    g_use: int,
    is_backfill: bool = False,
) -> Dict[str, Any]:
    import time

    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    # -------------------------
    # small helpers
    # -------------------------
    def _as_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _get(obj: Any, k: str, default=None):
        try:
            if isinstance(obj, dict):
                return obj.get(k, default)
            return getattr(obj, k, default)
        except Exception:
            return default

    def _safe_int(x: Any, d: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return d

    def _safe_float(x: Any, d: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(d)

    def _same_plan(out0: Any, cluster_id0: str, nodes0: List[str], g0: int) -> bool:
        if not isinstance(out0, dict):
            return False
        oc = out0.get("cluster_id", None)
        if oc is None:
            oc = out0.get("cluster", None)
        on = out0.get("nodes", None)
        og = out0.get("g", None)
        if og is None:
            og = out0.get("world_size", None)

        if oc is None and on is None and og is None:
            return False

        try:
            if oc is not None and str(oc) != str(cluster_id0):
                return False
        except Exception:
            return False

        try:
            if on is not None:
                on2 = [str(x) for x in (on or [])]
                if on2 != [str(x) for x in (nodes0 or [])]:
                    return False
        except Exception:
            return False

        try:
            if og is not None and int(og) != int(g0):
                return False
        except Exception:
            return False

        return True

    def _idempotent_ok_by_run(out0: Any, exp_run_id: str, exp_attempt: int) -> Tuple[bool, str]:
        if not isinstance(out0, dict):
            return False, "out_not_dict"

        run_id0 = _as_str(out0.get("run_id", "") or "").strip()
        attempt0 = _safe_int(out0.get("attempt", None), -1)

        if not run_id0:
            return False, "missing_run_id_in_worker_resp"
        if run_id0 != str(exp_run_id):
            return False, f"run_id_mismatch exp={exp_run_id} got={run_id0}"

        if attempt0 <= 0:
            return True, "run_id_match_attempt_missing_ok"
        if int(attempt0) != int(exp_attempt):
            return False, f"attempt_mismatch exp={exp_attempt} got={attempt0}"

        return True, "run_id_attempt_match_ok"

    def _set_backfill_tags_locked(jr: Any, *, pin: str, hol_for: Optional[str], now0: float) -> None:
        """
        HOL waiting이 있는 클러스터에서 launch되는 non-gang job은 victim화.
        hol_for(=HOL job_id or norm id)를 같이 박아야 victim 매칭이 안정됨.
        """
        try:
            jr.is_backfill = True
        except Exception:
            pass
        try:
            jr.is_hol_backfill = True
        except Exception:
            pass
        try:
            jr.queue_kind = "HOL_BACKFILL"
        except Exception:
            pass
        try:
            jr.hol_backfill_pin = str(pin)
        except Exception:
            pass
        # ✅ 핵심: victim이 어느 HOL을 위한 건지 SSOT로 기록
        try:
            v = str(hol_for or "").strip()
            jr.hol_backfill_for = (v if v else None)
        except Exception:
            pass

        ddl0 = _safe_float(_get(jr, "backfill_deadline_ts", 0.0) or 0.0, 0.0)
        if ddl0 <= 0.0 or ddl0 < now0:
            try:
                slice_sec = float(globals().get("BACKFILL_SLICE_SEC", 60.0) or 60.0)
            except Exception:
                slice_sec = 60.0
            try:
                jr.backfill_deadline_ts = float(now0 + slice_sec)
            except Exception:
                pass

    def _term_set() -> set:
        try:
            return set(_TERMINAL)
        except Exception:
            return {"FINISHED", "FAILED", "CANCELLED", "COMPLETED", "PREEMPTED"}

    # -------------------------
    # normalize inputs
    # -------------------------
    cluster_id = _as_str(cluster_id).strip()
    node_names = [_as_str(x).strip() for x in (node_names or []) if _as_str(x).strip()]

    try:
        g_use = int(g_use or 0)
    except Exception:
        g_use = 0

    jid = _as_str(_get(job, "job_id", None) or _get(job, "id", None) or "").strip()
    if not jid:
        return {"ok": False, "reason": "missing_job_id"}

    if g_use <= 0:
        return {"ok": False, "reason": "invalid_g_use", "g_use": g_use}

    if len(node_names) != g_use:
        return {"ok": False, "reason": "nodes_len_mismatch", "g_use": g_use, "nodes": node_names}

    # gang 판정
    try:
        gang_need = int(_job_gang_required_g(job) or 0)
    except Exception:
        gang_need = 0

    if gang_need == 4 and (g_use != 4 or len(node_names) != 4):
        return {"ok": False, "reason": "gang4_requires_ws4", "jid": jid, "g_use": g_use, "nodes": node_names}

    snap: Dict[str, Any] = {
        "jid": str(jid),
        "cluster_id": str(cluster_id),
        "nodes": list(node_names),
        "g_use": int(g_use),
        "is_backfill": bool(is_backfill),
        "gang_need": int(gang_need),
        "model": "",
        "dataset": "",
        "epochs": 0,
        "batch_size_per_gpu": 0,
        "user_request": "",
        "run_id": "",
        "attempt": 1,
        "resume": None,
        "status0": "",
        "started_emit_key": "",
        "g_target": 0,
        "g_alloc": int(g_use),
        "launch_dispatched_prev": False,
        "forced_hol_backfill": False,
        "hol_pin": "",
        "hol_jid": "",
    }

    # =========================================================
    # Phase 1) LOCK: 제약/멱등/SSOT(run_id/attempt) 확정 + snap 구성
    #   - 여기서는 reserve_nodes() 호출 금지
    #   - 여기서는 STARTING/status 변경 금지 (reserve 성공 후에만)
    # =========================================================
    with STATE_LOCK:
        st = get_global_state()
        jr = (getattr(st, "jobs", {}) or {}).get(jid)
        if jr is None:
            return {"ok": False, "reason": "job_not_found", "jid": jid}

        status0 = _as_str(_get(jr, "status", "")).upper()
        snap["status0"] = status0

        terminal_set = _term_set()
        if status0 in terminal_set:
            return {"ok": False, "reason": "terminal_job", "status": status0}

        # ✅ reserve 전에 blocked/cooldown 체크 (드레인/롤백 루프 방지)
        now_chk = float(time.time())
        try:
            bu = float(_get(jr, "blocked_until", 0.0) or 0.0)
        except Exception:
            bu = 0.0
        try:
            cd = float(_get(jr, "launch_cooldown_until_ts", 0.0) or 0.0)
        except Exception:
            cd = 0.0
        if max(bu, cd) > now_chk:
            return {
                "ok": False,
                "reason": "blocked_until",
                "jid": jid,
                "blocked_until": float(max(bu, cd)),
                "now": float(now_chk),
            }

        try:
            s0 = str(status0).upper()
        except Exception:
            s0 = status0

        try:
            is_hbf = bool(_get(jr, "is_hol_backfill", False)) or (str(_get(jr, "queue_kind", "") or "").upper() == "HOL_BACKFILL")
        except Exception:
            is_hbf = False

        # "HOL reclaim lock"이 살아있으면(=HoL 뜰 때까지) victim은 막아야 함
        hold_until = 0.0
        try:
            hold_until = float(getattr(st, "hol_reclaim_until_ts", 0.0) or 0.0)
        except Exception:
            hold_until = 0.0

        if is_hbf and hold_until > now_chk and s0 in ("STOPPED", "PREEMPTED", "PREEMPTING", "STOPPING"):
            # blocked_until을 확실히 걸어두고 리턴 (즉시 재런치 방지)
            try:
                prev_bu = float(_get(jr, "blocked_until", 0.0) or 0.0)
                jr.blocked_until = float(max(prev_bu, hold_until + 5.0))
            except Exception:
                pass

            return {
                "ok": False,
                "reason": "hol_reclaim_hold_victim",
                "jid": jid,
                "status": s0,
                "now": float(now_chk),
                "hold_until": float(hold_until),
            }
        # ✅ RUNNING: 동일 plan이면 idempotent_ok
        if status0 == "RUNNING":
            cur_cid = _as_str(_get(jr, "cluster_id", "") or "").strip()
            try:
                cur_nodes = [_as_str(x).strip() for x in (_get(jr, "nodes", []) or [])]
                cur_nodes = [x for x in cur_nodes if x]
            except Exception:
                cur_nodes = []
            cur_g = _safe_int(_get(jr, "g_cur", 0) or 0, 0)

            if cur_cid == cluster_id and cur_nodes == node_names and cur_g == int(g_use):
                return {
                    "ok": True,
                    "job_id": jid,
                    "cluster_id": cluster_id,
                    "g": int(g_use),
                    "nodes": list(node_names),
                    "note": "idempotent_already_running_same_plan",
                }
            return {
                "ok": False,
                "reason": "already_running_conflict",
                "cur_cluster": cur_cid,
                "cur_nodes": cur_nodes,
                "cur_g": cur_g,
                "req_cluster": cluster_id,
                "req_nodes": node_names,
                "req_g": int(g_use),
            }

        # 이동 금지 조건
        ckpt_local_only = bool(_get(jr, "checkpoint_local_only", False))
        resume = _get(jr, "resume_from_checkpoint", None) or None
        if resume is not None:
            resume = _as_str(resume).strip() or None
        ckpt_cluster = _as_str(_get(jr, "checkpoint_cluster_id", "") or "").strip()

        must_stay_local = bool(ckpt_local_only or resume or ckpt_cluster)
        if must_stay_local:
            allowed = (
                ckpt_cluster
                or _as_str(_get(jr, "home_cluster_id", "") or "").strip()
                or _as_str(_get(jr, "admitted_cluster_id", "") or "").strip()
            )
            if allowed and cluster_id != allowed:
                return {
                    "ok": False,
                    "reason": "checkpoint_local_cluster_mismatch",
                    "jid": jid,
                    "req_cluster": cluster_id,
                    "allowed_cluster": allowed,
                    "ckpt_cluster": ckpt_cluster or None,
                    "home": _get(jr, "home_cluster_id", None),
                    "admitted": _get(jr, "admitted_cluster_id", None),
                    "resume": resume,
                    "ckpt_local_only": ckpt_local_only,
                }

        # STARTING 멱등: 동일 (cluster,nodes)만 허용
        # ✅ (핵심수정) launch_dispatched=True라고 무조건 스킵하지 않는다.
        if status0 == "STARTING":
            cur_cid = _as_str(_get(jr, "cluster_id", "") or "").strip()
            try:
                cur_nodes = [_as_str(x).strip() for x in (_get(jr, "nodes", []) or [])]
                cur_nodes = [x for x in cur_nodes if x]
            except Exception:
                cur_nodes = []

            if cur_cid and cur_cid != cluster_id:
                return {
                    "ok": False,
                    "reason": "starting_conflict_cluster",
                    "cur_cluster": cur_cid,
                    "req_cluster": cluster_id,
                }
            if cur_nodes and cur_nodes != node_names:
                return {
                    "ok": False,
                    "reason": "starting_conflict_nodes",
                    "cur_nodes": cur_nodes,
                    "req_nodes": node_names,
                }

            dispatched_prev = bool(_get(jr, "launch_dispatched", False))
            snap["launch_dispatched_prev"] = bool(dispatched_prev)

            if dispatched_prev:
                try:
                    ts0 = float(_get(jr, "launch_dispatched_ts", 0.0) or 0.0)
                except Exception:
                    ts0 = 0.0
                now1 = float(time.time())
                try:
                    STALE_SEC = float(globals().get("LAUNCH_DISPATCH_STALE_SEC", 10.0) or 10.0)
                except Exception:
                    STALE_SEC = 10.0

                # 최근 디스패치면 스킵(워커가 아직 받을 수 있음)
                if ts0 > 0.0 and (now1 - ts0) < STALE_SEC:
                    return {
                        "ok": True,
                        "job_id": jid,
                        "cluster_id": cluster_id,
                        "g": int(g_use),
                        "nodes": list(node_names),
                        "note": "idempotent_already_starting_recent_dispatched_skip_executor",
                        "launch_dispatched_age_sec": float(now1 - ts0),
                    }

                # stale이면 재시도 허용: 디스패치/인플라이트를 내려서 아래 flow가 정상 진행되게
                try:
                    jr.launch_dispatched = False
                    jr.launch_dispatched_ts = 0.0
                except Exception:
                    pass
                try:
                    jr.launch_inflight = False
                    jr.launching_since_ts = 0.0
                except Exception:
                    pass

                # ✅ placement(클러스터/노드/할당) 잔재 제거 (충돌 루프 방지)
                try:
                    jr.cluster_id = None
                    jr.nodes = []
                    jr.g_cur = 0
                    try:
                        jr.g_alloc = 0
                        jr.world_size = 0
                        jr.actual_g = 0
                    except Exception:
                        pass
                except Exception:
                    pass

                # ✅ node_owner 잔재 제거 (free 계산/중복런치 꼬임 방지)
                try:
                    owners = getattr(st, "node_owner", None)
                    if isinstance(owners, dict):
                        for n, o in list(owners.items()):
                            if str(o) == str(jid):
                                owners.pop(n, None)
                except Exception:
                    pass

                # QUEUED로 수렴
                try:
                    jr.status = "QUEUED"
                    _clear_requeue_victim_tags_locked(jr)

                except Exception:
                    pass

                try:
                    logger.info(
                        "[START_STALE_ROLLBACK] jid=%s cid=%s g_use=%s requeue QUEUED",
                        jid, cluster_id, g_use
                    )
                except Exception:
                    pass

                # ✅ 중요: QUEUED로 내릴 때 "SSOT 큐"에 재삽입 (증발 방지)
                jid_s = str(jid)

                # 1) 전역 큐(list) SSOT면 거기에 넣기 (중복 방지)
                try:
                    gq = getattr(st, "global_queue", None)
                    if isinstance(gq, list):
                        if jid_s not in [str(x) for x in gq]:
                            gq.append(jid_s)
                except Exception:
                    pass

                # 2) 홈(또는 admitted) 클러스터 큐의 SSOT는 cq._jobs 임 (cq.q/cq.queue 아님)
                try:
                    home = _as_str(
                        _get(jr, "home_cluster_id", "") or _get(jr, "admitted_cluster_id", "") or ""
                    ).strip()

                    if home:
                        cq = (getattr(st, "cluster_queues", {}) or {}).get(str(home))
                        if cq is not None:
                            items = list(getattr(cq, "_jobs", []) or [])

                            # _jobs 안의 job_id 비교(중복 방지)
                            def _item_jid(it) -> str:
                                try:
                                    if isinstance(it, str):
                                        return str(it)
                                    if isinstance(it, dict):
                                        return str(it.get("job_id") or it.get("id") or "")
                                    if hasattr(it, "job_id"):
                                        return str(getattr(it, "job_id") or "")
                                    return str(it)
                                except Exception:
                                    return ""

                            already = False
                            for it in items:
                                if _item_jid(it) == jid_s:
                                    already = True
                                    break

                            if not already:
                                # g_req는 최소 1, 가능하면 job의 g_target 힌트 반영
                                try:
                                    g_req0 = int(_get(jr, "g_target", 1) or 1)
                                except Exception:
                                    g_req0 = 1
                                items.append({"job_id": jid_s, "g_req": max(1, int(g_req0))})
                                setattr(cq, "_jobs", items)
                except Exception:
                    pass

                status0 = "QUEUED"
                snap["status0"] = "QUEUED"

        if status0 != "QUEUED":
            # ✅ STARTING stuck 복구: SSOT release + QUEUED 수렴 + 큐 재삽입
            if str(status0).upper() == "STARTING":
                try:
                    # reserve가 남아있을 수 있으니 release 시도
                    try:
                        cid0 = _as_str(_get(jr, "cluster_id", "") or "").strip()
                    except Exception:
                        cid0 = ""
                    try:
                        release_nodes_for_job_locked(job_id=str(jid), cluster_id=str(cid0))
                    except Exception:
                        pass

                    for k, v in (
                        ("launch_inflight", False),
                        ("launching_since_ts", 0.0),
                        ("launch_dispatched", False),
                        ("launch_dispatched_ts", 0.0),
                        ("preempt_inflight", False),
                        ("preempting_since_ts", 0.0),
                    ):
                        try:
                            setattr(jr, k, v)
                        except Exception:
                            pass

                    try:
                        jr.status = "QUEUED"
                        _clear_requeue_victim_tags_locked(jr)

                    except Exception:
                        pass
                    status0 = "QUEUED"
                except Exception:
                    pass

            if str(status0).upper() != "QUEUED":
                return {"ok": False, "reason": "invalid_status_for_launch", "status": status0}

        # jr 기준 gang 재확인
        try:
            if gang_need == 0 and int(_job_gang_required_g(jr) or 0) == 4:
                gang_need = 4
        except Exception:
            pass
        if gang_need == 4 and (g_use != 4 or len(node_names) != 4):
            return {
                "ok": False,
                "reason": "gang4_requires_ws4",
                "jid": jid,
                "g_use": g_use,
                "nodes": node_names,
            }

        model = _as_str(_get(jr, "model", "") or _get(job, "model", "") or "")
        dataset = _as_str(_get(jr, "dataset", "") or _get(job, "dataset", "") or "")
        epochs = int(_get(jr, "epochs", None) or _get(job, "epochs", None) or 1)

        batch_size_per_gpu = int(
            _get(jr, "batch_size_per_gpu", None)
            or _get(job, "batch_size_per_gpu", None)
            or _get(jr, "batch_size", None)
            or _get(job, "batch_size", None)
            or 32
        )
        user_request = _as_str(_get(jr, "user_request", "") or _get(job, "user_request", "") or "")

        # ✅ 강제 HOL_BACKFILL 태그 결정
        hol_jid = ""
        hol_pin = ""
        try:
            hol_jid, hol_pin, _pin_free = _find_waiting_gang4_hol_locked(st)
        except Exception:
            hol_jid, hol_pin = "", ""

        hol_jid_s = _as_str(hol_jid).strip()
        hol_pin_s = _as_str(hol_pin).strip()

        waiting_gang4_here = bool(hol_jid_s and hol_pin_s and hol_pin_s == cluster_id)
        forced_hol_backfill = bool(waiting_gang4_here and (jid != hol_jid_s) and (int(gang_need) != 4))
        if forced_hol_backfill:
            is_backfill = True
            snap["forced_hol_backfill"] = True
            snap["hol_pin"] = str(cluster_id)
            snap["hol_jid"] = str(hol_jid_s)

        # ✅ reclaim 활성화 중이면 pin에 BACKFILL(또는 forced_hol_backfill) 런치 금지
        try:
            reclaim_active = bool(getattr(st, "hol_reclaim_active", False))
            reclaim_pin = _as_str(getattr(st, "hol_reclaim_pin", "") or "").strip()
            reclaim_until = float(getattr(st, "hol_reclaim_until_ts", 0.0) or 0.0)
        except Exception:
            reclaim_active, reclaim_pin, reclaim_until = False, "", 0.0

        if reclaim_active and reclaim_pin == str(cluster_id) and now_chk <= reclaim_until:
            # pin reclaim window 동안 backfill을 다시 올리면 무한 preempt 루프가 납니다.
            if bool(is_backfill) or bool(forced_hol_backfill):
                try:
                    jr.blocked_until = float(max(float(_get(jr, "blocked_until", 0.0) or 0.0), now_chk + 1.0))
                except Exception:
                    pass
                return {
                    "ok": False,
                    "reason": "hol_reclaim_active_block_backfill",
                    "jid": jid,
                    "pin": str(cluster_id),
                    "until_ts": float(reclaim_until),
                }

        # run_id/attempt SSOT (여긴 안전: 예약 전이라도 식별자만)
        run_id = _get(jr, "run_id", None) or None
        run_id = _as_str(run_id).strip() if run_id else ""
        if not run_id:
            try:
                run_id = _new_run_id()
            except Exception:
                run_id = hex(int(time.time() * 1e6))[2:]
            try:
                jr.run_id = run_id
            except Exception:
                pass

        attempt = _get(jr, "attempt", None)
        try:
            attempt = int(attempt) if attempt is not None else 1
            if attempt <= 0:
                attempt = 1
        except Exception:
            attempt = 1

        # ✅ 'prev_status'는 "변경 전" 상태로 판단해야 함 (status0 사용 권장)
        prev_status = str(status0).lower()  # <- 위에서 status0 = jr.status snapshot 해놨죠

        if prev_status in ("preempted", "preempting", "stopped"):
            attempt += 1
            try:
                run_id = _new_run_id()        # ✅ 로컬 변수도 같이 갱신
            except Exception:
                run_id = hex(int(time.time() * 1e6))[2:]
            try:
                jr.run_id = run_id
            except Exception:
                pass

        try:
            jr.attempt = int(attempt)
        except Exception:
            pass

        started_key = f"{run_id}:{int(attempt)}"
        snap["started_emit_key"] = started_key
        snap["run_id"] = str(run_id)
        snap["attempt"] = int(attempt)

        # ⚠️ SSOT 오염 방지: g_alloc/world_size/actual_g는 Phase3에서만 기록
        try:
            jr.metrics_cluster_id = str(cluster_id)
        except Exception:
            pass

        snap.update(
            {
                "model": str(model),
                "dataset": str(dataset),
                "epochs": int(epochs),
                "batch_size_per_gpu": int(batch_size_per_gpu),
                "user_request": str(user_request),
                "run_id": str(run_id),
                "attempt": int(attempt),
                "resume": resume,
                "is_backfill": bool(is_backfill),
                "gang_need": int(gang_need),
            }
        )

    # =========================================================
    # Phase 2) LOCK 밖: reserve_nodes() (데드락 방지)
    # =========================================================
    now0 = float(time.time())
    try:
        reserve_nodes(job_id=str(jid), cluster_id=str(cluster_id), nodes=list(node_names))
    except Exception as e:
        return {
            "ok": False,
            "reason": "reserve_nodes_failed",
            "jid": str(jid),
            "cluster_id": str(cluster_id),
            "nodes": list(node_names),
            "exc": repr(e),
        }

    # =========================================================
    # Phase 3) LOCK 재진입: reserve 성공한 경우에만 STARTING 수렴 + 태그 박기
    #   - 상태가 바뀌었으면 안전하게 처리
    # =========================================================
    with STATE_LOCK:
        st = get_global_state()
        jr = (getattr(st, "jobs", {}) or {}).get(jid)
        if jr is None:
            # reserve는 했는데 job이 사라짐 -> 즉시 release
            try:
                release_nodes_for_job_locked(job_id=str(jid), cluster_id=str(cluster_id))
            except Exception:
                pass
            return {"ok": False, "reason": "job_not_found_after_reserve", "jid": str(jid)}

        status1 = _as_str(_get(jr, "status", "")).upper()
        terminal_set = _term_set()

        if status1 in terminal_set:
            try:
                release_nodes_for_job_locked(job_id=str(jid), cluster_id=str(cluster_id))
            except Exception:
                pass
            return {"ok": False, "reason": "terminal_job_after_reserve", "status": status1, "jid": str(jid)}

        if status1 in ("RUNNING", "STARTING"):
            cur_cid = _as_str(_get(jr, "cluster_id", "") or "").strip()
            try:
                cur_nodes = [_as_str(x).strip() for x in (_get(jr, "nodes", []) or [])]
                cur_nodes = [x for x in cur_nodes if x]
            except Exception:
                cur_nodes = []
            cur_g = _safe_int(_get(jr, "g_cur", 0) or 0, 0)

            same = (cur_cid == cluster_id and cur_nodes == node_names and int(cur_g) == int(g_use))
            if same:
                # ✅ reserve를 "이미 같은 plan이 잡고 있었던 상태"일 수 있음
                #    여기서 executor를 재호출할지 여부는 Phase1에서 이미 결정했으니,
                #    이 지점에서는 'release하지 않고' 진행하면 유령 점유가 될 수 있음.
                #    따라서 STARTING+dispatched_recent 스킵으로 여기로 온 케이스가 아니라면 release 후 반환.
                dispatched_prev = bool(_get(jr, "launch_dispatched", False))
                if status1 == "STARTING" and dispatched_prev:
                    # 최근 dispatched 스킵은 Phase1에서 return 되었어야 함.
                    # 여기까지 왔다는 건 재시도 경로이므로, executor 호출을 진행해야 한다.
                    pass
                else:
                    # executor 호출을 안 할 거라면 우리가 잡은 reserve를 풀어야 안전.
                    try:
                        release_nodes_for_job_locked(job_id=str(jid), cluster_id=str(cluster_id))
                    except Exception:
                        pass
                    return {
                        "ok": True,
                        "job_id": jid,
                        "cluster_id": cluster_id,
                        "g": int(g_use),
                        "nodes": list(node_names),
                        "note": f"idempotent_already_{status1.lower()}_same_plan_release_reserved",
                    }

            # 충돌이면 우리가 잡은 reserve를 풀어야 함
            try:
                release_nodes_for_job_locked(job_id=str(jid), cluster_id=str(cluster_id))
            except Exception:
                pass

            return {
                "ok": False,
                "reason": f"{status1.lower()}_conflict_after_reserve",
                "cur_cluster": cur_cid,
                "cur_nodes": cur_nodes,
                "cur_g": cur_g,
                "req_cluster": cluster_id,
                "req_nodes": node_names,
                "req_g": int(g_use),
            }

        # QUEUED(혹은 기타): reserve 성공했으니 STARTING으로 수렴
        try:
            jr.cluster_id = str(cluster_id)
            jr.nodes = list(node_names)
            jr.g_cur = int(g_use)
            jr.launch_inflight = True
            jr.launching_since_ts = float(now0)
            jr.status = "STARTING"
        except Exception:
            pass

        # ✅ STARTING에서도 node_owner를 선점(SSOT)해서 free 계산/중복런치를 막는다
        try:
            owners = getattr(st, "node_owner", None)
            if not isinstance(owners, dict):
                owners = {}
                st.node_owner = owners
            for n in list(node_names):
                owners[str(n)] = str(jid)
        except Exception:
            pass

        # (선택) clusters[cluster_id].node_owner 뷰도 즉시 반영(디버그 가독성)
        try:
            cls = getattr(st, "clusters", None)
            if isinstance(cls, dict):
                cobj = cls.get(str(cluster_id))
                if cobj is not None:
                    view = {}
                    if isinstance(cobj, dict):
                        cnodes = [str(x).strip() for x in (cobj.get("nodes") or []) if str(x).strip()]
                    else:
                        cnodes = [str(x).strip() for x in (getattr(cobj, "nodes", None) or []) if str(x).strip()]
                    for nn in cnodes:
                        oo = owners.get(nn)
                        if oo is not None:
                            view[nn] = oo
                    if isinstance(cobj, dict):
                        cobj["node_owner"] = view
                    else:
                        setattr(cobj, "node_owner", view)
        except Exception:
            pass

        # ✅ SSOT 오염 방지: 여기서 alloc 관련 필드 확정 기록
        try:
            jr.g_alloc = int(g_use)
        except Exception:
            pass
        try:
            jr.world_size = int(g_use)
        except Exception:
            pass
        try:
            jr.actual_g = int(g_use)
        except Exception:
            pass

        try:
            jr.metrics_cluster_id = str(cluster_id)
        except Exception:
            pass

        # ✅ SSOT 태그 박기(STARTING 예약 시점) - 강화 버전
        if bool(snap.get("is_backfill")):
            # 1) 기본 backfill 마킹
            try:
                jr.is_backfill = True
            except Exception:
                pass

            try:
                qk0 = str(getattr(jr, "queue_kind", "") or "").upper().strip()
            except Exception:
                qk0 = ""

            # HOL_BACKFILL 후보: (이미 SSOT에 찍혀있거나) 이번 런치가 forced_hol_backfill인 경우
            try:
                ssot_hbf = bool(_get(jr, "is_hol_backfill", False)) or (qk0 == "HOL_BACKFILL")
            except Exception:
                ssot_hbf = False
            forced_flag = bool(snap.get("forced_hol_backfill"))
            hol_candidate = bool(ssot_hbf or forced_flag)

            # queue_kind 기본값
            try:
                if qk0 not in ("HOL_BACKFILL", "BACKFILL"):
                    jr.queue_kind = ("HOL_BACKFILL" if hol_candidate else "BACKFILL")
            except Exception:
                pass

            # 2) hol_for 결정: SSOT 우선 → 없으면 best-effort로 HOL 찾기
            hol_for = ""
            if hol_candidate:
                try:
                    hol_for = str(_get(jr, "hol_backfill_for", "") or "").strip()
                except Exception:
                    hol_for = ""

                if not hol_for:
                    # snap에서 넘어온 hol_jid가 있으면 우선 사용
                    try:
                        hol_for = str(snap.get("hol_jid") or "").strip()
                    except Exception:
                        hol_for = ""

                if not hol_for:
                    # 마지막 수단: 지금 시점 HOL 탐색(best-effort)
                    try:
                        hj, hp, _pf = _find_waiting_gang4_hol_locked(st)
                        hj = _as_str(hj).strip()
                        hp = _as_str(hp).strip()
                        if hj and hp == str(cluster_id):
                            hol_for = hj
                    except Exception:
                        pass

            # 3) hol_for가 잡혔으면 "victim SSOT" 강제 기록
            if hol_candidate and hol_for:
                _set_backfill_tags_locked(
                    jr,
                    pin=str(cluster_id),
                    hol_for=str(hol_for),
                    now0=float(now0),
                )
                try:
                    jr.preempt_ok_only_by_hol = True
                except Exception:
                    pass
            else:
                # 4) 일반 backfill: deadline만 보장
                ddl0 = _safe_float(_get(jr, "backfill_deadline_ts", 0.0) or 0.0, 0.0)
                if ddl0 <= 0.0 or ddl0 < float(now0):
                    try:
                        slice_sec = float(globals().get("BACKFILL_SLICE_SEC", 60.0) or 60.0)
                    except Exception:
                        slice_sec = 60.0
                    try:
                        jr.backfill_deadline_ts = float(float(now0) + slice_sec)
                    except Exception:
                        pass


    # -----------------------------
    # LOCK 밖: executor launch
    # -----------------------------
    t_req = float(time.time())
    req_id = f'{snap["jid"]}:{snap["run_id"]}:{snap["attempt"]}'

    if rl:
        try:
            rl.http_event(
                event="launch_req",
                job_id=str(snap["jid"]),
                cluster=str(cluster_id),
                op="launch_or_reuse",
                method="POST",
                url=f"/run_task?cluster={cluster_id}",
                http_status=0,
                ok=False,
                req_id=req_id,
                latency_ms=0.0,
                req={
                    "cluster_id": str(cluster_id),
                    "job_id": str(snap["jid"]),
                    "model": str(snap["model"]),
                    "dataset": str(snap["dataset"]),
                    "world_size": int(snap["g_use"]),
                    "epochs": int(snap["epochs"]),
                    "batch_size_per_gpu": int(snap["batch_size_per_gpu"]),
                    "nodes": list(snap["nodes"]),
                    "is_backfill": bool(snap["is_backfill"]),
                    "extra": {
                        "run_id": str(snap["run_id"]),
                        "attempt": int(snap["attempt"]),
                        "resume_from_checkpoint": snap["resume"],
                        "reason": "BACKFILL" if bool(snap["is_backfill"]) else "LAUNCH",
                        "forced_hol_backfill": bool(snap.get("forced_hol_backfill")),
                        "hol_pin": str(snap.get("hol_pin") or ""),
                        "hol_jid": str(snap.get("hol_jid") or ""),
                    },
                },
                resp={},
                ts=t_req,
            )
        except Exception:
            pass

    # ✅ executor 호출 직전에 "dispatch 됐다" SSOT (여기서 True로 만들고, stale이면 Phase1에서 풀린다)
    with STATE_LOCK:
        stx = get_global_state()
        jrx = (getattr(stx, "jobs", {}) or {}).get(jid)
        if jrx is not None:
            try:
                jrx.launch_dispatched = True
                jrx.launch_dispatched_ts = float(time.time())
            except Exception:
                pass

    try:
        out = launch_or_reuse(
            cluster_id=str(cluster_id),
            job_id=str(jid),
            model=str(snap["model"]),
            dataset=str(snap["dataset"]),
            world_size=int(g_use),
            epochs=int(snap["epochs"]),
            batch_size_per_gpu=int(snap["batch_size_per_gpu"]),
            user_request=str(snap["user_request"]) if snap["user_request"] is not None else None,
            nodes=list(node_names),
            is_backfill=bool(snap["is_backfill"]),
            extra={
                "run_id": str(snap["run_id"]),
                "attempt": int(snap["attempt"]),
                "resume_from_checkpoint": snap["resume"],
                "reason": "BACKFILL" if bool(snap["is_backfill"]) else "LAUNCH",
                "forced_hol_backfill": bool(snap.get("forced_hol_backfill")),
                "hol_pin": str(snap.get("hol_pin") or ""),
                "hol_jid": str(snap.get("hol_jid") or ""),
            },
        )

        if not isinstance(out, dict):
            out = {"ok": False, "reason": "executor_return_not_dict", "detail": out}

    except Exception as e:
        out = {"ok": False, "reason": "executor_launch_exception", "exc": repr(e)}

    # -------------------------------------------------
    # ✅ 409 node_draining이면 drain_until 기록 + 즉시 롤백(멈춤 방지)
    # -------------------------------------------------
    rollback_409 = False
    drain_node = ""
    drain_until_f = 0.0

    try:
        if isinstance(out, dict):
            http_s = int(out.get("http_status", 0) or out.get("status_code", 0) or 0)
            r = str(out.get("reason", "") or "").lower()

            # ✅ node/drain_until은 top-level에도 있을 수 있으니 먼저 읽기
            try:
                drain_node = str(out.get("node") or "").strip()
            except Exception:
                drain_node = ""
            try:
                drain_until_f = float(out.get("drain_until", 0.0) or 0.0)
            except Exception:
                drain_until_f = 0.0

            body = out.get("body")
            detail = out.get("detail")
            payload = body if isinstance(body, dict) else (detail if isinstance(detail, dict) else {})

            if isinstance(payload, dict) and isinstance(payload.get("detail"), dict):
                payload = payload.get("detail") or {}

            payload_reason = str((payload or {}).get("reason", "") or "").lower()

            # ✅ 중첩 payload에도 정보가 있으면 보강
            if not drain_node:
                try:
                    drain_node = str((payload or {}).get("node") or "").strip()
                except Exception:
                    drain_node = ""
            if drain_until_f <= 0.0:
                try:
                    drain_until_f = float((payload or {}).get("drain_until", 0.0) or 0.0)
                except Exception:
                    drain_until_f = 0.0

            is_node_draining = (http_s == 409) and (("node_draining" in r) or (payload_reason == "node_draining"))
            if is_node_draining:
                rollback_409 = True

                # ✅ drain_until이 없으면 스핀 방지용 최소 backoff
                if drain_until_f <= 0.0:
                    drain_until_f = float(time.time() + 2.0)
    except Exception:
        pass

    if rollback_409:
        now = float(time.time())

        try:
            with STATE_LOCK:
                st_rb0 = get_global_state()

                nd = getattr(st_rb0, "node_drain_until_by_node", None)
                if not isinstance(nd, dict):
                    nd = {}
                    try:
                        st_rb0.node_drain_until_by_node = nd
                    except Exception:
                        pass
                try:
                    du0 = float(drain_until_f or 0.0)
                except Exception:
                    du0 = 0.0
                if du0 <= 0.0:
                    du0 = float(time.time() + 2.0)

                targets = []
                if str(drain_node or "").strip():
                    targets = [str(drain_node).strip()]
                else:
                    # ✅ node 정보가 없으면 요청했던 노드 전체를 drain 처리
                    targets = [str(x).strip() for x in (node_names or []) if str(x).strip()]

                for nn in targets:
                    try:
                        nd[nn] = float(max(float(nd.get(nn, 0.0) or 0.0), du0))
                    except Exception:
                        pass

                jd = getattr(st_rb0, "job_drain_deadline_by_job", None)
                if not isinstance(jd, dict):
                    jd = {}
                    try:
                        st_rb0.job_drain_deadline_by_job = jd
                    except Exception:
                        pass

                jid_s = str(jid)
                if jid_s and jid_s not in jd:
                    jd[jid_s] = now + 10.0
                deadline = float(jd.get(jid_s) or (now + 10.0))

                try:
                    alpha = float(getattr(st_rb0, "DRAIN_RETRY_ALPHA_SEC", 0.5) or 0.5)
                except Exception:
                    alpha = 0.5

                if float(drain_until_f or 0.0) > 0.0:
                    unblock = min(float(drain_until_f) + alpha, deadline)
                else:
                    unblock = min(now + 0.2, deadline)

                jr0 = (getattr(st_rb0, "jobs", {}) or {}).get(jid_s)
                if jr0 is not None:
                    try:
                        cur = float(getattr(jr0, "blocked_until", 0.0) or 0.0)
                        jr0.blocked_until = max(cur, float(unblock))
                    except Exception:
                        pass
        except Exception:
            pass

        # (1) best-effort node release
        try:
            with STATE_LOCK:
                release_nodes_for_job_locked(job_id=str(jid), cluster_id=str(cluster_id))
        except Exception:
            try:
                with STATE_LOCK:
                    st_rel = get_global_state()
                    _ss_release_nodes_for_job_locked(st_rel, job_id=str(jid), nodes=list(node_names))
            except Exception:
                pass

        # ✅ 보험: node_owner 잔재 제거 (release_nodes_for_job_locked가 안 지우는 구현이면 필수)
        try:
            with STATE_LOCK:
                st_tmp = get_global_state()
                owners = getattr(st_tmp, "node_owner", None)
                if isinstance(owners, dict):
                    for nn in [str(x).strip() for x in (node_names or []) if str(x).strip()]:
                        if str(owners.get(nn) or "") == str(jid):
                            owners.pop(nn, None)
        except Exception:
            pass

        # (2) job 상태를 QUEUED로 복귀
        with STATE_LOCK:
            st_rb = get_global_state()
            jr_rb = (getattr(st_rb, "jobs", {}) or {}).get(str(jid))
            if jr_rb is not None:
                try:
                    jr_rb.launch_dispatched = False
                    jr_rb.launch_dispatched_ts = 0.0
                except Exception:
                    pass
                try:
                    jr_rb.launch_inflight = False
                    jr_rb.launching_since_ts = 0.0
                except Exception:
                    pass
                try:
                    jr_rb.status = "QUEUED"
                    _clear_requeue_victim_tags_locked(jr_rb)
                except Exception:
                    pass
                # ✅ placement clear (QUEUE로 내릴 때 같이 비워야 STARTING 멱등/충돌 루프가 안 남음)
                try:
                    jr_rb.cluster_id = None
                    jr_rb.nodes = []
                    jr_rb.g_cur = 0
                    try:
                        jr_rb.g_alloc = 0
                        jr_rb.world_size = 0
                        jr_rb.actual_g = 0
                    except Exception:
                        pass
                except Exception:
                    pass

                # ✅ QUEUED로 내리면 SSOT 큐에 재삽입(증발 방지)
                try:
                    jid_s = str(jid)
                    gq = getattr(st_rb, "global_queue", None)
                    if isinstance(gq, list) and jid_s not in [str(x) for x in gq]:
                        gq.append(jid_s)
                except Exception:
                    pass

                try:
                    home = _as_str(
                        _get(jr_rb, "home_cluster_id", "") or _get(jr_rb, "admitted_cluster_id", "") or ""
                    ).strip()
                    if home:
                        cq = (getattr(st_rb, "cluster_queues", {}) or {}).get(str(home))
                        if cq is not None:
                            items = list(getattr(cq, "_jobs", []) or [])
                            if jid_s not in [
                                (str(it.get("job_id") or it.get("id") or "") if isinstance(it, dict)
                                 else (str(getattr(it, "job_id")) if hasattr(it, "job_id") else str(it)))
                                for it in items
                            ]:
                                items.append({"job_id": jid_s, "g_req": max(1, int(_get(jr_rb, "g_target", 1) or 1))})
                                setattr(cq, "_jobs", items)
                except Exception:
                    pass

        try:
            logger.info(
                "[DRAIN_409_BACKOFF] jid=%s node=%s drain_until=%.3f -> blocked_until~=min(drain+alpha, first+10s)",
                str(jid), str(drain_node), float(drain_until_f or 0.0)
            )
        except Exception:
            pass

        return {
            "ok": False,
            "reason": "node_draining",
            "detail": out,
            "run_id": str(snap["run_id"]),
            "attempt": int(snap["attempt"]),
            "cluster_id": cluster_id,
            "nodes": list(node_names),
            "g": int(g_use),
            "idempotent_mode": None,
        }

    # -----------------------------
    # 일반 성공/실패 수렴
    # -----------------------------
    ok0 = bool(out.get("ok")) if isinstance(out, dict) else False
    status_s = str((out or {}).get("status", "") or "").lower()
    reason_s = str((out or {}).get("reason", "") or "").lower()

    idempotent_hint = bool(
        status_s in ("already_running", "running", "idempotent_ok")
        or ("already_running" in reason_s)
        or ("idempotent" in reason_s)
    )

    ok = False
    idempotent_mode = None
    if ok0:
        ok = True
        idempotent_mode = "out_ok_true"
    elif idempotent_hint:
        if _same_plan(out, cluster_id, node_names, int(g_use)):
            ok = True
            idempotent_mode = "idempotent_same_plan"
        else:
            ok_run, why = _idempotent_ok_by_run(out, str(snap["run_id"]), int(snap["attempt"]))
            ok = bool(ok_run)
            idempotent_mode = f"idempotent_run_match:{why}" if ok else f"idempotent_reject:{why}"

    t_resp = float(time.time())
    if rl:
        try:
            rl.http_event(
                event="launch_resp",
                job_id=str(snap["jid"]),
                cluster=str(cluster_id),
                op="launch_or_reuse",
                method="POST",
                url=f"/run_task?cluster={cluster_id}",
                http_status=int(out.get("http_status", 200 if ok else 500)) if isinstance(out, dict) else (200 if ok else 500),
                ok=bool(ok),
                req_id=req_id,
                latency_ms=(t_resp - t_req) * 1000.0,
                req={},
                resp=(out if isinstance(out, dict) else {"detail": out}),
                ts=t_resp,
            )
        except Exception:
            pass

    if ok:
        emit_started = False
        with STATE_LOCK:
            st2 = get_global_state()
            jr2 = (getattr(st2, "jobs", {}) or {}).get(jid)
            if jr2 is not None:
                try:
                    jr2.status = "RUNNING"

                    # ✅ SSOT node ownership 확정 (RUNNING 수렴 시점)
                    try:
                        owners = getattr(st2, "node_owner", None)
                        if not isinstance(owners, dict):
                            owners = {}
                            st2.node_owner = owners
                    except Exception:
                        owners = getattr(st2, "node_owner", {})
                        if not isinstance(owners, dict):
                            owners = {}

                    try:
                        n2c = getattr(st2, "node_cluster", None)
                        if not isinstance(n2c, dict):
                            n2c = {}
                            st2.node_cluster = n2c
                    except Exception:
                        n2c = getattr(st2, "node_cluster", {})
                        if not isinstance(n2c, dict):
                            n2c = {}

                    # ✅ node_owner는 node_names 전체에 대해 갱신해야 함 (n 미정의 버그 수정)
                    for n in list(node_names):
                        owners[n] = jid
                        if n not in n2c or not str(n2c.get(n) or "").strip():
                            n2c[n] = cluster_id

                    try:
                        cls = getattr(st2, "clusters", None)
                        if isinstance(cls, dict):
                            cobj = cls.get(str(cluster_id))
                            if cobj is not None:
                                cnodes = []
                                if isinstance(cobj, dict):
                                    cnodes = [str(x).strip() for x in (cobj.get("nodes") or []) if str(x).strip()]
                                else:
                                    cnodes = [str(x).strip() for x in (getattr(cobj, "nodes", None) or []) if str(x).strip()]
                                view = {}
                                for nn in cnodes:
                                    oo = owners.get(nn)
                                    if oo is not None:
                                        view[nn] = oo
                                if isinstance(cobj, dict):
                                    cobj["node_owner"] = view
                                else:
                                    setattr(cobj, "node_owner", view)
                    except Exception:
                        pass

                    if not float(_get(jr2, "start_ts", 0.0) or 0.0):
                        jr2.start_ts = float(time.time())
                    jr2.launch_inflight = False
                    jr2.launching_since_ts = 0.0

                    jr2.cluster_id = str(cluster_id)
                    jr2.nodes = list(node_names)
                    jr2.g_cur = int(g_use)
                    try:
                        jr2.g_alloc = int(g_use)
                    except Exception:
                        pass
                    try:
                        jr2.world_size = int(g_use)
                    except Exception:
                        pass
                    try:
                        jr2.actual_g = int(g_use)
                    except Exception:
                        pass
                    try:
                        jr2.metrics_cluster_id = str(cluster_id)
                    except Exception:
                        pass

                    started_key = str(snap.get("started_emit_key") or "")
                    prev_key = str(_get(jr2, "started_event_key", "") or "")
                    if started_key and prev_key != started_key:
                        jr2.started_event_key = started_key
                        emit_started = True
                except Exception:
                    pass

        if rl and emit_started:
            try:
                rl.job_event(
                    event="started",
                    job_id=str(snap["jid"]),
                    cluster=str(cluster_id),
                    world_size=int(snap["g_use"]),
                    note=(
                        f"started_on={cluster_id}, g_alloc={snap['g_use']}, nodes={snap['nodes']}, "
                        f"backfill={bool(snap['is_backfill'])}, forced_hol_backfill={bool(snap.get('forced_hol_backfill'))}, "
                        f"run_id={snap['run_id']}, attempt={snap['attempt']} idempotent_mode={idempotent_mode or ''}"
                    ),
                    metadata={
                        "model": snap["model"],
                        "dataset": snap["dataset"],
                        "nodes": list(snap["nodes"]),
                        "run_id": snap["run_id"],
                        "attempt": int(snap["attempt"]),
                        "resume_from_checkpoint": snap["resume"],
                        "gang_need": int(snap["gang_need"]),
                        "is_backfill": bool(snap["is_backfill"]),
                        "forced_hol_backfill": bool(snap.get("forced_hol_backfill")),
                        "hol_pin": str(snap.get("hol_pin") or ""),
                        "hol_jid": str(snap.get("hol_jid") or ""),
                        "g_target": int(snap.get("g_target") or 0),
                        "g_alloc": int(snap.get("g_alloc") or snap["g_use"]),
                        "idempotent_mode": idempotent_mode,
                    },
                    ts=float(time.time()),
                    stream="job",
                )
            except Exception:
                pass

        return {
            "ok": True,
            "job_id": jid,
            "cluster_id": cluster_id,
            "g": int(g_use),
            "nodes": list(node_names),
            "run_id": str(snap["run_id"]),
            "attempt": int(snap["attempt"]),
            "detail": out,
            "started_emitted": bool(emit_started),
            "idempotent_mode": idempotent_mode,
        }

    # 실패: inflight 내림 + dispatch 롤백 + STARTING -> QUEUED 수렴(재시도 가능)
    try:
        with STATE_LOCK:
            release_nodes_for_job_locked(job_id=str(jid), cluster_id=str(cluster_id))
    except Exception:
        pass

    with STATE_LOCK:
        st3 = get_global_state()
        jr3 = (getattr(st3, "jobs", {}) or {}).get(jid)
        if jr3 is not None:
            try:
                jr3.status = "QUEUED"
                _clear_requeue_victim_tags_locked(jr3)
            except Exception:
                pass

            # ✅ 일반 실패에서도 dispatch/inflight를 반드시 내려야 다음 tick에서 재시도 됨
            try:
                jr3.launch_dispatched = False
                jr3.launch_dispatched_ts = 0.0
            except Exception:
                pass
            try:
                jr3.launch_inflight = False
                jr3.launching_since_ts = 0.0
            except Exception:
                pass

            # ✅ QUEUED로 내리면 SSOT 큐에 재삽입(증발 방지)
            try:
                jid_s = str(jid)
                gq = getattr(st3, "global_queue", None)
                if isinstance(gq, list) and jid_s not in [str(x) for x in gq]:
                    gq.append(jid_s)
            except Exception:
                pass

            try:
                home = _as_str(
                    _get(jr3, "home_cluster_id", "") or _get(jr3, "admitted_cluster_id", "") or ""
                ).strip()
                if home:
                    cq = (getattr(st3, "cluster_queues", {}) or {}).get(str(home))
                    if cq is not None:
                        items = list(getattr(cq, "_jobs", []) or [])
                        if jid_s not in [
                            (str(it.get("job_id") or it.get("id") or "") if isinstance(it, dict)
                             else (str(getattr(it, "job_id")) if hasattr(it, "job_id") else str(it)))
                            for it in items
                        ]:
                            items.append({"job_id": jid_s, "g_req": max(1, int(_get(jr3, "g_target", 1) or 1))})
                            setattr(cq, "_jobs", items)
            except Exception:
                pass

    if rl:
        try:
            rl.job_event(
                event="launch_failed",
                job_id=str(snap["jid"]),
                cluster=str(cluster_id),
                world_size=int(snap["g_use"]),
                note=(
                    f"launch_failed reason={(out.get('reason') if isinstance(out, dict) else None) or 'executor_launch_failed'} "
                    f"run_id={snap['run_id']} attempt={snap['attempt']} idempotent_mode={idempotent_mode or ''}"
                ),
                metadata={
                    "detail": (out if isinstance(out, dict) else {"detail": out}),
                    "nodes": list(snap["nodes"]),
                    "is_backfill": bool(snap["is_backfill"]),
                    "forced_hol_backfill": bool(snap.get("forced_hol_backfill")),
                    "hol_pin": str(snap.get("hol_pin") or ""),
                    "hol_jid": str(snap.get("hol_jid") or ""),
                    "idempotent_hint": bool(idempotent_hint),
                    "idempotent_mode": idempotent_mode,
                },
                ts=float(time.time()),
            )
        except Exception:
            pass

    return {
        "ok": False,
        "reason": (out.get("reason") if isinstance(out, dict) else None) or "executor_launch_failed",
        "detail": out,
        "run_id": str(snap["run_id"]),
        "attempt": int(snap["attempt"]),
        "cluster_id": cluster_id,
        "nodes": list(node_names),
        "g": int(g_use),
        "idempotent_mode": idempotent_mode,
    }

def _remove_job_from_all_cluster_queues_locked(
    cluster_queues: Dict[str, Any],
    job_id: str,
) -> int:
    job_id = str(job_id)
    removed = 0

    for _, q in (cluster_queues or {}).items():
        if q is None:
            continue

        # Case A: custom queue object API
        # - try q.remove(job_id) if exists
        if hasattr(q, "remove") and callable(getattr(q, "remove")):
            try:
                # 기대: remove가 bool 또는 count를 반환할 수도 있음
                r = q.remove(job_id)
                if isinstance(r, bool):
                    removed += 1 if r else 0
                elif isinstance(r, int):
                    removed += r
                else:
                    # 반환값 불명확하면, 내부에서 제거됐다고 가정하지 말고 아래 fallback도 수행
                    pass
                continue
            except Exception:
                # remove 실패 시 fallback으로 계속 진행
                pass

        # Case B: internal list `_jobs`
        try:
            lst = list(getattr(q, "_jobs", []) or [])
        except Exception:
            lst = []

        if not lst:
            continue

        # normalize to str for safe compare
        before = len(lst)
        lst2 = [x for x in lst if str(x) != job_id]
        after = len(lst2)

        if after != before:
            removed += (before - after)
            try:
                setattr(q, "_jobs", lst2)
            except Exception:
                # setattr 실패해도 여기서 throw하지 않음 (caller가 전체 스케줄러를 죽이면 더 최악)
                pass

    return removed

def _cleanup_preempt_inflight_locked(state: Any, now_ts: float) -> None:
    jobs = getattr(state, "jobs", {}) or {}

    cluster_block = getattr(state, "cluster_launch_blocked_until", None)
    node_drain = getattr(state, "node_drain_until", None)

    if not isinstance(cluster_block, dict):
        cluster_block = {}
        try:
            state.cluster_launch_blocked_until = cluster_block
        except Exception:
            pass
    if not isinstance(node_drain, dict):
        node_drain = {}
        try:
            state.node_drain_until = node_drain
        except Exception:
            pass

    try:
        hard_to = float(PREEMPT_INFLIGHT_TIMEOUT_SEC)
    except Exception:
        hard_to = 60.0

    # 너무 짧으면 위험, 너무 길면 stuck이 길어짐 (60 권장)
    try:
        force_release_default = float(getattr(state, "FORCE_RELEASE_SEC", 60.0) or 60.0)
    except Exception:
        force_release_default = 60.0
    if force_release_default < 30.0:
        force_release_default = 30.0

    owners = getattr(state, "node_owner", None)
    if not isinstance(owners, dict):
        owners = {}
        try:
            state.node_owner = owners
        except Exception:
            pass

    def _clear_hol_backfill_tags(jr: Any) -> None:
        # hol_backfill / backfill 태그는 “임시 실행” 표식이므로, PREEMPT 수렴 시 반드시 제거
        try:
            jr.is_hol_backfill = False
        except Exception:
            pass
        try:
            jr.is_backfill = False
        except Exception:
            pass
        try:
            jr.backfill_deadline_ts = 0.0
        except Exception:
            pass

        for k in ("hol_backfill_for", "hol_backfill_pin"):
            try:
                setattr(jr, k, None)
            except Exception:
                pass
        try:
            jr.hol_backfill_since_ts = 0.0
        except Exception:
            pass

        # queue_kind가 BACKFILL/HOL_BACKFILL로 고정돼 있으면 복구
        try:
            qk = str(getattr(jr, "queue_kind", "") or "")
            if qk.upper() in ("HOL_BACKFILL", "BACKFILL"):
                setattr(jr, "queue_kind", None)
        except Exception:
            pass

    def _clear_pending_meta(jr: Any) -> None:
        # pending_*는 preempt 시 스냅샷/재큐잉 컨텍스트라서, 수렴 후 남으면 SSOT 오염
        for k in (
            "pending_requeue_kind",
            "pending_prev_queue_kind",
            "pending_prev_queue_cluster_id",
            "pending_prev_nodes",
            "pending_checkpoint",
            "pending_drain_until",
            "preempt_reason",
        ):
            try:
                setattr(jr, k, None)
            except Exception:
                pass

    for jid, jr in list(jobs.items()):
        jid = str(jid)
        try:
            if not bool(getattr(jr, "preempt_inflight", False)):
                continue

            # 기준 시각
            t0 = float(getattr(jr, "preempting_since_ts", 0.0) or 0.0)
            if t0 <= 0.0:
                t0 = float(getattr(jr, "last_preempt_try_ts", 0.0) or 0.0)
            if t0 <= 0.0:
                # 기준 시각이 없으면 지금으로 박고 대기
                try:
                    jr.preempting_since_ts = float(now_ts)
                    jr.last_preempt_try_ts = float(now_ts)
                    jr.status = "PREEMPTING"
                    jr.last_preempt_error = "missing_preempt_ts_seeded"
                except Exception:
                    pass
                continue

            # ✅ nodes snapshot 필수
            nodes = list(getattr(jr, "pending_prev_nodes", None) or [])
            nodes = [str(n) for n in nodes if str(n)]
            if not nodes:
                # ACK만 기다리다 영원히 못 푸는게 싫으면 deadline만 설정
                try:
                    if float(getattr(jr, "force_release_deadline_ts", 0.0) or 0.0) <= 0.0:
                        jr.force_release_deadline_ts = float(t0 + hard_to + force_release_default)
                    jr.status = "PREEMPTING"
                    jr.last_preempt_error = "no_pending_prev_nodes_skip_force_release_wait_ack"
                    jr.last_preempt_try_ts = float(now_ts)
                except Exception:
                    pass
                continue

            # drain gate: node 단위 drain을 최우선 신뢰
            drain_until = 0.0
            for n in nodes:
                try:
                    drain_until = max(drain_until, float(node_drain.get(str(n), 0.0) or 0.0))
                except Exception:
                    pass

            # cluster block은 참고만 (cluster_id 없을 수 있음)
            try:
                cid = str(getattr(jr, "cluster_id", "") or "").strip()
            except Exception:
                cid = ""
            blocked_until = 0.0
            if cid:
                try:
                    blocked_until = float(cluster_block.get(cid, 0.0) or 0.0)
                except Exception:
                    blocked_until = 0.0

            gate = max(drain_until, blocked_until, t0 + float(PREEMPT_DRAIN_SEC_DEFAULT))
            hard_timeout = t0 + hard_to

            # stop 요청이 실제로 나갔는지(안 나갔으면 더 보수적)
            try:
                issued = bool(getattr(jr, "preempt_issued", False))
            except Exception:
                issued = False

            # 1) gate 이전 & hard_timeout 이전이면 대기
            if now_ts < gate and now_ts < hard_timeout:
                continue

            # 2) deadline 계산/갱신 (이미 값이 있으면 존중)
            ddl = float(getattr(jr, "force_release_deadline_ts", 0.0) or 0.0)
            if ddl <= 0.0:
                extra = force_release_default * (2.0 if not issued else 1.0)
                ddl = float(max(gate, hard_timeout) + extra)
                try:
                    jr.force_release_deadline_ts = float(ddl)
                except Exception:
                    pass

            # 3) ddl 전이면 상태만 유지
            if now_ts < ddl:
                try:
                    jr.status = "PREEMPTING"
                    jr.last_preempt_error = "preempt_inflight_waiting_ack_or_force_release"
                    jr.last_preempt_try_ts = float(now_ts)
                except Exception:
                    pass
                continue

            # ✅ 4) 최후수단 force_release (nodes=None 금지)
            before_owned = [n for n in nodes if str(owners.get(n) or "") == jid]

            released = 0
            try:
                released = int(_ss_release_nodes_for_job_locked(state, job_id=jid, nodes=list(nodes)) or 0)
            except Exception:
                released = 0

            after_owned = [n for n in nodes if str(owners.get(n) or "") == jid]

            # ⚠️ owner가 남아있으면 QUEUED로 바꾸면 안 됨.
            if after_owned:
                try:
                    jr.status = "PREEMPTING"
                    jr.preempt_inflight = True
                    jr.last_preempt_try_ts = float(now_ts)
                    jr.last_preempt_error = (
                        f"force_release_attempted_but_owner_still_present "
                        f"released={released} still_owned={after_owned}"
                    )
                    jr.force_release_deadline_ts = float(now_ts + 3.0)  # 루프 폭주 방지
                except Exception:
                    pass
                continue

            # ✅ 5) 상태 수렴 (owner가 없어졌을 때만)
            # 여기서 hol_backfill/pending 메타를 반드시 정리해야 SSOT 오염이 안 남음
            try:
                jr.preempt_inflight = False
                jr.preempting_since_ts = 0.0
                jr.last_preempt_try_ts = float(now_ts)
                jr.preempt_issued = True  # “어쨌든 수렴시켰다” 표시

                jr.cluster_id = None
                jr.nodes = []
                jr.g_cur = 0
                jr.status = "QUEUED"
                _clear_requeue_victim_tags_locked(jr)


                jr.launch_inflight = False
                jr.launching_since_ts = 0.0

                jr.launch_cooldown_until_ts = float(now_ts + 2.0)
                jr.last_preempt_error = (
                    f"force_release_deadline_elapsed "
                    f"before_owned={len(before_owned)} released={released} after_owned=0"
                )
                jr.force_release_deadline_ts = 0.0
            except Exception:
                pass

            # ✅ hol_backfill/backfill 태그/pendings 정리(SSOT 오염 방지)
            try:
                if bool(getattr(jr, "pending_clear_backfill_flags", False)):
                    # requeue 수렴 시 플래그 기반으로 확실히 끄고, 마커도 제거
                    _clear_hol_backfill_tags(jr)
                    try:
                        jr.pending_clear_backfill_flags = False
                    except Exception:
                        pass
                else:
                    # 마커가 없어도 안전하게 한 번은 클리어 (idempotent)
                    _clear_hol_backfill_tags(jr)
            except Exception:
                pass

            try:
                _clear_pending_meta(jr)
            except Exception:
                pass

            # ✅ 6) 큐 단일화 수렴
            try:
                purge_job_from_all_queues_locked(state, jid, purge_global=False, purge_cluster=False, purge_home=True)
            except Exception:
                pass
            try:
                _global_queue_insert_sorted_locked(state, jid)
            except Exception:
                pass
            try:
                home = str(getattr(jr, "home_cluster_id", "") or getattr(jr, "admitted_cluster_id", "") or "").strip()
                if home:
                    ensure_job_single_queue_locked(state, jid, home)
            except Exception:
                pass

        except Exception:
            continue

def _feed_cluster_queues_from_home_mobile_locked(state: Any) -> None:
    import time

    # -----------------------------
    # helpers
    # -----------------------------
    def _as_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _norm_jid(x: Any) -> str:
        return _as_str(x).strip().lower()

    def _qitem_job_id(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return _as_str(x)
        if isinstance(x, dict):
            return _as_str(x.get("job_id") or x.get("id") or "")
        if hasattr(x, "job_id"):
            return _as_str(getattr(x, "job_id"))
        return _as_str(x)

    def _queue_list(q: Any) -> List[Any]:
        if q is None:
            return []
        if isinstance(q, list):
            return list(q)
        try:
            return list(getattr(q, "_jobs", []) or [])
        except Exception:
            return []

    def _set_queue_list(q: Any, items: List[Any]) -> None:
        if q is None:
            return
        if isinstance(q, list):
            q[:] = list(items)
            return
        try:
            setattr(q, "_jobs", list(items))
        except Exception:
            pass

    def _clusterq_normalize_items(items: List[Any]) -> List[Dict[str, Any]]:
        """
        clusterQ는 dict만 유지:
        - str -> {"job_id": str, "g_req": 1}
        - dict -> job_id/g_req 정규화
        - 기타 -> skip
        """
        out: List[Dict[str, Any]] = []
        for it in (items or []):
            if isinstance(it, dict):
                jid = _as_str(it.get("job_id") or it.get("id") or "").strip()
                if not jid:
                    continue
                try:
                    g = int(it.get("g_req") or 1)
                except Exception:
                    g = 1
                out.append({"job_id": jid, "g_req": int(max(1, g))})
                continue

            if isinstance(it, str):
                jid = _as_str(it).strip()
                if not jid:
                    continue
                out.append({"job_id": jid, "g_req": 1})
                continue

            if hasattr(it, "job_id"):
                jid = _as_str(getattr(it, "job_id", "") or "").strip()
                if not jid:
                    continue
                g = 1
                try:
                    g = int(getattr(it, "g_req", 1) or 1)
                except Exception:
                    g = 1
                out.append({"job_id": jid, "g_req": int(max(1, g))})
                continue
        return out

    def _queue_contains_clusterq(q: Any, jid: str) -> bool:
        sjid = _as_str(jid)
        sjid_norm = _norm_jid(sjid)
        items = _queue_list(q)
        for it in items:
            if _norm_jid(_qitem_job_id(it)) == sjid_norm:
                return True
        return False

    def _append_to_clusterq_dict_only(q: Any, jid: str, *, g_req_hint: int = 1) -> None:
        items_raw = _queue_list(q)
        items = _clusterq_normalize_items(items_raw)

        sjid = _as_str(jid)
        sjid_norm = _norm_jid(sjid)
        for it in items:
            if _norm_jid(it.get("job_id")) == sjid_norm:
                _set_queue_list(q, items)
                try:
                    setattr(q, "eta_dirty", True)
                except Exception:
                    pass
                return

        try:
            g = int(g_req_hint or 1)
        except Exception:
            g = 1

        items.append({"job_id": sjid, "g_req": int(max(1, g))})
        _set_queue_list(q, items)
        try:
            setattr(q, "eta_dirty", True)
        except Exception:
            pass

    def _is_gang4(jr: Any) -> bool:
        try:
            return int(_job_gang_required_g(jr) or 0) == 4
        except Exception:
            return False

    def _job_status(jr: Any) -> str:
        try:
            return _as_str(getattr(jr, "status", "") or "").upper()
        except Exception:
            return ""

    def _job_enqueue_ts(jr: Any) -> float:
        if jr is None:
            return float(1e30)
        for k in ("enqueue_ts", "submit_ts"):
            try:
                v = getattr(jr, k, None)
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return float(1e30)

    def _job_preferred_cluster(jr: Any) -> Optional[str]:
        try:
            pref = _as_str(getattr(jr, "preferred_cluster_id", "") or "").strip()
        except Exception:
            pref = ""
        return pref or None

    def _job_pin_constraint_cluster(jr: Any) -> Optional[str]:
        """
        hard pin constraints:
        - user_pinned_cluster_id
        - checkpoint_cluster_id
        - checkpoint_local_only/resume_from_checkpoint -> home/preferred/admitted로 수렴
        """
        try:
            pinned_user = _as_str(getattr(jr, "user_pinned_cluster_id", "") or "").strip()
        except Exception:
            pinned_user = ""
        try:
            ckpt_cid = _as_str(getattr(jr, "checkpoint_cluster_id", "") or "").strip()
        except Exception:
            ckpt_cid = ""
        try:
            ckpt_local = bool(getattr(jr, "checkpoint_local_only", False))
        except Exception:
            ckpt_local = False
        try:
            if getattr(jr, "resume_from_checkpoint", None):
                ckpt_local = True
        except Exception:
            pass

        if pinned_user:
            return pinned_user
        if ckpt_cid:
            return ckpt_cid
        if ckpt_local:
            try:
                pref = _as_str(getattr(jr, "preferred_cluster_id", "") or "").strip()
            except Exception:
                pref = ""
            try:
                home = _as_str(getattr(jr, "home_cluster_id", "") or "").strip()
            except Exception:
                home = ""
            try:
                adm = _as_str(getattr(jr, "admitted_cluster_id", "") or "").strip()
            except Exception:
                adm = ""
            return pref or home or adm or None
        return None

    def _job_g_hint_for_cluster(jr: Any, cid: str) -> int:
        cid = _as_str(cid)
        try:
            mp = getattr(jr, "g_hint_by_cluster", None)
            if isinstance(mp, dict):
                v = mp.get(cid)
                if v is not None:
                    return int(max(1, int(v)))
        except Exception:
            pass

        for k in ("g_target_hint", "g_target"):
            try:
                v = getattr(jr, k, None)
                if v is not None and int(v) > 0:
                    return int(max(1, int(v)))
            except Exception:
                pass
        return 1

    def _job_requeue_target_cluster(jr: Any) -> Optional[str]:
        """
        ✅ reclaim-preempt victim이 돌아왔을 때, pin cluster로 되돌려 보내기 위한 SSOT.
        cluster_tick에서 victim에 jr.requeue_target_cluster_id 를 심어둔 전제.
        """
        try:
            v = _as_str(getattr(jr, "requeue_target_cluster_id", "") or "").strip()
        except Exception:
            v = ""
        return v or None

    # -----------------------------
    # snapshot
    # -----------------------------
    jobs = getattr(state, "jobs", {}) or {}
    clusters = getattr(state, "clusters", {}) or {}

    hqs = getattr(state, "home_cluster_queues", None) or {}
    qs = getattr(state, "cluster_queues", None) or {}
    if not isinstance(hqs, dict):
        hqs = {}
        try:
            state.home_cluster_queues = hqs
        except Exception:
            pass
    if not isinstance(qs, dict):
        qs = {}
        try:
            state.cluster_queues = qs
        except Exception:
            pass

    # cluster-level blocked/drain (node drain은 _free_nodes_in_cluster에서 처리)
    drains = getattr(state, "drain_until_by_cluster", {}) or {}
    cb = getattr(state, "cluster_launch_blocked_until", {}) or {}
    if not isinstance(drains, dict):
        drains = {}
    if not isinstance(cb, dict):
        cb = {}

    now_ts = float(time.time())

    def _cluster_blocked(cid: str) -> bool:
        cid = _as_str(cid)
        try:
            bu = float(cb.get(cid, 0.0) or 0.0)
        except Exception:
            bu = 0.0
        if bu > 0.0 and now_ts < bu:
            return True

        try:
            du = float(drains.get(cid, 0.0) or 0.0)
        except Exception:
            du = 0.0
        if du > 0.0 and now_ts < du:
            return True

        return False

    # -----------------------------
    # 1) HoL(gang4) pin target(backfill 목적) 계산
    # -----------------------------
    hol_pin_target: Optional[str] = None
    try:
        hol_jid = _as_str(getattr(state, "hol_job_id", "") or "").strip()
        hol_pin = _as_str(getattr(state, "hol_pin_cluster", "") or "").strip()
    except Exception:
        hol_jid, hol_pin = "", ""

    pin_free = 0
    if hol_jid and hol_pin and hol_pin in clusters and (not _cluster_blocked(hol_pin)):
        try:
            pin_free = int(len(_free_nodes_in_cluster(state, hol_pin) or []))
        except Exception:
            pin_free = 0

        # backfill은 free 1~3일 때만 의미 있음
        if 1 <= pin_free <= 3:
            hol_pin_target = hol_pin

    # -----------------------------
    # 2) feeder 실행
    # -----------------------------
    MAX_FEED_PER_TICK = int(getattr(state, "FEED_MAX_PER_TICK", 8) or 8)
    moved = 0

    # HOME 큐 순회
    for home_cid, hq in list(hqs.items()):
        if moved >= MAX_FEED_PER_TICK:
            break

        home_cid = _as_str(home_cid)
        hitems = _queue_list(hq)
        if not hitems:
            continue

        # HOME은 str만 유지(혼입 정리)
        hitems_norm: List[str] = []
        for it in hitems:
            jid = _qitem_job_id(it)
            jid = _as_str(jid).strip()
            if jid:
                hitems_norm.append(jid)

        # 정렬: requeue_front 우선 -> enqueue_ts -> jid
        def _sort_key(jid: str) -> Tuple[int, float, str]:
            jr = jobs.get(str(jid))
            try:
                prio = 0 if bool(getattr(jr, "requeue_front", False)) else 1
            except Exception:
                prio = 1
            try:
                enq = float(_job_enqueue_ts(jr))
            except Exception:
                enq = float(1e30)
            return (prio, enq, _norm_jid(jid))

        try:
            hitems_sorted = sorted(list(hitems_norm), key=_sort_key)
        except Exception:
            hitems_sorted = list(hitems_norm)

        moved_jids: Set[str] = set()

        for jid in hitems_sorted:
            if moved >= MAX_FEED_PER_TICK:
                break
            if not jid or jid in moved_jids:
                continue

            jr = jobs.get(str(jid))
            if jr is None:
                continue

            # QUEUED만 이동
            if _job_status(jr) != "QUEUED":
                continue

            # gang4는 feeder가 절대 이동시키지 않음
            if _is_gang4(jr):
                continue

            # inflight/preempt 중이면 이동 금지
            try:
                if bool(getattr(jr, "launch_inflight", False)) or bool(getattr(jr, "preempt_inflight", False)):
                    continue
            except Exception:
                pass

            # blocked/cooldown이면 지금 옮겨도 실행이 안 됨(불필요한 흔들림 방지)
            try:
                if now_ts < float(getattr(jr, "blocked_until", 0.0) or 0.0):
                    continue
                if now_ts < float(getattr(jr, "launch_cooldown_until_ts", 0.0) or 0.0):
                    continue
            except Exception:
                pass

            # -----------------------------
            # target cluster 결정 (우선순위가 중요)
            # -----------------------------
            target_cid: Optional[str] = None

            # (우선 0) ✅ reclaim-preempt victim의 재큐잉 목적지
            rq = _job_requeue_target_cluster(jr)
            if rq and rq in clusters and (not _cluster_blocked(rq)):
                target_cid = _as_str(rq)

            # (우선 1) hard constraint(pin/ckpt)
            if target_cid is None:
                must_cid = _job_pin_constraint_cluster(jr)
                if must_cid:
                    target_cid = _as_str(must_cid)

            # (우선 2) hol pin target (backfill 목적) — 단, hard constraint를 깨면 안 됨
            if target_cid is None and hol_pin_target:
                must_cid2 = _job_pin_constraint_cluster(jr)
                if (must_cid2 is None) or (_as_str(must_cid2) == _as_str(hol_pin_target)):
                    target_cid = _as_str(hol_pin_target)

            # (우선 3) preferred cluster — blocked 아니고 free>=1 이어야
            if target_cid is None:
                pref_cid = _job_preferred_cluster(jr)
                if pref_cid and pref_cid in clusters and (not _cluster_blocked(pref_cid)):
                    try:
                        free2 = int(len(_free_nodes_in_cluster(state, pref_cid) or []))
                    except Exception:
                        free2 = 0
                    if free2 >= 1:
                        target_cid = _as_str(pref_cid)

            # (우선 4) 그 외: free>=1 & blocked 아닌 곳 선택 (home 우선)
            if target_cid is None:
                cand: List[Tuple[int, int, str]] = []
                for cid2 in list(clusters.keys()):
                    cid2 = _as_str(cid2)
                    if _cluster_blocked(cid2):
                        continue
                    try:
                        free2 = int(len(_free_nodes_in_cluster(state, cid2) or []))
                    except Exception:
                        free2 = 0
                    if free2 >= 1:
                        # home 우선 + free 많은 곳 우선
                        cand.append((1 if cid2 == home_cid else 0, free2, cid2))
                if cand:
                    cand.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    target_cid = _as_str(cand[0][2])

            if not target_cid:
                continue
            if target_cid not in qs:
                continue

            cq = qs.get(str(target_cid))
            if cq is None:
                continue

            # 중복 방지(정규화 비교)
            if _queue_contains_clusterq(cq, str(jid)):
                moved_jids.add(str(jid))
                # clusterQ에 올라간 순간 requeue_front 특혜 종료
                try:
                    if bool(getattr(jr, "requeue_front", False)):
                        jr.requeue_front = False
                except Exception:
                    pass
                # requeue_target도 사용했으면 해제(무한 강제수렴 방지)
                try:
                    if _job_requeue_target_cluster(jr):
                        jr.requeue_target_cluster_id = None
                except Exception:
                    pass
                continue

            # ✅ clusterQ item에는 g_req 대신 "cluster별 g_hint"를 기록
            g_hint = _job_g_hint_for_cluster(jr, str(target_cid))
            _append_to_clusterq_dict_only(cq, str(jid), g_req_hint=int(g_hint))

            moved += 1
            moved_jids.add(str(jid))

            # clusterQ로 옮겼으면 requeue_front 특혜 종료
            try:
                if bool(getattr(jr, "requeue_front", False)):
                    jr.requeue_front = False
            except Exception:
                pass

            # ✅ requeue_target은 “1회 수렴” 용도. 옮겼으면 제거.
            try:
                if _job_requeue_target_cluster(jr):
                    jr.requeue_target_cluster_id = None
            except Exception:
                pass

        # HOME 큐 재구성(str-only 유지)
        if moved_jids:
            new_hitems: List[str] = []
            for jid0 in hitems_norm:
                if jid0 and str(jid0) in moved_jids:
                    continue
                new_hitems.append(str(jid0))
            _set_queue_list(hq, new_hitems)
        else:
            _set_queue_list(hq, list(hitems_norm))

        try:
            setattr(hq, "eta_dirty", True)
        except Exception:
            pass

def _cluster_tick_once(cluster_id: str) -> None:
    import time

    cluster_id = str(cluster_id)
    now_ts = float(time.time())

    # --- logger ---
    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    # -----------------------------
    # helpers
    # -----------------------------
    def _as_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _norm_jid(jid: Any) -> str:
        return _as_str(jid).strip().lower()

    def _qitem_job_id(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return _as_str(x)
        if isinstance(x, dict):
            return _as_str(x.get("job_id") or x.get("id") or "")
        if hasattr(x, "job_id"):
            return _as_str(getattr(x, "job_id"))
        return _as_str(x)

    def _qitem_g_req(x: Any, default: int = 1) -> int:
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

    def _status(jr: Any) -> str:
        try:
            return _as_str(getattr(jr, "status", "") or "").upper()
        except Exception:
            return ""

    def _is_queued(jr: Any) -> bool:
        return _status(jr) == "QUEUED"

    def _is_running(jr: Any) -> bool:
        return _status(jr) == "RUNNING"

    def _is_gang4(jr: Any) -> bool:
        try:
            return int(_job_gang_required_g(jr) or 0) == 4
        except Exception:
            return False

    def _job_launchable_now(jr: Any, t: float) -> bool:
        try:
            if t < float(getattr(jr, "blocked_until", 0.0) or 0.0):
                return False
            if t < float(getattr(jr, "launch_cooldown_until_ts", 0.0) or 0.0):
                return False
            if bool(getattr(jr, "launch_inflight", False)):
                return False
            if bool(getattr(jr, "preempt_inflight", False)):
                return False
        except Exception:
            pass
        return True

    def _iter_cluster_q_items(q: Any) -> List[Any]:
        if q is None:
            return []

        # 1) list 자체
        if isinstance(q, list):
            return list(q)

        # 2) 흔한 내부 필드들
        for name in ("_jobs", "jobs", "items", "_items", "q", "_q", "deque", "_deque", "data", "_data"):
            try:
                v = getattr(q, name, None)
            except Exception:
                v = None
            if isinstance(v, list) and v:
                return list(v)
            # deque 같은 경우
            try:
                if v is not None and hasattr(v, "__iter__"):
                    vv = list(v)
                    if vv:
                        return vv
            except Exception:
                pass

        # 3) 메서드 기반(있으면)
        for m in ("to_list", "dump", "snapshot", "get_items", "get_jobs"):
            try:
                fn = getattr(q, m, None)
                if callable(fn):
                    vv = list(fn() or [])
                    if vv:
                        return vv
            except Exception:
                pass

        return []

    def _clear_backfill_tags_if_queued(jr: Any) -> None:
        if not jr:
            return
        if _status(jr) != "QUEUED":
            return
        for k, v in (
            ("is_hol_backfill", False),
            ("is_backfill", False),
            ("backfill_deadline_ts", 0.0),
            ("hol_backfill_for", None),
            ("hol_backfill_pin", None),
            ("hol_backfill_since_ts", 0.0),
        ):
            try:
                setattr(jr, k, v)
            except Exception:
                pass
        try:
            qk = _as_str(getattr(jr, "queue_kind", "") or "")
            if qk.upper() in ("HOL_BACKFILL", "BACKFILL"):
                jr.queue_kind = None
        except Exception:
            pass

    def _job_enqueue_ts(jr: Any) -> float:
        for k in ("enqueue_ts", "submit_ts"):
            try:
                v = getattr(jr, k, None)
                if v is not None:
                    return float(v)
            except Exception:
                pass
        return float(now_ts)

    def _job_max_g_cap(jr: Any) -> Optional[int]:
        for k in ("max_g", "max_world_size", "g_target", "world_size"):
            try:
                v = getattr(jr, k, None)
                if v is None:
                    continue
                iv = int(v)
                if iv > 0:
                    return iv
            except Exception:
                continue
        return None

    def _job_g_hint_for_cluster(jr: Any, cid: str, fallback: int = 1) -> int:
        cid = str(cid)
        try:
            mp = getattr(jr, "g_hint_by_cluster", None)
            if isinstance(mp, dict):
                v = mp.get(cid)
                if v is not None:
                    return int(max(1, int(v)))
        except Exception:
            pass
        for k in ("g_target_hint", "g_target"):
            try:
                v = getattr(jr, k, None)
                if v is not None and int(v) > 0:
                    return int(max(1, int(v)))
            except Exception:
                pass
        return int(max(1, fallback))

    # -----------------------------
    # preempt emitter (프로젝트 함수명 차이 흡수)
    # -----------------------------
    def _emit_preempt(victim_job_id: str, reason: str) -> bool:
        # ✅ tick에서는 어떤 이유로든 preempt를 절대 발행하지 않는다.
        #    (completion handler에서만 /preempt_job 허용)
        try:
            if rl:
                rl.queue_event(
                    event="preempt_suppressed_in_cluster_tick",
                    job_id=str(victim_job_id),
                    queue_len=-1,
                    qlen_clusterq=-1,
                    note=f"cluster={cluster_id} reason={_as_str(reason)}",
                    ts=float(time.time()),
                )
        except Exception:
            pass
        return False

    def _cluster_node_owner_map(st0: Any, cid: str) -> Dict[str, Any]:
        clusters0 = getattr(st0, "clusters", {}) or {}
        cr = clusters0.get(str(cid))
        if cr is None:
            return {}
        try:
            no = getattr(cr, "node_owner", None)
        except Exception:
            no = None
        if isinstance(no, dict):
            return no
        if isinstance(cr, dict) and isinstance(cr.get("node_owner"), dict):
            return cr.get("node_owner") or {}
        return {}

    # -----------------------------
    # cluster blocked + reclaim lock
    # -----------------------------
    def _get_cluster_blocked_until(st0: Any, cid: str) -> float:
        try:
            cb = getattr(st0, "cluster_launch_blocked_until", {}) or {}
            if isinstance(cb, dict):
                return float(cb.get(str(cid), 0.0) or 0.0)
        except Exception:
            pass
        return 0.0

    def _set_cluster_blocked_until(st0: Any, cid: str, until_ts: float) -> None:
        try:
            cb = getattr(st0, "cluster_launch_blocked_until", None)
            if not isinstance(cb, dict):
                cb = {}
                setattr(st0, "cluster_launch_blocked_until", cb)
            prev = float(cb.get(str(cid), 0.0) or 0.0)
            cb[str(cid)] = float(max(prev, float(until_ts)))
        except Exception:
            pass

    def _reclaim_lock_active(st0: Any) -> bool:
        try:
            act = bool(getattr(st0, "hol_reclaim_active", False))
        except Exception:
            act = False
        if not act:
            return False
        try:
            until = float(getattr(st0, "hol_reclaim_until_ts", 0.0) or 0.0)
        except Exception:
            until = 0.0
        if until > 0.0 and float(time.time()) > until:
            try:
                st0.hol_reclaim_active = False
            except Exception:
                pass
            return False
        return True

    def _set_reclaim_lock(st0: Any, *, hol_norm: str, pin: str, now_ts0: float, hold_sec: float = 30.0) -> None:
        try:
            st0.hol_reclaim_active = True
            st0.hol_reclaim_hol_job_id = str(hol_norm)
            st0.hol_reclaim_pin = str(pin)
            st0.hol_reclaim_until_ts = float(now_ts0 + float(hold_sec))
        except Exception:
            pass

    def _clear_reclaim_lock(st0: Any) -> None:
        for k, v in (
            ("hol_reclaim_active", False),
            ("hol_reclaim_hol_job_id", None),
            ("hol_reclaim_pin", None),
            ("hol_reclaim_until_ts", 0.0),
        ):
            try:
                setattr(st0, k, v)
            except Exception:
                pass

    # -----------------------------
    # ✅ NEW: HOL 선택을 "하나로 고정" (여러 gang4 대기 버그 해결)
    # -----------------------------
    def _iter_global_queue_ids(st0: Any) -> List[str]:
        for name in ("global_queue", "queue", "jobs_queue", "pending_queue"):
            q = getattr(st0, name, None)
            if q is None:
                continue

            # 1) list 형태
            if isinstance(q, list):
                out = []
                for it in q:
                    jid = _qitem_job_id(it)
                    if jid:
                        out.append(str(jid))
                if out:
                    return out

            # 2) 큐 객체 형태: _jobs 보유
            try:
                items = list(getattr(q, "_jobs", None) or [])
            except Exception:
                items = []
            if items:
                out = []
                for it in items:
                    jid = _qitem_job_id(it)
                    if jid:
                        out.append(str(jid))
                if out:
                    return out

        return []

    def _job_pin_cluster(jr: Any) -> str:
        for k in ("pinned_cluster", "pinned_cluster_id", "admitted_cluster_id", "home_cluster_id"):
            try:
                v = str(getattr(jr, k, "") or "").strip()
                if v:
                    return v
            except Exception:
                pass
        return ""

    def _select_hol_gang4_for_pin_locked(st0: Any, jobs0: dict, pin_cid: str) -> Tuple[Optional[str], Optional[str], int]:
        pin_cid = str(pin_cid)
        try:
            pin_free = int(len(_free_nodes_in_cluster(st0, pin_cid) or []))
        except Exception:
            pin_free = 0

        gq_ids = _iter_global_queue_ids(st0)

        # (1) global queue order 기반
        for jid in gq_ids:
            jr = jobs0.get(str(jid))
            if jr is None:
                continue
            if (not _is_queued(jr)) or (not _is_gang4(jr)):
                continue

            pin = _job_pin_cluster(jr)
            if pin and pin != pin_cid:
                continue

            # 체크포인트 로컬 강제도 같은 방식으로만 보조
            try:
                ckpt_cid = str(getattr(jr, "checkpoint_cluster_id", "") or "").strip()
            except Exception:
                ckpt_cid = ""
            if ckpt_cid and ckpt_cid != pin_cid:
                continue

            return (str(jid), _norm_jid(jid), pin_free)

        # (2) fallback: enqueue_ts 오래된 queued gang4 (pin 일치만)
        cand: List[Tuple[float, str]] = []
        for jid_key, jr in (jobs0 or {}).items():
            if jr is None:
                continue
            if (not _is_queued(jr)) or (not _is_gang4(jr)):
                continue
            pin = _job_pin_cluster(jr)
            if pin and pin != pin_cid:
                continue
            cand.append((_job_enqueue_ts(jr), str(jid_key)))

        if cand:
            cand.sort(key=lambda x: (x[0], x[1]))
            hol_real = cand[0][1]
            return (hol_real, _norm_jid(hol_real), pin_free)

        return (None, None, pin_free)

    # -----------------------------
    # 계획 변수(락 밖에서 launch 수행)
    # -----------------------------
    job_to_start: Optional[str] = None
    start_nodes: Optional[List[str]] = None
    g_use: int = 0
    chosen_idx: int = -1
    chosen_item: Any = None
    chosen_item_greq: int = 1

    hol_to_launch: Optional[str] = None
    hol_nodes: Optional[List[str]] = None

    # logging snapshot
    snap_note = ""
    snap_tag_note = ""
    snap_tagged = False
    snap_deq = False
    snap_deq_job_id = ""
    snap_qlen_after = -1

    # --------------------------------------------
    # 0) LOCK: 상태 결정
    # --------------------------------------------
    with STATE_LOCK:
        st = get_global_state()

        # cluster blocked
        bu = _get_cluster_blocked_until(st, cluster_id)
        if now_ts < bu:
            return

        clusters = getattr(st, "clusters", {}) or {}
        jobs = getattr(st, "jobs", {}) or {}
        if cluster_id not in clusters:
            return

        free_nodes = list(_free_nodes_in_cluster(st, cluster_id) or [])
        free_n = int(len(free_nodes))

        # -----------------------------
        # ✅ selected HOL (pin=cluster_id) 고정
        # -----------------------------
        selected_hol_real: Optional[str] = None
        selected_hol_norm: Optional[str] = None
        selected_pin_free: int = 0

        # reclaim lock이 있으면 그 HOL을 SSOT로 사용
        if _reclaim_lock_active(st):
            try:
                lock_pin = str(getattr(st, "hol_reclaim_pin", "") or "")
                lock_hol_norm = _norm_jid(getattr(st, "hol_reclaim_hol_job_id", "") or "")
            except Exception:
                lock_pin, lock_hol_norm = "", ""
            if lock_pin == cluster_id and lock_hol_norm:
                # lock_hol_norm에 해당하는 real job_id 역조회
                real = None
                for jid_key in (jobs or {}).keys():
                    if _norm_jid(jid_key) == lock_hol_norm:
                        real = str(jid_key)
                        break
                if real is None:
                    _clear_reclaim_lock(st)
                    return
                selected_hol_real = real
                selected_hol_norm = lock_hol_norm
                try:
                    selected_pin_free = int(len(_free_nodes_in_cluster(st, cluster_id) or []))
                except Exception:
                    selected_pin_free = free_n
            else:
                # reclaim lock이 다른 핀에 걸려있으면 이 클러스터는 일반 tick만
                pass

        # reclaim lock이 없으면 새로 HOL을 선택
        if selected_hol_real is None:
            hol_real, hol_norm, pin_free = _select_hol_gang4_for_pin_locked(st, jobs, cluster_id)
            selected_hol_real = hol_real
            selected_hol_norm = hol_norm
            selected_pin_free = int(pin_free)

        hol_waiting_here = bool(selected_hol_real and selected_hol_norm)

        # HOL 유효성 재검증(queued+gang4)
        if hol_waiting_here:
            hol_jr = jobs.get(str(selected_hol_real))
            if hol_jr is None or (not _is_queued(hol_jr)) or (not _is_gang4(hol_jr)):
                hol_waiting_here = False
                selected_hol_real = None
                selected_hol_norm = None

        # ✅ reclaim lock이 pin인데 HOL이 유효하지 않으면 lock 해제
        if _reclaim_lock_active(st):
            try:
                lock_pin = str(getattr(st, "hol_reclaim_pin", "") or "")
            except Exception:
                lock_pin = ""
            if lock_pin == cluster_id and (not hol_waiting_here):
                _clear_reclaim_lock(st)

        # (A) reclaim lock 중이면: 이 클러스터가 pin이면 gang4만 재시도
        if _reclaim_lock_active(st):
            try:
                lock_pin = str(getattr(st, "hol_reclaim_pin", "") or "")
                lock_hol_norm = _norm_jid(getattr(st, "hol_reclaim_hol_job_id", "") or "")
            except Exception:
                lock_pin, lock_hol_norm = "", ""

            if lock_pin == cluster_id and lock_hol_norm:
                if free_n >= 4:
                    hol_to_launch = str(selected_hol_real or "")
                    hol_nodes = list(free_nodes[:4])
                else:
                    # ✅ 너무 빡빡하게 tick 돌면 다른 루트가 victim 재런치해서 4-free 창이 안 생김
                    _set_cluster_blocked_until(st, cluster_id, float(now_ts + 0.2))
                    return

        # (B) ✅ HOL 대기 중이고 free<4이면 "기다리기 + backfill만" (NO PREEMPT)
        #     - 기존 RUNNING job은 절대 건드리지 않는다.
        #     - free가 4가 되면 HoL을 즉시 launch한다.
        #     - free가 1~3이면 아래 (C) 로직에서 backfill job을 띄워 idle만 줄인다.
        if hol_to_launch is None and hol_waiting_here and selected_hol_real and selected_hol_norm:
            if free_n >= 4:
                hol_to_launch = str(selected_hol_real)
                hol_nodes = list(free_nodes[:4])
            else:
                # ✅ 여기서 preempt/reclaim 금지
                # (C)에서 free_n(1~3)로 backfill만 수행하게 둔다.
                if rl:
                    try:
                        rl.queue_event(
                            event="hol_wait_no_preempt",
                            job_id=str(selected_hol_real),
                            queue_len=-1,
                            qlen_clusterq=-1,
                            note=f"pin={cluster_id} free={free_n} hol={selected_hol_norm} action=wait_backfill_only",
                            ts=float(time.time()),
                        )
                    except Exception:
                        pass
                # return 하지 말고 계속 진행해서 (C)에서 backfill launch 기회를 준다.
                pass


        # ✅ reclaim_lock 활성화 + 이 클러스터가 pin이면: gang4 말고는 절대 launch하지 않음
        if _reclaim_lock_active(st):
            try:
                lock_pin = str(getattr(st, "hol_reclaim_pin", "") or "")
            except Exception:
                lock_pin = ""
            if lock_pin == cluster_id and hol_to_launch is None:
                # ✅ reclaim 중에는 "HOL launch만" 허용, 그 외 launch는 차단
                _set_cluster_blocked_until(st, cluster_id, float(now_ts + 0.2))
                return

        # (C) HOL이 없거나, HOL인데 reclaim 못하면: clusterQ에서 job 하나 launch
        if hol_to_launch is None:
            q = (getattr(st, "cluster_queues", {}) or {}).get(cluster_id)
            if q is None:
                return

            try:
                qitems = _iter_cluster_q_items(q)
                if rl:
                    rl.queue_event(
                        event="clusterq_snapshot",
                        job_id="-",
                        queue_len=-1,
                        qlen_clusterq=len(qitems),
                        note=f"cluster={cluster_id} free={free_n} qtype={type(q).__name__} first3={[ _qitem_job_id(x) for x in qitems[:3] ]}",
                        ts=float(time.time()),
                    )
            except Exception:
                qitems = []
            if not qitems:
                return
            if free_n <= 0:
                return

            # ✅ NEW: clusterQ 안에 gang4가 있으면, free>=4일 때 최우선으로 바로 launch
            if free_n >= 4:
                gang4_idx = -1
                gang4_jid = None
                gang4_enq = 1e30
                for i, item in enumerate(qitems):
                    jid = _qitem_job_id(item)
                    if not jid:
                        continue
                    jr = jobs.get(str(jid))
                    if jr is None or (not _is_queued(jr)) or (not _is_gang4(jr)):
                        continue
                    if not _job_launchable_now(jr, now_ts):
                        continue
                    enq = _job_enqueue_ts(jr)
                    if enq < gang4_enq:
                        gang4_enq = enq
                        gang4_idx = i
                        gang4_jid = str(jid)

                if gang4_jid:
                    # clusterQ에서 제거 (중복 launch 방지)
                    try:
                        qitems.pop(gang4_idx)
                        setattr(q, "_jobs", qitems)
                    except Exception:
                        pass

                    hol_to_launch = str(gang4_jid)
                    hol_nodes = list(free_nodes[:4])
                    # reclaim lock도 걸어두면 더 안정적(다른 launch가 끼는 것 방지)
                    _set_reclaim_lock(st, hol_norm=_norm_jid(gang4_jid), pin=cluster_id, now_ts0=float(now_ts), hold_sec=30.0)
                    _set_cluster_blocked_until(st, cluster_id, float(now_ts + 0.1))


            # ✅ backfill 조건: selected HOL이 있고, free 1~3일 때만
            is_backfill_run = bool(hol_waiting_here and selected_hol_norm and 1 <= free_n <= 3)

            # 후보 선택: (backfill이면 작은 g_hint 우선, 아니면 FIFO)
            candidates: List[Tuple[int, float, int, str, int]] = []
            for i, item in enumerate(qitems):
                jid = _qitem_job_id(item)
                if not jid:
                    continue
                jr = jobs.get(str(jid))
                if jr is None:
                    continue
                if not _is_queued(jr):
                    continue

                try:
                    _clear_backfill_tags_if_queued(jr)
                except Exception:
                    pass

                if not _is_queued(jr):
                    continue
                if _is_gang4(jr):
                    continue
                if not _job_launchable_now(jr, now_ts):
                    continue

                item_greq = _qitem_g_req(item, 1)
                enq_ts = _job_enqueue_ts(jr)

                score = int(item_greq) if is_backfill_run else 0
                candidates.append((int(score), float(enq_ts), int(i), str(jid), int(item_greq)))

            if not candidates:
                return

            candidates.sort(key=lambda x: (x[0], x[1], x[2]))
            _, _, chosen_idx, chosen_jid, chosen_item_greq = candidates[0]

            job_to_start = str(chosen_jid)
            jr_sel = jobs.get(job_to_start)

            cap = _job_max_g_cap(jr_sel)
            g_hint = _job_g_hint_for_cluster(jr_sel, cluster_id, fallback=int(chosen_item_greq))

            if cap is None:
                g_use = int(max(1, min(int(free_n), int(g_hint))))
            else:
                g_use = int(max(1, min(int(free_n), int(g_hint), int(cap))))

            start_nodes = list(free_nodes[:g_use])

            # pop from clusterQ
            try:
                chosen_item = qitems[chosen_idx]
                qitems.pop(chosen_idx)
                setattr(q, "_jobs", qitems)
                snap_qlen_after = len(qitems)
            except Exception:
                return

            # reserve nodes
            try:
                reserve_nodes(job_id=job_to_start, cluster_id=cluster_id, nodes=list(start_nodes))
            except Exception:
                # restore
                try:
                    qitems2 = list(getattr(q, "_jobs", []) or [])
                    ins = min(max(0, int(chosen_idx)), len(qitems2))
                    qitems2.insert(
                        ins,
                        chosen_item if chosen_item is not None else {"job_id": str(job_to_start), "g_req": int(max(1, chosen_item_greq))}
                    )
                    setattr(q, "_jobs", qitems2)
                except Exception:
                    pass
                return

            # ✅ tag backfill victim (반드시 selected_hol_norm으로 고정!)
            jr2 = jobs.get(job_to_start)
            if jr2 is not None:
                try:
                    if is_backfill_run:
                        jr2.is_backfill = True
                        jr2.is_hol_backfill = True
                        jr2.hol_backfill_for = str(selected_hol_norm)   # ✅ SSOT
                        jr2.hol_backfill_pin = cluster_id
                        jr2.hol_backfill_since_ts = float(now_ts)
                        try:
                            ddl0 = float(getattr(jr2, "backfill_deadline_ts", 0.0) or 0.0)
                        except Exception:
                            ddl0 = 0.0
                        try:
                            jr2.queue_kind = "HOL_BACKFILL"
                        except Exception:
                            pass
                        snap_tagged = True
                        snap_tag_note = f"tagged victim={job_to_start} hol={selected_hol_norm} pin={cluster_id} g={g_use}"
                    else:
                        jr2.is_backfill = False
                        jr2.is_hol_backfill = False
                        jr2.backfill_deadline_ts = 0.0
                        jr2.hol_backfill_for = None
                        jr2.hol_backfill_pin = None
                        jr2.hol_backfill_since_ts = 0.0
                except Exception:
                    pass

            snap_deq = True
            snap_deq_job_id = str(job_to_start)
            snap_note = (
                f"cluster={cluster_id} kind=CLUSTERQ backfill={is_backfill_run} "
                f"g={g_use} g_hint={g_hint} nodes={list(start_nodes)} "
                f"hol_waiting={hol_waiting_here} hol={selected_hol_norm or ''} free={free_n}"
            )

    # -----------------------------
    # LOG (락 밖)
    # -----------------------------
    if rl and snap_deq:
        try:
            rl.queue_event(
                event="dequeue_clusterq",
                job_id=str(snap_deq_job_id),
                queue_len=-1,
                qlen_clusterq=int(snap_qlen_after),
                note=str(snap_note),
                ts=float(time.time()),
            )
        except Exception:
            pass

    if rl and snap_tagged:
        try:
            rl.queue_event(
                event="hol_backfill_tagged",
                job_id=str(snap_deq_job_id),
                queue_len=-1,
                qlen_clusterq=int(snap_qlen_after),
                note=str(snap_tag_note),
                ts=float(time.time()),
            )
        except Exception:
            pass

    # 1) gang4 launch (락 밖)
    if hol_to_launch and hol_nodes and len(hol_nodes) >= 4:
        try:
            stx = get_global_state()
            jobsx = getattr(stx, "jobs", {}) or {}
            hol_job = jobsx.get(str(hol_to_launch))
        except Exception:
            hol_job = None

        if hol_job is None:
            with STATE_LOCK:
                try:
                    st4 = get_global_state()
                    _set_cluster_blocked_until(st4, cluster_id, float(time.time() + 0.2))
                except Exception:
                    pass
            return

        # ✅ HOL launch 직전에도 한 번 더 가드(다른 루트/플래그 꼬임 방지)
        if not _job_launchable_now(hol_job, float(time.time())):
            with STATE_LOCK:
                st4 = get_global_state()
                # reclaim lock은 유지하고, 짧게만 block
                _set_cluster_blocked_until(st4, cluster_id, float(time.time() + 0.2))
            return

        # ✅ NEW: HOL이 STARTING stuck(최근 dispatched 스킵) 상태면 재런치 가능하게 플래그를 깨준다
        # - "STARTING인데 launch 로그가 안 뜨는" 상황의 주범을 제거
        try:
            if _status(hol_job) == "STARTING":
                hol_job.launch_dispatched = False
                hol_job.launch_dispatched_ts = 0.0
                hol_job.launch_inflight = False
                hol_job.launching_since_ts = 0.0
        except Exception:
            pass

        out = _start_job_on_cluster(
            job=hol_job,
            cluster_id=cluster_id,
            node_names=list(hol_nodes[:4]),
            g_use=4,
            is_backfill=False,
        )

        ok = bool((out or {}).get("ok"))
        if ok:
            with STATE_LOCK:
                try:
                    st4 = get_global_state()
                    _clear_reclaim_lock(st4)
                except Exception:
                    pass
            if rl:
                try:
                    rl.queue_event(
                        event="hol_gang4_launch_ok",
                        job_id=str(hol_to_launch),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"pin={cluster_id} nodes={list(hol_nodes[:4])}",
                        ts=float(time.time()),
                    )
                except Exception:
                    pass
            return

        reason = _as_str((out or {}).get("reason", "") or "").lower()
        if reason == "node_draining":
            drain_until = 0.0
            for k in ("drain_until", "drain_until_ts", "blocked_until", "blocked_until_ts"):
                try:
                    v = (out or {}).get(k, None)
                    if v is not None:
                        drain_until = float(v)
                        break
                except Exception:
                    continue
            if drain_until <= 0.0:
                drain_until = float(time.time() + 1.0)
            with STATE_LOCK:
                st4 = get_global_state()
                # reclaim_lock 유지 (selected HOL 유지)
                try:
                    hol_norm = _norm_jid(getattr(st4, "hol_reclaim_hol_job_id", "") or "") or _norm_jid(hol_to_launch)
                except Exception:
                    hol_norm = _norm_jid(hol_to_launch)
                _set_reclaim_lock(st4, hol_norm=str(hol_norm), pin=cluster_id, now_ts0=float(time.time()), hold_sec=30.0)
                _set_cluster_blocked_until(st4, cluster_id, float(drain_until + 0.05))
            if rl:
                try:
                    rl.queue_event(
                        event="hol_gang4_launch_deferred_draining",
                        job_id=str(hol_to_launch),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"pin={cluster_id} retry_at={drain_until:.3f} resp={_safe_json(out or {})}",
                        ts=float(time.time()),
                    )
                except Exception:
                    pass
            return

        with STATE_LOCK:
            st4 = get_global_state()
            try:
                hol_norm = _norm_jid(getattr(st4, "hol_reclaim_hol_job_id", "") or "") or _norm_jid(hol_to_launch)
            except Exception:
                hol_norm = _norm_jid(hol_to_launch)
            _set_reclaim_lock(st4, hol_norm=str(hol_norm), pin=cluster_id, now_ts0=float(time.time()), hold_sec=30.0)
            _set_cluster_blocked_until(st4, cluster_id, float(time.time() + 1.0))
        if rl:
            try:
                rl.queue_event(
                    event="hol_gang4_launch_failed_retry",
                    job_id=str(hol_to_launch),
                    queue_len=-1,
                    qlen_clusterq=-1,
                    note=f"pin={cluster_id} reason={reason or 'unknown'} resp={_safe_json(out or {})}",
                    ts=float(time.time()),
                )
            except Exception:
                pass
        return

    # 2) normal/backfill launch (락 밖)
    if not (job_to_start and start_nodes and g_use > 0):
        return

    try:
        stx = get_global_state()
        jobx = (getattr(stx, "jobs", {}) or {}).get(str(job_to_start))
    except Exception:
        jobx = None

    is_backfill_reason = False
    try:
        if jobx is not None:
            is_backfill_reason = bool(getattr(jobx, "is_backfill", False) or getattr(jobx, "is_hol_backfill", False))
    except Exception:
        is_backfill_reason = False

    out = _start_job_on_cluster(
        job=jobx,
        cluster_id=cluster_id,
        node_names=list(start_nodes),
        g_use=int(g_use),
        is_backfill=bool(is_backfill_reason),
    )

    ok = bool((out or {}).get("ok"))
    if ok:
        if rl:
            try:
                rl.queue_event(
                    event="launch_ok_clusterq",
                    job_id=str(job_to_start),
                    queue_len=-1,
                    qlen_clusterq=-1,
                    note=f"cluster={cluster_id} backfill={is_backfill_reason} g={g_use} nodes={list(start_nodes)} ok=True",
                    ts=float(time.time()),
                )
            except Exception:
                pass
        return

    if rl:
        try:
            rl.queue_event(
                event="launch_failed_clusterq",
                job_id=str(job_to_start),
                queue_len=-1,
                qlen_clusterq=-1,
                note=f"cluster={cluster_id} backfill={is_backfill_reason} g={g_use} nodes={list(start_nodes)} resp={_safe_json(out or {})}",
                ts=float(time.time()),
            )
        except Exception:
            pass

    with STATE_LOCK:
        st3 = get_global_state()
        jobs3 = getattr(st3, "jobs", {}) or {}
        q3 = (getattr(st3, "cluster_queues", {}) or {}).get(cluster_id)

        # ✅ launch 실패 시: 방금 reserve한 노드는 "현재 tick이 가진 start_nodes" 기준으로 확실히 반납해야 함
        try:
            release_nodes_for_job(st3, str(job_to_start), list(start_nodes))
        except Exception:
            try:
                _ss_release_nodes_for_job_locked(st3, job_id=str(job_to_start), nodes=list(start_nodes))
            except Exception:
                pass


        job3 = jobs3.get(job_to_start)
        if job3 is not None:
            try:
                job3.launch_inflight = False
                job3.launching_since_ts = 0.0
                job3.cluster_id = None
                job3.nodes = []
                job3.g_cur = 0
                job3.status = "QUEUED"
                reason = _as_str((out or {}).get("reason", "") or "").lower()
                if reason != "node_draining":
                    job3.launch_cooldown_until_ts = float(time.time() + 1.0)
                _clear_requeue_victim_tags_locked(job3)

            except Exception:
                pass

        try:
            if q3 is not None:
                items3 = list(getattr(q3, "_jobs", []) or [])
                ins = min(max(0, int(chosen_idx)), len(items3))
                if chosen_item is None:
                    chosen_item = {"job_id": str(job_to_start), "g_req": int(max(1, chosen_item_greq))}
                items3.insert(ins, chosen_item)
                setattr(q3, "_jobs", items3)
        except Exception:
            pass

def _backfill_tick_global_safe() -> None:
    import time

    now_ts = float(time.time())

    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    # -------------------------
    # helpers
    # -------------------------
    def _as_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _norm_jid(x: Any) -> str:
        return _as_str(x).strip().lower()

    def _status(jr: Any) -> str:
        try:
            return _as_str(getattr(jr, "status", "") or "").upper()
        except Exception:
            return ""

    def _is_running(jr: Any) -> bool:
        return _status(jr) == "RUNNING"

    def _is_queued(jr: Any) -> bool:
        return _status(jr) == "QUEUED"

    def _is_gang4(jr: Any) -> bool:
        try:
            return int(_job_gang_required_g(jr) or 0) == 4
        except Exception:
            return False

    def _safe_to_preempt(jr: Any) -> bool:
        if jr is None:
            return False
        try:
            if bool(getattr(jr, "launch_inflight", False)):
                return False
            if bool(getattr(jr, "preempt_inflight", False)):
                return False
        except Exception:
            pass
        return True

    def _active_meta_for_job(st0: Any, jid: str) -> Dict[str, Any]:
        jid = str(jid)
        try:
            aj = getattr(st0, "active_jobs", None)
            if isinstance(aj, dict):
                v = aj.get(jid)
                if isinstance(v, dict):
                    return dict(v)
        except Exception:
            pass
        try:
            aj2 = globals().get("ACTIVE_JOBS")
            if isinstance(aj2, dict):
                v2 = aj2.get(jid)
                if isinstance(v2, dict):
                    return dict(v2)
        except Exception:
            pass
        return {}

    # ✅ NOTE:
    # - 당신 요구사항 때문에 tick에서는 preempt 금지.
    # - completion handler에서 preempt를 실제로 쏠 때는
    #   reason prefix를 "HOL_GANG_PREEMPT_AFTER_COMPLETION" 로 쓰는 걸 권장.
    # - 아래 guard는 "HOL_GANG_PREEMPT" (즉 tick/preempt 즉시발사)만 막고,
    #   AFTER_COMPLETION은 막지 않도록 해둠.
    ALLOW_HOL_RECLAIM_PREEMPT = bool(globals().get("ALLOW_HOL_RECLAIM_PREEMPT", False))

    def _emit_preempt(victim_job_id: str, reason: str) -> bool:
        # -------------------------------------------------
        # ✅ HARD GUARD:
        #   tick에서 즉시 preempt(HOL_GANG_PREEMPT) 쓰는 걸 전면 금지.
        #   단, completion 이후에 쏘는 AFTER_COMPLETION은 허용.
        # -------------------------------------------------
        try:
            rs = str(reason or "").upper()
            # "HOL_GANG_PREEMPT" 뒤에 공백이 있든 없든, prefix든 상관없이 차단
            if ("HOL_GANG_PREEMPT" in rs) and ("AFTER_COMPLETION" not in rs) and (not ALLOW_HOL_RECLAIM_PREEMPT):
                try:
                    logger.info(f"[PREEMPT_SUPPRESSED] victim={victim_job_id} reason={reason}")
                except Exception:
                    pass
                return False
        except Exception:
            pass

        last_err = None
        for name in ("preempt_job", "request_preempt_job", "_preempt_job", "issue_preempt", "request_preempt"):
            fn = globals().get(name)
            if not callable(fn):
                continue
            try:
                try:
                    out = fn(victim_job_id, reason=reason)
                except TypeError:
                    out = fn(victim_job_id, reason)

                if isinstance(out, dict):
                    return bool(out.get("ok", True))
                return True

            except Exception as e:
                last_err = e
                continue

        try:
            if last_err is not None:
                logger.warning(f"[PREEMPT_FAILED] victim={victim_job_id} reason={reason} last_err={last_err!r}")
            else:
                logger.warning(f"[PREEMPT_FAILED] victim={victim_job_id} reason={reason} no_callable_fn")
        except Exception:
            pass

        return False

    def _set_cluster_blocked_until(st0: Any, cid: str, until_ts: float) -> None:
        try:
            cb = getattr(st0, "cluster_launch_blocked_until", None)
            if not isinstance(cb, dict):
                cb = {}
                setattr(st0, "cluster_launch_blocked_until", cb)
            prev = float(cb.get(str(cid), 0.0) or 0.0)
            cb[str(cid)] = float(max(prev, float(until_ts)))
        except Exception:
            pass

    def _victim_gcur(jr: Any) -> int:
        for k in ("g_cur", "world_size", "g", "gpus"):
            try:
                v = getattr(jr, k, None)
                if v is None:
                    continue
                iv = int(v)
                if iv > 0:
                    return iv
            except Exception:
                continue
        try:
            nodes = getattr(jr, "nodes", None)
            if isinstance(nodes, list) and len(nodes) > 0:
                return int(len(nodes))
        except Exception:
            pass
        return 1

    def _cluster_node_owner_map(st0: Any, cid: str) -> Dict[str, Any]:
        clusters0 = getattr(st0, "clusters", {}) or {}
        cr = clusters0.get(str(cid))
        if cr is None:
            return {}
        try:
            no = getattr(cr, "node_owner", None)
        except Exception:
            no = None
        if isinstance(no, dict):
            return no
        if isinstance(cr, dict) and isinstance(cr.get("node_owner"), dict):
            return cr.get("node_owner") or {}
        return {}

    def _job_runs_in_cluster_by_owner(st0: Any, cid: str, jid: str) -> bool:
        no = _cluster_node_owner_map(st0, cid)
        nj = _norm_jid(jid)
        for _, owner in (no or {}).items():
            if _norm_jid(owner) == nj:
                return True
        return False

    def _find_waiting_gang4_in_pin_locked(st0: Any, pin: str) -> Optional[str]:
        jobs0 = getattr(st0, "jobs", {}) or {}
        gq = getattr(st0, "global_queue", None)
        if not isinstance(gq, list):
            for jid_key, jr in (jobs0 or {}).items():
                if jr is None:
                    continue
                if not _is_queued(jr):
                    continue
                if not _is_gang4(jr):
                    continue
                p = (
                    str(getattr(jr, "pinned_cluster", "") or "").strip()
                    or str(getattr(jr, "pinned_cluster_id", "") or "").strip()
                    or str(getattr(jr, "admitted_cluster_id", "") or "").strip()
                    or str(getattr(jr, "home_cluster_id", "") or "").strip()
                )
                if p and p == str(pin):
                    return str(getattr(jr, "job_id", "") or jid_key)
            return None

        def _job_id(x: Any) -> str:
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

        for item in gq:
            jid = _job_id(item)
            if not jid:
                continue
            jr = jobs0.get(jid)
            if jr is None:
                continue
            if not _is_queued(jr):
                continue
            if not _is_gang4(jr):
                continue
            p = (
                str(getattr(jr, "pinned_cluster", "") or "").strip()
                or str(getattr(jr, "pinned_cluster_id", "") or "").strip()
                or str(getattr(jr, "admitted_cluster_id", "") or "").strip()
                or str(getattr(jr, "home_cluster_id", "") or "").strip()
            )
            if p and p == str(pin):
                return str(jid)

        return None

    def _is_hol_backfill_victim(st0: Any, pin: str, jid: str, jr: Any) -> bool:
        if jr is None or not _is_running(jr):
            return False
        if not _safe_to_preempt(jr):
            return False

        if not _job_runs_in_cluster_by_owner(st0, str(pin), str(jid)):
            try:
                if str(getattr(jr, "cluster_id", "") or "") != str(pin):
                    return False
            except Exception:
                return False

        meta = _active_meta_for_job(st0, jid)

        try:
            qk = _as_str(getattr(jr, "queue_kind", "") or "").strip().upper()
        except Exception:
            qk = ""
        mqk = _as_str(meta.get("queue_kind") or meta.get("queueKind") or "").strip().upper()
        mreason = _as_str(meta.get("reason") or "").strip().upper()

        try:
            is_hbf = bool(getattr(jr, "is_hol_backfill", False))
        except Exception:
            is_hbf = False
        try:
            is_bf = bool(getattr(jr, "is_backfill", False))
        except Exception:
            is_bf = False

        tagged_hol = bool(
            is_hbf
            or qk == "HOL_BACKFILL"
            or mqk == "HOL_BACKFILL"
            or ("HOL_BACKFILL" in mreason)
        )

        # pin 매칭(있으면 엄격)
        try:
            hpin = _as_str(getattr(jr, "hol_backfill_pin", "") or "").strip()
        except Exception:
            hpin = ""
        mhpin = _as_str(meta.get("hol_backfill_pin") or meta.get("holBackfillPin") or "").strip()
        eff_pin = hpin or mhpin
        if eff_pin and eff_pin != str(pin):
            return False

        if tagged_hol:
            return True

        # 태그 유실 구제: armed 상태일 때만 BACKFILL 허용
        try:
            reclaim_active = bool(getattr(st0, "hol_reclaim_active", False))
            reclaim_pin = _as_str(getattr(st0, "hol_reclaim_pin", "") or "").strip()
            reclaim_until = float(getattr(st0, "hol_reclaim_until_ts", 0.0) or 0.0)
        except Exception:
            reclaim_active, reclaim_pin, reclaim_until = False, "", 0.0

        if not (reclaim_active and reclaim_pin == str(pin) and now_ts <= reclaim_until):
            return False

        if is_bf or qk == "BACKFILL" or mqk == "BACKFILL" or ("BACKFILL" in mreason):
            return True

        return False

    # -------------------------
    # LOCK: SSOT 기반 판단/armed 설정
    # -------------------------
    with STATE_LOCK:
        st = get_global_state()
        clusters = getattr(st, "clusters", {}) or {}
        jobs = getattr(st, "jobs", {}) or {}

        # pin에서 기다리는 gang4 HOL 하나라도 있나?
        try:
            hol_jid, hol_pin, pin_free = _find_waiting_gang4_hol_locked(st)
        except Exception:
            hol_jid, hol_pin, pin_free = None, None, 0

        hol_pin = _as_str(hol_pin).strip()
        if not hol_pin or hol_pin not in clusters:
            # reclaim 락 정리
            try:
                if bool(getattr(st, "hol_reclaim_active", False)):
                    st.hol_reclaim_active = False
            except Exception:
                pass
            return

        # pin에 실제 gang4가 대기하는지 재확인(더 신뢰)
        hol_in_pin = _find_waiting_gang4_in_pin_locked(st, hol_pin)
        if not hol_in_pin:
            try:
                if bool(getattr(st, "hol_reclaim_active", False)):
                    st.hol_reclaim_active = False
            except Exception:
                pass
            return

        # pin free 최신 계산
        try:
            free_nodes_pin = list(_free_nodes_in_cluster(st, hol_pin) or [])
            free_n_pin = int(len(free_nodes_pin))
        except Exception:
            free_n_pin = int(pin_free or 0)

        # reclaim은 free=1~3에서만 "준비" (0이면 completion을 더 기다려야 하고, 4면 필요없음)
        if not (1 <= int(free_n_pin) <= 3):
            return

        # victim 수집
        victims: List[Tuple[str, int]] = []
        for jid_key, jr in (jobs or {}).items():
            if jr is None or not _is_running(jr):
                continue
            if not _safe_to_preempt(jr):
                continue
            if not _is_hol_backfill_victim(st, hol_pin, str(jid_key), jr):
                continue
            victims.append((str(jid_key), int(max(1, _victim_gcur(jr)))))

        if not victims:
            if rl:
                try:
                    rl.queue_event(
                        event="hol_reclaim_no_victim",
                        job_id=str(hol_in_pin),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"pin={hol_pin} free={free_n_pin} running={sum(1 for _,j in (jobs or {}).items() if j and _is_running(j))}",
                        ts=float(time.time()),
                    )
                except Exception:
                    pass
            return

        reclaimable = sum(g for _, g in victims)
        if int(free_n_pin) + int(reclaimable) < 4:
            # 충분히 못 만들면 건드리지 않음(안전)
            if rl:
                try:
                    rl.queue_event(
                        event="hol_reclaim_insufficient",
                        job_id=str(hol_in_pin),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"pin={hol_pin} free={free_n_pin} reclaimable={reclaimable} victims={len(victims)}",
                        ts=float(time.time()),
                    )
                except Exception:
                    pass
            return

        # 이미 armed 상태면, completion까지 기다린다 (tick에서 victims/need churn 방지)
        try:
            if bool(getattr(st, "hol_reclaim_active", False)):
                apin  = str(getattr(st, "hol_reclaim_pin", "") or "").strip()
                until = float(getattr(st, "hol_reclaim_until_ts", 0.0) or 0.0)
                if apin == str(hol_pin) and now_ts <= until:
                    return
        except Exception:
            pass

        need = int(max(0, 4 - int(free_n_pin)))
        if need <= 0:
            return

        # 횟수 최소화(큰 것부터)
        victims.sort(key=lambda x: int(x[1]), reverse=True)

        # ✅ 여기서 절대 preempt 하지 않는다. "armed"만 한다.
        reason = f"HOL_GANG_RECLAIM_PENDING hol={_as_str(hol_in_pin).upper()} pin={hol_pin}"

        try:
            st.hol_reclaim_active = True
            st.hol_reclaim_pin = str(hol_pin)
            st.hol_reclaim_until_ts = float(now_ts + 30.0)

            st.hol_reclaim_need = int(need)
            st.hol_reclaim_hol_job_id = str(hol_jid)

            # ⚠️ PREEMPT 아님. 반드시 PENDING
            st.hol_reclaim_reason = f"HOL_GANG_RECLAIM_PENDING hol={hol_jid} pin={hol_pin}"

            # victims는 primitive로만
            st.hol_reclaim_victims = [{"job_id": v} for v in victims]
        except Exception:
            pass

        # 이번 tick에서 backfill이 바로 다시 pin을 채우지 못하게 짧게 block
        _set_cluster_blocked_until(st, hol_pin, float(now_ts + 0.5))

        if rl:
            try:
                rl.queue_event(
                    event="hol_reclaim_armed_wait_completion",
                    job_id=str(hol_in_pin),
                    queue_len=-1,
                    qlen_clusterq=-1,
                    note=f"pin={hol_pin} free={free_n_pin} need={need} reclaimable={reclaimable} victims={victims[:5]}",
                    ts=float(time.time()),
                )
            except Exception:
                pass

        # ✅ 핵심: armed만 하고 종료
        return

def _schedule_once() -> None:
    now_ts = float(_now())

    # 0) STATE 정리 + feeder (LOCK)
    with STATE_LOCK:
        st = get_global_state()

        # inflight/timeout 수렴
        try:
            _cleanup_preempt_inflight_locked(st, now_ts=now_ts)
        except Exception:
            pass
        try:
            _cleanup_launch_inflight_locked(st, now_ts)
        except Exception:
            pass
        try:
            _cleanup_drains_locked(st, now_ts)
        except Exception:
            pass

        # terminal/중복 정리
        try:
            _global_queue_compact_locked(st)
        except Exception:
            pass

        # ✅ 핵심: HoL pin 우선으로 HOME→CLUSTERQ feed를 먼저 수행
        try:
            _feed_cluster_queues_from_home_mobile_locked(st)
        except Exception:
            logger.exception("[tick] feed_cluster_queues_from_home_mobile_locked failed")

        cluster_ids = list((getattr(st, "clusters", {}) or {}).keys())
        cluster_ids = [str(c) for c in cluster_ids]

    # 1) HoL dispatch (GLOBAL) — preempt 계획/발행 우선
    try:
        _global_tick_once()
    except Exception:
        logger.exception("[tick] global_tick_once failed")

    # 2) clusterQ backfill tick (각 클러스터) — 유휴 메우기
    for cid in cluster_ids:
        try:
            _cluster_tick_once(str(cid))
        except Exception:
            logger.exception("[tick] cluster_tick_once failed cid=%s", cid)

    # 3) rebalance (elastic resize / launch decision)
    try:
        st3 = get_global_state()
        rebalance_tick_global(state=st3, now_ts=now_ts, reason="periodic")
    except Exception:
        logger.exception("[tick] rebalance failed")

    try:
        st2 = get_global_state()
        _backfill_tick_global_safe(st2)
    except Exception:
        logger.exception("[tick] backfill failed")

    try:
        _emit_csp_metrics_tick()
    except Exception:
        logger.exception("[tick] emit_csp_metrics failed")

def schedule_tick_global() -> None:
    if not _TICK_RUN_LOCK.acquire(blocking=False):
        request_tick()
        return
    try:
        _schedule_once()
    finally:
        _TICK_RUN_LOCK.release()

def compute_avg_cap_norm_util() -> float:
    def _safe_int(x, default=0) -> int:
        try:
            v = int(x)
            return v
        except Exception:
            return default

    def _safe_float(x, default=0.0) -> float:
        try:
            v = float(x)
            if not math.isfinite(v):
                return default
            return v
        except Exception:
            return default

    st = get_global_state()

    with STATE_LOCK:
        clusters = getattr(st, "clusters", {}) or {}
        node_owner = getattr(st, "node_owner", {}) or {}

        if not clusters:
            return 0.0

        vals = []
        for cid, cr in clusters.items():
            # nodes
            try:
                nodes = list(getattr(cr, "nodes", []) or [])
            except Exception:
                # dict 형태 대응
                nodes = list((cr.get("nodes") or [])) if isinstance(cr, dict) else []

            # total_gpus
            total = 0
            if isinstance(cr, dict):
                total = _safe_int(cr.get("total_gpus"), 0)
            else:
                total = _safe_int(getattr(cr, "total_gpus", 0), 0)

            if total <= 0:
                total = len(nodes)

            if total <= 0:
                continue

            # used_gpus: node_owner 기반
            used = 0
            for n in nodes:
                o = node_owner.get(str(n))
                if o is not None and str(o) != "":
                    used += 1

            # clamp
            if used < 0:
                used = 0
            if used > total:
                used = total

            u = _safe_float(used / total, 0.0)
            if u < 0.0:
                u = 0.0
            if u > 1.0:
                u = 1.0

            vals.append(u)

        if not vals:
            return 0.0

        return float(sum(vals) / len(vals))

def _queue_enter_locked(jr: Any, now_ts: float) -> None:
    t = getattr(jr, "last_queue_enter_ts", None)
    if t is None:
        try:
            jr.queue_enter_ts = float(now_ts)
        except Exception:
            pass
        try:
            jr.last_queue_enter_ts = float(now_ts)
        except Exception:
            pass

def enqueue_job_to_home_cluster_queue(job_id: str, home_cluster_id: str, *, why: str = "submit") -> Dict[str, Any]:
    jid = str(job_id)
    hc = str(home_cluster_id)

    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    now_ts = float(time.time())

    def _bump_seq_locked(jr: Any) -> int:
        try:
            s = int(getattr(jr, "event_seq", 0) or 0) + 1
            jr.event_seq = int(s)
            return int(s)
        except Exception:
            return -1

    def _snap_prev_locked(jr: Any) -> tuple[str, str]:
        pk = str(getattr(jr, "queue_kind", None) or "NONE")
        pc = str(
            getattr(jr, "queue_cluster_id", None)
            or getattr(jr, "home_cluster_id", None)
            or getattr(jr, "admitted_cluster_id", None)
            or "UNKNOWN"
        )
        return pk, pc

    def _stable_submit_key(jr: Any) -> float:
        """
        ✅ 절대 변하지 않는 순서키.
        - submit_ts(권장) → queue_seq(있으면 더 좋음) → fallback(now)
        """
        # queue_seq가 있다면 그걸 1순위로 써도 됨(숫자 안정)
        try:
            qs = getattr(jr, "queue_seq", None)
            if qs is not None:
                return float(qs)
        except Exception:
            pass

        try:
            ts0 = getattr(jr, "submit_ts", None)
            if ts0 is None:
                ts0 = getattr(jr, "submitted_ts", None)
            if ts0 is not None:
                return float(ts0)
        except Exception:
            pass

        # 진짜 최후 fallback: now (하지만 이러면 순서가 흔들릴 수 있음)
        return float(now_ts)

    def _list_insert_sorted_unique_by_key(lst: List[str], key_fn, item: str) -> None:
        """
        lst: job_id 리스트
        - 이미 있으면 아무 것도 안 함(=자리 유지)
        - 없으면 key 기준으로 적절한 위치에 삽입
        """
        if not isinstance(lst, list):
            return
        if item in lst:
            return

        # bisect 구현(외부 import 없이)
        k_item = None
        try:
            k_item = float(key_fn(item))
        except Exception:
            k_item = None

        if k_item is None:
            lst.append(item)
            return

        lo, hi = 0, len(lst)
        while lo < hi:
            mid = (lo + hi) // 2
            try:
                k_mid = float(key_fn(lst[mid]))
            except Exception:
                # 중간 원소 키가 깨지면 그냥 뒤로 밀림
                lo = mid + 1
                continue
            if k_mid <= k_item:
                lo = mid + 1
            else:
                hi = mid
        lst.insert(lo, item)

    with STATE_LOCK:
        st = get_global_state()
        jobs = getattr(st, "jobs", {}) or {}
        jr = jobs.get(jid)
        if jr is None:
            return {"ok": False, "reason": "job_not_found", "job_id": jid}

        seq = _bump_seq_locked(jr)
        prev_kind, prev_qcid = _snap_prev_locked(jr)

        # HOME 클러스터 id를 상태에 박아둠(없으면)
        try:
            if not getattr(jr, "home_cluster_id", None):
                jr.home_cluster_id = hc
        except Exception:
            pass

        # ✅ submit 순서키를 고정해서 사용(재큐잉 시 now_ts로 흔들리지 않게)
        submit_key = _stable_submit_key(jr)

        try:
            pass
        except Exception:
            pass

        # (A) cluster_queues 제거
        try:
            qs = getattr(st, "cluster_queues", None) or {}
            if isinstance(qs, dict):
                for _, q in qs.items():
                    if isinstance(q, list):
                        q[:] = [str(x) for x in q if str(x) != jid]
                        continue
                    removed = False
                    try:
                        if hasattr(q, "remove"):
                            q.remove(jid)
                            removed = True
                    except Exception:
                        pass
                    if not removed:
                        try:
                            jobs_list = list(getattr(q, "_jobs", []) or [])
                            new_jobs = []
                            for x in jobs_list:
                                xjid = str(getattr(x, "job_id", x) or "")
                                if xjid != jid:
                                    new_jobs.append(x)
                            setattr(q, "_jobs", new_jobs)
                        except Exception:
                            pass
                    try:
                        q.eta_dirty = True
                    except Exception:
                        pass
        except Exception:
            pass

        # (B) home_cluster_queues 제거(모든 home 큐에서 제거 후, hc에 재삽입)
        try:
            hqmap = getattr(st, "home_cluster_queues", None)
            if hqmap is None:
                st.home_cluster_queues = {}
                hqmap = st.home_cluster_queues
            if isinstance(hqmap, dict):
                for c0, q0 in list(hqmap.items()):
                    if isinstance(q0, list):
                        q0[:] = [str(x) for x in q0 if str(x) != jid]
        except Exception:
            pass

        # 2) global_queue SSOT: ✅ 이미 있으면 “그 자리 유지”
        try:
            gq = getattr(st, "global_queue", None)
            if gq is None or not isinstance(gq, list):
                st.global_queue = []
                gq = st.global_queue

            # global_queue 내부 key_fn은 “해당 job의 submit_key”로
            def _key_of_jobid(xjid: str) -> float:
                jj = jobs.get(str(xjid))
                if jj is None:
                    return float("inf")
                return _stable_submit_key(jj)

            # 이미 있으면 그대로(=원래 자리 유지)
            if jid not in gq:
                _list_insert_sorted_unique_by_key(gq, _key_of_jobid, jid)
        except Exception:
            pass

        # 3) home queue insert: ✅ submit 순서대로 삽입
        try:
            hqmap = getattr(st, "home_cluster_queues", None)
            if hqmap is None:
                st.home_cluster_queues = {}
                hqmap = st.home_cluster_queues
            if isinstance(hqmap, dict):
                q = hqmap.get(hc)
                if q is None or not isinstance(q, list):
                    q = []
                    hqmap[hc] = q

                def _key_in_home(xjid: str) -> float:
                    jj = jobs.get(str(xjid))
                    if jj is None:
                        return float("inf")
                    return _stable_submit_key(jj)

                _list_insert_sorted_unique_by_key(q, _key_in_home, jid)
        except Exception:
            pass

        # 3.5) ✅ 실행 큐(cluster_queues)에도 반드시 넣기 (SSOT: cluster_tick이 보는 큐)
        #      - 지금 버그는 HOME에만 있고 clusterQ에 없어서 tick이 못 집는 것
        try:
            qs = getattr(st, "cluster_queues", None)
            if qs is None or not isinstance(qs, dict):
                st.cluster_queues = {}
                qs = st.cluster_queues

            cq = qs.get(hc)
            if cq is None:
                # 코드베이스에 따라 cluster queue가 list이거나 객체일 수 있음
                # 여기서는 list로 기본 생성
                cq = []
                qs[hc] = cq

            # job의 g_req(=gang/target 힌트) 보정
            try:
                g_req0 = int(getattr(jr, "g_target", 0) or getattr(jr, "world_size", 0) or 1)
            except Exception:
                g_req0 = 1
            if g_req0 <= 0:
                g_req0 = 1

            inserted_clusterq = False

            # (1) list 기반 clusterQ
            if isinstance(cq, list):
                if jid not in [str(x) for x in cq]:
                    cq.append(jid)
                    inserted_clusterq = True

            # (2) 객체 기반 clusterQ: _jobs 리스트 사용
            else:
                try:
                    jobs_list = list(getattr(cq, "_jobs", []) or [])
                except Exception:
                    jobs_list = []

                exists = False
                for x in jobs_list:
                    try:
                        xjid = str(getattr(x, "job_id", x) or "")
                    except Exception:
                        xjid = str(x or "")
                    if xjid == jid:
                        exists = True
                        break

                if not exists:
                    jobs_list.append({"job_id": jid, "g_req": int(g_req0)})
                    try:
                        setattr(cq, "_jobs", jobs_list)
                    except Exception:
                        pass
                    inserted_clusterq = True

                try:
                    cq.eta_dirty = True
                except Exception:
                    pass

        except Exception:
            inserted_clusterq = False

        # 4) queue meta (null 금지)
        try:
            jr.queue_kind = "HOME"
            jr.queue_cluster_id = hc
        except Exception:
            pass

        # 5) queue enter ts 갱신(queued_delta 누적 기준)
        #    requeue여도 “대기 시작 시점”은 갱신되게 두는 게 일반적으로 맞음
        #    (누적은 queued_accum_sec에서 하니 last_queue_enter_ts는 새로 찍는게 정상)
        try:
            _queue_enter_locked(jr, now_ts)
        except Exception:
            try:
                jr.last_queue_enter_ts = float(now_ts)
            except Exception:
                pass

        # 6) qlen snapshots
        try:
            qlen_global = int(len(getattr(st, "global_queue", []) or []))
        except Exception:
            qlen_global = -1
        try:
            qlen_home = int(len((getattr(st, "home_cluster_queues", {}) or {}).get(hc, []) or []))
        except Exception:
            qlen_home = -1
        try:
            qlen_eligible = int(_cluster_eligible_queue_len_locked(st, hc))
        except Exception:
            qlen_eligible = -1

    # logs (락 밖)
    try:
        if rl:
            rl.queue_event(
                event="queue_enqueued",
                job_id=jid,
                queue_len=int(qlen_eligible),
                qlen_eligible=int(qlen_eligible),
                qlen_global=int(qlen_global),
                qlen_home=int(qlen_home),
                qlen_clusterq=-1,
                note=f"to_kind=HOME to_qcid={hc} why={why} from_kind={prev_kind} from_qcid={prev_qcid} submit_key={submit_key}",
                ts=float(_now()),
                seq=int(seq),
            )
            rl.job_event(
                event="enqueued_home",
                job_id=jid,
                cluster=hc,
                world_size=int(getattr(jr, "g_target", 0) or 0),
                note=f"why={why}",
                metadata={
                    "prev_queue_kind": prev_kind,
                    "prev_queue_cluster_id": prev_qcid,
                    "to_queue_kind": "HOME",
                    "to_queue_cluster_id": hc,
                    "submit_key": float(submit_key),
                },
                ts=float(_now()),
                seq=int(seq),
                stream="scheduler",
            )
            rl.job_event(
                event="enqueued_clusterq",
                job_id=jid,
                cluster=hc,
                world_size=int(getattr(jr, "g_target", 0) or 0),
                note=f"why={why}",
                metadata={
                    "to_queue_kind": "CLUSTERQ",
                    "to_queue_cluster_id": hc,
                },
                ts=float(_now()),
                seq=int(seq),
                stream="scheduler",
            )
    except Exception:
        pass

    return {"ok": True, "job_id": jid, "home_cluster_id": hc, "seq": int(seq)}

def _new_job_id() -> str:
    return "job-" + uuid.uuid4().hex[:12]

def _get_free_nodes_locked(st: Any, cluster_id: str) -> List[str]:
    """STATE_LOCK 안에서 호출. SSOT(node_owner) 기준 free node 리스트."""
    cluster_id = str(cluster_id)
    clusters = getattr(st, "clusters", {}) or {}
    node_owner: Dict[str, Any] = getattr(st, "node_owner", {}) or {}

    cr = clusters.get(cluster_id)
    if cr is None:
        return []

    try:
        nodes = [str(n) for n in (getattr(cr, "nodes", []) or [])]
    except Exception:
        if isinstance(cr, dict):
            nodes = [str(n) for n in (cr.get("nodes") or [])]
        else:
            nodes = []

    nodes = [n for n in nodes if n]
    return [n for n in nodes if node_owner.get(n) is None]

def _find_waiting_gang4_hol_locked(st: Any) -> Tuple[Optional[str], Optional[str], int]:
    jobs = getattr(st, "jobs", {}) or {}
    gq = getattr(st, "global_queue", None)
    if not isinstance(gq, list):
        return (None, None, 0)

    clusters = getattr(st, "clusters", {}) or {}

    def _job_id(x: Any) -> str:
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

    def _free_in_cluster(cid: str) -> int:
        cr = clusters.get(cid)
        if cr is None:
            return 0

        # nodes
        try:
            nodes = list(getattr(cr, "nodes", []) or [])
        except Exception:
            nodes = list(cr.get("nodes") or []) if isinstance(cr, dict) else []

        # node_owner는 cluster record 안에 있다
        try:
            no = getattr(cr, "node_owner", None)
        except Exception:
            no = None
        if not isinstance(no, dict) and isinstance(cr, dict):
            no = cr.get("node_owner")
        if not isinstance(no, dict):
            no = {}

        cnt = 0
        for n in nodes:
            if no.get(str(n)) is None:
                cnt += 1
        return int(cnt)

    for item in gq:
        jid = _job_id(item)
        if not jid:
            continue
        jr = jobs.get(jid)
        if jr is None:
            continue

        if str(getattr(jr, "status", "") or "").upper() != "QUEUED":
            continue

        try:
            need = int(_job_gang_required_g(jr) or 0)
        except Exception:
            need = 0
        if need != 4:
            continue

        pin = (
            str(getattr(jr, "pinned_cluster", "") or "").strip()
            or str(getattr(jr, "pinned_cluster_id", "") or "").strip()
            or str(getattr(jr, "admitted_cluster_id", "") or "").strip()
            or str(getattr(jr, "home_cluster_id", "") or "").strip()
        )

        # ✅ NEW: pin이 비어있으면 allowed/preferred로 보정
        if not pin:
            try:
                pin = str(
                    getattr(jr, "allowed_cluster", "")
                    or getattr(jr, "preferred_cluster", "")
                    or ""
                ).strip()
            except Exception:
                pin = ""

            if pin:
                # ✅ 여기서 “강제로 pin을 박아준다”
                try:
                    jr.pinned_cluster = pin
                except Exception:
                    pass
                try:
                    jr.admitted_cluster_id = pin
                except Exception:
                    pass

        if not pin:
            continue

        return (str(jid), str(pin), _free_in_cluster(pin))

    return (None, None, 0)

def _should_route_submit_to_clusterq_backfill_locked(
    st: Any,
    jr: Any,
    cid_sel: str,
    g_use: int,
    now_ts: float,
) -> Tuple[bool, Dict[str, Any]]:
    cid_sel = str(cid_sel)
    g_use = int(g_use or 0)

    hol_jid, hol_pin, pin_free = _find_waiting_gang4_hol_locked(st)

    if not hol_jid:
        return False, {"reason": "no_waiting_gang4_hol"}

    # 이미 gang4 자체면 submit-immediate는 별도(여긴 non-gang만 라우팅하려는 용도)
    try:
        need = int(_job_gang_required_g(jr) or 0)
    except Exception:
        need = 0
    if need == 4:
        return False, {"reason": "submit_is_gang4"}

    # 1) 정석 트리거: pin free 1~3
    if hol_pin and 1 <= int(pin_free) <= 3:
        return True, {
            "reason": "hol_gang4_waiting_pin_free_1_3",
            "hol_job_id": str(hol_jid),
            "hol_pin_cluster": str(hol_pin),
            "pin_free": int(pin_free),
        }

    # 2) 보수적 트리거: pin free==0인데, cid_sel에 free가 있어서 지금 즉시 띄울 수 있는 상황이면
    #    submit-immediate가 백필 태깅 없이 자원을 잠그는 걸 막기 위해 clusterq로 보냄
    #    (이게 없으면 'preemption은 영원히 안 나는' 상황이 계속 재현됨)
    free_sel = len(_get_free_nodes_locked(st, cid_sel))
    if int(pin_free) == 0 and free_sel > 0 and g_use > 0:
        return True, {
            "reason": "hol_gang4_waiting_pin_free_0_block_submit_immediate",
            "hol_job_id": str(hol_jid),
            "hol_pin_cluster": str(hol_pin) if hol_pin else None,
            "pin_free": int(pin_free),
            "sel_cluster": str(cid_sel),
            "sel_free": int(free_sel),
        }

    return False, {
        "reason": "hol_exists_but_no_route",
        "hol_job_id": str(hol_jid),
        "hol_pin_cluster": str(hol_pin) if hol_pin else None,
        "pin_free": int(pin_free),
    }

def _enqueue_to_cluster_queue_locked(st: Any, job_id: str, cluster_id: str, g_req: int) -> None:
    """STATE_LOCK 안에서 호출. cluster_queue에 item 삽입(중복 방지)."""
    job_id = str(job_id)
    cluster_id = str(cluster_id)
    g_req = int(g_req or 1)
    if g_req <= 0:
        g_req = 1

    cqs = getattr(st, "cluster_queues", {}) or {}
    q = cqs.get(cluster_id)
    if q is None:
        return

    try:
        items = list(getattr(q, "_jobs", []) or [])
    except Exception:
        items = []

    # 중복 방지
    def _jid(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return str(x.get("job_id") or x.get("id") or "")
        if hasattr(x, "job_id"):
            return str(getattr(x, "job_id"))
        return str(x)

    items = [x for x in items if _jid(x) and _jid(x) != job_id]

    items.append({"job_id": job_id, "g_req": g_req, "enq_ts": float(time.time())})
    try:
        setattr(q, "_jobs", items)
    except Exception:
        pass

def submit_job_core(req: Any) -> Dict[str, Any]:
    def _get(k: str, default=None):
        if req is None:
            return default
        if isinstance(req, dict):
            return req.get(k, default)
        return getattr(req, k, default)

    now_ts = float(time.time())

    # ---------- basics ----------
    model = _get("model") or _get("model_name")
    dataset = _get("dataset")
    if not model or not dataset:
        return {"ok": False, "reason": "missing model/dataset"}

    model = str(model).strip()
    dataset = str(dataset).strip()

    user_id = _get("user_id", None)
    epochs = int(_get("epochs", 20) or 20)
    batch_size_per_gpu = int(_get("batch_size_per_gpu", 32) or 32)

    # ---------- checkpoint locality inputs ----------
    resume_from_checkpoint = _get("resume_from_checkpoint", None) or _get("checkpoint_path", None)
    checkpoint_cluster_id = _get("checkpoint_cluster_id", None)

    user_pinned_cluster_id = _get("pinned_cluster_id", None) or _get("cluster_id", None)
    user_admitted_cluster_id = _get("admitted_cluster_id", None)

    # 1) policy
    user_request_text = _get("user_request", None) or _get("user_request_text", None) or ""
    try:
        policy = get_policy_from_user_request(user_request_text) or {}
    except Exception:
        policy = {}

    # 2) gang 판정
    tmp_for_gang = SimpleNamespace(
        model=model,
        dataset=dataset,
        is_gang=bool(_get("is_gang", False)),
        g_req=_get("g_req", None),
    )
    gang_need = int(_job_gang_required_g(tmp_for_gang) or 0)  # 4 or 0
    is_gang = bool(gang_need == 4)

    # user hint (고정 아님)
    g_req_in = _get("g_target", None) or _get("world_size", None) or _get("num_gpus", None) or _get("g_req", None)
    try:
        g_req_in = int(g_req_in) if g_req_in is not None else None
    except Exception:
        g_req_in = None

    # 3) profiling (submit 힌트)
    profiling: Optional[Dict[str, Any]] = None
    try:
        try:
            profiling = get_profiling_entry(model=model, dataset=dataset)
        except Exception:
            profiling = get_profiling_entry(model, dataset)
        if not isinstance(profiling, dict):
            profiling = None
    except Exception:
        profiling = None

    # 4) g_candidates
    if is_gang:
        g_candidates: List[int] = [4]
    else:
        recg = 0
        try:
            recg = int((profiling or {}).get("recommended_g", 0) or 0)
        except Exception:
            recg = 0

        base = [1, 2, 4]
        if recg > 0 and recg not in base:
            base.append(recg)
        if g_req_in is not None and g_req_in > 0:
            base.append(int(g_req_in))

        g_candidates = sorted(set([int(x) for x in base if int(x) > 0]))
        if not g_candidates:
            g_candidates = [1, 2, 4]

    # 5) admission (cluster 확정)
    chosen_cid: Optional[str] = None
    chosen_prof: Optional[Dict[str, Any]] = profiling
    chosen_score: Optional[float] = None
    admitted_g_hint: Optional[int] = None

    forced_by_pinned = bool(user_pinned_cluster_id)
    forced_by_admitted = bool((not user_pinned_cluster_id) and user_admitted_cluster_id)
    forced_by_gang_rr = False
    forced_by_checkpoint = False

    if resume_from_checkpoint and not checkpoint_cluster_id and not user_pinned_cluster_id and not user_admitted_cluster_id:
        return {
            "ok": False,
            "reason": "resume_requires_cluster_id",
            "detail": "Provide checkpoint_cluster_id (or pinned_cluster_id/admitted_cluster_id) when resume_from_checkpoint is used.",
        }

    with STATE_LOCK:
        st = get_global_state()
        clusters_all = getattr(st, "clusters", {}) or {}

        # -----------------------
        # (A) locality / user forced
        # -----------------------
        if checkpoint_cluster_id:
            cid0 = str(checkpoint_cluster_id)
            cr0 = clusters_all.get(cid0)
            if cr0 is None:
                return {"ok": False, "reason": f"invalid checkpoint_cluster_id={cid0}"}
            if user_pinned_cluster_id and str(user_pinned_cluster_id) != cid0:
                return {
                    "ok": False,
                    "reason": "checkpoint_cluster_conflict_with_pinned",
                    "detail": f"checkpoint_cluster_id={cid0} pinned_cluster_id={user_pinned_cluster_id}",
                }
            if user_admitted_cluster_id and str(user_admitted_cluster_id) != cid0:
                return {
                    "ok": False,
                    "reason": "checkpoint_cluster_conflict_with_admitted",
                    "detail": f"checkpoint_cluster_id={cid0} admitted_cluster_id={user_admitted_cluster_id}",
                }
            chosen_cid = cid0
            forced_by_checkpoint = True

        elif user_pinned_cluster_id:
            cid0 = str(user_pinned_cluster_id)
            cr0 = clusters_all.get(cid0)
            if cr0 is None:
                return {"ok": False, "reason": f"invalid pinned_cluster_id={cid0}"}
            chosen_cid = cid0

        elif user_admitted_cluster_id:
            cid0 = str(user_admitted_cluster_id)
            cr0 = clusters_all.get(cid0)
            if cr0 is None:
                return {"ok": False, "reason": f"invalid admitted_cluster_id={cid0}"}
            chosen_cid = cid0

        # -----------------------
        # ✅ (B) gang4: RR pin/admit SSOT 강제 (choose_best / fallback 금지)
        # -----------------------
        elif is_gang:
            forced_by_gang_rr = True
            pin = str(pick_next_gang_cluster_locked(st))
            if pin not in clusters_all:
                # 보수적으로 첫 클러스터
                try:
                    pin = str(sorted([str(x) for x in clusters_all.keys()])[0])
                except Exception:
                    pin = "clusterA"
            chosen_cid = pin

        # -----------------------
        # (C) non-gang: choose_best
        # -----------------------
        else:
            # choose_best는 non-gang에만
            try:
                job_tmp = SimpleNamespace(
                    model=model,
                    dataset=dataset,
                    batch_size_per_gpu=int(batch_size_per_gpu),
                    is_gang=False,
                    g_req=int(g_req_in) if (g_req_in is not None and int(g_req_in) > 0) else None,
                )
                cid, g_sel, prof_sel, S = _choose_best_cluster_and_g_theory(
                    state=st,
                    job=job_tmp,
                    policy=policy or {},
                    clusters=dict(clusters_all),
                    g_candidates=list(g_candidates),
                )
                chosen_cid = str(cid)
                admitted_g_hint = int(g_sel) if g_sel is not None else None
                chosen_prof = prof_sel if isinstance(prof_sel, dict) else (profiling or {})
                chosen_score = float(S)
            except Exception:
                # fallback: free node 많은 쪽
                node_owner = getattr(st, "node_owner", {}) or {}

                def _free_count(cid: str) -> int:
                    cr = (getattr(st, "clusters", {}) or {}).get(cid)
                    nodes = []
                    try:
                        nodes = list(getattr(cr, "nodes", []) or [])
                    except Exception:
                        if isinstance(cr, dict):
                            nodes = list(cr.get("nodes") or [])
                    free = 0
                    for n in nodes:
                        if node_owner.get(str(n)) is None:
                            free += 1
                    return int(free)

                best_cid = None
                best_free = -1
                for cid0 in (clusters_all or {}).keys():
                    f = _free_count(str(cid0))
                    if f > best_free:
                        best_free = f
                        best_cid = str(cid0)

                chosen_cid = best_cid or (str(next(iter(clusters_all.keys()))) if clusters_all else "clusterA")
                chosen_prof = profiling or {}
                chosen_score = None
                admitted_g_hint = None

    # 6) register
    job_id = _get("job_id", None) or _new_job_id()

    # g_target: gang은 4 고정, non-gang은 admitted_g_hint 있으면 힌트로 넣어둠(0도 가능하지만 submit-immediate에서 1로 깎이는 버그 유발)
    if is_gang:
        g_target = 4
    else:
        g_target = int(admitted_g_hint) if (admitted_g_hint is not None and int(admitted_g_hint) > 0) else 0

    jr = register_job_on_submit(
        job_id=str(job_id),
        model=model,
        dataset=dataset,
        user_id=str(user_id) if user_id is not None else None,
        policy=policy or {},
        g_target=int(g_target),
        profiling=chosen_prof if isinstance(chosen_prof, dict) else (profiling or {}),
        admitted_cluster_id=str(chosen_cid),
        user_pinned_cluster_id=str(user_pinned_cluster_id) if user_pinned_cluster_id else None,
        epochs=int(epochs),
        batch_size_per_gpu=int(batch_size_per_gpu),
    )

    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    # ✅ submit에서 job record 안정화(SSOT)
    seq_submitted = None
    with STATE_LOCK:
        st2 = get_global_state()
        jr2 = (getattr(st2, "jobs", {}) or {}).get(str(job_id))
        if jr2 is not None:
            try:
                s = int(getattr(jr2, "event_seq", 0) or 0) + 1
                jr2.event_seq = int(s)
                seq_submitted = int(s)
            except Exception:
                seq_submitted = None

            try:
                jr2.status = "QUEUED"
                _clear_requeue_victim_tags_locked(jr2)

            except Exception:
                pass

            # home/admit/pin SSOT
            try:
                # home은 "원래 의미"가 있다면 유지하고, 없으면 chosen으로 보정
                if not getattr(jr2, "home_cluster_id", None):
                    jr2.home_cluster_id = str(chosen_cid)

                jr2.admitted_cluster_id = str(chosen_cid)
                jr2.queue_kind = "HOME"
                jr2.queue_cluster_id = str(chosen_cid)
            except Exception:
                pass

            # ✅ gang4면 pin을 반드시 박아야 HoL/victim 로직이 pin 기반으로 돈다
            if is_gang:
                try:
                    jr2.pinned_cluster = str(chosen_cid)
                except Exception:
                    pass

            # stale 제거
            try:
                jr2.cluster_id = None
                jr2.nodes = []
                jr2.g_cur = 0
            except Exception:
                pass
            try:
                jr2.launch_inflight = False
                jr2.launching_since_ts = 0.0
                jr2.preempt_inflight = False
                jr2.preempting_since_ts = 0.0
                jr2.preempt_issued = False
            except Exception:
                pass
            try:
                jr2.blocked_until = 0.0
                jr2.launch_cooldown_until_ts = 0.0
            except Exception:
                pass

            if resume_from_checkpoint or checkpoint_cluster_id:
                try:
                    jr2.checkpoint_local_only = True
                    jr2.checkpoint_cluster_id = str(chosen_cid)
                    jr2.resume_from_checkpoint = str(resume_from_checkpoint) if resume_from_checkpoint else None
                except Exception:
                    pass
            else:
                try:
                    jr2.checkpoint_local_only = False
                    jr2.resume_from_checkpoint = None
                    jr2.checkpoint_cluster_id = None
                except Exception:
                    pass

            try:
                jr2.g_candidates = list(g_candidates)
            except Exception:
                pass
            try:
                if is_gang:
                    jr2.g_req = 4
                    jr2.g_target = 4
                else:
                    jr2.g_req = int(g_req_in) if (g_req_in is not None and int(g_req_in) > 0) else None
            except Exception:
                pass
            try:
                jr2.admitted_g_hint = int(admitted_g_hint) if admitted_g_hint is not None else None
            except Exception:
                pass

            # ✅ submit 경로에서는 backfill 플래그를 절대 켜지지 않게 초기화
            try:
                jr2.is_backfill = False
                jr2.is_hol_backfill = False
                jr2.hol_backfill_for = None
                jr2.hol_backfill_pin = None
                jr2.hol_backfill_since_ts = 0.0
            except Exception:
                pass

    if rl:
        try:
            rl.job_event(
                event="submitted",
                job_id=str(job_id),
                cluster=str(chosen_cid),
                world_size=int(g_target),
                note=f"model={model} dataset={dataset} gang={'1' if is_gang else '0'}",
                metadata={
                    "model": model,
                    "dataset": dataset,
                    "epochs": int(epochs),
                    "batch_size_per_gpu": int(batch_size_per_gpu),
                    "policy": policy or {},
                    "g_target_submit": int(g_target),
                    "g_candidates": list(g_candidates),
                    "admitted_g_hint": int(admitted_g_hint) if admitted_g_hint is not None else None,
                    "chosen_cluster": str(chosen_cid),
                    "forced_by_checkpoint": bool(forced_by_checkpoint),
                    "forced_by_pinned": bool(forced_by_pinned),
                    "forced_by_admitted": bool(forced_by_admitted),
                    "forced_by_gang_rr": bool(forced_by_gang_rr),
                },
                ts=float(time.time()),
                seq=int(seq_submitted) if isinstance(seq_submitted, int) else None,
                stream="scheduler",
            )
        except Exception:
            pass

    # 7) enqueue (HOME)
    enqueue_home = {}
    try:
        enqueue_home = enqueue_job_to_home_cluster_queue(str(job_id), str(chosen_cid))
    except Exception as e:
        enqueue_home = {"ok": False, "reason": "enqueue_home_exception", "exc": repr(e)}

    # 단일화
    try:
        with STATE_LOCK:
            st3 = get_global_state()
            try:
                ensure_job_single_queue_locked(st3, str(job_id), str(chosen_cid))
            except Exception:
                pass
    except Exception:
        pass

    # submit 즉시 배치
    # - gang4는 free>=4일 때만 바로 launch 시도
    # - non-gang은 HoL(gang4) 대기 상황이면 cluster_queue로 라우팅(진짜 백필)
    launch_attempted = False
    launch_result: Dict[str, Any] = {"ok": False, "reason": "not_attempted"}

    with STATE_LOCK:
        st4 = get_global_state()
        jobs4 = getattr(st4, "jobs", {}) or {}
        job_obj = jobs4.get(str(job_id))

        if job_obj is None:
            return {
                "ok": True,
                "job_id": str(job_id),
                "model": model,
                "dataset": dataset,
                "status": "QUEUED",
                "admitted_cluster_id": str(chosen_cid),
                "enqueue": {"home": enqueue_home},
                "launch": {"attempted": False, "reason": "job_record_missing_after_register"},
            }

        need_gang = (int(_job_gang_required_g(job_obj) or 0) == 4)

        # drain check
        drains = getattr(st4, "drain_until_by_cluster", {}) or {}
        drain_until = float(drains.get(str(chosen_cid), 0.0) or 0.0)
        if now_ts < drain_until:
            return {
                "ok": True,
                "job_id": str(job_id),
                "model": model,
                "dataset": dataset,
                "status": "QUEUED",
                "admitted_cluster_id": str(chosen_cid),
                "enqueue": {"home": enqueue_home},
                "launch": {"attempted": False, "reason": "cluster_draining", "drain_until": drain_until},
            }

        free_nodes = list(_free_nodes_in_cluster(st4, str(chosen_cid)) or [])
        free_nodes = [str(x) for x in free_nodes if str(x)]
        free_n = len(free_nodes)

        if need_gang and free_n < 4:
            return {
                "ok": True,
                "job_id": str(job_id),
                "model": model,
                "dataset": dataset,
                "status": "QUEUED",
                "admitted_cluster_id": str(chosen_cid),
                "enqueue": {"home": enqueue_home},
                "launch": {"attempted": False, "reason": "gang4_need_more_free", "free_n": free_n},
            }

        if free_n <= 0:
            return {
                "ok": True,
                "job_id": str(job_id),
                "model": model,
                "dataset": dataset,
                "status": "QUEUED",
                "admitted_cluster_id": str(chosen_cid),
                "enqueue": {"home": enqueue_home},
                "launch": {"attempted": False, "reason": "no_free_nodes", "free_n": free_n},
            }

        # ---- g_use / nodes_use 결정 ----
        if need_gang:
            g_use = 4
            nodes_use = free_nodes[:4]
        else:
            g_fixed = int(getattr(job_obj, "g_target", 0) or 0)
            if g_fixed <= 0:
                g_fixed = int(getattr(job_obj, "admitted_g_hint", 0) or 0) or 1
            g_use = min(int(g_fixed), len(free_nodes))
            if g_use <= 0:
                g_use = 1
            nodes_use = list(free_nodes[:g_use])

        # ✅ non-gang만: HoL(gang4) 대기면 submit 즉시 launch 금지 → cluster_queue로 라우팅
        if (not need_gang):
            route_ok, route_dbg = _should_route_submit_to_clusterq_backfill_locked(
                st=st4,
                jr=job_obj,
                cid_sel=str(chosen_cid),
                g_use=int(g_use),
                now_ts=float(now_ts),
            )

            if route_ok:
                # ✅ submit은 절대 clusterQ로 보내지 않는다.
                #    (clusterQ는 "preempt된 backfill job" 전용)
                try:
                    purge_job_from_all_queues_locked(
                        st4,
                        str(job_id),
                        purge_global=True,
                        purge_cluster=True,
                        purge_home=True,
                    )
                except Exception:
                    pass

                # ✅ globalQ로 넣고, admitted_cluster_id는 "힌트"로만 남긴다
                try:
                    _global_queue_insert_sorted_locked(st4, str(job_id))
                except Exception:
                    pass

                try:
                    ensure_job_single_queue_locked(st4, str(job_id), str(chosen_cid))
                except Exception:
                    pass

                try:
                    job_obj.queue_kind = "GLOBAL"
                    job_obj.queue_cluster_id = None
                except Exception:
                    pass

                if rl:
                    try:
                        rl.queue_event(
                            event="submit_route_globalq",
                            job_id=str(job_id),
                            queue_len=-1,
                            qlen_clusterq=-1,
                            note=f"submit_route=True -> GLOBALQ (no clusterQ) admitted={chosen_cid} g_hint={int(g_use)} dbg={route_dbg}",
                            ts=float(time.time()),
                        )
                    except Exception:
                        pass

                return {
                    "ok": True,
                    "job_id": str(job_id),
                    "model": model,
                    "dataset": dataset,
                    "status": "QUEUED",
                    "admitted_cluster_id": str(chosen_cid),
                    "enqueue": {"home": enqueue_home, "globalq": {"ok": True, "dbg": route_dbg}},
                    "launch": {
                        "attempted": False,
                        "reason": "routed_to_globalq_due_to_hol_wait",
                        "dbg": route_dbg,
                    },
                }


        # ---- reserve + 상태 마킹 ----
        try:
            reserve_nodes(job_id=str(job_id), cluster_id=str(chosen_cid), nodes=list(nodes_use))
        except Exception as e:
            return {
                "ok": True,
                "job_id": str(job_id),
                "model": model,
                "dataset": dataset,
                "status": "QUEUED",
                "admitted_cluster_id": str(chosen_cid),
                "enqueue": {"home": enqueue_home},
                "launch": {"attempted": False, "reason": "reserve_conflict", "exc": repr(e)},
            }

        try:
            job_obj.cluster_id = str(chosen_cid)
            job_obj.nodes = list(nodes_use)
            job_obj.g_cur = int(g_use)
            job_obj.launch_inflight = True
            job_obj.launching_since_ts = float(now_ts)
            job_obj.status = "STARTING"
            # ✅ submit 즉시 launch는 backfill 절대 아님
            job_obj.is_backfill = False
            job_obj.is_hol_backfill = False
        except Exception:
            pass

        launch_attempted = True
        chosen_nodes_final = list(nodes_use)
        chosen_g_final = int(g_use)

    # ---- LOCK 밖 실제 launch ----
    try:
        launch_result = _start_job_on_cluster(
            job_obj,
            str(chosen_cid),
            list(chosen_nodes_final),
            int(chosen_g_final),
            is_backfill=False,  # ✅ 절대 True 금지(이번 버그 포인트)
        )
    except Exception as e:
        launch_result = {"ok": False, "reason": "start_exception", "exc": repr(e)}

    # ---- 실패 rollback ----
    if not bool((launch_result or {}).get("ok")):
        with STATE_LOCK:
            st5 = get_global_state()
            job5 = (getattr(st5, "jobs", {}) or {}).get(str(job_id))
            if job5 is not None:
                try:
                    release_nodes_for_job(st5, str(job_id), list(getattr(job5, "nodes", []) or []))
                except Exception:
                    try:
                        _ss_release_nodes_for_job_locked(st5, job_id=str(job_id), nodes=list(getattr(job5, "nodes", []) or []))
                    except Exception:
                        pass

                try:
                    job5.launch_inflight = False
                    job5.launching_since_ts = 0.0
                    job5.cluster_id = None
                    job5.nodes = []
                    job5.g_cur = 0
                    job5.status = "QUEUED"
                    job5.blocked_until = float(time.time() + 1.0)
                    # submit 실패 후에도 backfill 아님
                    job5.is_backfill = False
                    job5.is_hol_backfill = False
                except Exception:
                    pass

                try:
                    purge_job_from_all_queues_locked(st5, str(job_id))
                    _global_queue_insert_sorted_locked(st5, str(job_id))
                    ensure_job_single_queue_locked(st5, str(job_id), str(chosen_cid))
                except Exception:
                    pass

    try:
        logger.info(
            "[SUBMIT_IMMEDIATE] job_id=%s cid=%s attempted=%s g=%s nodes=%s ok=%s reason=%s gang=%s",
            str(job_id),
            str(chosen_cid),
            bool(launch_attempted),
            (launch_result.get("g") if isinstance(launch_result, dict) else None) or "",
            (launch_result.get("nodes") if isinstance(launch_result, dict) else None) or "",
            bool(launch_result.get("ok")) if isinstance(launch_result, dict) else False,
            str(launch_result.get("status") or launch_result.get("reason") or ""),
            "1" if is_gang else "0",
        )
    except Exception:
        pass

    return {
        "ok": True,
        "job_id": str(job_id),
        "model": model,
        "dataset": dataset,
        "admitted_cluster_id": str(chosen_cid),
        "status": ("RUNNING" if bool((launch_result or {}).get("ok")) else "QUEUED"),
        "enqueue": {"home": enqueue_home},
        "launch": {"attempted": bool(launch_attempted), "result": launch_result},
        "admission": {
            "chosen_cluster": str(chosen_cid),
            "score": (float(chosen_score) if chosen_score is not None else None),
            "g_candidates": list(g_candidates),
            "admitted_g_hint": (int(admitted_g_hint) if admitted_g_hint is not None else None),
            "is_gang": bool(is_gang),
            "forced_by_checkpoint": bool(forced_by_checkpoint),
            "forced_by_pinned": bool(forced_by_pinned),
            "forced_by_admitted": bool(forced_by_admitted),
            "forced_by_gang_rr": bool(forced_by_gang_rr),
            "note": (
                "gang4 uses RR pin/admit (no choose/fallback). "
                "submit-immediate launch is never backfill; "
                "true backfill happens via cluster_queue only."
            ),
        },
    }

def _now() -> float:
    import time
    return float(time.time())

def _job_is_running(jr: Any) -> bool:
    return str(getattr(jr, "status", "") or "").upper() == "RUNNING"

_TICK_LOCK = threading.Lock()
_TICK_EVENT = threading.Event()

def request_tick() -> None:
    global _TICK_PENDING
    ensure_tick_thread_started()
    with _TICK_LOCK:
        _TICK_PENDING = True
    _TICK_EVENT.set()

def _tick_thread_main() -> None:
    global _TICK_PENDING
    import os, time

    while not _TICK_STOP.is_set():
        woke = _TICK_EVENT.wait(timeout=TICK_PERIOD_SEC)
        _TICK_EVENT.clear()

        with _TICK_LOCK:
            pending = bool(_TICK_PENDING)
            _TICK_PENDING = False

        try:
            _schedule_once()
        except Exception:
            logger.exception("[tick] tick_thread_main failed")

def _pick_victims_for_gang_locked(
    st: Any,
    *args,
    **kwargs,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # ---------- 1) alias 흡수 / 정규화 ----------
    pin = (
        kwargs.pop("pin", None)
        or kwargs.pop("pin_cluster_id", None)
        or kwargs.pop("cluster_id", None)
        or kwargs.pop("cid", None)
    )
    pin = str(pin).strip() if pin is not None else ""

    need_free = (
        kwargs.pop("need_free", None)
        or kwargs.pop("need", None)
        or kwargs.pop("need_g", None)
        or kwargs.pop("need_gpus", None)
        or kwargs.pop("need_slots", None)
    )
    try:
        need_free = int(need_free) if need_free is not None else 0
    except Exception:
        need_free = 0

    # 참고용(없어도 됨): hol 관련
    hol_jid = kwargs.pop("hol_jid", None)
    hol_norm = kwargs.pop("hol_norm", None)
    hol_pin = kwargs.pop("hol_pin", None)
    free_n = kwargs.pop("free_n", None)

    # 옵션들(없으면 기본값)
    only_hol_backfill = bool(kwargs.pop("only_hol_backfill", True))  # ✅ 기본 True: reclaim victim은 HOL_BACKFILL만
    exclude_gang4 = bool(kwargs.pop("exclude_gang4", True))
    safe_only = bool(kwargs.pop("safe_only", True))
    prefer_large_first = bool(kwargs.pop("prefer_large_first", True))
    max_victims = int(kwargs.pop("max_victims", 8) or 8)

    # 예상 못한 나머지 키워드는 dbg에만 남기고 버림
    extra_kwargs = dict(kwargs)

    dbg: Dict[str, Any] = {
        "pin": pin,
        "need_free": need_free,
        "hol_jid": (str(hol_jid) if hol_jid else None),
        "hol_norm": (str(hol_norm) if hol_norm else None),
        "hol_pin": (str(hol_pin) if hol_pin else None),
        "free_n": (int(free_n) if free_n is not None else None),
        "only_hol_backfill": only_hol_backfill,
        "exclude_gang4": exclude_gang4,
        "safe_only": safe_only,
        "prefer_large_first": prefer_large_first,
        "max_victims": max_victims,
        "extra_kwargs": extra_kwargs,
        "drop_reasons": {},  # ✅ 왜 탈락했는지 카운팅
    }

    if not pin or need_free <= 0:
        dbg["reason"] = "invalid_pin_or_need"
        return [], dbg

    # ---------- helpers ----------
    def _as_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _status(jr: Any) -> str:
        try:
            return str(getattr(jr, "status", "") or "").upper()
        except Exception:
            return ""

    def _is_running(jr: Any) -> bool:
        return _status(jr) == "RUNNING"

    def _safe_to_touch(jr: Any) -> bool:
        if jr is None:
            return False
        try:
            if bool(getattr(jr, "launch_inflight", False)) or bool(getattr(jr, "preempt_inflight", False)):
                return False
        except Exception:
            pass
        return True

    def _victim_gcur(jr: Any) -> int:
        for k in ("g_cur", "world_size", "g"):
            try:
                v = getattr(jr, k, None)
                if v is not None and int(v) > 0:
                    return int(v)
            except Exception:
                pass
        try:
            nd = getattr(jr, "nodes", None)
            if isinstance(nd, list) and nd:
                return int(len(nd))
        except Exception:
            pass
        return 1

    def _allowed_cluster(jr: Any) -> str:
        # "찜/허용" 클러스터: pinned > admitted > home
        try:
            pinned = _as_str(getattr(jr, "pinned_cluster", "") or "").strip()
        except Exception:
            pinned = ""
        try:
            adm = _as_str(getattr(jr, "admitted_cluster_id", "") or "").strip()
        except Exception:
            adm = ""
        try:
            home = _as_str(getattr(jr, "home_cluster_id", "") or "").strip()
        except Exception:
            home = ""
        return pinned or adm or home

    def _is_hol_victim(jr: Any, *, pin0: str) -> Tuple[bool, str]:
        """
        victim 허용 조건(강제):
        - is_hol_backfill 또는 queue_kind == HOL_BACKFILL
        - hol_backfill_pin == pin0
        - allowed_cluster == pin0  (✅ 찜하지 않은 cluster면 victim 금지)
        """
        try:
            qk = _as_str(getattr(jr, "queue_kind", "") or "").upper().strip()
        except Exception:
            qk = ""
        try:
            is_hbf = bool(getattr(jr, "is_hol_backfill", False)) or (qk == "HOL_BACKFILL")
        except Exception:
            is_hbf = False
        if not is_hbf:
            return False, "not_hol_backfill"

        try:
            hpin = _as_str(getattr(jr, "hol_backfill_pin", "") or "").strip()
        except Exception:
            hpin = ""
        if not hpin or hpin != str(pin0):
            return False, "hol_pin_mismatch"

        allowed = _allowed_cluster(jr)
        if not allowed or allowed != str(pin0):
            return False, "not_allowed_cluster"

        return True, "ok"

    def _drop(reason: str) -> None:
        try:
            mp = dbg.get("drop_reasons")
            if not isinstance(mp, dict):
                mp = {}
                dbg["drop_reasons"] = mp
            mp[reason] = int(mp.get(reason, 0) or 0) + 1
        except Exception:
            pass

    # ---------- 3) victim 선택 로직 ----------
    jobs = getattr(st, "jobs", {}) or {}
    cands: List[Tuple[str, int]] = []

    for jid, jr in (jobs or {}).items():
        if jr is None:
            _drop("jr_none")
            continue

        if not _is_running(jr):
            _drop("not_running")
            continue

        if safe_only and (not _safe_to_touch(jr)):
            _drop("unsafe_inflight")
            continue

        try:
            cid = _as_str(getattr(jr, "cluster_id", "") or "").strip()
        except Exception:
            cid = ""
        if cid != str(pin):
            _drop("not_on_pin")
            continue

        if exclude_gang4:
            try:
                if int(_job_gang_required_g(jr) or 0) == 4:
                    _drop("exclude_gang4")
                    continue
            except Exception:
                pass

        # ✅ reclaim victim은 기본적으로 HOL_BACKFILL만
        if only_hol_backfill:
            ok_v, why = _is_hol_victim(jr, pin0=str(pin))
            if not ok_v:
                _drop(why)
                continue

        gcur = max(1, int(_victim_gcur(jr)))
        cands.append((str(jid), int(gcur)))

    # 정렬/선택
    cands.sort(key=lambda x: int(x[1]), reverse=bool(prefer_large_first))

    picked: List[Dict[str, Any]] = []
    got = 0

    for vjid, vg in cands:
        if len(picked) >= int(max_victims):
            break
        picked.append({"job_id": str(vjid), "g": int(vg)})
        got += int(vg)
        if got >= int(need_free):
            break

    dbg["candidates"] = int(len(cands))
    dbg["picked"] = int(len(picked))
    dbg["got"] = int(got)
    dbg["picked_ids"] = [str(x.get("job_id")) for x in picked]
    if got < int(need_free):
        dbg["reason"] = "insufficient_victims"
    else:
        dbg["reason"] = "ok"

    return picked, dbg

def _global_tick_once() -> None:
    import time, uuid
    from typing import Any, Dict, List, Tuple

    def _find_hol_backfill_victims_locked(st0: Any, hol_jid0: str, pin0: str) -> List[str]:
        jobs0 = getattr(st0, "jobs", {}) or {}
        out = []
        for jid, jr0 in (jobs0 or {}).items():
            if jr0 is None:
                continue
            try:
                if not bool(getattr(jr0, "is_hol_backfill", False)):
                    continue
                if str(getattr(jr0, "hol_for", "") or "") != str(hol_jid0):
                    continue
                if str(getattr(jr0, "hol_pin", "") or "") != str(pin0):
                    continue
                su = str(getattr(jr0, "status", "") or "").upper()
                if su not in ("STARTING", "RUNNING", "LAUNCHING", "DISPATCHED"):
                    continue
            except Exception:
                continue
            out.append(str(jid))
        return out


    now_ts = float(time.time())

    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    launch_plan = None
    preempt_plan = None

    # -------------------------
    # LOCK: 계획 수립(+필요한 SSOT 정리)
    # -------------------------
    with STATE_LOCK:
        st = get_global_state()

        hol_jid, hol_pin, _pin_free_hint = _find_waiting_gang4_hol_locked(st)
        if not hol_jid:
            return

        jobs = getattr(st, "jobs", {}) or {}
        hol = jobs.get(str(hol_jid))
        if hol is None:
            return

        # hol_pin이 없으면 여기서도 보정(혹시 _find_가 pin을 못 준 케이스)
        if not hol_pin:
            hol_pin = (
                str(getattr(hol, "pinned_cluster", "") or "").strip()
                or str(getattr(hol, "pinned_cluster_id", "") or "").strip()
                or str(getattr(hol, "admitted_cluster_id", "") or "").strip()
                or str(getattr(hol, "home_cluster_id", "") or "").strip()
            )
            if not hol_pin:
                return

        # --- (A) HoL 시점에 gang4 pin 확정: pin이 비어있으면 여기서 강제로 박는다 ---
        def _get_pin(jr0: Any) -> str:
            try:
                return (
                    str(getattr(jr0, "pinned_cluster", "") or "").strip()
                    or str(getattr(jr0, "pinned_cluster_id", "") or "").strip()
                    or str(getattr(jr0, "admitted_cluster_id", "") or "").strip()
                    or str(getattr(jr0, "home_cluster_id", "") or "").strip()
                )
            except Exception:
                return ""

        try:
            need_g = int(_job_gang_required_g(hol) or 0)
        except Exception:
            need_g = 0

        if need_g == 4:
            pin0 = _get_pin(hol)
            if not pin0:
                chosen = str(pick_next_gang_cluster_locked(st))
                hol_pin = chosen
                try: hol.pinned_cluster = chosen
                except Exception: pass
                try: hol.pinned_cluster_id = chosen
                except Exception: pass
                try: hol.admitted_cluster_id = chosen
                except Exception: pass

                if rl:
                    try:
                        rl.queue_event(
                            event="gang_pin_committed_at_hol",
                            job_id=str(hol_jid),
                            queue_len=-1,
                            qlen_clusterq=-1,
                            note=f"pin_was_empty -> commit pin={chosen}",
                            ts=float(time.time()),
                        )
                    except Exception:
                        pass
            else:
                hol_pin = str(pin0)


        # --- 이미 terminal이면 무시 ---
        try:
            terminal_set = set(_TERMINAL)
        except Exception:
            terminal_set = {"FINISHED", "FAILED", "CANCELLED", "COMPLETED", "PREEMPTED"}

        try:
            st0 = str(getattr(hol, "status", "") or "").upper()
        except Exception:
            st0 = ""
        if st0 in terminal_set:
            return

        # ✅ NEW: node_draining 등으로 launch를 미루기로 했으면 그때까지는 tick에서 launch 시도 금지
        try:
            nle = float(getattr(hol, "next_launch_earliest_ts", 0.0) or 0.0)
        except Exception:
            nle = 0.0
        if nle > 0.0 and now_ts < nle:
            return

        # --- launcher thrash 방지(단, 영구락 방지) ---
        try:
            reclaim_active = bool(getattr(st, "hol_reclaim_active", False))

            li = bool(getattr(hol, "launch_inflight", False))
            ls = float(getattr(hol, "launching_since_ts", 0.0) or 0.0)

            # inflight는 짧게만 막는다 (RPC/워커 응답 지연 고려)
            #  - 2~5초면 충분. 길면 재시도가 막혀서 지금처럼 "안 다시 쏨"이 생김
            LAUNCH_INFLIGHT_HOLD_SEC = float(getattr(st, "LAUNCH_INFLIGHT_HOLD_SEC", 3.0) or 3.0)

            if li:
                # 최근에 쏜 직후면 return (중복 RPC 방지)
                if ls > 0.0 and (now_ts - ls) < LAUNCH_INFLIGHT_HOLD_SEC:
                    return

                # ✅ 타임아웃: inflight가 오래 유지되면 강제로 풀고 다음 단계로 진행(재시도 가능)
                try:
                    hol.launch_inflight = False
                except Exception:
                    pass
                try:
                    hol.launching_since_ts = 0.0
                except Exception:
                    pass

            # preempt_inflight는 reclaim_active가 아니면 tick을 막는다 (기존 로직 유지)
            if bool(getattr(hol, "preempt_inflight", False)) and (not reclaim_active):
                return
        except Exception:
            pass


        clusters = getattr(st, "clusters", {}) or {}
        node_owner = getattr(st, "node_owner", {}) or {}

        # cluster_launch_blocked_until 존중
        try:
            cb = getattr(st, "cluster_launch_blocked_until", {}) or {}
        except Exception:
            cb = {}

        def _free_nodes(cid: str) -> List[str]:
            # ✅ SSOT + drain-aware "available nodes"
            return list(_free_nodes_in_cluster(st, str(cid)) or [])

        def _free_count(cid: str) -> int:
            return int(len(_free_nodes(cid)))

        def _cluster_blocked(cid2: str) -> bool:
            cid2 = str(cid2)
            try:
                bu = float(cb.get(cid2, 0.0) or 0.0)
            except Exception:
                bu = 0.0
            return bool(bu > 0.0 and now_ts < bu)

        def _nodes_owned_by(jid: str, nodes: List[str]) -> bool:
            """reuse_nodes 안전성: 해당 노드들이 SSOT(node_owner)에서 실제로 hol에 묶여있는지 확인"""
            try:
                for n in (nodes or []):
                    if str(node_owner.get(str(n)) or "") != str(jid):
                        return False
                return True
            except Exception:
                return False

        # ---- pin SSOT ----
        hol_pin = str(hol_pin)

        # ---- pin move 금지 조건(체크포인트 로컬/리줌/클러스터 고정) ----
        def _hol_must_stay_local(jr: Any) -> Tuple[bool, str]:
            try:
                ckpt_local_only = bool(getattr(jr, "checkpoint_local_only", False))
            except Exception:
                ckpt_local_only = False
            try:
                resume = getattr(jr, "resume_from_checkpoint", None)
                resume = (str(resume).strip() if resume is not None else "") or ""
            except Exception:
                resume = ""
            try:
                ckpt_cluster = (str(getattr(jr, "checkpoint_cluster_id", "") or "").strip())
            except Exception:
                ckpt_cluster = ""

            if ckpt_cluster:
                return True, ckpt_cluster
            if ckpt_local_only or resume:
                # allowed는 home/admitted로 수렴
                try:
                    home = (str(getattr(jr, "home_cluster_id", "") or "").strip())
                except Exception:
                    home = ""
                try:
                    adm = (str(getattr(jr, "admitted_cluster_id", "") or "").strip())
                except Exception:
                    adm = ""
                allowed = ckpt_cluster or home or adm
                return True, allowed
            return False, ""

        must_stay, allowed_cluster = _hol_must_stay_local(hol)
        if must_stay and allowed_cluster and hol_pin != allowed_cluster:
            # pin이 이미 다른 곳으로 오염된 케이스: allowed로 되돌림
            hol_pin = str(allowed_cluster)
            try:
                hol.pinned_cluster = str(allowed_cluster)
            except Exception:
                pass
            try:
                hol.admitted_cluster_id = str(allowed_cluster)
            except Exception:
                pass

        # pin free 계산 (hint는 믿지 말고 SSOT node_owner로 재계산)
        pin_free = _free_count(hol_pin)

        # ---- pin move (이동 가능 + 목적지 blocked 아님 + free>=4) ----
        moved_from = None
        if pin_free < 4 and (not must_stay):
            best_cid = None
            for cid in sorted([str(x) for x in clusters.keys()]):
                if cid == hol_pin:
                    continue
                if _cluster_blocked(cid):
                    continue
                if _free_count(cid) >= 4:
                    best_cid = cid
                    break

            if best_cid:
                moved_from = hol_pin
                hol_pin = str(best_cid)
                try:
                    hol.pinned_cluster = str(best_cid)
                except Exception:
                    pass
                try:
                    hol.admitted_cluster_id = str(best_cid)
                except Exception:
                    pass

                pin_free = _free_count(hol_pin)

                if rl:
                    try:
                        rl.queue_event(
                            event="gang_pin_moved",
                            job_id=str(hol_jid),
                            queue_len=-1,
                            qlen_clusterq=-1,
                            note=f"moved_pin from={moved_from} to={hol_pin} free={pin_free}",
                            ts=float(time.time()),
                        )
                    except Exception:
                        pass

        # ✅ MIN FIX: pin을 옮겼는데 hol이 이전 클러스터에 STARTING 예약을 남기고 있으면 즉시 해제
        # (이게 없으면 A에 예약 남긴 채 B로 pin 이동 후 또 reserve -> 두 클러스터 점유 발생)
        try:
            st_h2 = str(getattr(hol, "status", "") or "").upper()
        except Exception:
            st_h2 = ""
        if moved_from and st_h2 == "STARTING":
            try:
                cur_cid2 = str(getattr(hol, "cluster_id", "") or "").strip()
            except Exception:
                cur_cid2 = ""
            try:
                cur_nodes2 = [str(n) for n in (getattr(hol, "nodes", []) or [])]
                cur_nodes2 = [n for n in cur_nodes2 if n]
            except Exception:
                cur_nodes2 = []

            if cur_cid2 and cur_cid2 != hol_pin:
                # 이전 pin(=cur_cid2)에 남아있는 예약/상태를 QUEUED로 수렴 + SSOT release
                try:
                    release_nodes_for_job_locked(job_id=str(hol_jid), cluster_id=str(cur_cid2))
                except Exception:
                    pass
                try:
                    hol.launch_inflight = False
                    hol.launching_since_ts = 0.0
                except Exception:
                    pass
                try:
                    hol.launch_dispatched = False
                    hol.launch_dispatched_ts = 0.0
                except Exception:
                    pass
                try:
                    hol.status = "QUEUED"
                except Exception:
                    pass
                try:
                    hol.cluster_id = None
                    hol.nodes = []
                    hol.g_cur = 0
                    hol.g_alloc = 0
                    hol.world_size = 0
                    hol.actual_g = 0
                except Exception:
                    pass

                if rl:
                    try:
                        rl.queue_event(
                            event="gang_pin_move_cleared_old_reserve",
                            job_id=str(hol_jid),
                            queue_len=-1,
                            qlen_clusterq=-1,
                            note=f"cleared_old_starting_reserve old_cluster={cur_cid2} old_nodes={cur_nodes2} new_pin={hol_pin}",
                            ts=float(time.time()),
                        )
                    except Exception:
                        pass

                # free 재계산
                pin_free = _free_count(hol_pin)

        # ---- 목적지 cluster가 blocked면 아무것도 하지 말기 ----
        if _cluster_blocked(hol_pin):
            return

        # 1) LAUNCH 경로: free>=4
        #    중복 reserve 방지: 이미 STARTING 예약 상태면 reserve 다시 치지 않는다.
        if pin_free >= 4:
            # 이미 STARTING인데 nodes/cluster가 잡혀 있으면 그걸 그대로 사용(멱등)
            try:
                st_h = str(getattr(hol, "status", "") or "").upper()
            except Exception:
                st_h = ""

            reuse_nodes = None
            if st_h == "STARTING":
                try:
                    cur_cid = str(getattr(hol, "cluster_id", "") or "").strip()
                except Exception:
                    cur_cid = ""
                try:
                    cur_nodes = [str(n) for n in (getattr(hol, "nodes", []) or [])]
                    cur_nodes = [n for n in cur_nodes if n]
                except Exception:
                    cur_nodes = []

                # 같은 pin에서 4개가 이미 잡혀있고, SSOT에서도 hol이 owner면 그대로 사용
                if cur_cid == hol_pin and len(cur_nodes) == 4 and _nodes_owned_by(str(hol_jid), cur_nodes):
                    reuse_nodes = list(cur_nodes)

            if reuse_nodes is not None:
                nodes_use = list(reuse_nodes)
            else:
                nodes_use = _free_nodes(hol_pin)[:4]
                if len(nodes_use) < 4:
                    return

                # reserve (최초 1회만)
                try:
                    reserve_nodes(job_id=str(hol_jid), cluster_id=str(hol_pin), nodes=list(nodes_use))
                except Exception:
                    return

            # launch 멱등 토큰 (디버그/증거용)
            try:
                lrid = (getattr(hol, "launch_request_id", None) or "").strip()
            except Exception:
                lrid = ""
            if not lrid:
                lrid = uuid.uuid4().hex[:12]
                try:
                    hol.launch_request_id = lrid
                except Exception:
                    pass

            # SSOT world_size / alloc / target 통일
            try:
                hol.g_target = 4
            except Exception:
                pass
            try:
                hol.cluster_id = str(hol_pin)
                hol.nodes = list(nodes_use)
                hol.g_cur = 4
                hol.g_alloc = 4
                hol.world_size = 4
                hol.launch_inflight = True
                hol.launching_since_ts = float(now_ts)
                hol.status = "STARTING"  # ✅ reserve 직후 STARTING으로 올려서 SSOT/표시 일치
            except Exception:
                pass

            launch_plan = (str(hol_jid), str(hol_pin), list(nodes_use), str(lrid))

            if rl:
                try:
                    rl.queue_event(
                        event="gang_launch_plan_committed",
                        job_id=str(hol_jid),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"pin={hol_pin} nodes={nodes_use} launch_request_id={lrid}",
                        ts=float(time.time()),
                    )
                except Exception:
                    pass

        # 2) PREEMPT 경로: pin_free 1~3
        #    이번 tick엔 launch 금지(반드시 다음 tick에서 free==4 확인 후 launch)
        else:
            if 0 < pin_free < 4:
                need = 4 - int(pin_free)

                if int(pin_free) == 3 and int(need) == 1:
                    bf_victims = _find_hol_backfill_victims_locked(st, hol_jid, hol_pin)
                    if bf_victims:
                        # tick에서도 바로 stop 발행하도록 preempt_plan을 만들어서 LOCK 밖에서 실행
                        preempt_plan = ("HOL_BACKFILL_ONLY", list(bf_victims), str(hol_jid), 1)

                        if rl:
                            rl.queue_event(
                                event="hol_backfill_preempt_planned_in_tick",
                                job_id=str(hol_jid),
                                queue_len=-1,
                                qlen_clusterq=-1,
                                note=f"pin={hol_pin} pin_free={pin_free} need=1 victims={bf_victims}",
                                ts=float(now_ts),
                            )

                # ✅ 1) 가장 신뢰 가능한 victim: "이미 pin에 떠있는 hol_backfill(victim)" 들
                bf_victims = _find_hol_backfill_victims_locked(st, hol_jid, hol_pin)

                # need 만큼만 우선 채움
                vlist: List[str] = []
                for vid in (bf_victims or []):
                    if len(vlist) >= int(need):
                        break
                    if vid and str(vid) != str(hol_jid):
                        vlist.append(str(vid))

                # ✅ 2) 부족하면 그때만 _pick_victims_for_gang_locked로 보조
                dbg = {}
                if len(vlist) < int(need):
                    victims2, dbg = _pick_victims_for_gang_locked(
                        st,
                        hol_jid=str(hol_jid),
                        pin=str(hol_pin),
                        need_g=int(need),
                        allow_preempt_normal=False,
                        allow_preempt_starting=True,
                    )

                    for v in (victims2 or []):
                        if len(vlist) >= int(need):
                            break
                        if isinstance(v, dict):
                            vid = str(v.get("job_id") or "").strip()
                        else:
                            vid = str(v).strip()
                        if vid and (vid not in vlist) and (vid != str(hol_jid)):
                            vlist.append(vid)

                # ✅ 3) 이제 vlist가 있으면 무조건 reclaim 무장(armed)
                if vlist:
                    try:
                        st.hol_reclaim_active = True
                    except Exception:
                        pass
                    try:
                        st.hol_reclaim_pin = str(hol_pin)
                    except Exception:
                        pass
                    try:
                        st.hol_reclaim_hol_job_id = str(hol_jid)
                    except Exception:
                        pass
                    try:
                        st.hol_reclaim_need = int(need)
                    except Exception:
                        pass
                    try:
                        st.hol_reclaim_until_ts = float(now_ts + 60.0)
                    except Exception:
                        pass
                    try:
                        st.hol_reclaim_reason = f"HOL_GANG_PREEMPT_RECLAIM_PENDING hol={str(hol_jid).upper()} pin={str(hol_pin).upper()}"
                    except Exception:
                        pass
                    try:
                        st.hol_reclaim_victims = [{"job_id": str(x), "g": 1} for x in vlist]
                    except Exception:
                        pass

                    if rl:
                        try:
                            rl.queue_event(
                                event="hol_reclaim_armed_in_tick",
                                job_id=str(hol_jid),
                                queue_len=-1,
                                qlen_clusterq=-1,
                                note=f"pin={hol_pin} pin_free={pin_free} need={need} bf_victims={bf_victims} final_vlist={vlist} dbg={_safe_json(dbg)}",
                                ts=float(now_ts),
                            )
                        except Exception:
                            pass
                else:
                    # victim이 하나도 없으면(=pin_free가 1~3인데도 잡이 없다는 뜻) 그냥 대기
                    return

                # completion-gated면 tick에서 stop 안 쏨
                preempt_plan = None


            else:
                return

    # LOCK 밖
    if preempt_plan:
        _tag, victims, holjid, _need = preempt_plan
        for vjid in victims:
            _stop_job_and_release_nodes(
                job_id=str(vjid),
                cluster_id=str(hol_pin),
                reason=f"HOL_GANG_PREEMPT_TICK hol={str(holjid)} pin={str(hol_pin)}",
                checkpoint=True,
                requeue_kind="HOL_BACKFILL",
            )
        _TICK_PENDING = True
        return

    if launch_plan:
        if len(launch_plan) == 5:
            jid0, cid0, nodes0, lrid, is_bf = launch_plan
        else:
            jid0, cid0, nodes0, lrid = launch_plan
            is_bf = False

        # jobx 조회는 LOCK으로 안전하게
        with STATE_LOCK:
            stx = get_global_state()
            jobx = (getattr(stx, "jobs", {}) or {}).get(str(jid0))

        if rl:
            try:
                rl.queue_event(
                    event="gang_launch_called",
                    job_id=str(jid0),
                    queue_len=-1,
                    qlen_clusterq=-1,
                    note=f"pin={cid0} nodes={nodes0} launch_request_id={lrid}",
                    ts=float(time.time()),
                )
            except Exception:
                pass

        stj = str(getattr(jobx, "status", "") or "").upper()
        if stj in ("STOPPED", "PREEMPTING", "PREEMPTED"):
            return {"ok": False, "reason": f"deny_launch_status_{stj}"}

        if bool(getattr(jobx, "preempt_inflight", False)) or bool(getattr(jobx, "preempt_issued", False)):
            return {"ok": False, "reason": "deny_launch_preempt_inflight"}

        out = None
        try:
            if rl:
                try:
                    rl.queue_event(
                        event="gang_launch_rpc_before",
                        job_id=str(jid0),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"calling _start_job_on_cluster pin={cid0} nodes={nodes0} launch_request_id={lrid}",
                        ts=float(time.time()),
                    )
                except Exception:
                    pass

            out = _start_job_on_cluster(
                job=jobx,
                cluster_id=str(cid0),
                node_names=list(nodes0),
                g_use=len(nodes0),
                is_backfill=bool(is_bf),
            )

            if rl:
                try:
                    rl.queue_event(
                        event="gang_launch_rpc_after",
                        job_id=str(jid0),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=f"returned out={_safe_json(out)}",
                        ts=float(time.time()),
                    )
                except Exception:
                    pass

        except Exception as e:
            out = {"ok": False, "reason": "start_exception", "detail": str(e)}

        try:
            ok = bool(out.get("ok")) if isinstance(out, dict) else False
        except Exception:
            ok = False

        # ✅ NEW: 409(node_draining) 신호를 state에 반영하기 위해 reason/drain_until 추출
        fail_reason = ""
        drain_until = 0.0
        try:
            if isinstance(out, dict):
                fail_reason = str(out.get("reason", "") or "").strip().lower()
                drain_until = float(out.get("drain_until", 0.0) or 0.0)
        except Exception:
            fail_reason = ""
            drain_until = 0.0
        if drain_until <= 0.0:
            drain_until = float(now_ts + 2.0)  # fallback

        # --- classify failure ---
        outd = out if isinstance(out, dict) else {}
        http_status = int(outd.get("http_status", 0) or outd.get("status_code", 0) or 0)
        r_reason = str(outd.get("reason", "") or "").lower()

        is_draining_409 = (http_status == 409) and ("drain" in r_reason or "node_draining" in r_reason)

        # drain_until 추출(가능한 한 많이)
        drain_until = 0.0
        try:
            drain_until = float(outd.get("drain_until", 0.0) or 0.0)
        except Exception:
            drain_until = 0.0

        # FIX: 성공했을 때만 STARTING 전이
        with STATE_LOCK:
            st2 = get_global_state()
            hol2 = (getattr(st2, "jobs", {}) or {}).get(str(jid0))
            if hol2 is not None:
                if ok:
                    try:
                        hol2.status = "STARTING"
                    except Exception:
                        pass
                else:
                    # 실패면 inflight 정리 + 상태 원복(멱등)
                    try:
                        hol2.launch_inflight = False
                        hol2.launching_since_ts = 0.0
                    except Exception:
                        pass
                    try:
                        hol2.status = "QUEUED"
                    except Exception:
                        pass
                    try:
                        hol2.cluster_id = None
                        hol2.nodes = []
                        hol2.g_cur = 0
                        hol2.g_alloc = 0
                        hol2.world_size = 0
                    except Exception:
                        pass

                    # ✅ NEW: 409(node_draining)이면 "언제 다시 쏠지"를 저장해서 스핀 방지 + 자동 재시도
                    try:
                        o_reason = str(out.get("reason", "") or "").lower() if isinstance(out, dict) else ""
                        o_http = int(out.get("http_status", 0) or out.get("status_code", 0) or 0) if isinstance(out, dict) else 0
                        o_du = float(out.get("drain_until", 0.0) or 0.0) if isinstance(out, dict) else 0.0
                    except Exception:
                        o_reason, o_http, o_du = "", 0, 0.0

                    if (o_http == 409) and ("node_draining" in o_reason or "draining" in o_reason):
                        # drain_until을 못 받으면 최소 2초는 미루기(스핀 방지)
                        if o_du <= 0.0:
                            o_du = float(now_ts + 2.0)
                        try:
                            hol2.next_launch_earliest_ts = float(max(
                                float(getattr(hol2, "next_launch_earliest_ts", 0.0) or 0.0),
                                float(o_du + 0.1),
                            ))
                        except Exception:
                            pass


        # ✅ release는 "진짜 실패"에서만 (409는 release 금지)
        if (not ok) and (not is_draining_409):
            try:
                release_nodes_for_job(job_id=str(jid0), cluster_id=str(cid0))
            except Exception:
                pass


            if rl:
                try:
                    ev = "gang_launch_deferred_node_draining" if str(fail_reason) == "node_draining" else "gang_launch_failed_rollback"
                    rl.queue_event(
                        event=ev,
                        job_id=str(jid0),
                        queue_len=-1,
                        qlen_clusterq=-1,
                        note=(
                            f"pin={cid0} nodes={nodes0} drain_until={drain_until} out={_safe_json(out)}"
                            if str(fail_reason) == "node_draining"
                            else f"rollback_after_start_failed pin={cid0} nodes={nodes0} out={_safe_json(out)}"
                        ),
                        ts=float(time.time()),
                    )
                except Exception:
                    pass

        return


BACKFILL_SLICE_SEC = float(os.getenv("OURS_BACKFILL_SLICE_SEC", "30"))
HOL_PREEMPT_AFTER_SEC = float(os.getenv("OURS_HOL_PREEMPT_AFTER_SEC", "10"))
MAX_PREEMPT_PER_JOB = int(os.getenv("OURS_MAX_PREEMPT_PER_JOB", "3"))

def _force_converge_preempt_inflight_locked(state: Any, now_ts: float) -> int:
    jobs = getattr(state, "jobs", {}) or {}
    node_owner = getattr(state, "node_owner", {}) or {}

    converged = 0

    # drain 테이블은 있으면 유지
    drains = getattr(state, "drain_until_by_cluster", {}) or {}

    def _is_terminal(stt: str) -> bool:
        s = (stt or "").upper()
        return s in ("FINISHED", "FAILED", "CANCELLED", "COMPLETED", "PREEMPTED")

    def _dedup_str_list(xs):
        out = []
        seen = set()
        for x in xs or []:
            sx = str(x)
            if not sx or sx in seen:
                continue
            seen.add(sx)
            out.append(sx)
        return out

    for jid, jr in list(jobs.items()):
        jid = str(jid)
        if jr is None:
            continue

        # preempt inflight만 대상
        try:
            inflight = bool(getattr(jr, "preempt_inflight", False))
        except Exception:
            inflight = False
        if not inflight:
            continue

        # 데드라인(없으면 스킵)
        try:
            ddl = float(getattr(jr, "force_release_deadline_ts", 0.0) or 0.0)
        except Exception:
            ddl = 0.0
        if ddl <= 0.0 or now_ts < ddl:
            continue

        # 상태 확인
        try:
            stt = str(getattr(jr, "status", "") or "").upper()
        except Exception:
            stt = ""
        terminal = _is_terminal(stt)

        # pending 정보(없으면 현재 값 fallback)
        try:
            prev_kind = str(
                getattr(jr, "pending_prev_queue_kind", None)
                or getattr(jr, "queue_kind", None)
                or "HOME"
            )
        except Exception:
            prev_kind = "HOME"
        prev_kind_u = prev_kind.upper() if prev_kind else "HOME"
        if prev_kind_u not in ("HOME", "ADMITTED"):
            prev_kind_u = "HOME"

        try:
            prev_qcid = str(
                getattr(jr, "pending_prev_queue_cluster_id", None)
                or getattr(jr, "queue_cluster_id", None)
                or getattr(jr, "admitted_cluster_id", None)
                or getattr(jr, "home_cluster_id", None)
                or ""
            ).strip()
        except Exception:
            prev_qcid = ""

        try:
            nodes = list(getattr(jr, "pending_prev_nodes", None) or getattr(jr, "nodes", None) or [])
        except Exception:
            nodes = []
        nodes = _dedup_str_list(nodes)

        # 1) node_owner 강제 해제 (SSOT)
        if nodes:
            for n in nodes:
                try:
                    if str(node_owner.get(n)) == jid:
                        node_owner[n] = None
                except Exception:
                    pass

        # 2) release_nodes_for_job / fallback (nodes=None 금지)
        if nodes:
            try:
                release_nodes_for_job(state, jid, list(nodes))
            except Exception:
                try:
                    _ss_release_nodes_for_job_locked(state, job_id=str(jid), nodes=list(nodes))  # type: ignore
                except Exception:
                    pass

        # 3) job 상태 수렴 (inflight 제거 + 점유정보 제거)
        try:
            jr.preempt_inflight = False
        except Exception:
            pass
        try:
            jr.preempting_since_ts = 0.0
        except Exception:
            pass
        try:
            jr.last_preempt_try_ts = float(now_ts)
        except Exception:
            pass
        try:
            jr.launch_inflight = False
            jr.launching_since_ts = 0.0
        except Exception:
            pass

        # cluster/nodes 점유 정보 제거
        try:
            jr.cluster_id = None
            jr.nodes = []
            jr.g_cur = 0
        except Exception:
            pass

        # 3.5) A안: backfill/hol_backfill 태그는 preempt 수렴 시 제거
        try:
            jr.is_backfill = False
        except Exception:
            pass
        try:
            jr.is_hol_backfill = False
        except Exception:
            pass
        for k in ("hol_backfill_for_job_id", "hol_backfill_pin_cluster", "backfill_for_hol", "backfill_pin_cluster"):
            try:
                setattr(jr, k, None)
            except Exception:
                pass

        # 4) 재큐잉(terminal이 아니면): 메타 오염 금지
        if not terminal:
            try:
                jr.status = "QUEUED"
            except Exception:
                pass

            # queue_kind/queue_cluster_id만 “가능하면” 복원 (home/admitted/user_pinned는 건드리지 않음)
            try:
                jr.queue_kind = prev_kind_u
            except Exception:
                pass
            if prev_qcid:
                try:
                    jr.queue_cluster_id = str(prev_qcid)
                except Exception:
                    pass

            # 너무 빠른 재런치 방지(짧게)
            try:
                jr.blocked_until = float(max(float(getattr(jr, "blocked_until", 0.0) or 0.0), now_ts + 0.5))
                jr.launch_cooldown_until_ts = float(max(float(getattr(jr, "launch_cooldown_until_ts", 0.0) or 0.0), now_ts + 0.5))
            except Exception:
                pass

            # global_queue 복구: 정렬 삽입 함수 있으면 사용, 없으면 중복 방지 append
            try:
                gq = getattr(state, "global_queue", None)
                if isinstance(gq, list):
                    if jid not in [str(x) for x in gq]:
                        try:
                            _global_queue_insert_sorted_locked(state, str(jid))  # 있으면 이게 정답
                        except Exception:
                            gq.append(str(jid))
            except Exception:
                pass

            # (가능하면) single-queue invariant 회복
            try:
                if prev_qcid:
                    ensure_job_single_queue_locked(state, str(jid), str(prev_qcid))
            except Exception:
                pass

        # 5) drain 반영(가능하면)
        try:
            du = float(getattr(jr, "pending_drain_until", 0.0) or 0.0)
            if du > 0.0 and prev_qcid:
                drains[str(prev_qcid)] = float(max(float(drains.get(str(prev_qcid), 0.0) or 0.0), du))
        except Exception:
            pass

        # 6) pending 필드 정리(다음 preempt에 오염 방지)
        for k in (
            "pending_prev_nodes",
            "pending_prev_queue_kind",
            "pending_prev_queue_cluster_id",
            "pending_requeue_kind",
            "pending_drain_until",
            "force_release_deadline_ts",
        ):
            try:
                setattr(jr, k, None)
            except Exception:
                pass
        try:
            jr.force_release_deadline_ts = 0.0
        except Exception:
            pass

        converged += 1

    # SSOT 반영
    try:
        state.node_owner = node_owner
    except Exception:
        pass
    try:
        state.drain_until_by_cluster = drains
    except Exception:
        pass

    return converged

def _schedule_once() -> None:
    now_ts = float(time.time())

    # 0) state snapshot + compact/feed (최소만 LOCK)
    with STATE_LOCK:
        st = get_global_state()

        try:
            _global_queue_compact_locked(st)
        except Exception:
            pass
        try:
            _feed_cluster_queues_from_home_mobile_locked(st)
        except Exception:
            pass

        # cluster 목록 snapshot
        cluster_ids = list((getattr(st, "clusters", {}) or {}).keys())

    # 1) HoL dispatch 먼저 (global_tick 내부에서 STATE_LOCK을 잡음)
    try:
        _global_tick_once()
    except Exception:
        logger.exception("[tick] global_tick_once failed")

    # 1.5) PREEMPT 강제 수렴은 global_tick 직후에 수행 (ACK 누락/꼬임 회수)
    #      - "멈췄는데 살아있음"을 여기서 끝냄
    try:
        with STATE_LOCK:
            st2 = get_global_state()
            _force_converge_preempt_inflight_locked(st2, now_ts=float(time.time()))
    except Exception:
        logger.exception("[tick] force_converge_preempt_inflight failed")

    # 2) backfill (cluster tick) 마지막
    for cid in cluster_ids:
        try:
            _cluster_tick_once(str(cid))
        except Exception:
            logger.exception("[tick] cluster_backfill failed cid=%s", cid)

def stop_scheduler_loop() -> None:
    global _TICK_THREAD
    _TICK_STOP.set()
    _TICK_EVENT.set()
    t = _TICK_THREAD
    _TICK_THREAD = None
    if t is not None and t.is_alive():
        t.join(timeout=2.0)
    logger.info("[tick] scheduler loop stopped")

def flush_cluster_telemetry_csv() -> None:
    try:
        st = get_global_state()
    except Exception:
        return

    buf = getattr(st, "cluster_telemetry_buffer", None)
    if not isinstance(buf, list) or not buf:
        return

    csv_path = getattr(st, "cluster_telemetry_csv_path", None)
    if not csv_path:
        # 기본 경로 (없으면 생성)
        try:
            run_id = getattr(st, "run_id", "default")
        except Exception:
            run_id = "default"
        csv_path = f"logs/cluster_telemetry_{run_id}.csv"
        try:
            st.cluster_telemetry_csv_path = csv_path
        except Exception:
            pass

    try:
        with STATE_LOCK:
            rows = list(buf)
            buf.clear()
    except Exception:
        return

    if not rows:
        return

    try:
        import os
        import csv

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = None
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if writer is None:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                writer.writerow(row)

    except Exception as e:
        try:
            logger.exception("[telemetry] failed to flush cluster telemetry csv")
        except Exception:
            pass

def _tick_thread_main() -> None:
    global _TICK_PENDING

    next_deadline = time.time() + float(TICK_PERIOD_SEC)

    while True:
        if _TICK_STOP.is_set():
            logger.info("[tick] stop requested; exiting tick thread")
            return

        now = time.time()

        timeout = 0.0 if _TICK_PENDING else max(0.0, next_deadline - now)

        _TICK_WAKE_EVENT.wait(timeout=timeout)
        _TICK_WAKE_EVENT.clear()

        if _TICK_STOP.is_set():
            logger.info("[tick] stop requested; exiting tick thread")
            return

        now = time.time()
        do_tick = _TICK_PENDING or (now >= next_deadline)
        if not do_tick:
            continue

        next_deadline = time.time() + float(TICK_PERIOD_SEC)

        if not _TICK_RUN_LOCK.acquire(blocking=False):
            _TICK_PENDING = True
            continue

        try:
            _TICK_PENDING = False
            _schedule_once()
            flush_cluster_telemetry_csv()
        except Exception:
            logger.exception("[tick] schedule_once failed")
        finally:
            _TICK_RUN_LOCK.release()

        if _TICK_PENDING:
            _TICK_WAKE_EVENT.set()

def report_job_completed_core(rep: JobCompleteReport) -> Dict[str, Any]:
    global _TICK_PENDING
    import time

    job_id = str(getattr(rep, "job_id", None) or "").strip()
    if not job_id:
        return {"ok": False, "reason": "missing job_id"}

    raw_status_u = str(getattr(rep, "status", "") or "").strip().upper()
    raw_reason_u = str(getattr(rep, "reason", "") or "").strip().upper()
    now_ts = float(time.time())

    # exit_code는 정책 판단에 중요 (특히 1, 15)
    try:
        rep_exit_code = int(getattr(rep, "exit_code", 0) or 0)
    except Exception:
        rep_exit_code = 0

    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    # ---- 로그/메트릭 스냅샷(락 밖에서 씀) ----
    snap_cluster = ""
    snap_model = ""
    snap_dataset = ""
    snap_ws = 0
    snap_submit_ts = None
    snap_start_ts = None
    snap_end_ts = None
    snap_final_acc = None
    was_terminal = False
    did_requeue = False
    snap_note = ""

    # -------------------------
    # HOL reclaim: completion 이후 preempt 발행용(LOCK 밖에서 수행)
    # -------------------------
    preempt_actions: List[Tuple[str, str, int]] = []  # (victim_job_id, reason, g)
    preempt_pin: str = ""
    preempt_issued_g_total: int = 0  # ✅ completion에서 실제로 '계획'한 reclaim g 합

    def _term_set() -> set:
        try:
            return set(_TERMINAL)
        except Exception:
            return {"FINISHED", "FAILED", "CANCELLED", "COMPLETED", "PREEMPTED"}

    def _get(jr0: Any, k: str, d=None):
        if jr0 is None:
            return d
        if isinstance(jr0, dict):
            return jr0.get(k, d)
        return getattr(jr0, k, d)

    def _set(jr0: Any, k: str, v: Any) -> None:
        if jr0 is None:
            return
        if isinstance(jr0, dict):
            jr0[k] = v
        else:
            setattr(jr0, k, v)

    def _pick_final_accuracy(rep0: Any, jr0: Any) -> Optional[float]:
        v = getattr(rep0, "final_accuracy", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        for k in ("final_accuracy", "current_accuracy", "current_acc", "latest_accuracy", "last_accuracy"):
            vv = _get(jr0, k, None)
            if vv is None:
                continue
            try:
                return float(vv)
            except Exception:
                continue
        return None

    def _is_preempt_completion(status_u: str, reason_u: str, rep0: Any, jr0: Any) -> bool:
        try:
            ec = int(getattr(rep0, "exit_code", 0) or 0)
        except Exception:
            ec = 0
        if ec == 15:
            return True

        if status_u == "PREEMPTED":
            return True
        if status_u == "STOPPED":
            return True
        if "PREEMPT" in reason_u or "RESIZE" in reason_u or "LAUNCH_ROLLBACK" in reason_u:
            return True
        try:
            if bool(_get(jr0, "preempt_inflight", False)):
                return True
        except Exception:
            pass
        return False

    # ---- home queue helpers (락 안에서만 사용) ----
    def _ensure_home_queue_obj(st0: Any, cid: str):
        hqs0 = getattr(st0, "home_cluster_queues", None)
        if not isinstance(hqs0, dict):
            hqs0 = {}
            try:
                st0.home_cluster_queues = hqs0
            except Exception:
                pass

        q = hqs0.get(cid)
        if q is None:
            q = []
            hqs0[cid] = q
        return q, hqs0

    def _qitem_job_id(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            try:
                return str(x.get("job_id") or x.get("id") or "")
            except Exception:
                return ""
        if hasattr(x, "job_id"):
            try:
                return str(getattr(x, "job_id") or "")
            except Exception:
                return ""
        try:
            return str(x)
        except Exception:
            return ""

    def _home_queue_list(q: Any) -> List[Any]:
        if q is None:
            return []
        if isinstance(q, list):
            return list(q)
        try:
            return list(getattr(q, "_jobs", []) or [])
        except Exception:
            return []

    def _home_queue_set(q: Any, items: List[Any]) -> None:
        if q is None:
            return
        if isinstance(q, list):
            q[:] = list(items)
            return
        try:
            setattr(q, "_jobs", list(items))
        except Exception:
            pass

    def _home_queue_append_fifo(q: Any, jid: str) -> None:
        items = _home_queue_list(q)
        if any(_qitem_job_id(it) == str(jid) for it in items):
            return
        items.append(str(jid))
        _home_queue_set(q, items)
        try:
            setattr(q, "eta_dirty", True)
        except Exception:
            pass

    # -------------------------
    # (NEW) HOL reclaim completion-gated helpers (LOCK 안에서만 사용)
    # -------------------------
    def _as_str(x: Any) -> str:
        try:
            return str(x)
        except Exception:
            return ""

    def _status_u(jr0: Any) -> str:
        return _as_str(_get(jr0, "status", "") or "").upper()

    def _is_active_for_reclaim(jr0: Any, st0: Any, pin: str, jid: str) -> bool:
        su = _status_u(jr0)
        if su in ("RUNNING", "STARTING", "LAUNCHING", "DISPATCHED"):
            return True
        # 상태가 이상해도 SSOT(node_owner)에 잡혀 있으면 “실제로 돌고 있음”으로 간주
        try:
            return _job_runs_in_cluster_by_owner(st0, pin, jid)
        except Exception:
            return False

    def _is_queued(jr0: Any) -> bool:
        return _status_u(jr0) == "QUEUED"

    def _safe_to_preempt(jr0: Any) -> bool:
        if jr0 is None:
            return False
        try:
            # ✅ HOL-backfill victim은 STARTING(launch_inflight)이어도 reclaim 허용
            is_hbf = bool(_get(jr0, "is_hol_backfill", False))
            ok_only = bool(_get(jr0, "preempt_ok_only_by_hol", False))
            if bool(_get(jr0, "preempt_inflight", False)):
                return False

            if bool(_get(jr0, "launch_inflight", False)):
                if not (is_hbf and ok_only):
                    return False
        except Exception:
            pass
        return True

    def _victim_gcur(jr0: Any) -> int:
        for k in ("g_cur", "world_size", "g", "gpus", "g_alloc"):
            try:
                v = _get(jr0, k, None)
                if v is None:
                    continue
                iv = int(v)
                if iv > 0:
                    return iv
            except Exception:
                continue
        try:
            nodes0 = _get(jr0, "nodes", None)
            if isinstance(nodes0, list) and len(nodes0) > 0:
                return int(len(nodes0))
        except Exception:
            pass
        return 1

    def _cluster_node_owner_map(st0: Any, cid: str) -> Dict[str, Any]:
        clusters0 = getattr(st0, "clusters", {}) or {}
        cr = clusters0.get(str(cid))
        if cr is None:
            return {}
        try:
            no = getattr(cr, "node_owner", None)
        except Exception:
            no = None
        if isinstance(no, dict):
            return no
        if isinstance(cr, dict) and isinstance(cr.get("node_owner"), dict):
            return cr.get("node_owner") or {}
        return {}

    def _job_runs_in_cluster_by_owner(st0: Any, cid: str, jid0: str) -> bool:
        no = _cluster_node_owner_map(st0, cid)
        jidn = str(jid0).strip().lower()
        for _, owner in (no or {}).items():
            if str(owner).strip().lower() == jidn:
                return True
        return False

    def _find_waiting_gang4_in_pin_locked(st0: Any, pin: str) -> Optional[str]:
        jobs0 = getattr(st0, "jobs", {}) or {}
        gq = getattr(st0, "global_queue", None)

        def _is_gang4(jr0: Any) -> bool:
            try:
                return int(_job_gang_required_g(jr0) or 0) == 4
            except Exception:
                return False

        def _pin_of(jr0: Any) -> str:
            return (
                str(_get(jr0, "pinned_cluster", "") or "").strip()
                or str(_get(jr0, "pinned_cluster_id", "") or "").strip()
                or str(_get(jr0, "admitted_cluster_id", "") or "").strip()
                or str(_get(jr0, "home_cluster_id", "") or "").strip()
                # ✅ 핵심: pin이 비어있는 경우 allowed_cluster로도 판단
                or str(_get(jr0, "allowed_cluster", "") or "").strip()
                or str(_get(jr0, "preferred_cluster", "") or "").strip()
            )

        def _job_id(x: Any) -> str:
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

        if isinstance(gq, list):
            for item in gq:
                jid = _job_id(item)
                if not jid:
                    continue
                jr0 = jobs0.get(jid)
                if jr0 is None:
                    continue
                if not _is_queued(jr0):
                    continue
                if not _is_gang4(jr0):
                    continue
                if _pin_of(jr0) == str(pin):
                    return str(jid)

        # fallback: 전역큐 없으면 jobs에서라도
        for jid_key, jr0 in (jobs0 or {}).items():
            if jr0 is None:
                continue
            if not _is_queued(jr0):
                continue
            try:
                if int(_job_gang_required_g(jr0) or 0) != 4:
                    continue
            except Exception:
                continue
            if _pin_of(jr0) == str(pin):
                return str(_get(jr0, "job_id", "") or jid_key)

        return None

    def _free_n_in_cluster_locked(st0: Any, cid: str) -> int:
        try:
            free_nodes = list(_free_nodes_in_cluster(st0, cid) or [])
            return int(len(free_nodes))
        except Exception:
            # fallback: owner-map 기반 대충 계산(가능한 경우)
            try:
                clusters0 = getattr(st0, "clusters", {}) or {}
                cobj = clusters0.get(str(cid))
                if cobj is None:
                    return 0
                if isinstance(cobj, dict):
                    cnodes = [str(x) for x in (cobj.get("nodes") or [])]
                else:
                    cnodes = [str(x) for x in (getattr(cobj, "nodes", None) or [])]
                cnodes = [x for x in cnodes if x]
                no = _cluster_node_owner_map(st0, cid) or {}
                used = 0
                for n in cnodes:
                    if n in no:
                        used += 1
                return max(0, int(len(cnodes) - used))
            except Exception:
                return 0

    def _set_cluster_blocked_until_locked(st0: Any, cid: str, until_ts: float) -> None:
        try:
            cb = getattr(st0, "cluster_launch_blocked_until", None)
            if not isinstance(cb, dict):
                cb = {}
                setattr(st0, "cluster_launch_blocked_until", cb)
            prev = float(cb.get(str(cid), 0.0) or 0.0)
            cb[str(cid)] = float(max(prev, float(until_ts)))
        except Exception:
            pass

    # -------------------------
    # LOCK: state mutation + (NEW) completion 직후 reclaim preempt 계획 생성
    # -------------------------
    with STATE_LOCK:
        st = get_global_state()
        jobs = getattr(st, "jobs", {}) or {}
        cluster_queues = getattr(st, "cluster_queues", {}) or {}

        jr = jobs.get(job_id)

        # job record 없으면: idempotent
        if jr is None:
            try:
                _remove_job_from_all_cluster_queues_locked(cluster_queues, job_id)
            except Exception:
                pass
            try:
                _global_queue_compact_locked(st)
            except Exception:
                pass
            _TICK_PENDING = True
            return {"ok": True, "job_id": job_id, "missing_job_record": True}

        terminal_set = _term_set()
        cur_status_u = str(_get(jr, "status", "") or "").upper()
        was_terminal = cur_status_u in terminal_set

        is_preempt = _is_preempt_completion(raw_status_u, raw_reason_u, rep, jr)

        # ✅ 스냅샷은 "지우기 전에"
        snap_cluster = str(_get(jr, "cluster_id", "") or _get(jr, "cluster", "") or "")
        snap_model = str(_get(jr, "model", "") or "")
        snap_dataset = str(_get(jr, "dataset", "") or "")
        snap_ws = int(
            _get(jr, "world_size", 0)
            or _get(jr, "g_alloc", 0)
            or _get(jr, "g_target", 0)
            or _get(jr, "g_cur", 0)
            or 0
        )
        snap_submit_ts = _get(jr, "submit_ts", None)
        snap_start_ts = _get(jr, "start_ts", None)

        # ---- 1) node_owner 해제 (✅ SSOT 통일: pop) ----
        try:
            release_nodes_for_job_locked(job_id=str(job_id), cluster_id=(str(snap_cluster).strip() or None))
        except Exception:
            owner = getattr(st, "node_owner", None)
            nclu = getattr(st, "node_cluster", None)
            if isinstance(owner, dict):
                for n, o in list(owner.items()):
                    if o is not None and str(o) == str(job_id):
                        owner.pop(n, None)
                        if isinstance(nclu, dict):
                            nclu.pop(n, None)

        # =========================================================
        # ✅ (NEW) COMPLETION-GATED HOL RECLAIM:
        #   - "armed" 되어 있으면, 지금 completion 직후에만 victim preempt를 발행한다.
        #   - 단, 이 completion 자체가 PREEMPT completion(=exit 15/STOP 등)인 경우엔
        #     연쇄 preempt를 막기 위해 reclaim 발행을 스킵한다.
        # =========================================================
        try:
            reclaim_active = bool(getattr(st, "hol_reclaim_active", False))
            reclaim_pin0 = str(getattr(st, "hol_reclaim_pin", "") or "").strip()
            reclaim_until = float(getattr(st, "hol_reclaim_until_ts", 0.0) or 0.0)

            # ✅ completion이 발생한 클러스터에서만 reclaim 발행
            comp_cid = str(snap_cluster or "").strip()

            allow_reclaim_issue = bool(
                (not is_preempt)
                and reclaim_active
                and reclaim_pin0
                and (now_ts <= reclaim_until)
                and (comp_cid == str(reclaim_pin0))
            )

            if allow_reclaim_issue:
                # HOL이 지금도 pin에서 대기 중인지 확인(없으면 reclaim 해제)
                hol_in_pin = _find_waiting_gang4_in_pin_locked(st, reclaim_pin0)

                # fallback: tick에서 저장한 hol id를 신뢰
                if not hol_in_pin:
                    try:
                        hol0 = str(getattr(st, "hol_reclaim_hol_job_id", "") or "").strip()
                    except Exception:
                        hol0 = ""
                    if hol0:
                        hol_in_pin = hol0  # 최소한 reclaim을 꺼버리진 않음

                # 그래도 없으면 그때만 해제
                if not hol_in_pin:
                    try:
                        st.hol_reclaim_active = False
                    except Exception:
                        pass

                else:
                    free_n = _free_n_in_cluster_locked(st, reclaim_pin0)

                    # free가 4면 굳이 preempt 필요 없음(launch tick에서 바로 시작되게)
                    if free_n >= 4:
                        # reclaim은 더 이상 필요 없다고 표시해도 됨(선택)
                        try:
                            st.hol_reclaim_need = 0
                            st.hol_reclaim_active = False
                        except Exception:
                            pass
                        # tick 당장 돌게
                        _TICK_PENDING = True
                    else:
                        need = 0
                        try:
                            need = int(getattr(st, "hol_reclaim_need", 0) or 0)
                        except Exception:
                            need = 0

                        # 필요량은 현재 free 기준으로 재계산 (st 값이 오래됐을 수 있음)
                        need = int(max(0, 4 - int(free_n)))
                        if need <= 0:
                            _TICK_PENDING = True
                        else:
                            # ✅ completion에서는 항상 AFTER_COMPLETION reason을 새로 생성 (tick에서 저장한 PENDING/임의 reason 무시)
                            reason0 = str(getattr(st, "hol_reclaim_reason", "") or "").strip()
                            r0u = reason0.upper()

                            if (not reason0) or ("PENDING" in r0u) or ("RECLAIM_PENDING" in r0u) or ("HOL_GANG_PREEMPT" in r0u and "AFTER_COMPLETION" not in r0u):
                                reason = f"HOL_GANG_PREEMPT_AFTER_COMPLETION hol={str(hol_in_pin).upper()} pin={reclaim_pin0}"
                            else:
                                reason = reason0

                            # victims SSOT 읽기
                            vraw = getattr(st, "hol_reclaim_victims", None)
                            victims_list: List[Dict[str, Any]] = []
                            if isinstance(vraw, list):
                                for it in vraw:
                                    if isinstance(it, dict) and it.get("job_id"):
                                        victims_list.append({"job_id": str(it.get("job_id")), "g": int(it.get("g") or 1)})
                                    elif isinstance(it, str):
                                        victims_list.append({"job_id": str(it), "g": 1})

                            got = 0
                            issued_local: List[Tuple[str, str, int]] = []

                            for it in victims_list:
                                if got >= need:
                                    break
                                vjid = str(it.get("job_id") or "").strip()
                                if not vjid or vjid == str(job_id):
                                    continue
                                vjr = jobs.get(vjid)
                                if vjr is None:
                                    continue
                                if not _is_active_for_reclaim(vjr, st, reclaim_pin0, vjid):
                                    continue
                                if not _safe_to_preempt(vjr):
                                    continue
                                # pin에서 실제로 돌고 있는지 (node_owner 기반)
                                if not _job_runs_in_cluster_by_owner(st, reclaim_pin0, vjid):
                                    # 보조로 cluster_id도 확인
                                    if str(_get(vjr, "cluster_id", "") or "") != str(reclaim_pin0):
                                        continue

                                vg = int(it.get("g") or _victim_gcur(vjr) or 1)
                                vg = max(1, vg)

                                # ✅ 여기서 "발행 계획"만 만들고, preempt_inflight 먼저 박아 중복 방지
                                try:
                                    _set(vjr, "preempt_inflight", True)
                                    _set(vjr, "preempting_since_ts", float(now_ts))
                                except Exception:
                                    pass

                                # requeue는 pin으로 유도(체크포인트/수렴)
                                # ✅ reclaim victim은 "돌아갈 목적지"를 SSOT로 박아야 feeder가 흔들리지 않는다.
                                # - 기본: home/admitted/ckpt(있으면 ckpt 우선)
                                try:
                                    ckpt_cid_v = str(_get(vjr, "checkpoint_cluster_id", "") or "").strip()
                                except Exception:
                                    ckpt_cid_v = ""

                                try:
                                    home_v = str(_get(vjr, "home_cluster_id", "") or "").strip()
                                except Exception:
                                    home_v = ""

                                try:
                                    adm_v = str(_get(vjr, "admitted_cluster_id", "") or "").strip()
                                except Exception:
                                    adm_v = ""

                                target_v = ckpt_cid_v or home_v or adm_v or ""

                                # ✅ victim이 다시 떠돌지 않게 "돌아갈 곳"을 SSOT로 박는다.
                                try:
                                    ckpt = str(_get(vjr, "checkpoint_cluster_id", "") or "").strip()
                                except Exception:
                                    ckpt = ""
                                try:
                                    home = str(_get(vjr, "home_cluster_id", "") or "").strip()
                                except Exception:
                                    home = ""
                                try:
                                    adm = str(_get(vjr, "admitted_cluster_id", "") or "").strip()
                                except Exception:
                                    adm = ""

                                target = ckpt or home or adm
                                try:
                                    if target:
                                        _set(vjr, "requeue_target_cluster_id", target)   # ✅ 핵심
                                    _set(vjr, "requeue_front", False)
                                except Exception:
                                    pass

                                issued_local.append((vjid, reason, vg))
                                got += vg

                            if issued_local:
                                preempt_actions = list(issued_local)
                                preempt_pin = str(reclaim_pin0)
                                preempt_issued_g_total = int(got)  # ✅ 락 밖에서 disarm 판단에 씀

                                # ✅ (중요) 이번 completion에서 발행한 만큼 need를 먼저 깎아둔다 (중복 발사 방지)
                                try:
                                    prev_need0 = int(getattr(st, "hol_reclaim_need", 0) or 0)
                                except Exception:
                                    prev_need0 = 0
                                new_need0 = max(0, int(prev_need0) - int(got))
                                try:
                                    st.hol_reclaim_need = int(new_need0)

                                    # need==0이면 “추가 preempt 발행”은 중단, 대신 launch까지 backfill을 막아줌
                                    if new_need0 <= 0:
                                        st.hol_reclaim_active = True   # 유지 (launch 완료/hol 사라질 때까지)
                                        st.hol_reclaim_reason = str(reason)  # reason 정규화
                                    else:
                                        st.hol_reclaim_active = True
                                except Exception:
                                    pass

                                # backfill 방지 시간을 더 늘려야 함 (0.5초는 너무 짧음)
                                _set_cluster_blocked_until_locked(st, reclaim_pin0, float(now_ts + 3.0))


                                if rl:
                                    try:
                                        rl.queue_event(
                                            event="hol_reclaim_issue_after_completion",
                                            job_id=str(hol_in_pin),
                                            queue_len=-1,
                                            qlen_clusterq=-1,
                                            note=f"pin={reclaim_pin0} free={free_n} need={need} issued={[x[0] for x in issued_local]}",
                                            ts=float(now_ts),
                                        )
                                    except Exception:
                                        pass

                                _TICK_PENDING = True
                            else:
                                # 이번 completion에서 발행 못 했으면 armed 유지 + 다음 completion에서 또 시도
                                if rl:
                                    try:
                                        rl.queue_event(
                                            event="hol_reclaim_issue_after_completion_no_victim",
                                            job_id=str(hol_in_pin),
                                            queue_len=-1,
                                            qlen_clusterq=-1,
                                            note=f"pin={reclaim_pin0} free={free_n} need={need} victims={len(victims_list)}",
                                            ts=float(now_ts),
                                        )
                                    except Exception:
                                        pass
        except Exception:
            # reclaim 로직은 실패해도 completion 처리 자체는 계속 진행
            pass

        # ✅ pin이 아닌 completion이면 이번엔 reclaim 발행하지 않는다(armed는 유지)
        if reclaim_active and reclaim_pin0:
            comp_cid = str(snap_cluster or "").strip()
            if comp_cid and (comp_cid != str(reclaim_pin0)):
                if rl:
                    try:
                        rl.queue_event(
                            event="hol_reclaim_skip_non_pin_completion",
                            job_id=str(job_id),
                            queue_len=-1,
                            qlen_clusterq=-1,
                            note=f"comp_cid={comp_cid} reclaim_pin={reclaim_pin0} is_preempt={is_preempt} status={raw_status_u}",
                            ts=float(now_ts),
                        )
                    except Exception:
                        pass

        # ---- 2) rep -> jr 반영 ----
        end_ts = float(getattr(rep, "end_ts", None) or now_ts)
        _set(jr, "end_ts", end_ts)

        fa = _pick_final_accuracy(rep, jr)
        if fa is not None:
            _set(jr, "final_accuracy", float(fa))

        for k in ("exit_code", "reason", "final_loss", "run_id", "attempt", "status"):
            v = getattr(rep, k, None)
            if v is not None:
                _set(jr, k, v)

        # ✅ 점유 정리(항상)
        _set(jr, "nodes", [])
        _set(jr, "cluster_id", None)
        _set(jr, "g_cur", 0)
        _set(jr, "launch_inflight", False)
        _set(jr, "launching_since_ts", 0.0)

        # ---- 3) PREEMPT/RESIZE면: REQUEUE로 수렴 ----
        if is_preempt:
            _set(jr, "preempt_inflight", False)
            _set(jr, "preempting_since_ts", 0.0)

            # enqueue_ts는 최초 1회만 (공정성 SSOT)
            if _get(jr, "enqueue_ts", None) is None:
                _set(jr, "enqueue_ts", float(_get(jr, "submit_ts", None) or now_ts))
            _set(jr, "last_enqueue_ts", float(now_ts))
            _set(jr, "requeue_ts", float(now_ts))

            # requeue 목적지 결정
            prev_qcid = str(
                _get(jr, "pending_prev_queue_cluster_id", None)
                or _get(jr, "queue_cluster_id", None)
                or _get(jr, "home_cluster_id", None)
                or _get(jr, "admitted_cluster_id", None)
                or "UNKNOWN"
            )

            # locality 강제
            try:
                ckpt_cid = str(_get(jr, "checkpoint_cluster_id", "") or "").strip()
            except Exception:
                ckpt_cid = ""
            try:
                ckpt_local_only = bool(_get(jr, "checkpoint_local_only", False))
            except Exception:
                ckpt_local_only = False
            try:
                resume = _get(jr, "resume_from_checkpoint", None)
            except Exception:
                resume = None
            if resume is not None:
                try:
                    resume = str(resume).strip() or None
                except Exception:
                    resume = None

            must_stay_local = bool(ckpt_cid or ckpt_local_only or resume)
            if ckpt_cid:
                prev_qcid = ckpt_cid
            elif must_stay_local:
                prev_qcid = str(
                    _get(jr, "admitted_cluster_id", "") or _get(jr, "home_cluster_id", "") or prev_qcid or "UNKNOWN"
                )

            if not prev_qcid or prev_qcid == "UNKNOWN":
                prev_qcid = str(_get(jr, "home_cluster_id", "") or _get(jr, "admitted_cluster_id", "") or "UNKNOWN")

            if ckpt_cid:
                _set(jr, "home_cluster_id", str(ckpt_cid))

            _set(jr, "queue_kind", "HOME")
            _set(jr, "queue_cluster_id", prev_qcid)
            _set(jr, "status", "QUEUED")
            _clear_requeue_victim_tags_locked(jr)

            # ✅ feeder 흔들림 방지: 이번 requeue 목적지도 1회 박고, feeder가 옮기면 None으로 지우게 두면 됨
            try:
                _set(jr, "requeue_target_cluster_id", str(prev_qcid))
            except Exception:
                pass

            # 쿨다운
            try:
                bu0 = float(_get(jr, "blocked_until", 0.0) or 0.0)
                cd0 = float(_get(jr, "launch_cooldown_until_ts", 0.0) or 0.0)

                # ✅ HOL reclaim에 의해 잘린 victim이면, reclaim window 동안 재런치 금지
                rep_reason_u = str(getattr(rep, "reason", "") or "").upper()
                hol_cut = ("HOL_GANG_PREEMPT" in rep_reason_u)

                hold_until = now_ts + 0.5  # default
                if hol_cut:
                    try:
                        # reclaim window가 state에 있으면 그걸 우선
                        ru = float(getattr(st, "hol_reclaim_until_ts", 0.0) or 0.0)
                    except Exception:
                        ru = 0.0
                    # 최소 10초는 막고(너무 짧으면 루프), 있으면 reclaim_until까지 막기
                    hold_until = max(now_ts + 10.0, ru if ru > 0.0 else 0.0)

                    # (선택) victim을 pin에 다시 안 올리려면 “앞줄”도 끄는 게 안전
                    try:
                        _set(jr, "requeue_front", False)
                    except Exception:
                        pass

                _set(jr, "blocked_until", float(max(bu0, hold_until)))
                _set(jr, "launch_cooldown_until_ts", float(max(cd0, hold_until)))
            except Exception:
                pass


            # ✅ 핵심: 모든 큐에서 제거 → HOME에만 재삽입
            try:
                purge_job_from_all_queues_locked(st, str(job_id), purge_global=True, purge_cluster=True, purge_home=True)
            except Exception:
                pass

            # ✅ HOME 큐로 재삽입 (SSOT)
            try:
                target_home_cid = str(prev_qcid)
                hq, hqs_map = _ensure_home_queue_obj(st, target_home_cid)
                _home_queue_append_fifo(hq, job_id)
                try:
                    st.home_cluster_queues = hqs_map
                except Exception:
                    pass
            except Exception:
                # 최후 안전망
                try:
                    gq = getattr(st, "global_queue", None)
                    if not isinstance(gq, list):
                        st.global_queue = []
                        gq = st.global_queue
                    if isinstance(gq, list) and str(job_id) not in [str(x) for x in gq]:
                        gq.append(str(job_id))
                except Exception:
                    pass

            did_requeue = True
            snap_note = f"PREEMPT->REQUEUE kind=HOME qcid={prev_qcid} ckpt_cid={ckpt_cid or ''}"

        else:
            # ---- 4) 정상 완료/실패면: TERMINAL 처리 ----
            # ✅ exit_code=1은 "job_completed"로 취급 (FAILED 금지)
            # ✅ exit_code=0도 FINISHED
            if raw_status_u not in ("FINISHED", "FAILED", "CANCELLED", "COMPLETED"):
                if rep_exit_code in (0, 1):
                    raw_status_u = "FINISHED"
                else:
                    raw_status_u = "FAILED"

            term_status = raw_status_u if raw_status_u in ("FINISHED", "FAILED", "CANCELLED", "COMPLETED") else "COMPLETED"
            _set(jr, "status", term_status)

            # ✅ terminal은 어떤 큐에도 남으면 안 됨
            try:
                purge_job_from_all_queues_locked(st, str(job_id), purge_global=True, purge_cluster=True, purge_home=True)
            except Exception:
                pass

            # ✅ (추가) terminal job은 state에서 제거(유령 방지)
            try:
                for attr in (
                    "job_drain_deadline_by_job",
                    "job_blocked_until_by_job",
                    "job_launch_cooldown_by_job",
                    "job_last_seen_by_job",
                ):
                    mp = getattr(st, attr, None)
                    if isinstance(mp, dict):
                        mp.pop(str(job_id), None)

                jobs.pop(str(job_id), None)
            except Exception:
                pass

        _TICK_PENDING = True

        snap_end_ts = float(_get(jr, "end_ts", None) or now_ts)
        snap_final_acc = _get(jr, "final_accuracy", None)

    # ✅ (NEW) LOCK 밖: completion 직후 preempt 발행 (stop 방식만 사용)
    if preempt_actions:
        failed: List[str] = []
        results: List[Dict[str, Any]] = []

        for vjid, reason, _vg in preempt_actions:
            try:
                out = _stop_job_and_release_nodes(
                    job_id=str(vjid),
                    cluster_id=str(preempt_pin),
                    reason=str(reason),
                    checkpoint=True,
                    requeue_kind="HOL_BACKFILL",
                )
                okp = bool(out.get("ok")) if isinstance(out, dict) else True
                if not okp:
                    failed.append(str(vjid))
                results.append(out if isinstance(out, dict) else {"ok": True, "resp_raw": str(out)})
            except Exception as e:
                failed.append(str(vjid))
                results.append({
                    "ok": False,
                    "job_id": str(vjid),
                    "cluster_id": str(preempt_pin),
                    "reason": "exception",
                    "detail": str(e),
                })

        if rl:
            try:
                rl.queue_event(
                    event="hol_reclaim_preempt_issued",
                    job_id=str(getattr(get_global_state(), "hol_reclaim_hol_job_id", "") or ""),
                    queue_len=-1,
                    qlen_clusterq=-1,
                    note=f"pin={preempt_pin} issued={len(preempt_actions)} failed={failed} results={_safe_json(results)}",
                    ts=float(time.time()),
                )
            except Exception:
                pass

        # 실패한 victim만 inflight 롤백 + need 롤백
        if failed:
            with STATE_LOCK:
                st = get_global_state()
                jobs = getattr(st, "jobs", {}) or {}

                for vjid in failed:
                    vjr = jobs.get(str(vjid))
                    if vjr is not None:
                        try:
                            _set(vjr, "preempt_inflight", False)
                            _set(vjr, "preempting_since_ts", 0.0)
                        except Exception:
                            pass

                # ✅ (중요) need 롤백
                try:
                    cur_need = int(getattr(st, "hol_reclaim_need", 0) or 0)
                except Exception:
                    cur_need = 0
                try:
                    st.hol_reclaim_need = int(cur_need + int(preempt_issued_g_total or 0))
                    st.hol_reclaim_active = True
                except Exception:
                    pass


    # ---- logs / metrics (락 밖) ----
    if rl:
        try:
            if did_requeue:
                ev = "preempted_requeued"
                note = f"requeued duplicated={bool(was_terminal)} {snap_note}"
            else:
                st_u = raw_status_u if raw_status_u else "COMPLETED"
                ev = "completed" if st_u in ("FINISHED", "COMPLETED") else st_u.lower()
                note = f"status={st_u} duplicated={bool(was_terminal)} exit_code={rep_exit_code}"

            rl.job_event(
                event=ev,
                job_id=str(job_id),
                cluster=str(snap_cluster),
                world_size=int(snap_ws),
                note=str(note),
                metadata={
                    "model": snap_model,
                    "dataset": snap_dataset,
                    "submitted_ts": snap_submit_ts,
                    "started_ts": snap_start_ts,
                    "end_ts": snap_end_ts,
                    "exit_code": getattr(rep, "exit_code", None),
                    "reason": getattr(rep, "reason", None),
                    "final_accuracy": snap_final_acc,
                    "final_loss": getattr(rep, "final_loss", None),
                    "requeued": bool(did_requeue),
                },
                ts=float(snap_end_ts or now_ts),
            )
        except Exception:
            pass

        try:
            queued_sec = (float(snap_start_ts) - float(snap_submit_ts)) if (snap_submit_ts and snap_start_ts) else None
            jct_sec = (float(snap_end_ts) - float(snap_submit_ts)) if (snap_submit_ts and snap_end_ts) else None

            rl.job_metrics_upsert_row(
                job_id=str(job_id),
                cluster=str(snap_cluster),
                model=str(snap_model),
                dataset=str(snap_dataset),
                world_size=int(snap_ws),
                submitted_ts=float(snap_submit_ts) if snap_submit_ts else now_ts,
                started_ts=float(snap_start_ts) if snap_start_ts else None,
                end_ts=float(snap_end_ts) if snap_end_ts else now_ts,
                queued_sec=queued_sec,
                jct_sec=jct_sec,
                final_accuracy=float(snap_final_acc) if (snap_final_acc is not None) else None,
                status=("PREEMPTED" if did_requeue else str(raw_status_u or "COMPLETED").upper()),
            )
        except Exception:
            pass

    if did_requeue:
        return {"ok": True, "job_id": job_id, "status": "QUEUED", "requeued": True, "duplicated": bool(was_terminal)}
    else:
        term_status = raw_status_u if raw_status_u in ("FINISHED", "FAILED", "CANCELLED", "COMPLETED") else "COMPLETED"
        return {"ok": True, "job_id": job_id, "status": term_status, "requeued": False, "duplicated": bool(was_terminal)}

def update_cluster_telemetry_locked(
    state: Any,
    cluster_id: str,
    util_avg: float,
    power_sum: float,
    mem_used_sum: float,
    mem_total_sum: float,
    ts: float,
) -> None:
    if not hasattr(state, "cluster_telemetry") or not isinstance(getattr(state, "cluster_telemetry"), dict):
        state.cluster_telemetry = {}

    state.cluster_telemetry[str(cluster_id)] = {
        "ts": float(ts),
        # util_avg: worker gpu_util(0~100) 평균 그대로
        "util_avg": float(util_avg),
        # power_sum: cluster 내 GPU power 합(W)
        "power_sum": float(power_sum),
        "mem_used_sum": float(mem_used_sum),
        "mem_total_sum": float(mem_total_sum),
    }

def handle_telemetry(payload: TelemetryIn, request: Request) -> Dict[str, Any]:
    now = time.time()

    # pydantic v1/v2 호환 dict 변환
    if hasattr(payload, "model_dump"):
        p = payload.model_dump()
    else:
        p = payload.dict()

    # -------------------------
    # helpers
    # -------------------------
    def _to_float(x, default=0.0) -> float:
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _clamp(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _as_pct(v: float) -> float:
        if v <= 1.5:   # 0~1 형태일 가능성
            return _clamp(v * 100.0, 0.0, 100.0)
        return _clamp(v, 0.0, 100.0)

    def _get_first(d: dict, keys: list[str]):
        for k in keys:
            if k in d and d.get(k) is not None:
                return d.get(k)
        return None

    def _extract_utils_pct_and_power_sum(obj) -> tuple[list[float], float]:
        utils: list[float] = []
        p_sum = 0.0

        if obj is None:
            return utils, p_sum

        # list 형태
        if isinstance(obj, list):
            for it in obj:
                if isinstance(it, (int, float, str)):
                    utils.append(_as_pct(_to_float(it, 0.0)))
                elif isinstance(it, dict):
                    u = _get_first(it, ["util", "gpu_util", "gpu_utilization", "gpu_util_pct", "util_pct"])
                    if u is not None:
                        utils.append(_as_pct(_to_float(u, 0.0)))
                    pw = _get_first(it, ["power_w", "power", "gpu_power_w", "power_draw_w"])
                    if pw is not None:
                        p_sum += _to_float(pw, 0.0)
            return utils, p_sum

        # dict 형태
        if isinstance(obj, dict):
            # 1) per-gpu util list 후보
            util_list = _get_first(obj, ["gpu_utils", "gpu_utils_pct", "gpu_util_list", "gpu_utilization_list"])
            if isinstance(util_list, list):
                utils = [_as_pct(_to_float(v, 0.0)) for v in util_list]

            # 2) per-gpu power list 후보
            p_list = _get_first(obj, ["gpu_powers_w", "gpu_power_list_w", "gpu_power_w_list"])
            if isinstance(p_list, list):
                for v in p_list:
                    p_sum += _to_float(v, 0.0)

            # 3) gpus: [{...}, ...] 형태면 그걸 최우선
            if isinstance(obj.get("gpus"), list):
                u2, p2 = _extract_utils_pct_and_power_sum(obj["gpus"])
                if u2:
                    utils = u2
                if p2 > 0.0:
                    p_sum = p2

            # 4) 중첩 구조 (nvml/telemetry/stats)
            for nest in ("nvml", "telemetry", "stats"):
                if isinstance(obj.get(nest), dict):
                    u3, p3 = _extract_utils_pct_and_power_sum(obj[nest])
                    if u3 and not utils:
                        utils = u3
                    if p3 > 0.0 and p_sum == 0.0:
                        p_sum = p3

            # 5) 단일 값 fallback
            if not utils:
                u_single = _get_first(obj, ["util_avg", "avg_gpu_util", "gpu_util", "gpu_utilization", "gpu_util_pct", "util"])
                if u_single is not None:
                    utils = [_as_pct(_to_float(u_single, 0.0))]

            if p_sum == 0.0:
                pw_single = _get_first(obj, ["power_sum", "power_w", "power_current_w", "node_power_w", "power"])
                if pw_single is not None:
                    p_sum = _to_float(pw_single, 0.0)

            return utils, p_sum

        return utils, p_sum

    cluster_id = (
        p.get("cluster_id")
        or p.get("cluster")
        or p.get("cid")
        or request.headers.get("x-cluster-id")
        or request.headers.get("X-Cluster-Id")
    )
    if not cluster_id:
        return {"ok": False, "reason": "missing_cluster_id"}

    cluster_id = str(cluster_id)

    node_id = (
        p.get("node_id")
        or p.get("node")
        or p.get("hostname")
        or request.headers.get("x-node-id")
        or request.headers.get("X-Node-Id")
        or (request.client.host if request.client else "unknown")
    )
    node_id = str(node_id)

    utils_pct, power_sum_w = _extract_utils_pct_and_power_sum(p)

    # reported GPU count (없으면 util list 길이)
    n_reported = 0
    n_key = _get_first(p, ["num_gpus", "gpu_count", "ngpus", "gpus_count"])
    if n_key is not None:
        try:
            n_reported = int(n_key)
        except Exception:
            n_reported = 0
    if n_reported <= 0:
        n_reported = len(utils_pct)

    util_sum_pct = float(sum(utils_pct))  # 각 gpu util%의 합

    # update scheduler state
    st = get_global_state()
    ttl = _to_float(getattr(st, "telemetry_ttl_sec", None), 15.0)
    if ttl <= 0:
        ttl = 15.0

    with STATE_LOCK:
        # per-node telemetry 저장소 (새로 추가해도 됨: state 객체에 attribute 하나 늘리는 수준)
        per_node = getattr(st, "cluster_telemetry_nodes", None)
        if per_node is None or not isinstance(per_node, dict):
            per_node = {}
            setattr(st, "cluster_telemetry_nodes", per_node)
        if cluster_id not in per_node or not isinstance(per_node.get(cluster_id), dict):
            per_node[cluster_id] = {}

        per_node[cluster_id][node_id] = {
            "ts": now,
            "util_sum_pct": util_sum_pct,   # sum of per-gpu util(%)
            "n_reported": int(n_reported),  # how many gpus contributed
            "power_sum_w": float(power_sum_w),
        }

        # prune TTL
        alive_samples = []
        for nid, s in list(per_node[cluster_id].items()):
            ts0 = _to_float(s.get("ts"), 0.0)
            if now - ts0 > ttl:
                del per_node[cluster_id][nid]
                continue
            alive_samples.append(s)

        # cluster total_gpus 확보
        clusters = getattr(st, "clusters", {}) or {}
        cr = clusters.get(cluster_id)

        total_gpus = 0
        if cr is not None:
            try:
                total_gpus = int(getattr(cr, "total_gpus", 0) or 0)
            except Exception:
                total_gpus = 0

        # total_gpus가 0이면 reported 합으로 임시 추정 (그래도 분모 0 방지)
        if total_gpus <= 0:
            total_gpus = 0
            for s in alive_samples:
                total_gpus += int(s.get("n_reported", 0) or 0)
            if total_gpus <= 0:
                total_gpus = 1

        # ---- cluster 평균 util% 계산 ----
        # 핵심: "reported 평균"이 아니라 "cluster total 기준 평균"으로 만든다.
        agg_util_sum_pct = 0.0
        agg_power_sum_w = 0.0
        for s in alive_samples:
            agg_util_sum_pct += _to_float(s.get("util_sum_pct"), 0.0)
            agg_power_sum_w += _to_float(s.get("power_sum_w"), 0.0)

        util_avg_pct = agg_util_sum_pct / float(total_gpus)
        util_avg_pct = _clamp(util_avg_pct, 0.0, 100.0)

        try:
            rl = get_run_logger()
        except Exception:
            rl = None

        if rl:
            try:
                # per-node level (alive samples 기준)
                for nid, s in per_node.get(cluster_id, {}).items():
                    rl.telemetry_sample(
                        node_id=str(nid),
                        gpu_index=0,  # cluster-level aggregate라 index 의미 없음
                        gpu_util=float(s.get("util_sum_pct", 0.0)) / max(1, int(s.get("n_reported", 1))),
                        power_w=float(s.get("power_sum_w", 0.0)),
                        mem_used_mb=0.0,
                        mem_total_mb=0.0,
                        ts=now,
                    )
            except Exception as e:
                try:
                    rl.log_warn(f"[logfail] telemetry_sample failed: {e}")
                except Exception:
                    pass

        # cluster_telemetry dict 갱신 (emit 함수가 읽는 곳)
        ct = getattr(st, "cluster_telemetry", None)
        if ct is None or not isinstance(ct, dict):
            ct = {}
            setattr(st, "cluster_telemetry", ct)

        ct[cluster_id] = {
            "ts": now,
            "util_avg": float(util_avg_pct),        # 0..100
            "power_sum": float(agg_power_sum_w),    # W
        }

        # (선택) cluster record에도 최신값을 심어두면 fallback에서도 덜 망가짐
        if cr is not None:
            setattr(cr, "util_raw", float(util_avg_pct))         # 0..100
            setattr(cr, "power_current_w", float(agg_power_sum_w))

    return {"ok": True, "accepted": True, "cluster": cluster_id, "node": node_id}

def report_checkpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = payload.get("job_id")
    if not job_id:
        return {"ok": False, "reason": "missing job_id"}

    epochs_done = payload.get("epochs_done") or payload.get("epoch_done")
    epochs_total = payload.get("epochs_total") or payload.get("epochs")

    new_end = update_job_eta_on_event(
        str(job_id),
        epochs_done=epochs_done,
        epochs_total=epochs_total,
        source="checkpoint",
    )
    return {"ok": True, "job_id": str(job_id), "expected_end_ts": new_end}

def _finalize_queue_time_on_start_locked(jr: Any, now_ts: float) -> float:
    if not hasattr(jr, "queued_total_sec"):
        jr.queued_total_sec = 0.0
    if not hasattr(jr, "queue_enter_ts"):
        jr.queue_enter_ts = None

    delta = 0.0
    if jr.queue_enter_ts is not None:
        try:
            delta = max(0.0, float(now_ts) - float(jr.queue_enter_ts))
        except Exception:
            delta = 0.0
        jr.queued_total_sec = float(jr.queued_total_sec) + float(delta)
        jr.queue_enter_ts = None

    return float(delta)

def mark_job_started_locked(
    state: Any,
    *,
    job_id: str,
    cluster_id: str,
    nodes: List[str],
    world_size: int,
    gang: bool = False,
    note: str = "",
    is_backfill: bool = False,
    now_ts: Optional[float] = None,
) -> Dict[str, Any]:
    if now_ts is None:
        now_ts = time.time()
    now_ts = float(now_ts)

    jr = state.jobs.get(str(job_id)) if hasattr(state, "jobs") else None
    if jr is None:
        return {"ok": False, "reason": "job_not_found", "job_id": str(job_id)}

    # ---- core fields (RUNNING commit) ----
    try:
        jr.cluster_id = str(cluster_id)
    except Exception:
        pass
    try:
        jr.nodes = list(nodes or [])
    except Exception:
        pass
    try:
        jr.g_cur = int(world_size)
    except Exception:
        pass

    try:
        jr.status = "RUNNING"
    except Exception:
        pass

    try:
        jr.is_backfill = bool(is_backfill)
    except Exception:
        pass
    # (선택) gang 플래그 유지용: 있는 필드면 기록
    for k in ("is_gang", "gang", "gang_required"):
        try:
            if hasattr(jr, k):
                setattr(jr, k, bool(getattr(jr, k) or gang))
        except Exception:
            pass

    submit_ts = None
    try:
        submit_ts = getattr(jr, "submit_ts", None)
    except Exception:
        submit_ts = None
    if submit_ts is None:
        try:
            submit_ts = getattr(jr, "submitted_ts", None)
        except Exception:
            submit_ts = None

    if submit_ts is None:
        submit_ts = float(now_ts)

    try:
        if getattr(jr, "submit_ts", None) is None:
            jr.submit_ts = float(submit_ts)
    except Exception:
        pass
    try:
        if getattr(jr, "submitted_ts", None) is None:
            jr.submitted_ts = float(submit_ts)
    except Exception:
        pass

    # ---- queue time finalize + start_ts ----
    queued_delta = 0.0
    try:
        queued_delta = float(_finalize_queue_time_on_start_locked(jr, now_ts))
    except Exception:
        # fallback: 최소한 queued_total_sec라도 안전하게
        try:
            # 기존 누적이 있으면 유지
            prev_total = float(getattr(jr, "queued_total_sec", 0.0) or 0.0)
            # submit_ts 기준으로 이번 queued를 계산
            this_q = max(0.0, float(now_ts) - float(submit_ts))
            jr.queued_total_sec = max(prev_total, this_q)
            queued_delta = this_q
        except Exception:
            queued_delta = 0.0

    # start_ts는 없으면 세팅 (이미 있으면 덮어쓰지 않음)
    try:
        if getattr(jr, "start_ts", None) is None:
            jr.start_ts = float(now_ts)
    except Exception:
        pass

    queued_total = 0.0
    try:
        queued_total = float(getattr(jr, "queued_total_sec", 0.0) or 0.0)
    except Exception:
        queued_total = 0.0

    qkind = None
    qcid = None
    try:
        qkind = getattr(jr, "queue_kind", None)
    except Exception:
        qkind = None
    try:
        qcid = getattr(jr, "queue_cluster_id", None)
    except Exception:
        qcid = None

    return {
        "ok": True,
        "job_id": str(job_id),
        "cluster_id": str(cluster_id),
        "queued_total_sec": float(queued_total),
        "queued_delta_sec": float(queued_delta),
        "gang": bool(gang),
        "is_backfill": bool(is_backfill),
        "queue_kind": qkind,
        "queue_cluster_id": qcid,
        "model": getattr(jr, "model", None),
        "dataset": getattr(jr, "dataset", None),
        "preempt_count": int(getattr(jr, "preempt_count", 0) or 0),
        "note": note,
        "ts": float(now_ts),
    }

def mark_job_started(
    job_id: str,
    cluster_id: str,
    nodes: List[str],
    world_size: int,
    gang: bool = False,
    note: str = "",
    *,
    is_backfill: bool = False,
) -> Dict[str, Any]:
    now_ts = time.time()

    with STATE_LOCK:
        state = get_global_state()
        info = mark_job_started_locked(
            state,
            job_id=str(job_id),
            cluster_id=str(cluster_id),
            nodes=list(nodes),
            world_size=int(world_size),
            gang=bool(gang),
            note=str(note or ""),
            is_backfill=bool(is_backfill),
            now_ts=float(now_ts),
        )

    if not info.get("ok"):
        return info

    # ---- logging (LOCK 밖) ----
    try:
        rl = get_run_logger()
    except Exception:
        rl = None

    queued_total = float(info["queued_total_sec"])
    queued_delta = float(info["queued_delta_sec"])
    qkind = info.get("queue_kind")
    qcid = info.get("queue_cluster_id")

    if rl:
        try:
            rl.queue_event(
                event="dequeue",
                job_id=str(job_id),
                queue_len=-1,
                note=f"cluster={cluster_id} kind={qkind} backfill={bool(is_backfill)} "
                     f"queued_delta={queued_delta:.3f} queued_total={queued_total:.3f}",
                ts=float(now_ts),
            )
        except Exception:
            pass

        try:
            rl.job_event(
                event="started",
                job_id=str(job_id),
                cluster=str(cluster_id),
                world_size=int(world_size),
                note=(note or f"started_on={cluster_id}, g={world_size}, nodes={nodes}, queued_sec={queued_total:.3f}, "
                              f"gang={gang}, backfill={bool(is_backfill)}"),
                metadata={
                    "model": info.get("model"),
                    "dataset": info.get("dataset"),
                    "nodes": list(nodes),
                    "queued_total_sec": queued_total,
                    "queued_delta_sec": queued_delta,
                    "submitted_ts": None,
                    "started_ts": float(now_ts),
                    "gang": bool(gang),
                    "is_backfill": bool(is_backfill),
                    "preempt_count": int(info.get("preempt_count", 0) or 0),
                    "queue_kind": qkind,
                    "queue_cluster_id": qcid,
                },
                ts=float(now_ts),
                stream="job",
            )
        except Exception:
            pass

        try:
            rl.job_event(
                event="queue_dequeued",
                job_id=str(job_id),
                cluster=str(cluster_id),
                world_size=int(world_size),
                note=f"kind={qkind} backfill={bool(is_backfill)} queued_delta={queued_delta:.3f} queued_total={queued_total:.3f}",
                metadata={
                    "queue_kind": qkind,
                    "queue_cluster_id": qcid,
                    "queued_delta_sec": queued_delta,
                    "queued_total_sec": queued_total,
                    "is_backfill": bool(is_backfill),
                },
                ts=float(now_ts),
                stream="queue",
            )
        except Exception:
            pass

    return info

def _ensure_eta_fields_locked(jr: Any) -> None:
    def _setdefault(obj: Any, k: str, v: Any) -> None:
        try:
            if isinstance(obj, dict):
                obj.setdefault(k, v)
            else:
                if not hasattr(obj, k) or getattr(obj, k) is None:
                    setattr(obj, k, v)
        except Exception:
            pass

    _setdefault(jr, "last_eta_update_ts", None)      # float | None
    _setdefault(jr, "expected_end_ts", None)         # float | None
    _setdefault(jr, "eta_source", None)              # str | None

    _setdefault(jr, "eta_stale_sec", 120.0)          # stale 기준
    _setdefault(jr, "remaining_floor_sec", 60.0)     # stale일 때 최소 남은시간 하한

    _setdefault(jr, "epochs", None)                  # total epochs
    _setdefault(jr, "epochs_total", None)
    _setdefault(jr, "steps_total", None)

    _setdefault(jr, "_eta_rate_ewma", None)          # float | None
    _setdefault(jr, "_eta_last_progress", None)      # float | None
    _setdefault(jr, "_eta_last_progress_ts", None)   # float | None

def update_job_eta_on_event_locked(
    state: Any,
    job_id: str,
    *,
    epochs_done: Optional[float] = None,
    epochs_total: Optional[float] = None,
    steps_done: Optional[float] = None,
    steps_total: Optional[float] = None,
    steps_per_sec: Optional[float] = None,
    epochs_per_sec: Optional[float] = None,
    source: str = "event",
    now_ts: Optional[float] = None,
) -> Optional[float]:
    jid = str(job_id)
    jr = (getattr(state, "jobs", {}) or {}).get(jid)
    if jr is None:
        return None

    st = str(getattr(jr, "status", "") or "").upper()
    if st not in ("RUNNING", "QUEUED"):
        return None

    _ensure_eta_fields_locked(jr)
    now = float(now_ts if now_ts is not None else time.time())

    # ---- 1) total / done 정규화 ----
    def _to_float(x) -> Optional[float]:
        try:
            if x is None:
                return None
            v = float(x)
            if not math.isfinite(v):
                return None
            return v
        except Exception:
            return None

    if epochs_total is None:
        epochs_total = _to_float(getattr(jr, "epochs_total", None))
        if epochs_total is None:
            epochs_total = _to_float(getattr(jr, "epochs", None))

    if steps_total is None:
        steps_total = _to_float(getattr(jr, "steps_total", None))

    epochs_done = _to_float(epochs_done)
    steps_done = _to_float(steps_done)
    epochs_total = _to_float(epochs_total)
    steps_total = _to_float(steps_total)

    if epochs_done is None:
        epochs_done = _to_float(getattr(jr, "epoch_done", None))
        if epochs_done is None:
            epochs_done = _to_float(getattr(jr, "epochs_done", None))

    # ---- 2) rate 추정(우선순위: explicit rate > EWMA deriv) ----
    def _clamp_rate(v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if v <= 0:
            return None
        if not math.isfinite(v):
            return None
        return v

    steps_per_sec = _clamp_rate(_to_float(steps_per_sec))
    epochs_per_sec = _clamp_rate(_to_float(epochs_per_sec))

    last_p = _to_float(getattr(jr, "_eta_last_progress", None))
    last_t = _to_float(getattr(jr, "_eta_last_progress_ts", None))

    # 어떤 progress를 쓰는지 결정: steps가 있으면 steps 우선(더 촘촘)
    use_steps = (steps_done is not None and steps_total is not None) or (steps_done is not None and steps_total is None)
    cur_p = steps_done if use_steps else epochs_done

    derived_rate = None
    if cur_p is not None and last_p is not None and last_t is not None:
        dt = now - float(last_t)
        dp = float(cur_p) - float(last_p)
        if dt > 0.5 and dp > 0:  # 너무 짧은 간격/역행은 무시
            derived_rate = dp / dt
            derived_rate = _clamp_rate(derived_rate)

    # EWMA로 rate 안정화
    ewma = _to_float(getattr(jr, "_eta_rate_ewma", None))
    alpha = 0.35  # 이벤트 빈도 고려: 너무 작으면 느리고, 너무 크면 출렁
    chosen_rate = None

    # explicit가 있으면 우선
    if use_steps and steps_per_sec is not None:
        chosen_rate = steps_per_sec
    elif (not use_steps) and epochs_per_sec is not None:
        chosen_rate = epochs_per_sec
    else:
        # derivative가 있으면 EWMA 갱신
        if derived_rate is not None:
            if ewma is None:
                ewma = derived_rate
            else:
                ewma = (1 - alpha) * float(ewma) + alpha * float(derived_rate)
            try:
                jr._eta_rate_ewma = float(ewma)
            except Exception:
                pass
        chosen_rate = _clamp_rate(ewma)

    # progress memory update
    try:
        jr._eta_last_progress = cur_p
        jr._eta_last_progress_ts = now
    except Exception:
        pass

    # rate 없으면 계산 불가
    if chosen_rate is None:
        # 이벤트는 받았으니 timestamp는 찍되, ETA는 유지
        try:
            jr.last_eta_update_ts = now
            jr.eta_source = str(source)
        except Exception:
            pass
        return _to_float(getattr(jr, "expected_end_ts", None))

    # ---- 3) remaining 계산 ----
    remaining = None
    if use_steps:
        if steps_total is not None and steps_done is not None:
            remaining = max(0.0, float(steps_total) - float(steps_done))
    else:
        if epochs_total is not None and epochs_done is not None:
            remaining = max(0.0, float(epochs_total) - float(epochs_done))

    # remaining이 없으면 계산 불가
    if remaining is None:
        try:
            jr.last_eta_update_ts = now
            jr.eta_source = str(source)
        except Exception:
            pass
        return _to_float(getattr(jr, "expected_end_ts", None))

    remaining_sec = remaining / float(chosen_rate)
    if not math.isfinite(remaining_sec) or remaining_sec < 0:
        remaining_sec = 0.0

    computed_end = now + float(remaining_sec)

    # ---- 4) 감소폭 제한(ETA가 너무 갑자기 짧아지는 것을 막음) ----
    old_end = _to_float(getattr(jr, "expected_end_ts", None))
    if old_end is not None:
        min_end = now + 0.85 * max(0.0, float(old_end) - now)
        new_end = computed_end if computed_end >= min_end else min_end
    else:
        new_end = computed_end

    # ---- 5) write back ----
    try:
        jr.expected_end_ts = float(new_end)
        jr.last_eta_update_ts = now
        jr.eta_source = str(source)
    except Exception:
        if isinstance(jr, dict):
            jr["expected_end_ts"] = float(new_end)
            jr["last_eta_update_ts"] = now
            jr["eta_source"] = str(source)

    return float(new_end)

def update_job_eta_on_event(
    job_id: str,
    *,
    epochs_done: Optional[float] = None,
    epochs_total: Optional[float] = None,
    steps_done: Optional[float] = None,
    steps_total: Optional[float] = None,
    steps_per_sec: Optional[float] = None,
    epochs_per_sec: Optional[float] = None,
    source: str = "event",
) -> Optional[float]:
    st = get_global_state()
    with STATE_LOCK:
        return update_job_eta_on_event_locked(
            st,
            str(job_id),
            epochs_done=epochs_done,
            epochs_total=epochs_total,
            steps_done=steps_done,
            steps_total=steps_total,
            steps_per_sec=steps_per_sec,
            epochs_per_sec=epochs_per_sec,
            source=source,
            now_ts=time.time(),
        )

def apply_resize_ssot_locked(
    st: Any,
    *,
    job_id: str,
    cid: str,
    new_nodes: List[str],
    now_ts: Optional[float] = None,
) -> None:
    job_id = str(job_id)
    cid = str(cid)
    now_ts = float(now_ts or __import__("time").time())
    nodes = [str(n) for n in (new_nodes or []) if n]

    # 1) node_owner: 기존 job 소유 노드 제거 후 새 노드 할당
    node_owner = getattr(st, "node_owner", {}) or {}
    if not isinstance(node_owner, dict):
        node_owner = {}
        setattr(st, "node_owner", node_owner)

    for n, owner in list(node_owner.items()):
        if str(owner) == job_id:
            node_owner.pop(str(n), None)

    for n in nodes:
        node_owner[str(n)] = job_id

    # 2) jobs[job_id] 업데이트
    jobs = getattr(st, "jobs", {}) or {}
    jr = jobs.get(job_id)
    if jr is not None:
        try:
            if isinstance(jr, dict):
                jr["nodes"] = list(nodes)
                jr["g_cur"] = int(len(nodes))
                jr.setdefault("cluster_id", cid)
                jr["cluster_id"] = cid
            else:
                setattr(jr, "nodes", list(nodes))
                setattr(jr, "g_cur", int(len(nodes)))
                setattr(jr, "cluster_id", cid)
        except Exception:
            pass

    # 3) 클러스터 카운터 재계산(있으면)
    try:
        _recompute_cluster_counters_locked(st)
    except Exception:
        pass

def can_resize_locked(st: Any, job_id: str, now_ts: float) -> bool:
    job_id = str(job_id)

    infl = getattr(st, "resize_inflight", None)
    if isinstance(infl, dict) and infl.get(job_id):
        return False

    cd = getattr(st, "resize_cooldown_until", None)
    if isinstance(cd, dict):
        try:
            until = float(cd.get(job_id) or 0.0)
            if until > float(now_ts):
                return False
        except Exception:
            pass

    return True

def mark_resize_inflight_locked(st: Any, job_id: str, now_ts: float) -> None:
    job_id = str(job_id)
    infl = getattr(st, "resize_inflight", None)
    if not isinstance(infl, dict):
        infl = {}
        setattr(st, "resize_inflight", infl)
    infl[job_id] = {"ts": float(now_ts)}

def clear_resize_inflight_locked(st: Any, job_id: str) -> None:
    job_id = str(job_id)
    infl = getattr(st, "resize_inflight", None)
    if isinstance(infl, dict):
        infl.pop(job_id, None)

def set_resize_cooldown_locked(st: Any, job_id: str, now_ts: float, cooldown_sec: float) -> None:
    job_id = str(job_id)
    cd = getattr(st, "resize_cooldown_until", None)
    if not isinstance(cd, dict):
        cd = {}
        setattr(st, "resize_cooldown_until", cd)
    cd[job_id] = float(now_ts + float(cooldown_sec))