# app/backfill_policy.py
from __future__ import annotations

import logging
import time, math, threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterable, Set

from app.queue import ClusterQueue
from app.profiling import get_profiling_entry
from app.logger import get_run_logger


logger = logging.getLogger(__name__)
STATE_LOCK = threading.RLock()

@dataclass
class BackfillConfig:
    enable: bool = True
    skip_if_waiting_gang_hol: bool = False

    max_backfills_per_cluster: int = 1
    launch_fail_cooldown_sec: float = 5.0
    prefer_small_g: bool = True
    max_g_per_backfill_job: int = 1
    max_estimated_remaining_sec: float = 1e18

CFG = BackfillConfig()
_GLOBAL_STATE = None

def _now() -> float:
    return time.time()

def _get_jobs(state: Any) -> Dict[str, Any]:
    return (getattr(state, "jobs", {}) or {})

def _get_clusters(state: Any) -> Dict[str, Any]:
    return (getattr(state, "clusters", {}) or {})

def _get_global_queue(state: Any) -> List[str]:
    # 1) global_queue가 있으면 그걸 SSOT로 사용
    q = getattr(state, "global_queue", None)
    if isinstance(q, list) and q:
        return [str(x) for x in q if str(x)]

    # 2) fallback: cluster_queues + home_cluster_queues에서 QUEUED를 모아 전역 후보 풀 구성
    jobs = getattr(state, "jobs", {}) or {}

    def _pull_from_queue_obj(qobj: Any) -> List[str]:
        out: List[str] = []
        if qobj is None:
            return out
        if isinstance(qobj, list):
            return [str(x) for x in qobj if str(x)]
        # ClusterQueue 류: _jobs 안에 QueueJob or str
        try:
            qjobs = list(getattr(qobj, "_jobs", []) or [])
            for it in qjobs:
                jid = getattr(it, "job_id", None)
                if jid is None and isinstance(it, str):
                    jid = it
                if jid:
                    out.append(str(jid))
        except Exception:
            pass
        return out

    seen = set()
    merged: List[str] = []

    # cluster_queues 우선
    cq_map = getattr(state, "cluster_queues", None) or {}
    if isinstance(cq_map, dict):
        for _cid, cq in cq_map.items():
            for jid in _pull_from_queue_obj(cq):
                jr = jobs.get(str(jid))
                if jr is None or not _is_queued(jr):
                    continue
                if jid in seen:
                    continue
                merged.append(jid)
                seen.add(jid)

    # home_cluster_queues 다음
    hq_map = getattr(state, "home_cluster_queues", None) or {}
    if isinstance(hq_map, dict):
        for _cid, hq in hq_map.items():
            for jid in _pull_from_queue_obj(hq):
                jr = jobs.get(str(jid))
                if jr is None or not _is_queued(jr):
                    continue
                if jid in seen:
                    continue
                merged.append(jid)
                seen.add(jid)

    # submit 순서로 정렬(가능하면)
    try:
        merged.sort(key=lambda jid: float(_job_sort_key_for_global_order(jobs.get(jid))))
    except Exception:
        pass

    return merged

def _is_queued_locked(st: Any, job_id: str) -> Dict[str, Any]:
    """
    현재 job이 어느 큐에 들어가 있는지 진단용(정합성 체크용)
    return 예:
      {"in_global": True, "in_home": ["clusterA"], "in_mobile": False, "in_clusterq": ["clusterB"]}
    """
    jid = str(job_id)
    out = {"in_global": False, "in_home": [], "in_mobile": False, "in_clusterq": []}

    # global
    try:
        gq = list(getattr(st, "global_queue", []) or [])
        out["in_global"] = (jid in set(str(x) for x in gq))
    except Exception:
        pass

    # home_cluster_queues: Dict[cid, List[jid]]
    try:
        hqmap = getattr(st, "home_cluster_queues", {}) or {}
        if isinstance(hqmap, dict):
            for cid, q in hqmap.items():
                if isinstance(q, list) and jid in set(str(x) for x in q):
                    out["in_home"].append(str(cid))
    except Exception:
        pass

    # mobile_queue: List[jid]
    try:
        mq = getattr(st, "mobile_queue", None)
        if isinstance(mq, list):
            out["in_mobile"] = (jid in set(str(x) for x in mq))
    except Exception:
        pass

    # cluster_queues: Dict[cid, ClusterQueue]
    # ClusterQueue는 보통 _jobs에 QueueJob 객체가 들어있음
    try:
        cqmap = getattr(st, "cluster_queues", {}) or {}
        if isinstance(cqmap, dict):
            for cid, cq in cqmap.items():
                items = []
                try:
                    items = list(getattr(cq, "_jobs", []) or [])
                except Exception:
                    items = []
                for it in items:
                    it_jid = getattr(it, "job_id", None) or getattr(it, "id", None) or None
                    if it_jid is not None and str(it_jid) == jid:
                        out["in_clusterq"].append(str(cid))
                        break
    except Exception:
        pass

    return out

CLUSTERS: Dict[str, List[str]] = {
    "clusterA": ["node_a", "node_b", "node_c", "node_d"],
    "clusterB": ["node_e", "node_f", "node_g", "node_h"],
}

@dataclass
class ClusterRuntimeState:
    cluster_id: str
    nodes: List[str]

    total_gpus: int
    used_gpus: int = 0
    free_gpus: int = 0

    util: float = 0.0
    power_current_w: float = 0.0
    power_budget_w: float = 1.0

    speed_factor: float = 1.0
    price_per_gpu_hour: float = 1.0

    # 실행 중 job들(백필 포함)
    running_jobs: Dict[str, "JobRuntimeState"] = field(default_factory=dict)

@dataclass
class JobRuntimeState:
    job_id: str
    model: str
    dataset: str

    status: str = "QUEUED"  # QUEUED / RUNNING / FINISHED / FAILED / CANCELLED / PREEMPTED
    user_id: Optional[str] = None

    # 선택된 타겟 g (admission/global 결과)
    g_target: int = 1
    g_cur: int = 0

    # 선택된(추천) 클러스터: 아직 시작 전이면 여기 보고 배치
    admitted_cluster_id: Optional[str] = None

    # 실제 할당된 클러스터(시작 후 고정)
    cluster_id: Optional[str] = None
    nodes: List[str] = field(default_factory=list)

    submit_ts: float = 0.0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

    profiling: Any = None
    policy: Any = None

    is_backfill: bool = False
    last_scaled_ts: float = 0.0
    preempt_count: int = 0

    final_accuracy: Optional[float] = None

@dataclass
class GlobalSchedulerState:
    clusters: Dict[str, ClusterRuntimeState] = field(default_factory=dict)
    jobs: Dict[str, JobRuntimeState] = field(default_factory=dict)

    global_queue: List[Any] = field(default_factory=list)

    pinned_cluster: Dict[str, str] = field(default_factory=dict)
    node_owner: Dict[str, str] = field(default_factory=dict)
    node_cluster: Dict[str, str] = field(default_factory=dict)
    cluster_queues: Dict[str, ClusterQueue] = field(default_factory=dict)

    last_elastic_ts: float = 0.0
    elastic_cooldown_sec: float = 300.0

def init_global_state() -> Any:
    global _GLOBAL_STATE

    if _GLOBAL_STATE is None:
        class _State: 
            pass
        _GLOBAL_STATE = GlobalSchedulerState()

    st = _GLOBAL_STATE

    # containers
    if not hasattr(st, "clusters"):
        st.clusters = {}
    if not hasattr(st, "jobs"):
        st.jobs = {}

    # deque 금지: ours_core가 pop(0)/insert(0)/[0]을 list 전제로 사용
    if not hasattr(st, "global_queue") or st.global_queue is None:
        st.global_queue = []
    else:
        # QueueJob/기타 섞여있으면 str(job_id)로 정규화
        new_gq = []
        for x in list(st.global_queue):
            if isinstance(x, str):
                jid = x
            else:
                jid = getattr(x, "job_id", None) or getattr(x, "id", None)
            if jid:
                new_gq.append(str(jid))
        st.global_queue = new_gq

    if not hasattr(st, "cluster_queues"):
        st.cluster_queues = {}
    if not hasattr(st, "node_owner"):
        st.node_owner = {}
    if not hasattr(st, "node_cluster"):
        st.node_cluster = {}
    if not hasattr(st, "pinned_cluster"):
        st.pinned_cluster = {}

    for cid, nodes in CLUSTERS.items():
        if cid not in st.clusters:
            try:
                st.clusters[cid] = ClusterRuntimeState(
                    cluster_id=cid,
                    nodes=list(nodes),
                    total_gpus=len(nodes),
                    used_gpus=0,
                    free_gpus=len(nodes),
                )
            except Exception:
                st.clusters[cid] = {
                    "cluster_id": cid,
                    "nodes": list(nodes),
                    "total_gpus": len(nodes),
                    "used_gpus": 0,
                    "free_gpus": len(nodes),
                    "util": 0.0,
                    "power_current_w": 0.0,
                    "power_budget_w": 1.0,
                    "speed_factor": 1.0,
                    "price_per_gpu_hour": 1.0,
                    "running_jobs": {},
                }
        else:
            cr = st.clusters[cid]
            try:
                if not getattr(cr, "nodes", None):
                    cr.nodes = list(nodes)
                if int(getattr(cr, "total_gpus", 0) or 0) <= 0:
                    cr.total_gpus = len(nodes)
                # free_gpus는 node_owner 단일진실로 매번 계산 가능하지만,
                # 초기값 비어있으면 채워둠
                if int(getattr(cr, "free_gpus", 0) or 0) <= 0:
                    cr.free_gpus = int(getattr(cr, "total_gpus", len(nodes)))
            except Exception:
                if isinstance(cr, dict):
                    cr.setdefault("nodes", list(nodes))
                    cr["total_gpus"] = int(cr.get("total_gpus") or len(nodes))
                    cr["free_gpus"] = int(cr.get("free_gpus") or cr["total_gpus"])

    # HoL state
    if not hasattr(st, "hol_job_id"):
        st.hol_job_id = None
    if not hasattr(st, "hol_pin_cluster"):
        st.hol_pin_cluster = None
    if not hasattr(st, "hol_saved_running"):
        st.hol_saved_running = {}  # {cluster_id: set(job_ids)}

    # cluster_queues 생성/보정
    for cid in list(st.clusters.keys()):
        if cid not in st.cluster_queues:
            st.cluster_queues[cid] = ClusterQueue(cluster_id=cid)

    return st

def get_global_state() -> Any:
    global _GLOBAL_STATE
    if _GLOBAL_STATE is None:
        init_global_state()
    return _GLOBAL_STATE

def _is_queued(jr: Any) -> bool:
    # ⚠️ circular import 방지용 로컬 헬퍼
    try:
        st = str(getattr(jr, "status", "") or "").upper()
        return st == "QUEUED"
    except Exception:
        return False

def _is_gang(jr: Any) -> bool:
    return bool(getattr(jr, "is_gang", False) or getattr(jr, "gang", False) or getattr(jr, "gang_required", False))

def _job_g_req(jr: Any) -> int:
    # best_g/target_g/world_size_req 등 여러 필드 호환
    for k in ("g_req", "g_target", "world_size_req", "world_size", "best_g"):
        v = getattr(jr, k, None)
        if v is None:
            continue
        try:
            iv = int(v)
            if iv > 0:
                return iv
        except Exception:
            pass
    return 1

def _job_priority(jr: Any) -> float:
    # 낮을수록 더 높은 우선순위라고 가정(없으면 0)
    v = getattr(jr, "priority", None)
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0


def _job_remaining_sec(jr: Any, get_epoch_time_fn: Optional[Callable[[Any], float]]) -> Optional[float]:
    if get_epoch_time_fn is None:
        return None

    try:
        total_epochs = int(getattr(jr, "epochs", None) or 0)
        cur_epoch = int(getattr(jr, "current_epoch", None) or 0)
        if total_epochs <= 0:
            return None
        rem_epochs = max(0, total_epochs - cur_epoch)
        ep_sec = float(get_epoch_time_fn(jr))
        if ep_sec <= 0:
            return None
        return float(rem_epochs) * float(ep_sec)
    except Exception:
        return None

def _ensure_dict_attr(state: Any, name: str) -> dict:
    m = getattr(state, name, None)
    if not isinstance(m, dict):
        m = {}
        setattr(state, name, m)
    return m

def _free_nodes_in_cluster(state: Any, cid: str) -> List[str]:
    import time

    cid = str(cid)
    now = float(time.time())

    # ---- node_owner (SSOT)
    # 1) cluster별 node_owner가 있으면 그걸 우선
    # 2) 없으면 state.node_owner fallback
    node_owner = {}
    try:
        clusters0 = getattr(state, "clusters", None)
        if isinstance(clusters0, dict):
            cr0 = clusters0.get(cid)
        else:
            cr0 = None

        if cr0 is not None:
            if isinstance(cr0, dict):
                no0 = cr0.get("node_owner")
            else:
                no0 = getattr(cr0, "node_owner", None)
            if isinstance(no0, dict):
                node_owner = dict(no0)  # shallow copy
    except Exception:
        node_owner = {}

    if not node_owner:
        no = getattr(state, "node_owner", None)
        if isinstance(no, dict):
            node_owner = no
        else:
            node_owner = {}


    # ---- cluster nodes
    clusters = getattr(state, "clusters", None)
    if not isinstance(clusters, dict):
        clusters = {}

    cr = clusters.get(cid)

    nodes: List[str] = []
    try:
        if isinstance(cr, dict):
            nodes = [str(x) for x in (cr.get("nodes") or []) if x]
        else:
            nodes = [str(x) for x in (getattr(cr, "nodes", []) or []) if x]
    except Exception:
        nodes = []

    # ---- drain map
    nd_map = getattr(state, "node_drain_until_by_node", None)
    if not isinstance(nd_map, dict):
        nd_map = {}
        try:
            state.node_drain_until_by_node = nd_map
        except Exception:
            pass

    # ---- alpha (thrash 방지)
    try:
        alpha = float(getattr(state, "DRAIN_RETRY_ALPHA_SEC", 0.0) or 0.0)
    except Exception:
        alpha = 0.0
    if alpha <= 0.0:
        # 0.5s는 짧아서 409 루프를 만들기 쉬움. 기본을 좀 보수적으로.
        alpha = 2.0

    # ---- 옵션: 여기서 drain pop 할지 여부 (디버깅/일관성 측면에서 기본 False 권장)
    #   * True로 켜면: 만료된 drain 기록을 여기서 제거
    #   * False면: 만료되더라도 free에는 포함시키되, 기록 정리는 별도 cleanup에서
    pop_expired = bool(getattr(state, "POP_EXPIRED_DRAINS_IN_FREE_FN", False))

    out: List[str] = []

    for n in nodes:
        nn = str(n)

        # 1) owner 체크 (보수적으로: 키가 존재하면 점유)
        #    - 값이 None이면 "비점유"로 볼 수 있게 예외 처리
        try:
            if nn in node_owner:
                o = node_owner.get(nn)
                # None / "" / "none" / "null" 은 비점유로 취급
                if o is not None:
                    s = str(o).strip().lower()
                    if s not in ("", "none", "null"):
                        continue  # 점유 중
        except Exception:
            continue

        # 2) drain 체크
        du = None
        try:
            du = nd_map.get(nn)
        except Exception:
            du = None

        if du is not None:
            try:
                du_f = float(du)
            except Exception:
                du_f = 0.0

            if du_f > 0.0:
                if now < (du_f + alpha):
                    # 아직 드레인 안정화 구간
                    continue

                # 만료됨: free에는 포함시키되, pop은 옵션
                if pop_expired:
                    try:
                        nd_map.pop(nn, None)
                    except Exception:
                        pass

        out.append(nn)

    out.sort()
    return out

def _job_gang_required_g(job_obj: Any) -> int:
    try:
        model = str(getattr(job_obj, "model", None) or getattr(job_obj, "model_name", None) or "").strip()
        dataset = str(getattr(job_obj, "dataset", None) or "").strip()

        # ✅ 오직 이 조합만 자동 gang=4
        if model == "DenseNet-121" and dataset == "TinyImageNet":
            return 4

        if bool(getattr(job_obj, "is_gang", False)):
            g_req = int(getattr(job_obj, "g_req", 0) or 0)
            return (g_req if g_req > 0 else 4)

    except Exception:
        pass
    return 0

def _job_sort_key_for_global_order(jr: Any) -> float:
    # queue_seq는 "순서" 자체이므로, 있으면 submit_ts보다 강하게 사용
    try:
        qs = getattr(jr, "queue_seq", None)
        if qs is not None:
            v = float(qs)
            if v > 0:
                return v
    except Exception:
        pass

    try:
        es = getattr(jr, "event_seq", None)
        if es is not None:
            v = float(es)
            if v > 0:
                # event_seq도 순서로 쓸 수 있음
                return v
    except Exception:
        pass

    try:
        ts = getattr(jr, "submit_ts", None)
        if ts is not None:
            v = float(ts)
            if v > 0:
                # submit_ts는 보통 epoch time
                return v
    except Exception:
        pass

    return float(time.time())

# Normalization references
SPS_REF    = float(__import__("os").getenv("OURS_SPS_REF", "10.0"))
TIME_REF   = float(__import__("os").getenv("OURS_TIME_REF_SEC", "3600.0"))
COST_REF   = float(__import__("os").getenv("OURS_COST_REF", "0.05"))
ENERGY_REF = float(__import__("os").getenv("OURS_ENERGY_REF_KWH", "1.0"))
MU_FAIR    = float(__import__("os").getenv("OURS_MU_FAIR", "0.2"))

def _score_S_jcg(
    policy,
    sps: float,
    g: int,
    price_per_gpu_hour: float,
    p_current_w_cluster: float = 0.0,
    p_current_w: float = 0.0,
    p_budget_w_cluster: float = 1.0,
    f_fair_c: float = 1.0,
    # ✅ NEW: profiling 기반 power (per-gpu)
    avg_power_w_per_gpu: float = 0.0,
) -> float:
    import math
    from typing import Any, Dict

    # --- policy normalize ---
    if policy is None:
        policy_d: Dict[str, Any] = {}
    elif isinstance(policy, dict):
        policy_d = policy
    else:
        if hasattr(policy, "model_dump") and callable(getattr(policy, "model_dump")):
            try:
                policy_d = dict(policy.model_dump())
            except Exception:
                policy_d = {}
        elif hasattr(policy, "dict") and callable(getattr(policy, "dict")):
            try:
                policy_d = dict(policy.dict())
            except Exception:
                policy_d = {}
        else:
            try:
                policy_d = dict(getattr(policy, "__dict__", {}) or {})
            except Exception:
                policy_d = {}

    def _f(x: Any, d: float) -> float:
        try:
            v = float(x)
            if not math.isfinite(v):
                return float(d)
            return v
        except Exception:
            return float(d)

    lam_t = _f(policy_d.get("lambda_time", 0.34), 0.34)
    lam_c = _f(policy_d.get("lambda_cost", 0.33), 0.33)
    lam_e = _f(policy_d.get("lambda_energy", 0.33), 0.33)

    lam_t = max(0.0, min(1.0, lam_t))
    lam_c = max(0.0, min(1.0, lam_c))
    lam_e = max(0.0, min(1.0, lam_e))
    s = lam_t + lam_c + lam_e
    if s <= 1e-9:
        lam_t, lam_c, lam_e = 0.34, 0.33, 0.33
    else:
        lam_t, lam_c, lam_e = lam_t / s, lam_c / s, lam_e / s

    sps = _f(sps, 0.0)
    if sps <= 0.0:
        return -1e18

    g = int(g) if int(g or 0) > 0 else 1
    price_per_gpu_hour = _f(price_per_gpu_hour, 1.0)

    # ✅ power 우선순위: profiling(per-gpu) > cluster telemetry/field
    ap = _f(avg_power_w_per_gpu, 0.0)
    if ap > 0.0:
        pcur_total_w = ap * float(g)   # ✅ 총전력(W)
    else:
        pcur_total_w = _f(p_current_w_cluster, 0.0)
        if pcur_total_w <= 0.0:
            pcur_total_w = _f(p_current_w, 0.0)

    f_fair_c = _f(f_fair_c, 1.0)

    # --- terms ---
    sps_term = (sps / float(SPS_REF)) if float(SPS_REF) > 0 else 0.0

    cost_per_step = (float(g) * float(price_per_gpu_hour)) / (3600.0 * float(sps))
    cost_term = (cost_per_step / float(COST_REF)) if float(COST_REF) > 0 else 0.0

    energy_kwh_per_step = (float(pcur_total_w) * (1.0 / float(sps))) / 3_600_000.0
    energy_term = (energy_kwh_per_step / float(ENERGY_REF)) if float(ENERGY_REF) > 0 else 0.0

    fair_excess = max(0.0, float(f_fair_c) - 1.0)
    fair_penalty = float(MU_FAIR) * fair_excess

    S = (lam_t * sps_term) - (lam_c * cost_term) - (lam_e * energy_term) - fair_penalty
    if not math.isfinite(S):
        return -1e18
    return float(S)

def _compute_f_fair_all_clusters(state: Any) -> Dict[str, float]:
    capnorm: Dict[str, float] = {}
    for cid, cr in getattr(state, "clusters", {}).items():
        util = float(getattr(cr, "util", 0.0) or 0.0)
        total_gpus = int(getattr(cr, "total_gpus", 0) or 0)
        speed = float(getattr(cr, "speed_factor", 1.0) or 1.0)
        denom = float(max(1, total_gpus)) * float(max(1e-9, speed))
        capnorm[str(cid)] = util / denom

    if not capnorm:
        return {}

    avg = sum(capnorm.values()) / float(len(capnorm))
    if avg <= 0.0:
        return {cid: 1.0 for cid in capnorm.keys()}

    return {cid: (capnorm[cid] / avg) for cid in capnorm.keys()}

def _choose_best_cluster_and_g_theory(
    *,
    state: Any,
    job: Any,
    policy: Dict[str, Any],
    clusters: Dict[str, Any],
    g_candidates: List[int],
) -> Tuple[str, int, Dict[str, Any], float]:
    import inspect
    import math

    def _f(x, d=0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(d)

    def _i(x, d=0) -> int:
        try:
            return int(x)
        except Exception:
            return int(d)

    # 0) job spec
    model = getattr(job, "model", None) or getattr(job, "model_name", None)
    dataset = getattr(job, "dataset", None)

    batch_size = (
        getattr(job, "batch_size", None)
        or getattr(job, "batch_size_per_gpu", None)
        or getattr(job, "local_batch", None)
    )
    batch_size = _i(batch_size, 0)

    if not model or not dataset:
        raise RuntimeError(f"invalid_job_spec: model={model!r} dataset={dataset!r}")

    # 1) gang 판정
    is_gang = bool(
        getattr(job, "is_gang", False)
        or getattr(job, "gang", False)
        or getattr(job, "gang_required", False)
        or ((str(model) == "DenseNet-121") and (str(dataset) == "TinyImageNet"))
    )

    if is_gang:
        g_candidates_eff = [4]
    else:
        g_candidates_eff = list(g_candidates or []) or [1]

        g_req = getattr(job, "g_req", None)
        try:
            g_req_i = int(g_req) if g_req is not None else 0
        except Exception:
            g_req_i = 0

        if g_req_i and g_req_i > 1:
            g_candidates_eff = sorted(set([_i(x, 0) for x in g_candidates_eff] + list(range(1, g_req_i + 1))))
        else:
            g_candidates_eff = sorted(set([_i(x, 0) for x in g_candidates_eff] + [1]))

        g_candidates_eff = [g for g in g_candidates_eff if _i(g, 0) > 0] or [1]

    # 2) fairness map
    f_fair_map = _compute_f_fair_all_clusters(state) or {}

    # 3) helpers
    def _cluster_nodes(cr: Any) -> List[str]:
        if isinstance(cr, dict):
            return [str(x) for x in (cr.get("nodes") or []) if x]
        return [str(x) for x in (getattr(cr, "nodes", None) or []) if x]

    def _cluster_total_gpus(cr: Any) -> int:
        if isinstance(cr, dict):
            return _i(cr.get("total_gpus", 0), 0)
        return _i(getattr(cr, "total_gpus", 0), 0)

    def _cluster_speed_factor(cr: Any) -> float:
        if isinstance(cr, dict):
            return _f(cr.get("speed_factor", 1.0), 1.0)
        return _f(getattr(cr, "speed_factor", 1.0), 1.0)

    def _free_now(state: Any, cid: str, cr: Any) -> int:
        # SSOT 우선
        try:
            fn = globals().get("_free_nodes_in_cluster", None)
            if callable(fn):
                return int(len(fn(state, str(cid)) or []))
        except Exception:
            pass

        # fallback: node_owner 기반
        try:
            node_owner = getattr(state, "node_owner", {}) or {}
            nodes = _cluster_nodes(cr)
            if nodes and isinstance(node_owner, dict):
                free = 0
                for n in nodes:
                    if not node_owner.get(str(n)):
                        free += 1
                return int(free)
        except Exception:
            pass

        # fallback: cr.free_gpus
        try:
            if isinstance(cr, dict):
                return max(0, _i(cr.get("free_gpus", 0), 0))
            return max(0, _i(getattr(cr, "free_gpus", 0), 0))
        except Exception:
            return 0

    def _queue_len_hint(state: Any, cid: str) -> int:
        try:
            cq_map = getattr(state, "cluster_queues", {}) or {}
            cq = cq_map.get(cid)
            if cq is None:
                return 0
            try:
                return int(len(getattr(cq, "_jobs", []) or []))
            except Exception:
                if hasattr(cq, "__len__"):
                    return int(len(cq))
        except Exception:
            pass
        return 0

    def _get_prof(model: str, dataset: str, cid: str, g: int, batch: int) -> Optional[Dict[str, Any]]:
        try:
            sig = inspect.signature(get_profiling_entry)
            params = list(sig.parameters.keys())
        except Exception:
            params = []

        try:
            if params:
                kwargs: Dict[str, Any] = {}

                if "model_name" in params:
                    kwargs["model_name"] = model
                elif "model" in params:
                    kwargs["model"] = model

                if "dataset" in params:
                    kwargs["dataset"] = dataset

                if "cluster_id" in params:
                    kwargs["cluster_id"] = cid
                elif "cid" in params:
                    kwargs["cid"] = cid

                if "gpu_count" in params:
                    kwargs["gpu_count"] = int(g)
                elif "g" in params:
                    kwargs["g"] = int(g)

                if batch > 0:
                    if "batch_size" in params:
                        kwargs["batch_size"] = int(batch)
                    elif "batch" in params:
                        kwargs["batch"] = int(batch)

                if kwargs:
                    prof = get_profiling_entry(**kwargs)
                    if isinstance(prof, dict) and prof:
                        return prof

            # legacy positional: (model, dataset, cid, g)
            prof2 = get_profiling_entry(model, dataset, cid, int(g))
            if isinstance(prof2, dict) and prof2:
                if batch > 0 and ("batch_size" not in prof2):
                    prof2["batch_size"] = int(batch)
                return prof2
        except Exception:
            return None

        return None

    # ✅ profiling fallback (다른 클러스터 prof를 speed_factor로 스케일)
    def _get_prof_with_fallback(model: str, dataset: str, cid: str, g: int, batch: int) -> Optional[Dict[str, Any]]:
        prof = _get_prof(model, dataset, cid, g, batch)
        if isinstance(prof, dict) and prof:
            return prof

        try:
            target_cr = (clusters or {}).get(cid)
            sf_t = _cluster_speed_factor(target_cr) if target_cr is not None else 1.0

            for cid2, cr2 in (clusters or {}).items():
                cid2 = str(cid2)
                if cid2 == cid:
                    continue

                prof2 = _get_prof(model, dataset, cid2, g, batch)
                if not (isinstance(prof2, dict) and prof2):
                    continue

                sps2 = _f(prof2.get("throughput_sps"), 0.0)
                if not (sps2 > 0.0 and math.isfinite(sps2)):
                    continue

                sf_2 = _cluster_speed_factor(cr2)
                ratio = (sf_t / sf_2) if (sf_2 and sf_2 > 0.0) else 1.0

                out = dict(prof2)
                out["throughput_sps"] = float(sps2) * float(ratio)

                # ✅ NEW: power(per-gpu) 키 유지 (스케일은 보수적으로 “그대로”)
                if "avg_power_w_per_gpu" in prof2:
                    out["avg_power_w_per_gpu"] = float(prof2.get("avg_power_w_per_gpu") or 0.0)

                out["fallback_from_cluster"] = str(cid2)
                out["fallback_speed_ratio"] = float(ratio)
                return out
        except Exception:
            return None

        return None

    # ✅ “즉시 시작 가능한 클러스터가 존재하면” free=0 클러스터는 스킵
    any_can_start_now = False
    if is_gang:
        for cid0, cr0 in (clusters or {}).items():
            if _free_now(state, str(cid0), cr0) >= 4:
                any_can_start_now = True
                break
    else:
        for cid0, cr0 in (clusters or {}).items():
            if _free_now(state, str(cid0), cr0) > 0:
                any_can_start_now = True
                break

    best: Optional[Dict[str, Any]] = None
    missing_profiling = 0
    bad_sps = 0
    skipped_gang_not4 = 0
    skipped_cap = 0
    skipped_free0_when_can_start = 0

    W_FREE = float(getattr(state, "W_ADMISSION_FREE", 0.08) or 0.08)
    W_QUEUE = float(getattr(state, "W_ADMISSION_QUEUE", 0.05) or 0.05)

    for cid0, cr in (clusters or {}).items():
        cid = str(cid0)

        price_per_gpu_hour = _f(cr.get("price_per_gpu_hour", 1.0), 1.0) if isinstance(cr, dict) else _f(getattr(cr, "price_per_gpu_hour", 1.0), 1.0)
        p_current_w = _f(cr.get("power_current_w", 0.0), 0.0) if isinstance(cr, dict) else _f(getattr(cr, "power_current_w", 0.0), 0.0)
        p_budget_w  = _f(cr.get("power_budget_w", 1.0), 1.0) if isinstance(cr, dict) else _f(getattr(cr, "power_budget_w", 1.0), 1.0)

        f_fair_c = _f(f_fair_map.get(cid, 1.0), 1.0)

        total_g = _cluster_total_gpus(cr)
        nodes_n = len(_cluster_nodes(cr))
        cap = max(total_g, nodes_n)
        free_now = _free_now(state, cid, cr)
        qlen = _queue_len_hint(state, cid)

        if is_gang and cap < 4:
            skipped_cap += 1
            continue

        if any_can_start_now:
            if is_gang:
                if int(free_now) < 4:
                    skipped_free0_when_can_start += 1
                    continue
            else:
                if int(free_now) <= 0:
                    skipped_free0_when_can_start += 1
                    continue

        seen_geff: set[int] = set()

        for g0 in (g_candidates_eff or []):
            g = _i(g0, 0)
            if g <= 0:
                continue

            if is_gang and g != 4:
                skipped_gang_not4 += 1
                continue

            if is_gang:
                if int(free_now) < 4:
                    continue
                g_eff = 4
                need = 0
            else:
                if int(free_now) > 0:
                    g_eff = max(1, min(int(g), int(free_now)))
                    need = 0
                else:
                    g_eff = 1
                    need = 1

                if g_eff in seen_geff:
                    continue
                seen_geff.add(g_eff)

            prof = _get_prof_with_fallback(str(model), str(dataset), str(cid), int(g_eff), int(batch_size))

            # 최후 fallback: g 프로파일로 다운스케일
            if (not prof) and (not is_gang) and (int(g_eff) != int(g)):
                prof_hi = _get_prof_with_fallback(str(model), str(dataset), str(cid), int(g), int(batch_size))
                if isinstance(prof_hi, dict) and prof_hi:
                    sps_hi = _f(prof_hi.get("throughput_sps"), 0.0)
                    if sps_hi > 0.0 and math.isfinite(sps_hi):
                        prof = dict(prof_hi)
                        prof["throughput_sps"] = float(sps_hi) * (float(g_eff) / float(max(1, int(g))))
                        prof["scaled_from_g"] = int(g)

            if not prof:
                missing_profiling += 1
                continue

            sps = _f(prof.get("throughput_sps"), 0.0)
            if not (sps > 0.0 and math.isfinite(sps)):
                bad_sps += 1
                continue

            # ✅ NEW: profiling power(per-gpu)
            avg_p = _f(prof.get("avg_power_w_per_gpu"), 0.0)

            S_base = _score_S_jcg(
                policy=policy,
                sps=float(sps),
                g=int(g_eff),
                price_per_gpu_hour=float(price_per_gpu_hour),
                p_current_w_cluster=float(p_current_w),
                p_budget_w_cluster=float(p_budget_w),
                f_fair_c=float(f_fair_c),
                avg_power_w_per_gpu=float(avg_p),   # ✅ 핵심
            )

            denom = float(cap if cap > 0 else max(1, int(g_eff)))
            free_ratio = float(max(0, free_now)) / float(denom)
            q_pen = float(qlen) / float(max(1, cap))
            need_pen = float(need) / float(max(1, cap))

            S = float(S_base) * (1.0 + W_FREE * free_ratio) - (W_QUEUE * q_pen) - (W_QUEUE * 1.5 * need_pen)

            if best is None or float(S) > float(best["S"]):
                best = {
                    "cid": str(cid),
                    "g": int(g_eff),
                    "S": float(S),
                    "S_base": float(S_base),
                    "sps": float(sps),
                    "avg_power_w_per_gpu": float(avg_p),
                    "price": float(price_per_gpu_hour),
                    "pcur": float(p_current_w),
                    "pbud": float(p_budget_w),
                    "f_fair": float(f_fair_c),
                    "free_now": int(free_now),
                    "cap": int(cap),
                    "qlen": int(qlen),
                    "need": int(need),
                    "profiling": dict(prof),
                    "is_gang": bool(is_gang),
                    "batch": int(batch_size or 0),
                    "any_can_start_now": bool(any_can_start_now),
                }

    if best is None:
        raise RuntimeError(
            f"No admissible (cluster,g) found "
            f"(gang={is_gang}, any_can_start_now={any_can_start_now}, "
            f"missing_profiling={missing_profiling}, bad_sps={bad_sps}, "
            f"skipped_free0_when_can_start={skipped_free0_when_can_start}, "
            f"skipped_gang_not4={skipped_gang_not4}, skipped_cap={skipped_cap})"
        )

    if bool(best.get("is_gang")) and int(best["g"]) != 4:
        raise RuntimeError(f"gang_invariant_broken: selected_g={best['g']} (must be 4)")

    logger.info(
        "[choose] cid=%s g=%d sps=%.3f S=%.4f (S_base=%.4f) "
        "avgP=%.1fW/gpu price=%.2f f_fair=%.3f free=%d cap=%d qlen=%d need=%d",
        best["cid"], best["g"], best["sps"], best["S"], best["S_base"],
        best["avg_power_w_per_gpu"], best["price"], best["f_fair"],
        best["free_now"], best["cap"], best["qlen"], best["need"],
    )

    return best["cid"], best["g"], best["profiling"], best["S"]

def choose_placement_now_for_cluster(
    *,
    state: Any,
    job: Any,
    policy: Dict[str, Any],
    clusters: Optional[Dict[str, Any]] = None,
    g_candidates: Optional[List[int]] = None,
    cluster_id: Optional[str] = None,
    free_nodes: Optional[List[str]] = None,
) -> Tuple[int, List[str], Dict[str, Any]]:

    def _i(x: Any, d=0) -> int:
        try:
            return int(x)
        except Exception:
            return int(d)

    def _free_nodes_for_cid(cid: str, cr: Any) -> List[str]:
        nodes: List[str] = []
        if isinstance(cr, dict):
            nodes = [str(n) for n in (cr.get("nodes") or []) if n]
        else:
            nodes = [str(n) for n in (getattr(cr, "nodes", None) or []) if n]

        owner = getattr(state, "node_owner", {}) or {}
        out: List[str] = []
        for n in nodes:
            if not owner.get(n):
                out.append(n)
        return out

    # ---- gang 판정 (Mode2에서도 동일하게 적용) ----
    model = getattr(job, "model", None) or getattr(job, "model_name", None)
    dataset = getattr(job, "dataset", None)
    is_gang = bool(
        getattr(job, "is_gang", False)
        or getattr(job, "gang", False)
        or getattr(job, "gang_required", False)
        or ((str(model) == "DenseNet-121") and (str(dataset) == "TinyImageNet"))
    )

    g_req = _i(getattr(job, "g_req", 0) or 0, 0)

    # -----------------------------
    # Mode 1) global-choice
    # -----------------------------
    if clusters is not None and g_candidates is not None:
        cid_best, g_best, prof_best, S_best = _choose_best_cluster_and_g_theory(
            state=state,
            job=job,
            policy=policy,
            clusters=clusters,
            g_candidates=list(g_candidates),
        )

        cr = clusters.get(str(cid_best))
        free = _free_nodes_for_cid(str(cid_best), cr) if cr is not None else []

        if len(free) < int(g_best):
            # 여기서 "실패"가 아니라 "대기"로 처리할지 여부는 상위 스케줄러 정책임.
            return 0, [], {
                "ok": False,
                "reason": "best_cluster_has_no_free_nodes_now",
                "cid_best": str(cid_best),
                "g_best": int(g_best),
                "free_n": int(len(free)),
                "free_needed": int(g_best),
                "S": float(S_best),
                "profiling": dict(prof_best or {}),
            }

        nodes_use = free[: int(g_best)]
        return int(g_best), list(nodes_use), {
            "ok": True,
            "reason": "global_choice",
            "cid": str(cid_best),
            "g": int(g_best),
            "S": float(S_best),
            "profiling": dict(prof_best or {}),
            "free_n": int(len(free)),
        }

    # -----------------------------
    # Mode 2) single-cluster immediate
    # -----------------------------
    cid = str(cluster_id or "")
    fn = [str(n) for n in (free_nodes or []) if n]

    if not cid or not fn:
        return 0, [], {"ok": False, "reason": "no_cluster_or_no_free_nodes", "cid": cid, "free_n": len(fn)}

    # ✅ 결정값 보존: g_target 우선
    g_fixed = _i(getattr(job, "g_target", 0) or 0, 0)

    # gang이면 무조건 4
    if is_gang:
        g_fixed = 4
    else:
        # g_req가 있으면 그 이상이 되도록(단, free 범위 내에서 결국 clamp됨)
        if g_fixed <= 0:
            if g_candidates:
                gc = sorted({max(1, _i(x, 1)) for x in g_candidates}, reverse=True)
            else:
                gc = [1]
            g_fixed = gc[0] if gc else 1

        if g_req and g_req > 1:
            # "최소 g_req"를 만족시키려는 의도라면 이렇게(하지만 free 부족이면 아래에서 줄어듦)
            g_fixed = max(int(g_fixed), int(g_req))

    g_use = min(int(g_fixed), int(len(fn)))
    nodes_use = fn[:g_use]

    # gang인데 free가 4 미만이면 즉시 불가
    if is_gang and g_use < 4:
        return 0, [], {
            "ok": False,
            "reason": "gang_needs_4_free_nodes",
            "cid": cid,
            "free_n": int(len(fn)),
            "g_need": 4,
        }

    return int(g_use), list(nodes_use), {
        "ok": True,
        "reason": "single_cluster_immediate",
        "cid": cid,
        "g_target": int(getattr(job, "g_target", 0) or 0),
        "g_req": int(g_req),
        "is_gang": bool(is_gang),
        "g_use": int(g_use),
        "free_n": int(len(fn)),
    }

def _ss_release_nodes_for_job_locked(
    state: Any,
    *,
    job_id: str,
    nodes: Optional[List[str]] = None,
) -> int:
    jid = str(job_id)

    owners = getattr(state, "node_owner", None)
    n2c = getattr(state, "node_cluster", None)

    if not isinstance(owners, dict):
        logger.warning("[release] missing map: node_owner")
        return 0

    # node_cluster는 없을 수도 있으니 "없으면 만들어 둠"
    if not isinstance(n2c, dict):
        n2c = {}
        try:
            state.node_cluster = n2c
        except Exception:
            pass

    released = 0
    affected_clusters: set[str] = set()

    def _mark_cluster(n: str) -> None:
        try:
            cid = n2c.get(n)
            if cid is not None:
                affected_clusters.add(str(cid))
        except Exception:
            pass

    if nodes is not None:
        req = [str(x).strip() for x in (nodes or []) if x is not None and str(x).strip()]
        for n in req:
            try:
                owner = owners.get(n)
                if owner is not None and str(owner) == jid:
                    _mark_cluster(n)
                    owners.pop(n, None)
                    n2c.pop(n, None)
                    released += 1
            except Exception:
                continue
    else:
        for n, owner in list(owners.items()):
            try:
                if owner is not None and str(owner) == jid:
                    _mark_cluster(n)
                    owners.pop(n, None)
                    n2c.pop(n, None)
                    released += 1
            except Exception:
                continue

    if released:
        # ✅ cluster counters / free_changed 갱신
        try:
            if affected_clusters:
                for cid in affected_clusters:
                    try:
                        _recompute_cluster_counters_locked(state, str(cid))
                    except Exception:
                        pass
                    try:
                        _set_free_changed_locked(state, str(cid))
                    except Exception:
                        pass
            else:
                # node_cluster가 비었거나 추적이 안 된 경우: 보수적으로 전체 갱신
                for cid in list((getattr(state, "clusters", {}) or {}).keys()):
                    try:
                        _recompute_cluster_counters_locked(state, str(cid))
                    except Exception:
                        pass
                    try:
                        _set_free_changed_locked(state, str(cid))
                    except Exception:
                        pass
        except Exception:
            pass

        logger.info("[release] ok job_id=%s released=%d affected_clusters=%s", jid, released, sorted(list(affected_clusters)))
    return released

def _recompute_cluster_counters_locked(state: Any, cluster_id: str) -> None:
    cid = str(cluster_id)
    cr = getattr(state, "clusters", {}).get(cid)
    if cr is None:
        return

    nodes = list(getattr(cr, "nodes", []) or [])
    total_cfg = int(getattr(cr, "total_gpus", 0) or 0)

    total = total_cfg if total_cfg > 0 else len(nodes)

    owner = getattr(state, "node_owner", {}) or {}
    used = 0
    for n in nodes:
        o = owner.get(n)
        if o is not None and str(o) != "":
            used += 1

    if used < 0:
        used = 0
    if total < 0:
        total = 0
    if used > total and total > 0:
        used = total

    free = total - used
    if free < 0:
        free = 0

    try:
        cr.used_gpus = int(used)
    except Exception:
        pass
    try:
        cr.free_gpus = int(free)
    except Exception:
        pass
    try:
        cr.total_gpus = int(total) if total > 0 else int(getattr(cr, "total_gpus", 0) or 0)
    except Exception:
        pass

    try:
        cr.util = (float(used) / float(total)) if total > 0 else 0.0
    except Exception:
        pass

    try:
        cr.node_owner = {n: owner.get(n) for n in nodes if owner.get(n) is not None and str(owner.get(n)) != ""}
    except Exception:
        pass

def _set_free_changed_locked(state: Any, cluster_id: str) -> None:
    if not hasattr(state, "cluster_events") or not isinstance(getattr(state, "cluster_events", None), dict):
        state.cluster_events = {}
    cid = str(cluster_id)
    rec = state.cluster_events.get(cid)
    if not isinstance(rec, dict):
        rec = {}
    rec["free_changed"] = True
    state.cluster_events[cid] = rec

def release_nodes_for_job_locked(*, job_id: str, cluster_id: Optional[str] = None) -> List[str]:
    """
    SSOT = state.node_owner + state.node_cluster 로 통일.

    ✅ 보장:
    - state.node_owner 에서 job_id가 점유한 노드를 확실히 pop
    - cluster_id가 주어져도 node_cluster 누락 때문에 release가 0개 되는 케이스를 막음
    - (호환) state.clusters[cid].node_owner 같은 "뷰"가 존재하면 SSOT로부터 동기화(갱신)함
      -> 이제 앞으로는 SSOT(node_owner, node_cluster)만 믿으면 됨. (cluster 내부 dict는 파생뷰)
    - 카운터/플래그 recompute는 영향을 받은 클러스터 기준으로 수행

    🔥 중요 수정:
    - node_cluster는 "노드 소속(정적)" 맵이므로 release 시 pop 금지 (유령락/비결정성 방지)
    - owner 비교는 대소문자/공백 차이에도 강건하게 (jid/o 모두 normalize)
    """
    jid_raw = str(job_id).strip()
    if not jid_raw:
        return []

    # normalize for robust owner matching (case/whitespace)
    jid_norm = jid_raw.strip().lower()

    cid = str(cluster_id).strip() if cluster_id is not None else None
    if cid is not None:
        cid = cid.strip()

    state = get_global_state()

    # --- SSOT maps ---
    owner = getattr(state, "node_owner", None)
    nclu = getattr(state, "node_cluster", None)

    if not isinstance(owner, dict):
        # SSOT가 dict가 아니면 지금 상태 자체가 깨진 것
        raise RuntimeError("state.node_owner is not a dict (SSOT broken)")

    if not isinstance(nclu, dict):
        nclu = {}
        try:
            state.node_cluster = nclu
        except Exception:
            pass

    clusters = getattr(state, "clusters", None)
    if not isinstance(clusters, dict):
        clusters = {}
        try:
            state.clusters = clusters
        except Exception:
            pass

    # --- helpers ---
    def _get_cluster_nodes(c: Any) -> List[str]:
        ns = None
        if isinstance(c, dict):
            ns = c.get("nodes", None)
        else:
            ns = getattr(c, "nodes", None)
        if not ns:
            return []
        out: List[str] = []
        for x in ns:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    def _get_cluster_owner_map(c: Any) -> Optional[Dict[str, Any]]:
        m = None
        if isinstance(c, dict):
            m = c.get("node_owner", None)
        else:
            m = getattr(c, "node_owner", None)
        return m if isinstance(m, dict) else None

    def _set_cluster_owner_map(c: Any, m: Dict[str, Any]) -> None:
        if isinstance(c, dict):
            c["node_owner"] = m
        else:
            try:
                setattr(c, "node_owner", m)
            except Exception:
                pass

    def _ensure_node_cluster_mapping() -> None:
        """
        nclu가 비어 있거나 일부 노드가 누락되어 있을 때,
        clusters[cid].nodes를 기준으로 n -> cid를 보강.
        """
        try:
            for ccid, cobj in list(clusters.items()):
                ccid_s = str(ccid).strip()
                if not ccid_s:
                    continue
                for n in _get_cluster_nodes(cobj):
                    if n not in nclu:
                        nclu[n] = ccid_s
        except Exception:
            pass

    def _sync_cluster_view_for(cids: Set[str]) -> None:
        """
        (호환) cluster 내부 node_owner가 존재하면 SSOT(node_owner,node_cluster)로부터 재구성.
        """
        for ccid in list(cids):
            cobj = clusters.get(ccid)
            if cobj is None:
                continue

            cnodes = _get_cluster_nodes(cobj)
            if not cnodes:
                # nodes 리스트가 없으면 정확한 재구성이 어려움 -> 최소한 job_id 잔존은 제거
                cmap = _get_cluster_owner_map(cobj)
                if isinstance(cmap, dict) and cmap:
                    for n, o in list(cmap.items()):
                        if o is None:
                            continue
                        if str(o).strip().lower() == jid_norm:
                            cmap.pop(n, None)
                    _set_cluster_owner_map(cobj, cmap)
                continue

            new_map: Dict[str, Any] = {}
            for n in cnodes:
                o = owner.get(n)
                if o is None:
                    continue
                new_map[n] = o
            _set_cluster_owner_map(cobj, new_map)

    # --- build mapping so cluster_id filter won't silently skip because node_cluster lacks entries ---
    _ensure_node_cluster_mapping()

    # --- 1) find candidate nodes in SSOT.owner ---
    released_nodes: List[str] = []
    affected_clusters: Set[str] = set()

    # cache for cluster nodes when we need conservative allow
    cluster_nodes_cache: Optional[Set[str]] = None
    if cid is not None:
        cobj0 = clusters.get(cid)
        if cobj0 is not None:
            try:
                cluster_nodes_cache = set(_get_cluster_nodes(cobj0))
            except Exception:
                cluster_nodes_cache = None

    for n, o in list(owner.items()):
        if o is None:
            continue
        if str(o).strip().lower() != jid_norm:
            continue

        n_cid = nclu.get(n)
        n_cid_s = str(n_cid).strip() if n_cid is not None else ""

        if cid is not None:
            # cluster_id가 주어졌는데 nclu가 비어 있으면, clusters[cid].nodes에 포함되면 release 허용
            if n_cid_s:
                if n_cid_s != cid:
                    continue
            else:
                if cluster_nodes_cache is not None:
                    if n not in cluster_nodes_cache:
                        continue
                # cluster 정보가 없거나 nodes가 비어있으면 유령락 방지를 위해 release 허용

        released_nodes.append(n)
        if n_cid_s:
            affected_clusters.add(n_cid_s)
        elif cid is not None:
            affected_clusters.add(cid)

    # --- 2) legacy: SSOT에 없는데 cluster view에만 남아있는 job_id 점유 제거 ---
    legacy_released: List[Tuple[str, str]] = []  # (node, cluster)
    if clusters:
        scan_cids = [cid] if cid is not None else list(clusters.keys())
        for ccid in scan_cids:
            ccid_s = str(ccid).strip()
            if not ccid_s:
                continue
            cobj = clusters.get(ccid)
            if cobj is None:
                continue
            cmap = _get_cluster_owner_map(cobj)
            if not isinstance(cmap, dict) or not cmap:
                continue
            for n, o in list(cmap.items()):
                if o is None:
                    continue
                if str(o).strip().lower() != jid_norm:
                    continue
                if n in released_nodes:
                    continue
                legacy_released.append((str(n), ccid_s))
                affected_clusters.add(ccid_s)

    # --- 3) pop from SSOT.owner ONLY ---
    for n in released_nodes:
        owner.pop(n, None)
        # 🔥 node_cluster는 정적 맵이므로 pop 금지

    # legacy: cluster 뷰에만 있었던 경우도 owner에서 혹시 남아있으면 pop 시도
    if legacy_released:
        for n, _ccid_s in legacy_released:
            owner.pop(n, None)
            # 🔥 node_cluster pop 금지

    # --- 4) sync cluster view(s) from SSOT (write-through) ---
    if affected_clusters:
        _sync_cluster_view_for(affected_clusters)
    else:
        # 보수적으로 전체 동기화
        try:
            _sync_cluster_view_for(set(str(x).strip() for x in clusters.keys() if str(x).strip()))
        except Exception:
            pass

    # --- 5) counters/flags ---
    if affected_clusters:
        for c in sorted(list(affected_clusters)):
            try:
                _recompute_cluster_counters_locked(state, str(c))
            except Exception:
                pass
            try:
                _set_free_changed_locked(state, str(c))
            except Exception:
                pass
    else:
        for c in list(clusters.keys()):
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

    # 최종 released 노드 목록 = SSOT에서 빠진 것 + legacy로 제거된 것
    if legacy_released:
        for n, _ in legacy_released:
            if n not in released_nodes:
                released_nodes.append(n)

    return released_nodes

def release_nodes_for_job(*, job_id: str, cluster_id: Optional[str] = None) -> List[str]:
    with STATE_LOCK:
        return release_nodes_for_job_locked(job_id=str(job_id), cluster_id=cluster_id)

def _hol_waiting_gang4_locked(state: Any, *, pin: str, hol_jid: str) -> bool:
    try:
        hj = str(getattr(state, "hol_job_id", "") or "").strip()
        hp = str(getattr(state, "hol_pin_cluster", "") or "").strip()
        if not hj or not hp:
            return False
        if hj != str(hol_jid) or hp != str(pin):
            return False

        jobs = getattr(state, "jobs", {}) or {}
        jr = jobs.get(hj)
        if jr is None:
            return False

        stt = str(getattr(jr, "status", "") or "").upper().strip()

        # ✅ "대기"는 시작 전 상태여야 함
        # - STARTING/RUNNING이면 이미 시작 단계라서 backfill 태깅 트리거로 쓰면 위험
        if stt in ("STARTING", "RUNNING"):
            return False

        # ✅ gang4 확인(우선순위: _job_gang_required_g)
        try:
            if int(_job_gang_required_g(jr) or 0) == 4:
                return True
        except Exception:
            pass

        # ✅ fallback: 흔히 쓰는 필드들로 4 이상이면 gang4로 간주
        for k in ("world_size", "world_size_req", "g_req", "g_target", "best_g", "admitted_g_hint"):
            try:
                if int(getattr(jr, k, 0) or 0) >= 4:
                    return True
            except Exception:
                pass

        return False
    except Exception:
        return False

def _tag_as_hol_backfill_locked(
    vjr: Any,
    *,
    hol_jid: str,
    pin: str,
    now_ts: float,
) -> None:
    # --- 0) 기본 backfill 표시 ---
    try:
        vjr.is_backfill = True
    except Exception:
        pass

    # --- 1) strict 필드 세트 ---
    try:
        vjr.is_hol_backfill = True
    except Exception:
        pass

    try:
        vjr.hol_backfill_for = str(hol_jid)
    except Exception:
        pass

    try:
        vjr.hol_backfill_pin = str(pin)
    except Exception:
        pass

    # --- 2) 정렬/디버그용 타임스탬프 ---
    # since는 "최초"만 유지
    try:
        prev = float(getattr(vjr, "hol_backfill_since_ts", 0.0) or 0.0)
        if prev <= 0.0:
            vjr.hol_backfill_since_ts = float(now_ts)
    except Exception:
        try:
            vjr.hol_backfill_since_ts = float(now_ts)
        except Exception:
            pass

    # last는 매번 갱신(진단용)
    try:
        vjr.hol_backfill_last_ts = float(now_ts)
    except Exception:
        pass

    # epoch 카운트(선택)
    try:
        k = int(getattr(vjr, "hol_backfill_epoch", 0) or 0)
        vjr.hol_backfill_epoch = int(k + 1)
    except Exception:
        pass

    # --- 3) 큐 표시(선택) ---
    try:
        vjr.queue_kind = "HOL_BACKFILL"
    except Exception:
        pass

    # --- 4) ✅ 재프리엠션 가드(핵심) ---
    # global tick이 같은 victim을 연속으로 잡는 걸 차단.
    # 값이 이미 더 미래면 유지(더 강한 가드)
    try:
        # 기본 2초. 전역 상수 있으면 그걸 쓰세요.
        try:
            guard_sec = float(globals().get("PREEMPT_MIN_INTERVAL_SEC", 2.0) or 2.0)
        except Exception:
            guard_sec = 2.0

        cur = float(getattr(vjr, "preempt_guard_until_ts", 0.0) or 0.0)
        nxt = float(now_ts + max(0.5, guard_sec))
        if cur < nxt:
            vjr.preempt_guard_until_ts = float(nxt)
    except Exception:
        pass

def make_backfill_launch_fn(
    *,
    state_getter: Callable[[], Any],
    reserve_nodes_fn: Callable[..., Any],
    start_job_fn: Callable[..., Dict[str, Any]],
) -> Callable[[str, str, List[str], bool, int], bool]:
    def launch_fn(job_id: str, cluster_id: str, nodes: List[str], is_backfill: bool, g_use: int) -> bool:
        jid = str(job_id)
        cid = str(cluster_id)
        nodes = [str(x) for x in (nodes or []) if str(x)]
        try:
            g_use_i = int(g_use or 0)
        except Exception:
            g_use_i = 0

        if g_use_i <= 0 or len(nodes) != g_use_i:
            return False

        jr_ref = None
        now = float(time.time())

        # -------------------------
        # 1) LOCK: tag + reserve + STARTING commit
        # -------------------------
        with STATE_LOCK:
            st = state_getter()
            jobs = getattr(st, "jobs", {}) or {}
            jr = jobs.get(jid)
            if jr is None:
                return False

            # QUEUED만 backfill launch 대상으로
            if str(getattr(jr, "status", "") or "").upper().strip() != "QUEUED":
                return False

            # gang job은 backfill 금지
            try:
                if int(_job_gang_required_g(jr) or 0) == 4:
                    return False
            except Exception:
                pass

            # drain 중이면 금지 (보수적으로)
            drains = getattr(st, "drain_until_by_cluster", {}) or {}
            if now < float(drains.get(cid, 0.0) or 0.0):
                return False

            # ✅ backfill 태깅 (SSOT 확정)
            if is_backfill:
                try:
                    jr.is_backfill = True
                except Exception:
                    pass
                try:
                    # 기본은 BACKFILL
                    jr.queue_kind = "BACKFILL"
                except Exception:
                    pass

                # ✅ HoL(gang4) 대기중 + pin==cid면 HOL_BACKFILL로 강제 태깅
                try:
                    hol = str(getattr(st, "hol_job_id", "") or "").strip()
                    pin = str(getattr(st, "hol_pin_cluster", "") or "").strip()
                    if hol and pin and pin == cid:
                        if _hol_waiting_gang4_locked(st, pin=pin, hol_jid=hol):
                            _tag_as_hol_backfill_locked(jr, hol_jid=str(hol), pin=str(pin), now_ts=float(now))
                            # tag func가 queue_kind를 HOL_BACKFILL로 박도록 이미 수정했으면 OK
                except Exception:
                    pass

                # ✅ slice deadline SSOT 보장 (없으면 여기서 세팅)
                try:
                    ddl0 = float(getattr(jr, "backfill_deadline_ts", 0.0) or 0.0)
                except Exception:
                    ddl0 = 0.0
                if ddl0 <= 0.0:
                    try:
                        slice_sec = float(globals().get("BACKFILL_SLICE_SEC", 60.0) or 60.0)
                    except Exception:
                        slice_sec = 60.0
                    try:
                        jr.backfill_deadline_ts = float(now + slice_sec)
                    except Exception:
                        pass

            # reserve
            try:
                reserve_nodes_fn(job_id=jid, cluster_id=cid, nodes=list(nodes))
            except Exception:
                return False

            # STARTING 커밋
            try:
                jr.cluster_id = cid
                jr.nodes = list(nodes)
                jr.g_cur = int(g_use_i)
                jr.launch_inflight = True
                jr.launching_since_ts = float(now)
                jr.status = "STARTING"
            except Exception:
                pass

            # 락 밖에서 start_job_fn에 넘길 참조 고정
            jr_ref = jr

        # -------------------------
        # 2) NO LOCK: 실제 launch
        # -------------------------
        try:
            out = start_job_fn(
                job=jr_ref,
                cluster_id=cid,
                node_names=list(nodes),
                g_use=int(g_use_i),
                is_backfill=bool(is_backfill),
            )
        except Exception as e:
            out = {"ok": False, "reason": "start_exception", "exc": repr(e)}

        ok = bool((out or {}).get("ok"))

        # 실패 시 rollback (reserve/STARTING 되돌리기)
        if not ok:
            with STATE_LOCK:
                st2 = state_getter()
                jr2 = (getattr(st2, "jobs", {}) or {}).get(jid)
                if jr2 is not None:
                    # ✅ nodes=None 금지: 반드시 스냅샷 nodes만 해제
                    nodes2 = list(getattr(jr2, "nodes", []) or nodes)
                    try:
                        release_nodes_for_job(st2, jid, list(nodes2))
                    except Exception:
                        try:
                            _ss_release_nodes_for_job_locked(st2, job_id=jid, nodes=list(nodes2))
                        except Exception:
                            pass

                    try:
                        jr2.launch_inflight = False
                        jr2.launching_since_ts = 0.0
                        jr2.cluster_id = None
                        jr2.nodes = []
                        jr2.g_cur = 0
                        jr2.status = "QUEUED"
                        jr2.blocked_until = float(time.time() + 0.5)
                    except Exception:
                        pass

        return ok

    return launch_fn
