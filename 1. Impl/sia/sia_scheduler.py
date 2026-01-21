from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import psycopg2
from pymoo.core.problem import Problem

log = logging.getLogger("SIA")
log.setLevel(logging.INFO)

SIA_COOLDOWN_SEC = int(os.getenv("SIA_COOLDOWN_SEC", "60"))

# 1. 프로파일 DB 연결 & Goodput(throughput) surface 로딩
PROF_DB_DSN = os.getenv(
    "PROF_DB_DSN",
    "postgresql://prof:profpw@163.180.117.216:5432/profdb",
)

GANG_JOBS: Dict[Tuple[str, str], int] = {
    # ("DenseNet-121", "TinyImageNet"): 4,
}

_GOODPUT_SAMPLES: Dict[Tuple[str, str, int, int, int], Dict[str, float]] = defaultdict(
    lambda: {"count": 0.0, "sps_sum": 0.0, "sps_last": 0.0, "ts_last": 0.0}
)

def _piecewise_linear_interp(xs: List[float], ys: List[float], x: float) -> float:
    # xs must be sorted
    if not xs:
        return 0.0
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    # find interval
    lo = 0
    hi = len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[lo + 1]
    y0, y1 = ys[lo], ys[lo + 1]
    if x1 == x0:
        return float(y0)
    t = (x - x0) / (x1 - x0)
    return float(y0 + t * (y1 - y0))


class SurfaceProjector:
    """
    현실 버전 Sia projection:
    - (g,b)->sps 표면이 있을 때: log(batch) 보간 + g 보간
    - 표면이 비어있을 때: fallback
    - 표면이 부분적으로 비어있을 때:
        * g=1 curve에서 batch 보간
        * g scaling은 간단히 '효율' 곡선으로 fit해서 예측
          efficiency(g) = sps(g,b*) / (g * sps(1,b*))
          => available g들로 eff(g)를 fit하고, missing g는 eff로 예측
    """
    def __init__(self, surface: Dict[int, Dict[int, float]]):
        self.surface = surface or {}
        self.g_list = sorted(self.surface.keys())

        # cache g=1 curve
        self.g1_curve = self.surface.get(1, {}) if self.surface else {}
        self._eff_fit = None  # (g_points, eff_points)

        self._build_efficiency_fit()

    def _build_efficiency_fit(self):
        if not self.surface or 1 not in self.surface:
            self._eff_fit = None
            return

        # pick a reference batch b* that exists in g=1 and most g's
        g1_batches = set(self.surface[1].keys())
        if not g1_batches:
            self._eff_fit = None
            return

        # choose batch that maximizes overlap
        best_b = None
        best_cnt = -1
        for b in g1_batches:
            cnt = 0
            for g in self.g_list:
                if b in self.surface.get(g, {}):
                    cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt
                best_b = b

        if best_b is None:
            self._eff_fit = None
            return

        sps1 = float(self.surface[1].get(best_b, 0.0) or 0.0)
        if sps1 <= 0:
            self._eff_fit = None
            return

        g_pts = []
        eff_pts = []
        for g in self.g_list:
            spsg = float(self.surface.get(g, {}).get(best_b, 0.0) or 0.0)
            if spsg <= 0:
                continue
            eff = spsg / (float(g) * sps1)
            eff = max(min(eff, 1.2), 0.01)
            g_pts.append(float(g))
            eff_pts.append(float(eff))

        if len(g_pts) < 2:
            self._eff_fit = None
            return

        # sort points
        order = np.argsort(np.array(g_pts))
        g_pts = [g_pts[i] for i in order]
        eff_pts = [eff_pts[i] for i in order]
        self._eff_fit = (g_pts, eff_pts)

    def _interp_batch_in_g(self, g: int, batch: int) -> float:
        batch_sps = self.surface.get(g, {})
        if not batch_sps:
            return 0.0
        if batch in batch_sps:
            return float(batch_sps[batch])

        # interpolate on log(batch) axis
        bs = sorted(batch_sps.keys())
        xs = [float(np.log(max(1, b))) for b in bs]
        ys = [float(batch_sps[b]) for b in bs]
        x = float(np.log(max(1, batch)))
        return _piecewise_linear_interp(xs, ys, x)

    def predict_sps(self, g: int, batch: int) -> float:
        if g <= 0:
            return 0.0
        if not self.surface:
            return 0.0

        # 1) if we have g row: interpolate batch
        if g in self.surface and self.surface[g]:
            return max(self._interp_batch_in_g(g, batch), 0.0)

        # 2) if g missing: try efficiency projection using g=1
        if not self.g1_curve:
            # no g=1 => fallback to closest g
            if self.g_list:
                g_closest = min(self.g_list, key=lambda x: abs(x - g))
                return max(self._interp_batch_in_g(g_closest, batch), 0.0)
            return 0.0

        sps1b = self._interp_batch_in_g(1, batch)
        if sps1b <= 0:
            return 0.0

        # efficiency(g) interpolation if we have fit
        if self._eff_fit is not None:
            g_pts, eff_pts = self._eff_fit
            eff = _piecewise_linear_interp(g_pts, eff_pts, float(g))
            eff = max(min(eff, 1.2), 0.01)
            return float(g) * float(sps1b) * float(eff)

        # fallback: closest g
        if self.g_list:
            g_closest = min(self.g_list, key=lambda x: abs(x - g))
            sps = self._interp_batch_in_g(g_closest, batch)
            return max(sps, 0.0)

        return 0.0

JOB_ONLINE_SPS_EWMA: Dict[str, Dict[str, Any]] = {}

def record_goodput_sample(
    job_id: str,
    g: int,
    local_batch: int,
    accum: int,
    sps: float,
    gns: Optional[float] = None,
    stat_eff: Optional[float] = None,
    cluster_id: Optional[str] = None,
) -> None:
    """
    Record an online throughput sample.

    NOTE:
    - Some callers (GlobalServer) pass cluster_id; older versions didn't.
    - For SIA baseline, we keep this as a no-op-ish recorder to avoid crashes.
    """
    try:
        jid = str(job_id)
        cid = str(cluster_id or "unknown")
        gg = max(1, int(g))
        b = max(1, int(local_batch))
        a = max(1, int(accum))
        s = float(sps)
    except Exception:
        return

    k = (jid, cid, gg, b, a)
    st = _GOODPUT_SAMPLES[k]
    st["count"] += 1.0
    st["sps_sum"] += s
    st["sps_last"] = s
    st["ts_last"] = float(time.time())

def get_recent_sps(
    job_id: str,
    cluster_id: Optional[str] = None,
    default: float = 0.0,
) -> float:
    """
    Get the most recent sps for a job (optionally within a cluster).
    """
    jid = str(job_id)
    cid = str(cluster_id or "unknown")
    best_ts = -1.0
    best_sps = default

    for (j, c, _g, _b, _a), st in _GOODPUT_SAMPLES.items():
        if j != jid:
            continue
        if cluster_id is not None and c != cid:
            continue
        ts = float(st.get("ts_last", 0.0) or 0.0)
        if ts > best_ts:
            best_ts = ts
            best_sps = float(st.get("sps_last", default) or default)

    return best_sps

def sia_best_local_config_for_g(
    job_id: str,
    model_name: str,
    dataset: str,
    g: int,
    default_batch: int,
    default_accum: int,
    default_sps: float,
    cluster_id: Optional[str] = None,
) -> Tuple[int, int, float, float, float]:
    """
    현실 baseline:
    - profiling DB가 batch=64 only 이므로 batch는 항상 64로 강제
    - accum은 surface에 없으니 default 유지
    - stat_eff는 1.0으로 둠 (batch 변화가 없으므로 의미 없음)
    - best_sps는 goodput을 그대로 sps로 취급 (goodput==sps * scale_factor, scale_factor=1.0)
    """
    cid = str(cluster_id or "clusterA")

    gp_fn = GoodputFunction(
        model_name=model_name,
        dataset=dataset,
        cluster_id=cid,
        scale_factor=1.0,
    )

    # ✅ batch 강제
    best_b = 64
    best_a = max(1, int(default_accum))

    gp = gp_fn.goodput(int(g), local_batch=64)
    if gp <= 0:
        # fallback: 그래도 baseline은 64 고정
        gp, _ = gp_fn.optimize(int(g))
    gp = float(gp)

    best_stat_eff = 1.0
    best_goodput = gp
    best_sps = gp if gp > 0 else float(default_sps)

    return int(best_b), int(best_a), float(best_sps), float(best_stat_eff), float(best_goodput)

def update_job_metrics_from_telemetry(job: RuntimeJobState, metrics: Dict) -> None:
    if not isinstance(metrics, dict):
        return

    # attained_service
    try:
        if "attained_service" in metrics and metrics["attained_service"] is not None:
            job.attained_service = float(metrics["attained_service"])
    except Exception:
        pass

    # last_sps / batch / accum
    try:
        if "last_sps" in metrics and metrics["last_sps"] is not None:
            job.last_sps = float(metrics["last_sps"])
    except Exception:
        pass

    try:
        if "current_local_batch" in metrics and metrics["current_local_batch"] is not None:
            job.current_local_batch = int(metrics["current_local_batch"])
    except Exception:
        pass

    try:
        if "current_grad_accum" in metrics and metrics["current_grad_accum"] is not None:
            job.current_grad_accum = max(1, int(metrics["current_grad_accum"]))
    except Exception:
        pass

    # (선택) restart stats를 외부에서 넘길 경우 대비
    for k in ["num_restarts", "total_run_time", "total_restart_overhead", "last_started_ts"]:
        if k in metrics and metrics[k] is not None:
            try:
                setattr(job, k, float(metrics[k]) if "total_" in k or k.endswith("_ts") else int(metrics[k]))
            except Exception:
                pass

def is_gang_model(model_name: Optional[str], dataset: Optional[str]) -> bool:
    if not model_name or not dataset:
        return False
    return (str(model_name), str(dataset)) in GANG_JOBS

def _connect_prof_db():
    if not PROF_DB_DSN:
        raise RuntimeError("PROF_DB_DSN is not set")
    return psycopg2.connect(PROF_DB_DSN)

def load_sps_surface(
    model_name: str,
    dataset: str,
    cluster_id: str,
) -> Dict[int, Dict[int, float]]:
    """
    minimal_profiling에서 (model_name, dataset, cluster_id)에 해당하는
    (gpu_count, batch_size)별 평균 throughput_sps를
    {g: {batch_size: sps}} 형태로 반환.
    """
    try:
        conn = _connect_prof_db()
    except Exception as e:
        log.error(f"[Sia] profiling DB connect failed in load_sps_surface: {e}")
        return {}

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT gpu_count,
                           batch_size,
                           AVG(throughput_sps) AS avg_sps
                    FROM minimal_profiling
                    WHERE model_name = %s
                      AND dataset    = %s
                      AND cluster_id = %s
                      AND throughput_sps IS NOT NULL
                      AND batch_size IS NOT NULL
                    GROUP BY gpu_count, batch_size
                    ORDER BY gpu_count ASC, batch_size ASC
                    """,
                    (model_name, dataset, cluster_id),
                )
                rows = cur.fetchall()
    except Exception as e:
        log.error(f"[Sia] error querying profiling DB in load_sps_surface: {e}")
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    surface: Dict[int, Dict[int, float]] = {}
    for g, batch_size, avg_sps in rows:
        if g is None or batch_size is None or avg_sps is None:
            continue
        g = int(g)
        b = int(batch_size)
        sps = float(avg_sps)
        surface.setdefault(g, {})
        surface[g][b] = sps

    if not surface:
        log.warning(
            f"[Sia] no profiling surface for model={model_name}, dataset={dataset}, "
            f"cluster_id={cluster_id}"
        )

    return surface

class GoodputFunction:
    """
    Sia baseline (profiling batch=64 only 현실 대응):
    - 실행(batch)은 그대로 두되, goodput 추정은 profiling이 있는 batch(기본 64)에서만 읽는다.
    - 요청 batch(local_batch)가 profiling batch와 다르면, 통계효율 페널티(stat_eff)를 곱해 보수적으로 추정한다.

    즉,
      gp_est(g, b_req) = scale_factor * sps_profile(g, b_profile=64) * stat_eff(g, b_req)
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        cluster_id: str,
        scale_factor: float = 1.0,
        profile_base_batch: int = 64,  # ✅ profiling DB에 존재하는 기준 batch (현재 환경: 64 only)
    ):
        self.model_name = model_name
        self.dataset = dataset
        self.cluster_id = cluster_id
        self.scale_factor = max(float(scale_factor), 1e-3)

        self.profile_base_batch = int(profile_base_batch) if int(profile_base_batch) > 0 else 64

        # surface[g][b] = sps
        self.surface = load_sps_surface(model_name, dataset, cluster_id)

        if not self.surface:
            log.warning(
                f"[Sia] empty (g, batch) surface; fallback for "
                f"model={model_name}, dataset={dataset}, cluster_id={cluster_id}"
            )
            # fallback: batch=64 기준으로만 둔다
            self.surface = {
                1: {self.profile_base_batch: 1.0},
                2: {self.profile_base_batch: 1.8},
                4: {self.profile_base_batch: 3.0},
            }

        self._g_list = sorted(self.surface.keys())

        # 각 g에 대해 profiling에 존재하는 batch 목록
        self._batch_cache: Dict[int, List[int]] = {
            g: sorted(bs.keys()) for g, bs in self.surface.items()
        }

        # ✅ stat_eff 페널티 강도 (너무 세면 추정이 과하게 낮아짐)
        # b_req가 커질수록(=global batch 커질수록) 통계효율이 떨어진다고 가정하는 단순 모델
        self._eff_gamma = 0.5

    def _closest_g(self, g: int) -> int:
        return min(self._g_list, key=lambda x: abs(x - g))

    def _get_profile_sps(self, g: int) -> float:
        """
        profiling이 있는 batch(=profile_base_batch)에서만 sps를 읽는다.
        없으면 해당 g에서 사용 가능한 batch 중 하나를 fallback으로 사용.
        """
        if g not in self.surface:
            g = self._closest_g(g)

        batch_sps = self.surface.get(g, {})
        if not batch_sps:
            return 0.0

        # 1) base batch가 있으면 그걸 사용 (현재 환경의 정답 루트)
        if self.profile_base_batch in batch_sps:
            return float(batch_sps[self.profile_base_batch])

        # 2) 없으면 가장 가까운 batch로 fallback
        # (DB가 늘어났을 때도 안정적으로 동작)
        b_list = self._batch_cache.get(g) or sorted(batch_sps.keys())
        if not b_list:
            return 0.0
        b_pick = min(b_list, key=lambda b: abs(int(b) - int(self.profile_base_batch)))
        return float(batch_sps.get(b_pick, 0.0))

    def _stat_eff(self, g: int, req_batch: int) -> float:
        """
        요청 batch가 base batch보다 커질수록 페널티를 주는 단순 stat_eff.
        - base_batch 대비 req_batch 비율로만 처리 (profiling이 64-only인 환경에 맞춘 보수적 근사)
        """
        b0 = max(1, int(self.profile_base_batch))
        b = max(1, int(req_batch))

        # b가 b0보다 작으면 효율이 "더 좋아질" 근거가 없으므로 1.0으로 clamp (보수적)
        if b <= b0:
            return 1.0

        ratio = b0 / float(b)
        eff = ratio ** float(self._eff_gamma)

        # 너무 작게 떨어지면 스케줄러가 과하게 보수적으로 변하므로 하한
        return max(float(eff), 0.1)

    def goodput(self, num_replicas: int, local_batch: Optional[int] = None) -> float:
        """
        goodput 추정:
          gp = scale_factor * sps_profile(g, base_batch=64) * stat_eff(g, local_batch_req)

        local_batch가 None이면 stat_eff=1로 본다 (즉, base batch 기준 gp).
        """
        if num_replicas <= 0:
            return 0.0

        g = int(num_replicas)
        if g not in self.surface:
            g = self._closest_g(g)

        sps0 = self._get_profile_sps(g)
        if sps0 <= 0.0:
            return 0.0

        if local_batch is None:
            eff = 1.0
        else:
            eff = self._stat_eff(g, int(local_batch))

        return float(self.scale_factor) * float(sps0) * float(eff)

    def optimize(self, num_replicas: int) -> Tuple[float, int]:
        """
        Sia baseline에서는 batch 튜닝을 하지 않는다.
        - goodput 최대화 관점에서 'batch 선택'을 하지 않고,
        - profiling 기준 batch(profile_base_batch)를 반환한다.

        반환:
          (gp_at_base_batch, best_b=profile_base_batch)
        """
        if num_replicas <= 0:
            return 0.0, 0

        g = int(num_replicas)
        if g not in self.surface:
            g = self._closest_g(g)

        sps0 = self._get_profile_sps(g)
        if sps0 <= 0.0:
            return 0.0, int(self.profile_base_batch)

        gp = float(self.scale_factor) * float(sps0)

        # ✅ "스케줄러가 바꾸는 batch"가 아니라 "profiling 기준 batch"를 의미하는 값으로 반환
        return gp, int(self.profile_base_batch)

@dataclass
class RuntimeJobState:
    job_id: str
    model_name: str
    dataset: str

    # ✅ "현재 어디서 도는지" + "이동 목표를 담을 수 있어야 함"
    cluster_id: str

    current_gpus: int
    progress: float = 0.0
    attained_service: float = 0.0
    last_scaled_at_ts: float = 0.0

    min_gpus: int = 1
    max_gpus: int = 4

    # ✅ global_server.py가 쓰는 필드명과 정합
    current_local_batch: Optional[int] = None
    current_grad_accum: int = 1
    last_sps: float = 0.0

    # Sia-style restart stats (현실 버전: baseline에서 “대충이라도” 들어가야 함)
    num_restarts: int = 0
    total_run_time: float = 0.0
    total_restart_overhead: float = 0.0
    last_started_ts: float = 0.0

    current_gpu_type: Optional[str] = None  # 옵션 (cluster_id가 타입이면 안 써도 됨)

@dataclass
class JobInfo:
    job_id: str
    goodput_fn: GoodputFunction
    fairness_weight: float
    min_gpus: int
    max_gpus: int
    restart_factor: float = 1.0   # r_i (재시작 페널티)
    min_goodput: float = 1.0      # 그 job이 선택 가능한 g 중 최소 goodput (row-normalization 기준)


class SiaProblem(Problem):
    RHO = 0.5

    def __init__(
        self,
        jobs: List[JobInfo],
        total_gpus: int,
    ):
        self.jobs = jobs
        self.total_gpus = total_gpus

        n_var = len(jobs)
        n_obj = 1
        n_constr = 1

        xl = np.array([j.min_gpus for j in jobs], dtype=int)
        xu = np.array([j.max_gpus for j in jobs], dtype=int)

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=n_constr,
            xl=xl,
            xu=xu,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        pop_size = X.shape[0]
        num_jobs = X.shape[1]

        F = np.zeros((pop_size, self.n_obj))
        G = np.zeros((pop_size, self.n_constr))

        X = X.astype(int)

        for k in range(pop_size):
            alloc = X[k]  # shape: (num_jobs,)

            # 제약: 전체 GPU 수
            total_g = int(np.sum(alloc))
            G[k, 0] = total_g - self.total_gpus  # <= 0 이어야 함

            # 목적: 여기서는 사용하지 않지만, 기존 구현과의 호환성을 위해 남겨둠
            total_obj = 0.0
            for i in range(num_jobs):
                g_i = alloc[i]
                job = self.jobs[i]
                if g_i <= 0:
                    continue
                gp, _ = job.goodput_fn.optimize(g_i)
                gp = max(gp, 0.0)
                gp_rho = gp ** self.RHO if gp > 0 else 0.0
                w = job.fairness_weight
                total_obj += w * gp_rho

            F[k, 0] = -total_obj

        out["F"] = F
        out["G"] = G

class SiaScheduler:
    def __init__(self):
        self.log = logging.getLogger("Sia.SCHED")
        self.log.setLevel(logging.INFO)

    def _compute_restart_factor(self, rj: RuntimeJobState) -> float:
        """
        r_i 근사:
        r_i = (T - N*S) / (T + S)

        - T: total_run_time
        - N: num_restarts
        - S: avg restart overhead (= total_restart_overhead / N)

        수정:
        - run_time이 작아도 계산 수행
        - 하한을 0.02로 내려서 restart 많은 job 페널티가 실제로 먹히게 함
        """
        if int(getattr(rj, "num_restarts", 0) or 0) <= 0:
            return 1.0

        T = max(float(getattr(rj, "total_run_time", 0.0) or 0.0), 1e-6)
        N = max(int(getattr(rj, "num_restarts", 0) or 0), 1)

        total_ovh = float(getattr(rj, "total_restart_overhead", 0.0) or 0.0)
        avg_S = total_ovh / float(N) if total_ovh > 0.0 else 0.0

        if avg_S <= 0.0:
            return 1.0

        numerator = T - float(N) * avg_S
        denominator = T + avg_S
        if denominator <= 0.0:
            return 1.0

        r = numerator / denominator
        r = max(min(float(r), 1.0), 0.02)
        return r

    def build_job_infos(self, runtime_jobs: List[RuntimeJobState]) -> List[JobInfo]:
        jobs: List[JobInfo] = []

        for rj in runtime_jobs:
            gp_fn = GoodputFunction(
                model_name=rj.model_name,
                dataset=rj.dataset,
                cluster_id=rj.cluster_id,
                scale_factor=self._estimate_scale_factor(rj),
            )

            # Pollux-style fairness + Sia r_i
            base_weight = 1.0 / (1.0 + max(0.0, rj.attained_service))
            r_factor = self._compute_restart_factor(rj)

            min_gpus = max(1, rj.min_gpus)
            max_gpus = max(min_gpus, rj.max_gpus)

            # job별 최소 goodput (row-normalization 기준)
            min_gp = None
            for g in range(min_gpus, max_gpus + 1):
                gp, _ = gp_fn.optimize(g)
                gp = max(gp, 0.0)
                if gp <= 0:
                    continue
                if min_gp is None or gp < min_gp:
                    min_gp = gp

            if min_gp is None or min_gp <= 0.0:
                min_gp = 1.0

            jobs.append(
                JobInfo(
                    job_id=rj.job_id,
                    goodput_fn=gp_fn,
                    fairness_weight=base_weight,
                    min_gpus=min_gpus,
                    max_gpus=max_gpus,
                    restart_factor=r_factor,
                    min_goodput=min_gp,
                )
            )

        return jobs

    def _estimate_scale_factor(self, rj: RuntimeJobState) -> float:
        """
        온라인 throughput(최근 sps)로 profiling surface 보정.
        - 당신 런타임(RuntimeJobState)은 recent_sps가 없고 last_sps를 씀.
        - 따라서 last_sps를 기준으로 scale_factor를 추정한다.
        """
        # ✅ throughput signal: last_sps 사용 (없으면 0으로 처리)
        sps_obs = float(getattr(rj, "last_sps", 0.0) or 0.0)
        if sps_obs <= 0.0:
            return 1.0

        # ✅ 현재 관측치가 어떤 (g, batch)에서 나왔는지 필요
        if getattr(rj, "current_local_batch", None) is None or int(getattr(rj, "current_gpus", 0) or 0) <= 0:
            return 1.0

        surface = load_sps_surface(rj.model_name, rj.dataset, rj.cluster_id)
        if not surface:
            return 1.0

        g = int(getattr(rj, "current_gpus", 0) or 0)
        if g not in surface:
            g_list = sorted(surface.keys())
            if not g_list:
                return 1.0
            g = min(g_list, key=lambda x: abs(x - g))

        batch_sps = surface.get(g, {})
        if not batch_sps:
            return 1.0

        b = int(getattr(rj, "current_local_batch", 0) or 0)
        sps0 = batch_sps.get(b)
        if not sps0 or float(sps0) <= 0.0:
            # profiling에 해당 batch가 없으면 보정 안 함
            return 1.0

        return max(float(sps_obs) / float(sps0), 0.1)

    def _compute_objective(
        self,
        job_infos: List[JobInfo],
        alloc_g: Dict[str, int],
    ) -> float:
        """
        Sia-style objective (homogeneous cluster 축소 버전).
        u_i(g) = w_i * (r_i * (gp_i(g) / min_goodput_i))^p
        """
        p = SiaProblem.RHO
        total_obj = 0.0

        for ji in job_infos:
            g_i = int(alloc_g.get(ji.job_id, ji.min_gpus))
            if g_i <= 0:
                continue

            gp, _ = ji.goodput_fn.optimize(g_i)
            gp = max(gp, 0.0)
            if gp <= 0.0:
                continue

            base = ji.min_goodput if ji.min_goodput > 0.0 else 1.0
            norm_gp = gp / base

            eff = ji.restart_factor * norm_gp
            if eff <= 0.0:
                continue

            try:
                val = eff ** p
            except OverflowError:
                val = 0.0

            total_obj += ji.fairness_weight * val

        return total_obj

    def optimize(
        self,
        runtime_jobs: List[RuntimeJobState],
        cluster_total_gpus: int,
    ) -> Dict[str, Tuple[int, int]]:
        """
        기존 Pollux greedy water-filling 구조 유지 + Sia objective만 교체.
        """
        if not runtime_jobs:
            self.log.info("[Sia] no jobs to schedule.")
            return {}

        job_infos = self.build_job_infos(runtime_jobs)
        jobinfo_by_id: Dict[str, JobInfo] = {j.job_id: j for j in job_infos}

        alloc_g: Dict[str, int] = {}
        total_g = 0

        for rj in runtime_jobs:
            ji = jobinfo_by_id.get(rj.job_id)
            if ji is None:
                g0 = max(1, rj.current_gpus)
            else:
                g0 = max(ji.min_gpus, min(rj.current_gpus, ji.max_gpus))
            alloc_g[rj.job_id] = g0
            total_g += g0

        obj = self._compute_objective(job_infos, alloc_g)

        # ---- shrink phase ----
        while total_g > cluster_total_gpus:
            best_job = None
            best_delta = None

            for ji in job_infos:
                j_id = ji.job_id
                g_curr = alloc_g.get(j_id, ji.min_gpus)
                if g_curr <= ji.min_gpus:
                    continue

                alloc_candidate = dict(alloc_g)
                alloc_candidate[j_id] = g_curr - 1

                obj_candidate = self._compute_objective(job_infos, alloc_candidate)
                delta = obj - obj_candidate

                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_job = j_id

            if best_job is None:
                self.log.warning(
                    "[Sia] shrink phase: cannot reduce GPUs further "
                    f"even though total_g={total_g} > cluster_total_gpus={cluster_total_gpus}"
                )
                break

            alloc_g[best_job] -= 1
            total_g -= 1
            obj = self._compute_objective(job_infos, alloc_g)

        # ---- expand phase ----
        while total_g < cluster_total_gpus:
            best_job = None
            best_gain = 0.0

            for ji in job_infos:
                j_id = ji.job_id
                g_curr = alloc_g.get(j_id, ji.min_gpus)
                if g_curr >= ji.max_gpus:
                    continue

                alloc_candidate = dict(alloc_g)
                alloc_candidate[j_id] = g_curr + 1

                obj_candidate = self._compute_objective(job_infos, alloc_candidate)
                gain = obj_candidate - obj

                if gain > best_gain:
                    best_gain = gain
                    best_job = j_id

            if best_job is None or best_gain <= 0.0:
                break

            alloc_g[best_job] += 1
            total_g += 1
            obj = self._compute_objective(job_infos, alloc_g)

        # ---- 최종 (g, local_batch) ----
        alloc_dict: Dict[str, Tuple[int, int]] = {}
        for ji in job_infos:
            j_id = ji.job_id
            new_g = int(alloc_g.get(j_id, ji.min_gpus))
            if new_g <= 0:
                new_g = ji.min_gpus

            gp, best_b = ji.goodput_fn.optimize(new_g)
            if best_b <= 0:
                best_b = 0  # fallback

            alloc_dict[j_id] = (new_g, best_b)

        self.log.info(f"[Sia] greedy allocation result (g, local_b): {alloc_dict}")
        return alloc_dict


def sia_reallocation_tick(
    cluster_id: str,
    cluster_total_gpus: int,
    runtime_jobs: List["RuntimeJobState"],
    scale_job_fn,
    now_ts: Optional[float] = None,

    # ✅ 아래는 global_server 쪽에서 넘기던 확장 인자들(호환용)
    cooldown_sec: Optional[int] = None,
    min_obj_gain: float = 0.0,
    max_scales_per_tick: int = 1,
    **kwargs,
) -> Dict[str, Tuple[int, int]]:
    """
    SIA reallocation tick (no cross-cluster migration).

    변경점(중요):
    - ✅ GANG job은 reallocation 대상에서 제외 (처음부터 끝까지 4 유지)
    - ✅ objective improvement(min_obj_gain) 없으면 스케일 안 함
    - ✅ cooldown 적용
    - ✅ tick당 scale 횟수 제한(max_scales_per_tick)
    - ✅ schedule_and_dispatch_jobs에서 넘기는 추가 kwargs 있어도 무시 (호환)
    """
    import time

    if now_ts is None:
        now_ts = time.time()

    if not runtime_jobs:
        log.info(f"[Sia] no running jobs in cluster {cluster_id}")
        return {}

    # cluster_id 정합성 체크 (migration 금지)
    for rj in runtime_jobs:
        if str(rj.cluster_id) != str(cluster_id):
            log.error(
                f"[Sia] runtime_jobs contains job {rj.job_id} with "
                f"mismatched cluster_id={rj.cluster_id} (expected {cluster_id})."
            )
            return {}

    # cooldown 기본값
    if cooldown_sec is None:
        cooldown_sec = SIA_COOLDOWN_SEC

    # ✅ GANG job 제외: "SIA baseline + gang 고정" 요구사항
    nongang_jobs: List["RuntimeJobState"] = []
    gang_jobs: List["RuntimeJobState"] = []
    for rj in runtime_jobs:
        if is_gang_model(rj.model_name, rj.dataset):
            gang_jobs.append(rj)
        else:
            nongang_jobs.append(rj)

    if not nongang_jobs:
        # gang만 있으면 reallocation 할 게 없음
        final_targets: Dict[str, Tuple[int, int]] = {}
        for rj in runtime_jobs:
            final_targets[rj.job_id] = (rj.current_gpus, rj.current_local_batch or 0)
        return final_targets

    sched = SiaScheduler()

    # ✅ optimize는 nongang만 대상으로 (gang을 건드리면 안 됨)
    alloc_dict = sched.optimize(nongang_jobs, int(cluster_total_gpus))
    if not alloc_dict:
        log.info(f"[Sia] no allocation change computed for cluster {cluster_id}")
        final_targets: Dict[str, Tuple[int, int]] = {}
        for rj in runtime_jobs:
            final_targets[rj.job_id] = (rj.current_gpus, rj.current_local_batch or 0)
        return final_targets

    # jobinfo 만들 때도 nongang만
    job_infos = sched.build_job_infos(nongang_jobs)
    jobinfo_by_id: Dict[str, "JobInfo"] = {j.job_id: j for j in job_infos}

    # 현재 / 신규 g 벡터 (nongang만 objective 비교)
    g_current: Dict[str, int] = {}
    for rj in nongang_jobs:
        ji = jobinfo_by_id.get(rj.job_id)
        if ji is not None:
            g_curr = max(ji.min_gpus, min(rj.current_gpus, ji.max_gpus))
        else:
            g_curr = max(rj.min_gpus, min(rj.current_gpus, rj.max_gpus))
        g_current[rj.job_id] = g_curr

    g_new: Dict[str, int] = {}
    for rj in nongang_jobs:
        desired = alloc_dict.get(rj.job_id, (rj.current_gpus, rj.current_local_batch or 0))
        desired_g, _ = desired
        ji = jobinfo_by_id.get(rj.job_id)
        if ji is not None:
            desired_g = max(ji.min_gpus, min(int(desired_g), ji.max_gpus))
        else:
            desired_g = max(rj.min_gpus, min(int(desired_g), rj.max_gpus))
        g_new[rj.job_id] = int(desired_g)

    obj_current = float(sched._compute_objective(job_infos, g_current))
    obj_new = float(sched._compute_objective(job_infos, g_new))
    gain = obj_new - obj_current

    if gain <= float(min_obj_gain):
        log.info(
            f"[Sia] skip scaling in cluster {cluster_id}: "
            f"obj_gain={gain:.6f} <= min_obj_gain={float(min_obj_gain):.6f} "
            f"(new={obj_new:.6f}, cur={obj_current:.6f})"
        )
        final_targets: Dict[str, Tuple[int, int]] = {}
        for rj in runtime_jobs:
            final_targets[rj.job_id] = (rj.current_gpus, rj.current_local_batch or 0)
        return final_targets

    # ✅ 실제 스케일 후보를 "gain 큰 것부터" 고르되 tick당 제한
    # (여기서는 단순히 '변화가 있는 job' 리스트로 만들고 앞에서부터 처리)
    candidates: List[Tuple[str, int, int]] = []
    for rj in nongang_jobs:
        desired_g, desired_b = alloc_dict.get(rj.job_id, (rj.current_gpus, rj.current_local_batch or 0))
        desired_g = int(desired_g)
        desired_b = int(desired_b or 0)

        curr_g = int(rj.current_gpus)
        curr_b = int(rj.current_local_batch or 0)

        if desired_g == curr_g and (desired_b == curr_b or desired_b == 0):
            continue
        candidates.append((rj.job_id, desired_g, desired_b))

    # tick당 scale 제한
    if max_scales_per_tick is None or int(max_scales_per_tick) <= 0:
        max_scales_per_tick = 1
    max_scales_per_tick = int(max_scales_per_tick)

    scaled_count = 0
    final_targets: Dict[str, Tuple[int, int]] = {}

    # 기본: 변화 없는 것들도 결과에 넣어줌
    for rj in runtime_jobs:
        final_targets[rj.job_id] = (int(rj.current_gpus), int(rj.current_local_batch or 0))

    for (jid, desired_g, desired_b) in candidates:
        if scaled_count >= max_scales_per_tick:
            break

        # runtime object 찾기
        rj = next((x for x in nongang_jobs if x.job_id == jid), None)
        if rj is None:
            continue

        # cooldown
        elapsed = float(now_ts - float(rj.last_scaled_at_ts or 0.0))
        if elapsed < float(cooldown_sec):
            log.info(
                f"[Sia] skip scaling job {jid}: cooldown "
                f"({elapsed:.1f}s < {int(cooldown_sec)}s)"
            )
            continue

        # bounds clamp
        ji = jobinfo_by_id.get(jid)
        if ji is not None:
            desired_g = max(int(ji.min_gpus), min(int(desired_g), int(ji.max_gpus)))
        else:
            desired_g = max(int(rj.min_gpus), min(int(desired_g), int(rj.max_gpus)))

        # batch는 0이면 "변경 없음"으로 취급
        if desired_b <= 0:
            desired_b = int(rj.current_local_batch or 0)

        # 실행
        try:
            ok = scale_job_fn(rj, int(desired_g), int(desired_b))
        except Exception as e:
            log.error(f"[Sia] scale_job_fn raised for job {jid}: {e}", exc_info=True)
            ok = False

        if ok:
            scaled_count += 1
            rj.last_scaled_at_ts = float(now_ts)
            rj.current_gpus = int(desired_g)
            rj.current_local_batch = int(desired_b)
            final_targets[jid] = (int(desired_g), int(desired_b))
            log.info(f"[Sia] scaled job {jid}: -> {desired_g} GPUs, local_batch={desired_b}")
        else:
            log.warning(f"[Sia] failed to scale job {jid} -> {desired_g} GPUs, local_batch={desired_b}")

    return final_targets