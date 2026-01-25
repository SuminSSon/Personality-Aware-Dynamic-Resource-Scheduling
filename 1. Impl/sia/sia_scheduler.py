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

def get_hardcoded_surface(model_name: str, cluster_id: str) -> Dict[int, Dict[int, float]]:
    # 1. 기존 32 배치 데이터 (보내주신 실제 값)
    base_data = {
        "clusterA": {
            "ResNet-50": {1: 17.05, 2: 18.26, 3: 15.01, 4: 18.08},
            "ResNet-18": {1: 31.09, 2: 30.54, 3: 29.53, 4: 30.15},
            "EfficientNetV2-S": {1: 8.5, 2: 8.98, 3: 7.84, 4: 8.85},
            "DistilBERT": {1: 16.84, 2: 15.71, 3: 9.36, 4: 8.91}
        },
        "clusterB": {
            "ResNet-50": {1: 11.93, 2: 9.54, 3: 7.43, 4: 7.27},
            "ResNet-18": {1: 28.87, 2: 17.35, 3: 13.81, 4: 13.11},
            "EfficientNetV2-S": {1: 7.82, 2: 6.55, 3: 6.07, 4: 6.06},
            "DistilBERT-base": {1: 17.14, 2: 4.52, 3: 3.17, 4: 3.04}
        }
    }

    model_results = base_data.get(cluster_id, {}).get(model_name, {})
    
    if not model_results:
        # 데이터가 아예 없는 모델은 기본값 반환
        return {g: {32: 10.0, 64: 12.0} for g in [1, 2, 4]}

    surface: Dict[int, Dict[int, float]] = {}
    
    for g, sps_32 in model_results.items():
        surface[g] = {32: sps_32}
        
        # [64 배치 데이터 추정 로직]
        # ClusterA (5070 Ti, 16GB): VRAM이 넉넉하지 않음. 64일 때 약 1.1배 상승 가정
        if cluster_id == "clusterA":
            sps_64 = sps_32 * 1.15 
        # ClusterB (A6000, 48GB): VRAM이 넉넉하여 대형 배치가 유리함. 약 1.25배 상승 가정
        else:
            sps_64 = sps_32 * 1.25
            
        surface[g][64] = round(sps_64, 2)

    return surface

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
    surface: Dict[int, Dict[int, float]] = {}
    
    # 1. DB에서 데이터 로드
    try:
        conn = _connect_prof_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT gpu_count, batch_size, AVG(throughput_sps) AS avg_sps
                    FROM minimal_profiling
                    WHERE model_name = %s AND dataset = %s AND cluster_id = %s
                    GROUP BY gpu_count, batch_size
                    """,
                    (model_name, dataset, cluster_id),
                )
                rows = cur.fetchall()
                for g, b, sps in rows:
                    surface.setdefault(int(g), {})[int(b)] = float(sps)
    except Exception as e:
        log.error(f"[Sia] DB error: {e}")

    # 2. 만약 DB가 비어있다면 하드코딩 데이터 사용
    if not surface:
        return get_hardcoded_surface(model_name, cluster_id)

    # 3. [핵심] DB에 32는 있는데 64가 없는 경우, 추정치로 채워넣기
    for g in surface:
        if 32 in surface[g] and 64 not in surface[g]:
            # ClusterA(5070ti)는 1.15배, ClusterB(A6000)는 1.25배 효율 상승 가정
            multiplier = 1.15 if cluster_id == "clusterA" else 1.25
            surface[g][64] = round(surface[g][32] * multiplier, 2)
            log.info(f"[Sia] Extrapolated 64 batch for {model_name} (g={g}) on {cluster_id}")

    return surface

class GoodputFunction:
    """
      gp_est(g, b_req) = scale_factor * sps_profile(g, b_profile=64) * stat_eff(g, b_req)
    """

    def __init__(
        self,
        model_name: str,
        dataset: str,
        cluster_id: str,
        scale_factor: float = 1.0,
        profile_base_batch: int = 64,
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
            self.surface = {
                1: {self.profile_base_batch: 1.0},
                2: {self.profile_base_batch: 1.8},
                4: {self.profile_base_batch: 3.0},
            }

        self._g_list = sorted(self.surface.keys())

        self._batch_cache: Dict[int, List[int]] = {
            g: sorted(bs.keys()) for g, bs in self.surface.items()
        }

        self._eff_gamma = 0.5

    def _closest_g(self, g: int) -> int:
        return min(self._g_list, key=lambda x: abs(x - g))

    def _get_profile_sps(self, g: int) -> float:
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
        g = self._closest_g(num_replicas)
        batch_dict = self.surface.get(g, {})
        if not batch_dict:
            return 0.0, self.profile_base_batch
        
        # 해당 g에서 SPS가 가장 높은 배치 사이즈 검색
        best_b = max(batch_dict, key=batch_dict.get)
        best_sps = batch_dict[best_b]
        
        return float(self.scale_factor) * float(best_sps), int(best_b)

@dataclass
class RuntimeJobState:
    job_id: str
    model_name: str
    dataset: str

    cluster_id: str

    current_gpus: int
    progress: float = 0.0
    attained_service: float = 0.0
    last_scaled_at_ts: float = 0.0

    min_gpus: int = 1
    max_gpus: int = 4

    current_local_batch: Optional[int] = None
    current_grad_accum: int = 1
    last_sps: float = 0.0

    num_restarts: int = 0
    total_run_time: float = 0.0
    total_restart_overhead: float = 0.0
    last_started_ts: float = 0.0

    current_gpu_type: Optional[str] = None

@dataclass
class JobInfo:
    job_id: str
    goodput_fn: GoodputFunction
    fairness_weight: float
    min_gpus: int
    max_gpus: int
    restart_factor: float = 1.0
    min_goodput: float = 1.0


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

    def build_job_infos(self, runtime_jobs):
        from types import SimpleNamespace
        job_infos = []
        
        for rj in runtime_jobs:
            # 1. Goodput 함수 생성
            gp_fn = GoodputFunction(
                model_name=rj.model_name, 
                dataset=rj.dataset, 
                cluster_id=rj.cluster_id
            )
            
            # 2. min_goodput 계산 (g=1일 때)
            min_gp, _ = gp_fn.optimize(1) 
            if min_gp <= 0: min_gp = 0.1
            
            # 3. 모든 필수 속성을 포함한 객체 생성
            info = SimpleNamespace(
                job_id=rj.job_id,
                model_name=rj.model_name,
                dataset=rj.dataset,
                current_gpus=rj.current_gpus if hasattr(rj, "current_gpus") else 0,
                min_gpus=getattr(rj, "min_gpus", 1),
                max_gpus=getattr(rj, "max_gpus", 4),
                attained_service=getattr(rj, "attained_service", 0.0),
                cluster_id=rj.cluster_id,
                goodput_fn=gp_fn,
                min_goodput=min_gp,
                # 재시작 오버헤드 계수 (기본값 1.0)
                restart_factor=getattr(rj, "restart_factor", 1.0),
                # 페어니스 가중치 (기본값 1.0, 에러 발생 지점)
                fairness_weight=getattr(rj, "fairness_weight", 1.0),
                # 혹시 모를 추가 필드 방어
                priority=getattr(rj, "priority", 1.0)
            )
            job_infos.append(info)
            
        return job_infos

    def _compute_objective(
        self,
        job_infos: List[JobInfo],
        alloc_g: Dict[str, int],
    ) -> float:
        """
        SIA 논문의 Objective Function (Section 4.1):
        U = sum( w_i * ( (r_i * GP_i(g)) / GP_i_min )^p )
        """
        p = 0.5  # SIA에서 권장하는 RHO 값
        total_obj = 0.0

        for ji in job_infos:
            g_i = alloc_g.get(ji.job_id, ji.min_gpus)
            if g_i <= 0: continue

            # 최적 배치 사이즈에서의 Goodput 계산
            gp, _ = ji.goodput_fn.optimize(g_i)
            if gp <= 0: continue

            # 정규화된 효율 계산 (SIA 핵심 수식)
            norm_gp = gp / ji.min_goodput
            
            # 리스타트 팩터(r_i)와 함께 목적 함수 계산
            # r_i는 이미 build_job_infos에서 계산되어 ji에 저장됨
            eff = ji.restart_factor * norm_gp
            total_obj += ji.fairness_weight * (eff ** p)

        return total_obj

    def optimize(
        self,
        runtime_jobs: List[RuntimeJobState],
        cluster_total_gpus: int,
    ) -> Dict[str, Tuple[int, int]]:
        if not runtime_jobs: return {}

        # 1. Job 정보 빌드 (여기서 r_i가 계산됨)
        job_infos = self.build_job_infos(runtime_jobs)
        if isinstance(job_infos, dict):
            # 이미 id를 키로 하는 딕셔너리라면 바로 사용
            jobinfo_by_id = job_infos 
        else:
            # 객체 리스트라면 기존 로직 유지
            jobinfo_by_id = {j.job_id: j for j in job_infos}

        # 2. 초기 할당 (현재 할당 상태 유지 또는 최소값 할당)
        alloc_g = {rj.job_id: max(rj.min_gpus, rj.current_gpus) for rj in runtime_jobs}
        total_g = sum(alloc_g.values())

        # 3. [Shrink Phase] 초과 자원 회수
        while total_g > cluster_total_gpus:
            best_job, best_delta = None, float('inf')
            current_obj = self._compute_objective(job_infos, alloc_g)

            for ji in job_infos:
                if alloc_g[ji.job_id] > ji.min_gpus:
                    temp_alloc = dict(alloc_g)
                    temp_alloc[ji.job_id] -= 1
                    delta = current_obj - self._compute_objective(job_infos, temp_alloc)
                    if delta < best_delta:
                        best_delta, best_job = delta, ji.job_id
            
            if not best_job: break
            alloc_g[best_job] -= 1
            total_g -= 1

        # 4. [Expand Phase] 여유 자원 배분
        while total_g < cluster_total_gpus:
            best_job, best_gain = None, 0.0
            current_obj = self._compute_objective(job_infos, alloc_g)

            for ji in job_infos:
                if alloc_g[ji.job_id] < ji.max_gpus:
                    temp_alloc = dict(alloc_g)
                    temp_alloc[ji.job_id] += 1
                    gain = self._compute_objective(job_infos, temp_alloc) - current_obj
                    
                    # [핵심 수정]: 리스타트 페널티 대비 이득 검증
                    # SIA 논문 컨셉: Gain이 리스타트 오버헤드를 상쇄할 만큼 큰가?
                    # 여기서는 단순 Gain이 아닌, r_i 페널티가 적용된 점수차를 이용
                    if gain > best_gain:
                        best_gain, best_job = gain, ji.job_id

            if not best_job or best_gain <= 1e-6: break
            alloc_g[best_job] += 1
            total_g += 1

        # 5. 최종 결과 도출 (Best Batch Size 포함)
        return {jid: (g, jobinfo_by_id[jid].goodput_fn.optimize(g)[1]) 
                for jid, g in alloc_g.items()}

def sia_reallocation_tick(
    cluster_id: str,
    cluster_total_gpus: int,
    runtime_jobs: List[RuntimeJobState],
    scale_job_fn,
    now_ts: Optional[float] = None,
    cooldown_sec: Optional[int] = None,
    min_obj_gain: float = 0.001,
    max_scales_per_tick: int = 1,
    **kwargs,
) -> Dict[str, Tuple[int, int]]:
    import time
    if now_ts is None: now_ts = time.time()
    if cooldown_sec is None: cooldown_sec = SIA_COOLDOWN_SEC

    if not runtime_jobs:
        return {}

    # 1. Gang Job(고정 자원)과 Non-Gang Job(가변 자원) 분리
    nongang_jobs = [rj for rj in runtime_jobs if not is_gang_model(rj.model_name, rj.dataset)]
    gang_jobs = [rj for rj in runtime_jobs if is_gang_model(rj.model_name, rj.dataset)]

    # 2. 가용 GPU 계산 (전체 - Gang 점유분)
    gang_occupied = sum(rj.current_gpus for rj in gang_jobs)
    available_gpus = max(0, cluster_total_gpus - gang_occupied)

    if not nongang_jobs:
        return {rj.job_id: (rj.current_gpus, rj.current_local_batch or 0) for rj in runtime_jobs}

    # 3. SIA 스케줄러 실행 (목적 함수 최적화)
    sched = SiaScheduler()
    # optimize 내부에서 r_i(restart factor)가 반영된 Greedy 할당 수행
    alloc_dict = sched.optimize(nongang_jobs, available_gpus)

    # 4. Objective 비교를 위한 JobInfo 빌드
    job_infos = sched.build_job_infos(nongang_jobs)
    jobinfo_by_id = {j.job_id: j for j in job_infos}

    # 5. 현재 상태 vs 제안된 상태의 점수(Objective) 계산
    g_current = {rj.job_id: rj.current_gpus for rj in nongang_jobs}
    g_new = {jid: g for jid, (g, b) in alloc_dict.items()}

    obj_current = sched._compute_objective(job_infos, g_current)
    obj_new = sched._compute_objective(job_infos, g_new)
    total_gain = obj_new - obj_current

    # 6. 스케일링 후보군 선별 및 리스타트 비용 검증
    candidates = []
    for rj in nongang_jobs:
        desired_g, desired_b = alloc_dict.get(rj.job_id, (rj.current_gpus, rj.current_local_batch or 0))
        
        # 변화가 있는 작업만 후보 등록
        if desired_g != rj.current_gpus or (desired_b != rj.current_local_batch and desired_b > 0):
            # 개별 작업의 Gain 기여도 계산 (단순 근사)
            ji = jobinfo_by_id[rj.job_id]
            
            # 리스타트 페널티(r_i)가 클수록(r_i가 작을수록) 더 큰 gain이 필요함
            # SIA 논문: 이득이 리스타트 비용을 상쇄할 수 있는가?
            penalty_threshold = min_obj_gain / ji.restart_factor 
            
            candidates.append({
                "rj": rj,
                "ji": ji,
                "desired_g": desired_g,
                "desired_b": desired_b,
                "penalty_threshold": penalty_threshold
            })

    # 전체 gain이 최소 기준 미달이면 중단
    if total_gain < min_obj_gain:
        log.info(f"[Sia] Total gain {total_gain:.6f} is too low. Skipping.")
        return {rj.job_id: (rj.current_gpus, rj.current_local_batch or 0) for rj in runtime_jobs}

    # 7. 실제 스케일링 적용 (Max Scale 제한 및 Cooldown 고려)
    final_targets = {rj.job_id: (rj.current_gpus, rj.current_local_batch or 0) for rj in runtime_jobs}
    scaled_count = 0

    # Gain 기여도가 높을 것으로 예상되는 후보부터 정렬 (여기서는 간단히 차이값 기준 가능)
    for cand in candidates:
        if scaled_count >= max_scales_per_tick: break

        rj, ji = cand["rj"], cand["ji"]
        
        # Cooldown 시간 체크
        if now_ts - rj.last_scaled_at_ts < cooldown_sec:
            log.info(f"[Sia] Job {rj.job_id} in cooldown. Skipping.")
            continue

        # 개별 작업의 리스타트 위험도 검증
        if total_gain < cand["penalty_threshold"]:
            log.info(f"[Sia] Job {rj.job_id} gain not enough to cover restart risk.")
            continue

        # 스케일링 실행
        ok = scale_job_fn(rj, cand["desired_g"], cand["desired_b"])
        if ok:
            rj.last_scaled_at_ts = now_ts
            rj.current_gpus = cand["desired_g"]
            rj.current_local_batch = cand["desired_b"]
            final_targets[rj.job_id] = (cand["desired_g"], cand["desired_b"])
            scaled_count += 1
            log.info(f"[Sia] Scaled {rj.job_id} -> {cand['desired_g']} GPUs (Gain: {total_gain:.4f})")

    return final_targets