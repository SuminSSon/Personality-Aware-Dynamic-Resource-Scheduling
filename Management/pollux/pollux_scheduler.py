# app/pollux.py
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import psycopg2

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

log = logging.getLogger("POLLUX")
log.setLevel(logging.INFO)

# 1. 프로파일 DB 연결 & Goodput(throughput) curve 로딩
PROF_DB_DSN = os.getenv(
    "PROF_DB_DSN",
    "postgresql://prof:profpw@163.180.117.216:5432/profdb",
)

def _connect_prof_db():
    if not PROF_DB_DSN:
        raise RuntimeError("PROF_DB_DSN is not set")
    return psycopg2.connect(PROF_DB_DSN)

def load_sps_curve(
    model_name: str,
    dataset: str,
    cluster_id: str,
) -> Dict[int, float]:
    """
    minimal_profiling에서 (model_name, dataset, cluster_id)에 해당하는
    gpu_count별 평균 throughput_sps를 {g: sps}로 반환.
    """
    try:
        conn = _connect_prof_db()
    except Exception as e:
        log.error(f"[Pollux] profiling DB connect failed in load_sps_curve: {e}")
        return {}

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT gpu_count, AVG(throughput_sps) AS avg_sps
                    FROM minimal_profiling
                    WHERE model_name = %s
                      AND dataset    = %s
                      AND cluster_id = %s
                      AND throughput_sps IS NOT NULL
                    GROUP BY gpu_count
                    ORDER BY gpu_count ASC
                    """,
                    (model_name, dataset, cluster_id),
                )
                rows = cur.fetchall()
    except Exception as e:
        log.error(f"[Pollux] error querying profiling DB in load_sps_curve: {e}")
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    curve: Dict[int, float] = {}
    for g, avg_sps in rows:
        if g is None or avg_sps is None:
            continue
        curve[int(g)] = float(avg_sps)

    if not curve:
        log.warning(
            f"[Pollux] no profiling rows for model={model_name}, dataset={dataset}, "
            f"cluster_id={cluster_id}"
        )
    return curve

class GoodputFunction:
    """
    minimal_profiling 기반 Pollux-style goodput(g) 함수.
      - model_name, dataset, cluster_id로 gpu_count별 avg throughput_sps를 로드.
      - goodput(g) ≈ throughput_sps(g) (통계 효율은 일단 1로 근사).
    """

    def __init__(self, model_name: str, dataset: str, cluster_id: str):
        self.model_name = model_name
        self.dataset = dataset
        self.cluster_id = cluster_id

        self.curve = load_sps_curve(model_name, dataset, cluster_id)

        if not self.curve:
            # profiling 없으면 대충 스케일링 fallback
            log.warning(
                f"[Pollux] empty sps curve; fallback for model={model_name}, dataset={dataset}"
            )
            self.curve = {1: 1.0, 2: 1.8, 4: 3.0}

        self._g_list = sorted(self.curve.keys())
        base_g = self._g_list[0]
        self._goodput_1 = self.curve[base_g]

    def goodput(self, num_replicas: int) -> float:
        if num_replicas <= 0:
            return 0.0
        if num_replicas in self.curve:
            return self.curve[num_replicas]

        # 없는 g는 가장 가까운 g로 근사
        closest_g = min(self._g_list, key=lambda x: abs(x - num_replicas))
        return self.curve[closest_g]

    def optimize(self, num_replicas: int):
        """
        SpeedupFunction 같은 래퍼와 호환용: (goodput, dummy_batch_size) 튜플.
        """
        gp = self.goodput(num_replicas)
        return gp, 0.0

# 2. 런타임 job 정보 + fairness state 정의
@dataclass
class RuntimeJobState:
    """
    Pollux 재할당 tick에서 사용되는 job 상태 (Cluster 내부).
    """
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


@dataclass
class JobInfo:
    job_id: str
    goodput_fn: GoodputFunction
    fairness_weight: float
    min_gpus: int
    max_gpus: int


# 3. PolluxProblem: fairness-weighted goodput 최대화
class PolluxProblem(Problem):
    """
    Pollux-style 최적화 문제:
      - 변수: 각 job의 GPU 개수 g_j (정수)
      - 제약: ∑ g_j <= total_gpus
      - 목적: ∑ w_j * goodput_j(g_j) 최대화  (여기서는 minimize(-sum))
    """

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

            # 목적: fairness-weighted goodput 합을 최대화
            total_goodput = 0.0
            for i in range(num_jobs):
                g_i = alloc[i]
                job = self.jobs[i]
                if g_i <= 0:
                    continue
                gp = job.goodput_fn.goodput(g_i)
                w = job.fairness_weight
                total_goodput += w * gp

            # pymoo는 minimize 문제이므로 부호 반전
            F[k, 0] = -total_goodput

        out["F"] = F
        out["G"] = G


# 4. PolluxScheduler: NSGA2 로 최적 g_j 찾기
class PolluxScheduler:
    """
    한 클러스터 내부에서 Pollux-style로 GPU를 재분배하는 스케줄러.
    """

    def __init__(self):
        self.log = logging.getLogger("POLLUX.SCHED")
        self.log.setLevel(logging.INFO)

    def build_job_infos(self, runtime_jobs: List[RuntimeJobState]) -> List[JobInfo]:
        jobs: List[JobInfo] = []

        for rj in runtime_jobs:
            # 1) profiling 기반 goodput_fn 생성
            gp_fn = GoodputFunction(
                model_name=rj.model_name,
                dataset=rj.dataset,
                cluster_id=rj.cluster_id,
            )

            # 2) attained-service 기반 fairness weight
            #    - 많이 서비스 받은 job일수록 weight 감소
            #    - 1 / (1 + attained_service) 형태로 단순 근사
            w = 1.0 / (1.0 + max(0.0, rj.attained_service))

            jobs.append(
                JobInfo(
                    job_id=rj.job_id,
                    goodput_fn=gp_fn,
                    fairness_weight=w,
                    min_gpus=max(1, rj.min_gpus),
                    max_gpus=max(rj.min_gpus, rj.max_gpus),
                )
            )

        return jobs

    def optimize(
        self,
        runtime_jobs: List[RuntimeJobState],
        cluster_total_gpus: int,
    ) -> Dict[str, int]:
        """
        runtime_jobs: 현재 클러스터에서 실행 중인 job 상태들
        cluster_total_gpus: 이 클러스터 전체 GPU 개수 (ex. 4)

        반환: {job_id: new_gpus}
        """
        if not runtime_jobs:
            self.log.info("[Pollux] no jobs to schedule.")
            return {}

        job_infos = self.build_job_infos(runtime_jobs)

        problem = PolluxProblem(
            jobs=job_infos,
            total_gpus=cluster_total_gpus,
        )

        # pop_size를 job 수 기반으로 설정
        pop_size = max(20, len(job_infos) * 10)
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=0.9, eta=15, vtype=int),
            mutation=PM(prob=0.1, eta=20, vtype=int),
            eliminate_duplicates=True,
        )

        res = minimize(
            problem,
            algorithm,
            ("n_gen", 60),
            verbose=False,
        )

        if res.F is None or len(res.F) == 0:
            self.log.error("[Pollux] optimization failed; keep current allocation.")
            return {rj.job_id: rj.current_gpus for rj in runtime_jobs}

        # 단일 objective이므로 F[:,0] 최소값 선택
        best_idx = int(np.argmin(res.F[:, 0]))
        best_alloc = res.X[best_idx].astype(int)

        alloc_dict: Dict[str, int] = {}
        for i, rj in enumerate(runtime_jobs):
            new_g = int(best_alloc[i])
            alloc_dict[rj.job_id] = new_g

        self.log.info(f"[Pollux] optimization result: {alloc_dict}")
        return alloc_dict

# 5. 주기적 재할당 tick 헬퍼
POLLUX_COOLDOWN_SEC = int(os.getenv("POLLUX_COOLDOWN_SEC", "180"))

def pollux_reallocation_tick(
    cluster_id: str,
    cluster_total_gpus: int,
    runtime_jobs: List[RuntimeJobState],
    scale_job_fn,
    now_ts: Optional[float] = None,
) -> Dict[str, int]:
    if now_ts is None:
        now_ts = time.time()

    if not runtime_jobs:
        log.info(f"[Pollux] no running jobs in cluster {cluster_id}")
        return {}

    # 1) PolluxScheduler로 새 GPU 배분 계산
    sched = PolluxScheduler()
    alloc_dict = sched.optimize(runtime_jobs, cluster_total_gpus)

    if not alloc_dict:
        log.info(f"[Pollux] no allocation change computed for cluster {cluster_id}")
        return {}

    # 2) world_size 변경 적용 (쿨다운 포함)
    final_targets: Dict[str, int] = {}

    for rj in runtime_jobs:
        desired_g = alloc_dict.get(rj.job_id, rj.current_gpus)
        # min/max clamp
        desired_g = max(rj.min_gpus, min(desired_g, rj.max_gpus))
        final_targets[rj.job_id] = desired_g

        # 변화 없으면 스킵
        if desired_g == rj.current_gpus:
            continue

        # 쿨다운 체크
        elapsed = now_ts - rj.last_scaled_at_ts
        if elapsed < POLLUX_COOLDOWN_SEC:
            log.info(
                f"[Pollux] skip scaling job {rj.job_id}: cooldown not expired "
                f"({elapsed:.1f}s < {POLLUX_COOLDOWN_SEC}s)"
            )
            continue

        # 실제 scale 수행
        try:
            ok = scale_job_fn(rj, desired_g)
        except Exception as e:
            log.error(f"[Pollux] scale_job_fn raised for job {rj.job_id}: {e}", exc_info=True)
            ok = False

        if ok:
            log.info(
                f"[Pollux] scaled job {rj.job_id}: {rj.current_gpus} -> {desired_g} GPUs "
                f"(cluster={cluster_id})"
            )
        else:
            log.warning(
                f"[Pollux] failed to scale job {rj.job_id} to {desired_g} GPUs "
                f"(cluster={cluster_id})"
            )

    return final_targets
