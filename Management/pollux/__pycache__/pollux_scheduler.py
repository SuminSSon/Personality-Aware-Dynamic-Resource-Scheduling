# pollux_scheduler.py (pymoo 임포트 최종 수정)

import numpy as np
import logging
from typing import Dict, List, Any

# [수정] pymoo 임포트 경로를 모두 수정합니다.
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2  # <- .moo. 추가
from pymoo.optimize import minimize

# [수정] factory 대신 실제 연산자(operator) 클래스를 직접 임포트합니다.
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# (GoodputFunction, SpeedupFunction, PolluxProblem 클래스는 이전과 동일)

class GoodputFunction:
    """
    논문의 Equation 4 (Goodput) 모델을 구현한 클래스 (Mockup).
    """
    def __init__(self, job_profiling_data, current_progress):
        self.params = job_profiling_data
        self.progress = current_progress

    def optimize(self, num_replicas):
        if num_replicas == 0:
            return 0.0, 0
        t_grad = self.params.get("t_grad", 0.1)
        t_sync = self.params.get("t_sync", 0.05)
        step_time_k = t_grad + t_sync * (num_replicas - 1) / num_replicas
        step_time_1 = t_grad
        system_throughput = step_time_1 / step_time_k
        gns_slope = self.params.get("gns_slope", 0.001)
        stat_efficiency = max(0.1, 1.0 - self.progress * gns_slope * num_replicas)
        goodput = system_throughput * stat_efficiency
        return goodput, 512

class SpeedupFunction:
    """
    Goodput 모델을 래핑하여 스케줄러가 호출하는 speedup 함수
    """
    def __init__(self, goodput_fn: GoodputFunction):
        self._goodput_fn = goodput_fn
        goodput_1_tuple = goodput_fn.optimize(num_replicas=1)
        self._goodput_1 = goodput_1_tuple[0] if goodput_1_tuple else 0.0
        self._cache = {}

    def __call__(self, num_nodes, num_replicas):
        if num_replicas not in self._cache:
            goodput_tuple = self._goodput_fn.optimize(num_replicas)
            goodput = goodput_tuple[0] if goodput_tuple else 0.0
            
            if self._goodput_1 == 0:
                speedup = 0.0
            else:
                speedup = goodput / self._goodput_1
            self._cache[num_replicas] = speedup
        return self._cache[num_replicas]

class PolluxProblem(Problem):
    """
    Pollux 유전 알고리즘(NSGA2)이 풀 최적화 문제 (Equation 14)
    """
    def __init__(self, jobs, nodes, max_replicas):
        super().__init__(n_var=len(jobs), n_obj=2, n_constr=1, xl=0, xu=max_replicas, vtype=int) # vtype=int 추가
        self.jobs = jobs
        self.nodes = nodes
        self.total_gpus = sum(n.resources.get("gpu", 0) for n in nodes)

    def _evaluate(self, states, out, *args, **kwargs):
        pop_size = states.shape[0]
        num_jobs = states.shape[1]
        
        objectives = np.zeros((pop_size, self.n_obj))
        constraints = np.zeros((pop_size, self.n_constr))

        states = states.astype(int)
        
        for k in range(pop_size):
            alloc = states[k]
            
            # Constraint 1: 총 할당 GPU <= 전체 GPU
            constraints[k, 0] = np.sum(alloc) - self.total_gpus
            
            # Objective 1: Goodput(Speedup) 합계 최대화
            total_speedup = 0
            for i in range(num_jobs):
                num_replicas = alloc[i]
                if num_replicas > 0:
                    job = self.jobs[i]
                    total_speedup += job.speedup_fn(num_replicas, num_replicas)
            
            objectives[k, 0] = -total_speedup
            
            # Objective 2: 사용 노드(Job) 수 최소화
            objectives[k, 1] = np.count_nonzero(alloc)

        out["F"] = objectives
        out["G"] = constraints

# --- 래핑 클래스 (Global Server가 사용할 인터페이스) ---

class JobInfo:
    def __init__(self, job_id, speedup_fn, min_replicas=1, max_replicas=16):
        self.job_id = job_id
        self.speedup_fn = speedup_fn
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

class NodeInfo:
    def __init__(self, node_id, resources: Dict[str, int]):
        self.node_id = node_id
        self.resources = resources

class PolluxScheduler:
    """
    Global Server가 사용할 Pollux 스케줄러 인터페이스
    """
    def __init__(self):
        self.log = logging.getLogger("PolluxScheduler")
        self.log.setLevel(logging.INFO) 
        
        # [수정] factory 대신 실제 클래스를 인스턴스화합니다.
        self.algorithm = NSGA2(
            pop_size=50,
            sampling=IntegerRandomSampling(), # "int_random"
            crossover=SBX(prob=0.9, eta=15, vtype=int), # "int_sbx"
            mutation=PM(prob=0.1, eta=20, vtype=int),   # "int_pm"
            eliminate_duplicates=True
        )

    def optimize(self, jobs: List[JobInfo], nodes: List[NodeInfo]) -> Dict[str, int]:
        self.log.info(f"Starting Pollux optimization for {len(jobs)} jobs and {len(nodes)} nodes...")
        
        if not jobs:
            self.log.info("No jobs to schedule.")
            return {}

        total_gpus = sum(n.resources.get("gpu", 0) for n in nodes)
        if total_gpus == 0:
            self.log.warning("No GPU resources available in nodes.")
            return {job.job_id: 0 for job in jobs}

        max_replicas_per_job = [min(job.max_replicas, total_gpus) for job in jobs]
        
        problem = PolluxProblem(jobs, nodes, np.array(max_replicas_per_job))
        
        res = minimize(
            problem,
            self.algorithm,
            ('n_gen', 100),
            verbose=False
        )

        if res.F is None or len(res.F) == 0:
            self.log.error("Pollux optimization failed to find a solution.")
            return {job.job_id: 0 for job in jobs}

        best_solution_index = np.argmin(res.F[:, 0])
        best_allocation_vector = res.X[best_solution_index].astype(int)
        
        final_allocations = {}
        for i, job in enumerate(jobs):
            final_allocations[job.job_id] = int(best_allocation_vector[i]) # int()로 명시적 변환
            
        self.log.info(f"Pollux optimization complete. Allocations: {final_allocations}")
        
        return final_allocations