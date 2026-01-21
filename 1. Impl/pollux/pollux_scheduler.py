from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable, Dict, Tuple, Optional
import threading
import math
import time
import random
from collections import defaultdict

# job_id → (g, local_batch) → {"ema": float, "count": int}

MIN_G = 1 
POLLUX_WARMUP_SEC = 120
POLLUX_SCALE_COOLDOWN_SEC = 0

def _eligible_for_elastic_scaling(job: RuntimeJobState, now: float) -> bool:
    # queued(=아직 start 안 한 job)는 reallocation에서 receiver로 올라갈 수 있어야 함
    status = str(getattr(job, "status", "running") or "running").lower()
    if status in ("queued", "pending"):
        return True

    start_ts = float(getattr(job, "start_ts", 0.0) or 0.0)
    if start_ts <= 0.0:
        return False
    if (now - start_ts) < POLLUX_WARMUP_SEC:
        return False
    return True

@dataclass
class RuntimeJobState:
    job_id: str
    model_name: str
    dataset: str
    cluster_id: str

    current_gpus: int
    current_local_batch: int
    current_grad_accum: int = 1

    progress: float = 0.0
    attained_service: float = 0.0
    last_scaled_at_ts: float = 0.0

    start_ts: float = 0.0
    # 기본적으로 g는 1~4 범위라고 보고 감
    min_gpus: int = 1
    max_gpus: int = 4

    last_sps: float = 0.0
    last_metric_ts: float = 0.0

    checkpoint_mb: float = 0.0

    # elastic 필터에서 쓰는 상태 값
    status: str = "running"   # GlobalServer에서 따로 안 건드려도 running으로 처리

# ==== Gang-fixed jobs 설정 ====
# (model_name, dataset) 기준으로 "무조건 g=4, 스케일 금지"인 job 정의
GANG_JOBS = {
    ("DenseNet-121", "TinyImageNet"): 4,
}

def is_gang_model(model_name: str, dataset: str) -> bool:
    return (model_name, dataset) in GANG_JOBS

def is_gang_job(job: RuntimeJobState) -> bool:
    return is_gang_model(job.model_name, job.dataset)

ALLOWED_G = [1, 2, 4]

def _next_up_g(g: int) -> Optional[int]:
    """현재 g에서 '위쪽'으로 가장 가까운 허용 g (없으면 None)"""
    for x in ALLOWED_G:
        if x > g:
            return x
    return None

def _next_down_g(g: int) -> Optional[int]:
    """현재 g에서 '아래쪽'으로 가장 가까운 허용 g (없으면 None)"""
    prev = None
    for x in ALLOWED_G:
        if x >= g:
            break
        prev = x
    return prev

class PolluxGoodputProfiler:
    def __init__(self, alpha: float = 0.5, explore_prob: float = 0.0):
        """
        ✅ 재시작(stop+requeue) 기반 시스템에서는 탐색(explore)이 오버헤드 폭탄이므로 0으로 둔다.
        """
        self._lock = threading.Lock()
        # key = (job_id, g, local_batch, accum) -> (sps, stat_eff, goodput)
        self._gp: Dict[Tuple[str, int, int, int], Tuple[float, float, float]] = {}
        self.alpha = alpha
        self.explore_prob = float(explore_prob)  # ✅ default 0.0

        # g-only prior (fallback)
        self._prior_by_g: Dict[int, List[Tuple[int, int, float, float]]] = {
            (1): [(32, 1, 120.0, 1.00)],
            (2): [(32, 1, 220.0, 0.98)],
            (4): [(32, 1, 400.0, 0.95)],
            (8): [(32, 1, 720.0, 0.90)],
        }

        # (model, dataset, g) prior ...
        self._prior_by_mdg: Dict[
            Tuple[str, str, int],
            List[Tuple[int, int, float, float]]
        ] = {
            # (당신이 적어둔 prior 그대로 유지)
            ("ResNet-50", "CIFAR-10", 1): [(32, 1, 17.0, 1.00)],
            ("ResNet-50", "CIFAR-10", 2): [(32, 1, 30.6, 0.90)],
            ("ResNet-50", "CIFAR-10", 4): [(32, 1, 37.4, 0.65)],

            ("EfficientNetV2-S", "MNIST", 1): [(32, 1, 8.5, 1.00)],
            ("EfficientNetV2-S", "MNIST", 2): [(32, 1, 14.5, 0.90)],
            ("EfficientNetV2-S", "MNIST", 4): [(32, 1, 19.0, 0.70)],

            ("ResNet-18", "CIFAR-100", 1): [(32, 1, 31.1, 1.00)],
            ("ResNet-18", "CIFAR-100", 2): [(32, 1, 52.9, 0.90)],
            ("ResNet-18", "CIFAR-100", 4): [(32, 1, 80.9, 0.75)],

            ("DeepSpeech2", "ARCTIC", 1): [(32, 1, 9.8, 1.00)],
            ("DeepSpeech2", "ARCTIC", 2): [(32, 1, 15.7, 0.90)],
            ("DeepSpeech2", "ARCTIC", 4): [(32, 1, 20.6, 0.70)],

            ("DenseNet-121", "TinyImageNet", 1): [(32, 1, 10.7, 1.00)],
            ("DenseNet-121", "TinyImageNet", 2): [(32, 1, 18.2, 0.90)],
            ("DenseNet-121", "TinyImageNet", 4): [(32, 1, 26.8, 0.70)],

            ("DistilBERT-base", "SST-2", 1): [(32, 1, 16.8, 1.00)],
            ("DistilBERT-base", "SST-2", 2): [(32, 1, 30.2, 0.90)],
            ("DistilBERT-base", "SST-2", 4): [(32, 1, 47.0, 0.70)],

            ("DistilBERT", "SST-2", 1): [(32, 1, 16.8, 1.00)],
            ("DistilBERT", "SST-2", 2): [(32, 1, 30.2, 0.90)],
            ("DistilBERT", "SST-2", 4): [(32, 1, 47.0, 0.70)],
        }

    @staticmethod
    def _stat_eff_from_gns(
        g: int,
        local_batch: int,
        accum: int,
        gns: float,
    ) -> float:
        """
        Pollux에서 쓰는 GNS 기반 stat. efficiency 근사.
        """
        if gns <= 0:
            return 1.0
        eff_batch = max(1.0, float(g) * float(local_batch) * float(accum))
        se = gns / (gns + eff_batch)
        se = max(0.05, min(se, 1.0))
        return se

    def _candidate_configs_for_g(
        self,
        default_batch: int,
        default_accum: int,
        g: int,
        max_global_batch: int = 4096,
    ) -> List[Tuple[int, int]]:
        """
        (g, local_batch, accum)에 대해 탐색할 후보 집합 생성.
        - default_batch, default_accum 주변의 몇 개 배치/accum 조합.
        - Pollux autotuner가 하는 'config search space' 역할의 단순 버전.
        """
        base_b = max(4, default_batch)
        base_a = max(1, default_accum)

        batch_candidates = sorted({
            max(4, min(512, base_b // 2)),
            max(4, min(512, base_b)),
            max(4, min(512, base_b * 2)),
        })

        accum_candidates = {1, base_a}
        if base_a == 1:
            accum_candidates.add(2)
        elif base_a < 4:
            accum_candidates.add(base_a * 2)

        configs: List[Tuple[int, int]] = []
        for b in batch_candidates:
            for a in accum_candidates:
                B = g * b * a
                if B <= 0:
                    continue
                if B > max_global_batch:
                    continue
                configs.append((int(b), int(a)))

        # fallback (최소 하나는 보장)
        if not configs:
            configs.append((base_b, base_a))

        return configs

    def _estimate_goodput_from_prior(
        self,
        model_name: str,
        dataset: str,
        g: int,
        local_batch: int,
        accum: int,
        default_sps: float,
    ) -> Tuple[float, float, float]:
        """
        prior (model, dataset, g) 기반으로 goodput(B) surface를 근사.
        아이디어:
          - prior에서 (b0, a0, sps0, se0) 하나 가져옴
          - global batch B0 = g * b0 * a0, B = g * local_batch * accum
          - se(B) ~ B_crit / (B_crit + B)  (Pollux의 GNS 기반 곡선 흉내)
          - B_crit은 se0를 만족하도록 역산
        """
        key_mdg = (model_name, dataset, g)
        prior_list = self._prior_by_mdg.get(key_mdg)
        if not prior_list:
            prior_list = self._prior_by_g.get(g, [])

        if prior_list:
            b0, a0, sps0, se0 = max(prior_list, key=lambda x: x[2] * x[3])
            sps0 = float(sps0)
            se0 = max(0.01, min(float(se0), 0.99))
        else:
            # prior도 없으면 default_sps만 사용
            sps0 = max(default_sps, 1e-3)
            se0 = 0.9
            b0, a0 = local_batch, accum

        B0 = max(1.0, float(g) * float(b0) * float(a0))
        B  = max(1.0, float(g) * float(local_batch) * float(accum))

        # se0 = B_crit / (B_crit + B0)  →  B_crit = se0 * B0 / (1 - se0)
        B_crit = se0 * B0 / max(1e-6, (1.0 - se0))
        se = B_crit / (B_crit + B)
        se = max(0.01, min(se, 1.0))

        sps = sps0  # sps는 g에 주로 비례한다고 가정하고, batch에 대해서는 se로만 조정
        gp = sps * se

        return float(sps), float(se), float(gp)

    def record_goodput_sample(
        self,
        job_id: str,
        g: int,
        local_batch: int,
        accum: int,
        sps: float,
        gns: Optional[float] = None,
        stat_eff: Optional[float] = None,
    ) -> None:
        """
        학습 코드에서 보내는 (sps, gns/stat_eff)를 EMA로 적재.
        Pollux의 online goodput profiler와 같은 역할.
        """
        if g <= 0 or local_batch <= 0 or accum <= 0 or sps <= 0:
            return

        sps = float(sps)
        g = int(g)
        local_batch = int(local_batch)
        accum = int(accum)

        if stat_eff is None:
            if gns is not None:
                stat_eff = self._stat_eff_from_gns(g, local_batch, accum, float(gns))
            else:
                stat_eff = 1.0

        stat_eff = float(stat_eff)
        stat_eff = max(0.01, min(stat_eff, 1.0))
        goodput = sps * stat_eff

        key = (job_id, g, local_batch, accum)

        with self._lock:
            prev = self._gp.get(key)
            if prev is None:
                self._gp[key] = (sps, stat_eff, goodput)
            else:
                a = self.alpha
                prev_sps, prev_se, prev_gp = prev
                new_sps = a * sps + (1.0 - a) * prev_sps
                new_se  = a * stat_eff + (1.0 - a) * prev_se
                new_gp  = a * goodput + (1.0 - a) * prev_gp
                self._gp[key] = (new_sps, new_se, new_gp)

    def best_for_g(
        self,
        job_id: str,
        model_name: str,
        dataset: str,
        g: int,
        default_batch: int,
        default_accum: int,
        default_sps: float,
    ) -> Tuple[int, int, float, float, float]:
        """
        주어진 g에서 (local_batch, accum)을 Pollux 스타일로 고름.
        - candidate (b,a) 집합 생성
        - 관측된 goodput 있으면 그 값 사용
        - 없으면 prior 기반 surface로 goodput 예측
        - ε-greedy: 일부는 '관측 안 된 config'를 선택해 탐색
        반환: (best_batch, best_accum, sps, stat_eff, goodput)
        """
        default_batch = max(1, int(default_batch))
        default_accum = max(1, int(default_accum))
        default_sps = float(default_sps)

        candidates = self._candidate_configs_for_g(
            default_batch=default_batch,
            default_accum=default_accum,
            g=g,
        )

        scored = []  # (b, a, sps, se, gp, seen)
        with self._lock:
            for b, a in candidates:
                key = (job_id, g, b, a)
                prev = self._gp.get(key)
                if prev is not None:
                    sps, se, gp = prev
                    sps = max(float(sps), 1e-3)
                    se  = max(float(se), 0.01)
                    gp  = max(float(gp), 1e-3)
                    scored.append((b, a, sps, se, gp, True))
                else:
                    # prior 기반 surface로 예측
                    sps, se, gp = self._estimate_goodput_from_prior(
                        model_name=model_name,
                        dataset=dataset,
                        g=g,
                        local_batch=b,
                        accum=a,
                        default_sps=default_sps,
                    )
                    sps = max(float(sps), 1e-3)
                    se  = max(float(se), 0.01)
                    gp  = max(float(gp), 1e-3)
                    scored.append((b, a, sps, se, gp, False))

        if not scored:
            # 아주 극단적 fallback
            base_sps = max(default_sps, 1e-3)
            return default_batch, default_accum, base_sps, 1.0, base_sps

        # ε-greedy: 아직 안 본 config들을 우선 탐색
        explore = (random.random() < self.explore_prob)
        best_entry = None

        if explore:
            unseen = [x for x in scored if not x[5]]
            if unseen:
                best_entry = max(unseen, key=lambda x: x[4])  # goodput 기준
        # exploitation 또는 unseen이 없을 때
        if best_entry is None:
            best_entry = max(scored, key=lambda x: x[4])

        best_b, best_a, best_sps, best_se, best_gp, _ = best_entry
        return int(best_b), int(best_a), float(best_sps), float(best_se), float(best_gp)

_profiler = PolluxGoodputProfiler(alpha=0.5, explore_prob=0.0)

def record_goodput_sample(
    job_id: str,
    g: int,
    local_batch: int,
    accum: int,
    sps: float,
    gns: Optional[float] = None,
    stat_eff: Optional[float] = None,
) -> None:
    _profiler.record_goodput_sample(job_id, g, local_batch, accum, sps, gns, stat_eff)


def record_throughput_sample(job_id: str, g: int, local_batch: int, sps: float) -> None:
    _profiler.record_goodput_sample(
        job_id=job_id,
        g=g,
        local_batch=local_batch,
        accum=1,
        sps=sps,
        gns=None,
        stat_eff=1.0,
    )

FAIRNESS_ALPHA = 0.3  # fairness 영향 완화용 스케일 팩터

def _fairness_weight(job: RuntimeJobState) -> float:
    """
    attained_service가 커져도 너무 빨리 weight가 0으로 떨어지지 않도록 완화.
    FAIRNESS_ALPHA가 작을수록 fairness 페널티가 약해짐.
    """
    s = max(0.0, float(job.attained_service))
    return 1.0 / (1.0 + FAIRNESS_ALPHA * s)

def _build_U_table(
    runtime_jobs: List[RuntimeJobState],
    max_g_per_job: int = 4,
) -> Dict[str, Dict[int, Tuple[int, int, float, float, float]]]:
    """
    U_table[job_id][g] = (best_batch, best_accum, U, goodput, sps)

    - PolluxGoodputProfiler (online + prior)를 사용해서
      각 (job, g)에 대해 best (batch, accum)를 선택.
    - U = log(1 + goodput) (Pollux-style goodput utility)

    여기서 goodput = sps * stat_eff
    """

    if not runtime_jobs:
        return {}

    U_table: Dict[str, Dict[int, Tuple[int, int, float, float, float]]] = {}

    for job in runtime_jobs:
        jid = job.job_id
        model = job.model_name
        dataset = job.dataset

        # gang job이면 g=4만 허용
        if is_gang_job(job):
            g_min = g_max = 4
        else:
            g_min = max(1, int(getattr(job, "min_gpus", 1)))
            g_max = min(
                max_g_per_job,
                int(getattr(job, "max_gpus", max_g_per_job)),
            )

        if g_min > g_max:
            continue

        per_g: Dict[int, Tuple[int, int, float, float, float]] = {}

        # online profiler + prior 기반으로 각 g에서 best config 선택
        for g in ALLOWED_G:
            if g < g_min or g > g_max:
                continue

            # 현재 config를 default로 사용 (관측치가 없을 때 fallback)
            default_batch = max(1, int(getattr(job, "current_local_batch", 32)))
            default_accum = max(1, int(getattr(job, "current_grad_accum", 1)))
            default_sps = float(job.last_sps if job.last_sps > 0.0 else 1.0)

            b, a, sps, stat_eff, gp = pollux_best_local_config_for_g(
                job_id=jid,
                model_name=model,
                dataset=dataset,
                g=g,
                default_batch=default_batch,
                default_accum=default_accum,
                default_sps=default_sps,
            )

            gp = max(float(gp), 0.0)
            if gp <= 0.0:
                # 이 g에서 유효한 goodput을 못 얻으면 skip
                continue

            # Pollux: utility = log(1 + goodput)
            U_val = math.log(1.0 + gp)

            per_g[g] = (
                int(b),          # best_batch
                int(a),          # best_accum
                float(U_val),    # utility
                float(gp),       # goodput
                float(sps),      # sps
            )

        if per_g:
            U_table[jid] = per_g

    if not U_table:
        print("[POLLUX] U_table is empty after profiler-based build.")

    return U_table

def _job_weighted_utility(
    job: RuntimeJobState,
    g: int,
    U_table: Dict[str, Dict[int, Tuple[int, int, float, float, float]]],
) -> float:
    """
    w_j * U_j(g)를 계산.
    - U_table[job_id][g] = (best_batch, best_accum, U, goodput, sps)
    - w_j는 attained_service 기반 fairness weight.
    """
    job_U = U_table.get(job.job_id, {})
    entry = job_U.get(g)
    if entry is None:
        return 0.0

    # entry = (best_batch, best_accum, U, goodput, sps)
    U_val = float(entry[2])
    if U_val <= 0.0:
        return 0.0

    w = _fairness_weight(job)
    return w * U_val

def _compute_marginal_gains(
    runtime_jobs: List[RuntimeJobState],
    assign_g: Dict[str, int],
    U_table: Dict[str, Dict[int, Tuple[int, int, float, float, float]]],
    min_g: Dict[str, int],
    allow_preemption: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    각 job에 대해:
      ΔU_up[j]   = w_j * (U_j(g_up)   - U_j(g_cur))  (g_cur -> next_up_g)
      ΔU_down[j] = w_j * (U_j(g_down) - U_j(g_cur))  (g_cur -> next_down_g)

    - g_down == 0 인 경우:
        * allow_preemption=True 이고 min_g[j] == 0 일 때만 preemption 고려
        * 그 외에는 ΔU_down을 0으로 두어 donor 후보에서 제외
    """
    delta_up: Dict[str, float] = {}
    delta_down: Dict[str, float] = {}

    for j in runtime_jobs:
        jid = j.job_id
        g_cur = assign_g.get(jid, j.current_gpus)

        job_U = U_table.get(jid, {})

        # --- up: g_cur -> next_up_g ---
        g_up = _next_up_g(g_cur)
        if g_up is not None and g_up in job_U and g_cur in job_U:
            u_cur = _job_weighted_utility(j, g_cur, U_table)
            u_next = _job_weighted_utility(j, g_up, U_table)
            delta_up[jid] = u_next - u_cur
        else:
            delta_up[jid] = 0.0

        # --- down: g_cur -> next_down_g ---
        if g_cur <= 0:
            delta_down[jid] = 0.0
            continue

        g_min = min_g.get(jid, 1)
        g_down = _next_down_g(g_cur)

        # (1) g_down >= max(1, g_min) 인 경우: 정상적인 scale-down
        if g_down is not None and g_down >= max(1, g_min) and g_down in job_U and g_cur in job_U:
            u_cur = _job_weighted_utility(j, g_cur, U_table)
            u_prev = _job_weighted_utility(j, g_down, U_table)
            delta_down[jid] = u_prev - u_cur  # 보통 음수
            continue

        # (2) g_down == 0 인 경우: preemption 후보는 allow_preemption=True & min_g=0일 때만
        if g_down == 0 and allow_preemption and g_min <= 0:
            u_cur = _job_weighted_utility(j, g_cur, U_table)
            u_prev = 0.0  # U(0) = 0
            delta_down[jid] = u_prev - u_cur
        else:
            delta_down[jid] = 0.0

    return delta_up, delta_down

def update_job_metrics_from_telemetry(job: RuntimeJobState, metrics: Dict):
    now = time.time()
    new_attained = float(metrics.get("attained_service", 0.0))
    job.attained_service = new_attained
    job.last_metric_ts = now
    # ✅ goodput 샘플은 /report_progress에서만 기록하도록 통일 (오염 방지)

def pollux_reallocation_tick(
    cluster_id: str,
    cluster_total_gpus: int,
    runtime_jobs: List[RuntimeJobState],
    scale_job_fn: Callable[[RuntimeJobState, int, int], bool],
    now_ts: float,
    cooldown_sec: float,
    min_delta_gain: float,
    *,
    restart_overhead_sec: float = 45.0,
    min_residency_sec: float = 180.0,
    busy_mode_enabled: bool = True,
) -> None:
    """
    ✅ Pollux-ish reallocation tick (restart 기반 stop+requeue 환경에 맞춘 버전)

    핵심 변경점:
    - queue_len 같은 외부 신호로 'realloc 자체를 끄는' 로직 제거.
    - pending(job.status in queued/pending)도 입력으로 받아 "가치를" 계산하되,
      ✅ 이 함수는 JOB_QUEUE를 직접 못 바꾸므로 pending을 '직접 scale' 하진 않는다.
      대신 pending의 가치가 높으면, running job을 downscale 해서 free GPU를 만들도록 유도한다
      (이후 placement 루프가 pending을 시작할 수 있게 됨).
    - 한 tick에 action은 최대 1개 (churn 방지).
    - restart_overhead_sec를 utility-게이트로 사용해 “움직일 만큼 이득일 때만” 움직인다.

    제한(현 구조상 어쩔 수 없음):
    - pending job의 desired_g를 직접 수정하려면, 이 함수 시그니처를 바꾸거나
      scale_job_fn 외에 “queue update 콜백”이 필요하다.
    """

    if cluster_total_gpus <= 0 or not runtime_jobs:
        return

    # -------------------------
    # helpers: cooldown/residency
    # -------------------------
    def _cooldown_ok(job: RuntimeJobState) -> bool:
        last = float(getattr(job, "last_scaled_at_ts", 0.0) or 0.0)
        if last <= 0.0:
            return True
        return (now_ts - last) >= float(cooldown_sec)

    def _residency_ok(job: RuntimeJobState) -> bool:
        # restart 기반에서는 스케일 직후/시작 직후 재스케일이 독
        last = float(getattr(job, "last_scaled_at_ts", 0.0) or 0.0)
        start = float(getattr(job, "start_ts", 0.0) or 0.0)

        anchor = 0.0
        if last > 0.0:
            anchor = last
        elif start > 0.0:
            anchor = start
        else:
            # start_ts가 없으면 warmup이 걸러줄 것이고, 여기서는 막지 않음
            return True

        return (now_ts - anchor) >= float(min_residency_sec)

    def _is_running(job: RuntimeJobState) -> bool:
        st = str(getattr(job, "status", "running") or "running").lower()
        return st not in ("queued", "pending")

    def _is_pending(job: RuntimeJobState) -> bool:
        st = str(getattr(job, "status", "running") or "running").lower()
        return st in ("queued", "pending")

    # -------------------------
    # current assignment
    # - running: current_gpus>=1
    # - pending: current_gpus=0 (build_queued_runtime_jobs_for_cluster가 그렇게 만듦)
    # -------------------------
    assign_g: Dict[str, int] = {}
    min_g: Dict[str, int] = {}

    used_running = 0
    for j in runtime_jobs:
        if _is_running(j):
            g = max(1, int(getattr(j, "current_gpus", 1) or 1))
            used_running += g
            assign_g[j.job_id] = g
        else:
            assign_g[j.job_id] = 0

        min_g[j.job_id] = int(getattr(j, "min_gpus", 1) or 1)

    free = max(0, int(cluster_total_gpus) - int(used_running))

    # -------------------------
    # utility table
    # -------------------------
    U_table = _build_U_table(runtime_jobs, max_g_per_job=4)
    if not U_table:
        return

    # restart overhead → utility 게이트(휴리스틱)
    overhead_gate = math.log(1.0 + max(1.0, float(restart_overhead_sec) / 10.0))

    # -------------------------
    # 후보 평가 함수들
    # -------------------------
    def _best_batch_for(job: RuntimeJobState, g: int) -> int:
        job_U = U_table.get(job.job_id, {})
        if g in job_U:
            return int(job_U[g][0])
        # fallback
        return int(getattr(job, "current_local_batch", 64) or 64)

    def _weighted_U(job: RuntimeJobState, g: int) -> float:
        if g <= 0:
            return 0.0
        return float(_job_weighted_utility(job, g, U_table))

    # -------------------------
    # 1) SCALE-UP (free GPU가 있을 때)
    # - running job만 scale_job_fn으로 실제 실행 가능
    # - pending은 여기서 직접 scale 불가 (JOB_QUEUE를 못 만지므로)
    # -------------------------
    if free > 0:
        best_job: Optional[RuntimeJobState] = None
        best_target_g: Optional[int] = None
        best_gain_total = 0.0
        best_gain_per_gpu = 0.0

        for j in runtime_jobs:
            if not _is_running(j):
                continue
            if is_gang_job(j):
                continue
            if not _eligible_for_elastic_scaling(j, now_ts):
                continue
            if not _cooldown_ok(j) or not _residency_ok(j):
                continue

            g_cur = assign_g.get(j.job_id, max(1, int(j.current_gpus)))
            g_up = _next_up_g(g_cur)
            if g_up is None:
                continue

            g_max = max(1, int(getattr(j, "max_gpus", 4) or 4))
            if g_up > g_max:
                continue

            need = int(g_up - g_cur)
            if need <= 0 or need > free:
                continue

            u_cur = _weighted_U(j, g_cur)
            u_up = _weighted_U(j, g_up)
            gain_total = u_up - u_cur
            if gain_total <= 0.0:
                continue

            gain_per_gpu = gain_total / float(need)

            # gate: 최소 이득 + 오버헤드 상쇄
            if gain_total <= (min_delta_gain + overhead_gate):
                continue

            if gain_per_gpu > best_gain_per_gpu:
                best_gain_per_gpu = gain_per_gpu
                best_gain_total = gain_total
                best_job = j
                best_target_g = g_up

        if best_job is not None and best_target_g is not None:
            target_batch = _best_batch_for(best_job, int(best_target_g))
            ok = scale_job_fn(best_job, int(best_target_g), int(target_batch))
            if ok:
                best_job.last_scaled_at_ts = float(now_ts)
            return  # ✅ 한 tick에 1 action

    # -------------------------
    # 2) BUSY 상황에서 “free GPU 만들기” (pending이 있을 때만 의미가 큼)
    # - running job을 한 단계 downscale 해서 free를 만들고,
    #   그 free로 placement 루프가 pending을 시작하도록 한다.
    #
    # 조건:
    # - busy_mode_enabled=False면 아무 것도 안 함 (원하면 강경하게 끌 수 있음)
    # -------------------------
    if not busy_mode_enabled:
        return

    pending_jobs = [j for j in runtime_jobs if _is_pending(j)]
    if not pending_jobs:
        return  # queue가 없다면 굳이 downscale로 free 만들 필요가 적음

    # pending의 “가치(=0→1, 0→2, 0→4 중 한 단계)” 중 최대를 구함
    best_pending_gain_per_gpu = 0.0
    best_pending_need = 0

    for pj in pending_jobs:
        if is_gang_job(pj):
            # gang pending이면 0->4 한 번에 필요
            g_target = 4
        else:
            g_target = _next_up_g(0) or 1  # 0->1

        # pending은 attained_service=0이므로 weight=1 쪽이 자연스럽다
        u0 = 0.0
        u1 = _weighted_U(pj, int(g_target))
        gain_total = u1 - u0
        need = int(g_target - 0)
        if need <= 0:
            continue
        gain_per_gpu = gain_total / float(need)
        if gain_per_gpu > best_pending_gain_per_gpu:
            best_pending_gain_per_gpu = gain_per_gpu
            best_pending_need = need

    if best_pending_gain_per_gpu <= 0.0:
        return

    # donor 후보: running 중 scale-down 가능한 job
    best_donor: Optional[RuntimeJobState] = None
    best_donor_target_g: Optional[int] = None
    best_trade_score = 0.0  # (pending_gain_per_gpu - donor_loss_per_gpu) 같은 느낌

    for dj in runtime_jobs:
        if not _is_running(dj):
            continue
        if is_gang_job(dj):
            continue
        if not _eligible_for_elastic_scaling(dj, now_ts):
            continue
        if not _cooldown_ok(dj) or not _residency_ok(dj):
            continue

        g_cur = assign_g.get(dj.job_id, max(1, int(dj.current_gpus)))
        g_down = _next_down_g(g_cur)
        if g_down is None:
            continue

        g_min_j = max(1, int(min_g.get(dj.job_id, 1)))
        if g_down < g_min_j:
            continue

        freed = int(g_cur - g_down)
        if freed <= 0:
            continue

        u_cur = _weighted_U(dj, g_cur)
        u_down = _weighted_U(dj, g_down)
        loss_total = u_cur - u_down
        if loss_total < 0.0:
            loss_total = 0.0
        loss_per_gpu = loss_total / float(freed)

        # “pending의 가치가 donor 손실 + 오버헤드보다 충분히 큰가?”
        # - downscale도 restart를 일으키므로 overhead_gate를 반영
        trade = best_pending_gain_per_gpu - (loss_per_gpu + overhead_gate)

        # min_delta_gain은 “움직임 최소 이득”으로 사용
        if trade <= float(min_delta_gain):
            continue

        if trade > best_trade_score:
            best_trade_score = trade
            best_donor = dj
            best_donor_target_g = g_down

    if best_donor is None or best_donor_target_g is None:
        return

    # 실행: donor downscale (free GPU 확보 → 이후 placement가 pending 시작)
    target_batch = _best_batch_for(best_donor, int(best_donor_target_g))
    ok = scale_job_fn(best_donor, int(best_donor_target_g), int(target_batch))
    if ok:
        best_donor.last_scaled_at_ts = float(now_ts)
    return

def pollux_initial_g(
    *,
    model_name: str,
    dataset: str,
    cluster_free_gpus: int,
    cluster_total_gpus: Optional[int] = None,   # ✅ 추가: scarcity 계산용
    default_batch: int = 64,
    default_accum: int = 1,
    default_sps: float = 1.0,
    min_gpus: int = 1,
    max_gpus: int = 4,
    job_id: Optional[str] = None,
    **extra: object,
) -> int:
    """
    Scarcity-aware Pollux-style initial GPU count selection.

    핵심 아이디어(네 환경 최적):
      - realloc이 거의 없으면 "initial_g"가 사실상 성능을 결정.
      - Pollux스럽게 하려면 단순 argmax(goodput)가 아니라
        '추가 GPU의 한계효용(ΔU/Δg)'이 'GPU의 그림자 가격(shadow price)'을 이길 때만 scale-up.
      - shadow price는 queue_len 같은 간접 신호 대신, 클러스터 여유도 free/total로 근사.

    U(g) = log(1 + goodput(g))
    scale-up 조건:
      (U(g_next) - U(g_cur)) / (g_next - g_cur) > price(free/total)

    tie-break:
      - 이득이 비슷하면 작은 g 선호(자원 낭비 방지)
    """

    import math
    from typing import Dict, Optional as _Opt

    free = int(cluster_free_gpus)
    if free <= 0:
        return 0

    # gang job: 고정
    if is_gang_model(model_name, dataset):
        g = int(GANG_JOBS[(model_name, dataset)])
        return g if g <= free else 0

    g_lo = max(1, int(min_gpus))
    g_hi = min(int(max_gpus), free)
    if g_lo > g_hi:
        return 0

    # 후보 g 구성 (ALLOWED_G 우선, 없으면 연속 범위)
    allowed = []
    try:
        allowed = list(ALLOWED_G)  # type: ignore[name-defined]
    except Exception:
        allowed = []

    if allowed:
        candidates = sorted({int(g) for g in allowed if g_lo <= int(g) <= g_hi})
    else:
        candidates = list(range(g_lo, g_hi + 1))

    if not candidates:
        return 0

    jid = job_id or f"{model_name}:{dataset}"

    # 1) 각 g의 goodput과 utility 계산
    GP: Dict[int, float] = {}
    U: Dict[int, float] = {}

    for g in candidates:
        gp = 0.0
        try:
            _, _, _, _, gp_raw = pollux_best_local_config_for_g(
                job_id=jid,
                model_name=model_name,
                dataset=dataset,
                g=int(g),
                default_batch=int(default_batch),
                default_accum=int(default_accum),
                default_sps=float(default_sps),
            )
            gp = float(gp_raw)
        except Exception:
            gp = 0.0

        if not math.isfinite(gp) or gp < 0.0:
            gp = 0.0

        GP[g] = gp
        U[g] = math.log(1.0 + gp)

    # 전부 0이면: 가능한 최소 g로 시작 (강제 2/4 금지)
    if max(GP.values() or [0.0]) <= 0.0:
        return int(min(candidates))

    # 2) shadow price 계산 (free/total 기반)
    # total이 없으면 "보수적"으로 잡아서 함부로 키우지 않게 함
    total = int(cluster_total_gpus) if cluster_total_gpus is not None else 0
    if total <= 0:
        # total을 모르면 "중간 정도로 바쁨" 가정 (너무 공격적/보수적 모두 피함)
        free_ratio = 0.25  # 25% free라고 가정
    else:
        free_ratio = max(0.0, min(1.0, float(free) / float(max(1, total))))

    # price 함수: free_ratio가 낮을수록 가격↑, 높을수록 가격↓
    # - 너무 민감하면 스케일업이 영영 안 일어나고
    # - 너무 둔하면 항상 4로 붙습니다.
    # 아래는 실험하기 좋은 형태(단조 감소) + 하이퍼파라미터 2개
    price_min = float(extra.get("pollux_price_min", 0.01))  # 널널할 때 최소 가격
    price_max = float(extra.get("pollux_price_max", 0.20))  # 빡빡할 때 최대 가격
    price_min = max(0.0, min(price_min, 1.0))
    price_max = max(price_min, min(price_max, 2.0))

    # 곡선 모양: free_ratio^p
    p = float(extra.get("pollux_price_shape", 2.0))
    p = max(0.5, min(p, 6.0))

    # free_ratio=1 -> price ~ price_min
    # free_ratio=0 -> price ~ price_max
    price = price_min + (price_max - price_min) * (1.0 - (free_ratio ** p))

    # 3) Pollux식 "한 단계씩" 올리기 (ΔU/Δg > price 일 때만)
    # 시작점: 최소 g(자원 절약)
    g_cur = int(min(candidates))
    if g_cur > free:
        return 0

    def next_up(x: int) -> _Opt[int]:
        for gg in candidates:
            if gg > x:
                return gg
        return None

    while True:
        g_nxt = next_up(g_cur)
        if g_nxt is None:
            break
        if g_nxt > free:
            break

        du = float(U.get(g_nxt, 0.0) - U.get(g_cur, 0.0))
        dg = float(g_nxt - g_cur)
        marg = du / max(1e-9, dg)

        # ✅ 핵심 게이트: 한계효용이 가격을 이기면 올림
        if marg > price:
            g_cur = g_nxt
            continue
        break

    # 4) near-tie 처리: best 대비 너무 미미한 차이면 작은 g로 (GPU 낭비 방지)
    # (원하면 0~0.05 정도로 튜닝)
    rel_tol = float(extra.get("pollux_rel_tol", 0.02))
    rel_tol = max(0.0, min(rel_tol, 0.2))

    # 여기서 "best"는 candidates 중 U 최대
    best_g = max(candidates, key=lambda g: (U.get(int(g), 0.0), -int(g)))
    best_u = float(U.get(int(best_g), 0.0))

    # best_u의 (1-rel_tol) 이상이면 작은 g 선호
    thresh = best_u * (1.0 - rel_tol)
    near_best = [g for g in candidates if float(U.get(int(g), 0.0)) >= thresh]

    if near_best:
        # 단, scarcity-up 결과(g_cur)가 near_best 밖으로 밀리면 g_cur 유지
        # (즉, near_best가 너무 공격적으로 줄이는 걸 방지)
        g_small = int(min(near_best))
        if float(U.get(g_cur, 0.0)) >= thresh:
            return g_small
        return int(g_cur)

    return int(g_cur)

def pollux_best_local_config_for_g(
    job_id: str,
    model_name: str,
    dataset: str,
    g: int,
    default_batch: int,
    default_accum: int,
    default_sps: float,
) -> Tuple[int, int, float, float, float]:
    """
    Pollux-style: 주어진 g에서 (local_batch, accum)을 선택하고,
    해당 config의 (sps, stat_eff, goodput)를 반환.

    내부적으로 PolluxGoodputProfiler.best_for_g()를 사용:
      - 관측된 goodput이 있으면 EMA 기반 값 사용
      - 없으면 (model, dataset, g) prior를 이용해 surface를 근사
      - ε-greedy로 미관측 config도 탐색
    """
    default_batch = max(1, int(default_batch))
    default_accum = max(1, int(default_accum))
    default_sps = float(default_sps if default_sps > 0.0 else 1.0)

    b, a, sps, stat_eff, gp = _profiler.best_for_g(
        job_id=job_id,
        model_name=model_name,
        dataset=dataset,
        g=g,
        default_batch=default_batch,
        default_accum=default_accum,
        default_sps=default_sps,
    )
    return int(b), int(a), float(sps), float(stat_eff), float(gp)
