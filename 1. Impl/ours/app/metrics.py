import os, time, threading, subprocess
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from app.logger import get_run_logger

AVG_WINDOW_SEC = int(os.getenv("METRICS_AVG_WINDOW_SEC", "60"))
SAMPLE_INTERVAL = int(os.getenv("METRICS_SAMPLE_INTERVAL", "5"))
POWER_BUDGET = float(os.getenv("POWER_BUDGET_W", "1800"))

PUSH_TTL_SEC = int(os.getenv("PUSH_TTL_SEC", "120"))

CLUSTERA_GPUS_ENV = "CLUSTERA_GPUS"

SPEED_FACTOR: Dict[str, float] = {
    "clusterA": 1.41,
    "clusterB": 1.0,
}

_LAST_TILDE_U: Dict[str, float] = {}

FAIR_EMA_ALPHA = float(os.getenv("METRICS_FAIR_EMA_ALPHA", "0.3"))


try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

_gpu_util_samples = deque(maxlen=max(1, AVG_WINDOW_SEC // max(1, SAMPLE_INTERVAL)))
_power_ratio_samples = deque(maxlen=max(1, AVG_WINDOW_SEC // max(1, SAMPLE_INTERVAL)))

_sampler_started = False
_sampler_lock = threading.Lock()

@dataclass
class TelemetryRecord:
    ts: float
    node_id: str
    gpu_index: int
    gpu_util: Optional[float]
    power_w: Optional[float]
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]

_TELEM_STORE: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2048))
_TELEM_LOCK = threading.Lock()


# NVML 유틸
def _nvml_read_util_power(indices: Optional[List[int]] = None) -> Optional[Tuple[float, float]]:
    if not _NVML_OK:
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpu_idxs = indices if indices is not None else list(range(count))

        utils_0_1: List[float] = []
        power_ws: List[float] = []

        for i in gpu_idxs:
            if i < 0 or i >= count:
                continue
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            ur = pynvml.nvmlDeviceGetUtilizationRates(h)  # .gpu (%)
            utils_0_1.append((ur.gpu or 0) / 100.0)
            try:
                power_ws.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)  # mW -> W
            except Exception:
                # 일부 환경은 전력 미지원
                pass

        avg_util = (sum(utils_0_1) / len(utils_0_1)) if utils_0_1 else 0.0
        total_power_w = sum(power_ws) if power_ws else 0.0
        return round(avg_util, 2), total_power_w
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def _smi_read_util_power_all() -> Tuple[float, float]:
    """nvidia-smi 폴백: 전체 GPU 평균 util(0~1), total power(W)"""
    util_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL
    ).decode().strip().splitlines()
    utils = [int(x) for x in util_out if x.strip().isdigit()]
    U = (sum(utils) / len(utils) / 100.0) if utils else 0.0

    power_out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        stderr=subprocess.DEVNULL
    ).decode().strip().splitlines()
    draws = [float(x) for x in power_out if x.strip()]
    total_draw = sum(draws) if draws else 0.0
    return round(U, 2), total_draw

def _smi_read_util_for_indices(idxs: List[int]) -> float:
    vals = []
    for i in idxs:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-i", str(i), "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                vals.append(int(out) / 100.0)
        except Exception:
            pass
    return round(sum(vals)/len(vals), 2) if vals else 0.0

def _smi_read_power_for_indices(idxs: List[int]) -> float:
    total_w = 0.0
    for i in idxs:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-i", str(i), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                total_w += float(out)
        except Exception:
            pass
    return total_w

def _env_gpu_list(key: str) -> Optional[List[int]]:
    s = os.getenv(key, "").strip()
    if not s:
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip()]
    except Exception:
        return None

# ClusterA Pull형 샘플러
def _sample_gpu_metrics_loop():
    while True:
        try:
            res = _nvml_read_util_power(indices=None)
            if res is None:
                U, total_draw = _smi_read_util_power_all()
            else:
                U, total_draw = res

            _gpu_util_samples.append(U)
            ratio = min(1.0, total_draw / POWER_BUDGET) if POWER_BUDGET > 0 else 0.0
            _power_ratio_samples.append(round(ratio, 2))

            rl = get_run_logger()

            rl.log_info(
                f"[TELEMETRY] cluster=clusterA util={U:.2f} power_w={total_draw:.1f}"
            )

            rl.telemetry_sample(
                node_id="clusterA-local",
                gpu_index=-1,
                gpu_util=U * 100.0,   # %
                power_w=total_draw,
                mem_used_mb=0.0,
                mem_total_mb=0.0,
            )

        except Exception as e:
            print(f"[WARN] GPU metric sample failed: {e}")
        time.sleep(max(1, SAMPLE_INTERVAL))

def _ensure_sampler_once():
    global _sampler_started
    with _sampler_lock:
        if not _sampler_started:
            t = threading.Thread(target=_sample_gpu_metrics_loop, daemon=True)
            t.start()
            _sampler_started = True


_ensure_sampler_once()


# ClusterB 푸시형 데이터 수집/집계
def ingest_telemetry(
    node_id: str,
    gpu_index: int,
    ts: Optional[float],
    gpu_util: Optional[float],
    power_w: Optional[float],
    mem_used_mb: Optional[float],
    mem_total_mb: Optional[float],
) -> None:
    rec = TelemetryRecord(
        ts=float(ts or time.time()),
        node_id=node_id,
        gpu_index=int(gpu_index),
        gpu_util=float(gpu_util) if gpu_util is not None else None,
        power_w=float(power_w) if power_w is not None else None,
        mem_used_mb=float(mem_used_mb) if mem_used_mb is not None else None,
        mem_total_mb=float(mem_total_mb) if mem_total_mb is not None else None,
    )

    # --- 1) 메모리 저장 + TTL 정리 ---
    now = time.time()
    cutoff = now - PUSH_TTL_SEC
    with _TELEM_LOCK:
        dq = _TELEM_STORE[node_id]
        dq.append(rec)
        while dq and dq[0].ts < cutoff:
            dq.popleft()

    # --- 2) 로그 기록 + telemetry.csv 기록 ---
    rl = get_run_logger()

    rl.log_info(
        f"[TELEMETRY] cluster=clusterB node={node_id} gpu={int(gpu_index)} "
        f"util={gpu_util} power_w={power_w}"
    )

    rl.telemetry_sample(
        node_id=node_id,
        gpu_index=int(gpu_index),
        gpu_util=float(gpu_util) if gpu_util is not None else 0.0,
        power_w=float(power_w) if power_w is not None else 0.0,
        mem_used_mb=float(mem_used_mb) if mem_used_mb is not None else 0.0,
        mem_total_mb=float(mem_total_mb) if mem_total_mb is not None else 0.0,
    )

def _aggregate_clusterb_recent() -> Tuple[float, float]:
    now = time.time()
    total_util = 0.0
    util_cnt = 0
    total_power_w = 0.0

    with _TELEM_LOCK:
        for node_id, dq in _TELEM_STORE.items():
            # deque는 오래된 것 일부 남아있을 수 있어 개별 정리
            while dq and dq[0].ts < (now - PUSH_TTL_SEC):
                dq.popleft()
            for r in dq:
                if r.gpu_util is not None:
                    total_util += (r.gpu_util / 100.0)  # % -> 0~1
                    util_cnt += 1
                if r.power_w is not None:
                    total_power_w += r.power_w

    avg_util = (total_util / util_cnt) if util_cnt > 0 else 0.0
    ratio = min(1.0, total_power_w / POWER_BUDGET) if POWER_BUDGET > 0 else 0.0
    return round(avg_util, 2), round(ratio, 2)

# 공용 API: 평균값 / CSP / 람다
def get_gpu_utilization_avg() -> float:
    if not _gpu_util_samples:
        return 0.0
    return round(sum(_gpu_util_samples) / len(_gpu_util_samples), 2)

def get_power_pressure_avg() -> float:
    if not _power_ratio_samples:
        return 0.0
    return round(sum(_power_ratio_samples) / len(_power_ratio_samples), 2)

def get_cluster_utilization_avg(cluster: str) -> float:
    name = (cluster or "").strip().lower()

    if name in ("clustera", "cluster_a", "cluster-a", "a", "clustera-local"):
        idxs = _env_gpu_list(CLUSTERA_GPUS_ENV)
        if idxs:
            res = _nvml_read_util_power(indices=idxs)
            if res is not None:
                U, _ = res
                return U
            return _smi_read_util_for_indices(idxs)
        return get_gpu_utilization_avg()

    if name in ("clusterb", "cluster_b", "cluster-b", "b"):
        U, _ = _aggregate_clusterb_recent()
        return U

    return get_gpu_utilization_avg()


def get_cluster_power_ratio_avg(cluster: str) -> float:
    name = (cluster or "").strip().lower()

    if name in ("clustera", "cluster_a", "cluster-a", "a", "clustera-local"):
        idxs = _env_gpu_list(CLUSTERA_GPUS_ENV)
        if idxs:
            res = _nvml_read_util_power(indices=idxs)
            if res is not None:
                _, power_w = res
                ratio = (power_w / POWER_BUDGET) if POWER_BUDGET > 0 else 0.0
                return round(min(1.0, ratio), 2)
            power_w = _smi_read_power_for_indices(idxs)
            ratio = (power_w / POWER_BUDGET) if POWER_BUDGET > 0 else 0.0
            return round(min(1.0, ratio), 2)
        return get_power_pressure_avg()

    if name in ("clusterb", "cluster_b", "cluster-b", "b"):
        _, ratio = _aggregate_clusterb_recent()
        return ratio

    return get_power_pressure_avg()

def compute_p_fair(queue_len: int, slots_total: int) -> float:
    if slots_total <= 0:
        return 0.0
    return round(min(1.0, queue_len / float(slots_total)), 2)

def lambda_from_csp(p_fair: float, E_t: float) -> Dict[str, float]:
    lam_f = max(0.0, min(1.0, p_fair))
    lam_c = max(0.0, min(1.0, E_t))
    lam_t = max(0.0, 1.0 - lam_f - lam_c)
    return {"time": lam_t, "cost": lam_c, "fair": lam_f}

def get_cluster_csp(cluster: str, slots_total: int) -> Dict[str, float]:
    raw = (cluster or "").strip()
    name = raw.lower()

    # 1) utilization / energy pressure
    U_t = get_cluster_utilization_avg(raw)   # 내부에서 lower 처리
    E_t = get_cluster_power_ratio_avg(raw)

    # 2) speed_factor 키도 통일
    # SPEED_FACTOR가 "clusterA"/"clusterB"로 되어 있으면 여기서 매핑
    if name in ("clustera", "cluster_a", "cluster-a", "a", "clustera-local"):
        sf = float(SPEED_FACTOR.get("clusterA", 1.0))
        key = "clusterA"
    elif name in ("clusterb", "cluster_b", "cluster-b", "b"):
        sf = float(SPEED_FACTOR.get("clusterB", 1.0))
        key = "clusterB"
    else:
        sf = 1.0
        key = raw

    cap_c = max(1.0, float(slots_total) * sf)
    tilde_u_raw = U_t / cap_c

    prev = _LAST_TILDE_U.get(key, tilde_u_raw)
    alpha = max(0.0, min(1.0, FAIR_EMA_ALPHA))
    tilde_u_ema = alpha * tilde_u_raw + (1.0 - alpha) * prev
    _LAST_TILDE_U[key] = tilde_u_ema

    vals = list(_LAST_TILDE_U.values())
    bar_u = (sum(vals) / len(vals)) if vals else tilde_u_ema

    f_fair = (tilde_u_ema / bar_u) if bar_u > 0 else 1.0

    return {"cluster": raw, "p_fair": float(f_fair), "U_t": float(U_t), "E_t": float(E_t)}