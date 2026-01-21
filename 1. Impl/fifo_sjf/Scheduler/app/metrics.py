import os, time, threading, subprocess, json
from collections import deque, defaultdict
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

AVG_WINDOW_SEC = int(os.getenv("METRICS_AVG_WINDOW_SEC", "60"))
SAMPLE_INTERVAL = int(os.getenv("METRICS_SAMPLE_INTERVAL", "5"))
POWER_BUDGET = float(os.getenv("POWER_BUDGET_W", "1800"))  # 노드(또는 클러스터) 전력 예산 합(와트)

PUSH_TTL_SEC = int(os.getenv("PUSH_TTL_SEC", "120"))

CLUSTERA_GPUS_ENV = "CLUSTERA_GPUS"

# ====== NVML 로딩 ======
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

# ====== 순환 버퍼(ClusterA pull형 전역 이동평균) ======
_gpu_util_samples = deque(maxlen=max(1, AVG_WINDOW_SEC // max(1, SAMPLE_INTERVAL)))
_power_ratio_samples = deque(maxlen=max(1, AVG_WINDOW_SEC // max(1, SAMPLE_INTERVAL)))

_sampler_started = False
_sampler_lock = threading.Lock()

# ====== ClusterB 푸시형 저장소 ======
@dataclass
class TelemetryRecord:
    ts: float
    node_id: str
    gpu_index: int
    gpu_util: Optional[float]  # 0~100 (%)
    power_w: Optional[float]   # 와트
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]

# node_id -> deque[TelemetryRecord]
_TELEM_STORE: Dict[str, deque] = defaultdict(lambda: deque(maxlen=2048))
_TELEM_LOCK = threading.Lock()


# =========================================================
# NVML 유틸
# =========================================================
def _nvml_read_util_power(indices: Optional[List[int]] = None) -> Optional[Tuple[float, float]]:
    """
    indices가 None이면 모든 GPU 포함. (avg_util_0_1, total_power_w)를 반환.
    실패 시 None.
    """
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


# =========================================================
# ClusterA Pull형 샘플러
# =========================================================
def _sample_gpu_metrics_loop():
    while True:
        try:
            # NVML 우선 시도(전체 GPU)
            res = _nvml_read_util_power(indices=None)
            if res is None:
                U, total_draw = _smi_read_util_power_all()
            else:
                U, total_draw = res

            _gpu_util_samples.append(U)
            ratio = min(1.0, total_draw / POWER_BUDGET) if POWER_BUDGET > 0 else 0.0
            _power_ratio_samples.append(round(ratio, 2))
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


# 모듈 import 시 1회만 시작
_ensure_sampler_once()


# =========================================================
# ClusterB 푸시형 데이터 수집/집계
# =========================================================
def ingest_telemetry(
    node_id: str,
    gpu_index: int,
    ts: Optional[float],
    gpu_util: Optional[float],
    power_w: Optional[float],
    mem_used_mb: Optional[float],
    mem_total_mb: Optional[float],
) -> None:
    """
    ClusterB 등 외부 워커가 푸시하는 텔레메트리 수신용.
    """
    rec = TelemetryRecord(
        ts=float(ts or time.time()),
        node_id=node_id,
        gpu_index=int(gpu_index),
        gpu_util=float(gpu_util) if gpu_util is not None else None,
        power_w=float(power_w) if power_w is not None else None,
        mem_used_mb=float(mem_used_mb) if mem_used_mb is not None else None,
        mem_total_mb=float(mem_total_mb) if mem_total_mb is not None else None,
    )
    with _TELEM_LOCK:
        _TELEM_STORE[node_id].append(rec)
        # 오래된 항목 정리
        cutoff = time.time() - PUSH_TTL_SEC
        dq = _TELEM_STORE[node_id]
        while dq and dq[0].ts < cutoff:
            dq.popleft()


def _aggregate_clusterb_recent() -> Tuple[float, float]:
    """
    최근 PUSH_TTL_SEC 내 ClusterB 샘플을 집계.
    반환: (avg_util_0_1, total_power_ratio_0_1)
    total_power_ratio = sum(power_w)/POWER_BUDGET
    """
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


# =========================================================
# 공용 API: 평균값 / CSP / 람다
# =========================================================
def get_gpu_utilization_avg() -> float:
    if not _gpu_util_samples:
        return 0.0
    return round(sum(_gpu_util_samples) / len(_gpu_util_samples), 2)


def get_power_pressure_avg() -> float:
    if not _power_ratio_samples:
        return 0.0
    return round(sum(_power_ratio_samples) / len(_power_ratio_samples), 2)


def get_cluster_utilization_avg(cluster: str) -> float:
    name = (cluster or "").lower()
    if name == "clustera":
        # 특정 GPU 인덱스만 포함하도록 지정된 경우
        idxs = _env_gpu_list(CLUSTERA_GPUS_ENV)
        if idxs:
            # NVML 우선
            res = _nvml_read_util_power(indices=idxs)
            if res is not None:
                U, _ = res
                return U
            # 폴백
            return _smi_read_util_for_indices(idxs)
        # 지정 없으면 전역 이동평균 사용
        return get_gpu_utilization_avg()

    elif name == "clusterb":
        U, _ = _aggregate_clusterb_recent()
        return U

    # 기본(미지정): 전역 이동평균
    return get_gpu_utilization_avg()


def get_cluster_power_ratio_avg(cluster: str) -> float:
    name = (cluster or "").lower()
    if name == "clustera":
        idxs = _env_gpu_list(CLUSTERA_GPUS_ENV)
        if idxs:
            # NVML 우선
            res = _nvml_read_util_power(indices=idxs)
            if res is not None:
                _, power_w = res
                ratio = (power_w / POWER_BUDGET) if POWER_BUDGET > 0 else 0.0
                return round(min(1.0, ratio), 2)
            # 폴백
            power_w = _smi_read_power_for_indices(idxs)
            ratio = (power_w / POWER_BUDGET) if POWER_BUDGET > 0 else 0.0
            return round(min(1.0, ratio), 2)
        return get_power_pressure_avg()

    elif name == "clusterb":
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


def get_cluster_csp(cluster: str, queue_len: int, slots_total: int) -> Dict[str, float]:
    U_t = get_cluster_utilization_avg(cluster)
    E_t = get_cluster_power_ratio_avg(cluster)
    p_fair = compute_p_fair(queue_len, slots_total)
    return {"cluster": cluster, "p_fair": p_fair, "U_t": U_t, "E_t": E_t}

