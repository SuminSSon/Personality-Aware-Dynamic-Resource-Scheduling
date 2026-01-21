from __future__ import annotations

from typing import Tuple
import math
import logging

logger = logging.getLogger(__name__)


def normalize_policy_values(
    lambda_time: float | None,
    lambda_cost: float | None,
    lambda_energy: float | None,
) -> Tuple[float, float, float]:
    """
    (λ_time, λ_cost, λ_energy)를 정규화한다.

    - 음수는 0으로 클램프
    - (0,0,0)이면 기본값 (0.33, 0.33, 0.34) 사용
    - 합이 0이 아니면 합이 1이 되도록 나눈다.
    """
    lt = max(0.0, float(lambda_time or 0.0))
    lc = max(0.0, float(lambda_cost or 0.0))
    le = max(0.0, float(lambda_energy or 0.0))

    s = lt + lc + le
    if s <= 0.0 or not math.isfinite(s):
        lt, lc, le = 0.33, 0.33, 0.34
    else:
        lt /= s
        lc /= s
        le /= s

    return lt, lc, le


def clamp_score_term(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    score 구성 요소(sps_norm, cost_norm 등)를 지정 범위로 클램프한다.
    """
    if not math.isfinite(x):
        return min_val
    if min_val > max_val:
        min_val, max_val = max_val, min_val
    return max(min_val, min(max_val, x))


def normalize_fairness(raw_fair: float) -> float:
    """
    f_fair(c)를 fairness penalty term으로 변환한다.

    설계 의도:
    - f_fair(c) ≈ 1  → penalty 0 (균형 상태)
    - f_fair(c) < 1  → penalty 0 (덜 사용된 클러스터는 페널티 없음)
    - f_fair(c) > 1  → penalty > 0 (과사용된 정도에 비례)
    - 상한을 1.0으로 클램프해서 score를 터뜨리지 않음

    즉, score에서는 다음처럼 쓰일 수 있다:
        S = ... - μ_fair * fairness_term
    """
    if not math.isfinite(raw_fair) or raw_fair <= 0.0:
        return 0.0

    dev = raw_fair - 1.0  # 평균 대비 deviation
    if dev <= 0.0:
        return 0.0

    # dev를 [0, 1] 범위로 클램프 (너무 큰 값 방지)
    return clamp_score_term(dev, 0.0, 1.0)


def safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    """
    0 나누기, NaN 등을 방지하는 안전한 나눗셈 헬퍼.
    """
    try:
        if denom == 0.0:
            return default
        v = numer / denom
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        logger.exception("safe_div failed (numer=%s, denom=%s)", numer, denom)
        return default

