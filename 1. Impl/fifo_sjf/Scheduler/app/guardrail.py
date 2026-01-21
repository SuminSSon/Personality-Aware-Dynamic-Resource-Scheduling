from typing import Dict, List

COST_WORDS = ["비용","저렴","절약","최소","cost","cheap","budget","energy","전력"]
SPEED_WORDS = ["빠르게","빨리","속도","asap","deadline","speed","fast"]
BAL_WORDS = ["균형","balance","balanced","default"]

def _is_perm_1_4(xs: List[int]) -> bool:
    return sorted(xs) == [1,2,3,4]

def validate_policy_json(data: Dict) -> Dict:
    for k in ["job_id","η_perf","clusters","rationale"]:
        if k not in data:
            raise ValueError(f"missing field: {k}")

    try:
        eta = float(data["η_perf"])
    except:
        raise ValueError("η_perf must be float")
    data["η_perf"] = 0.0 if eta < 0 else 1.0 if eta > 1 else eta

    cl = data["clusters"]
    if not isinstance(cl, list) or len(cl) == 0:
        raise ValueError("clusters empty")
    for c in cl:
        if "cluster" not in c or "pref_g" not in c or "g_rank" not in c:
            raise ValueError("invalid cluster policy")
        if c["pref_g"] not in [1,2,3,4]:
            raise ValueError("pref_g must be 1..4")
        if not _is_perm_1_4(c["g_rank"]):
            raise ValueError("g_rank must be a permutation of 1..4")

    if not isinstance(data["rationale"], list):
        data["rationale"] = []
    return data

def _infer_intent_kind(text: str) -> str:
    t = (text or "").lower()
    if any(w.lower() in t for w in COST_WORDS) or any(w in text for w in COST_WORDS): return "cost"
    if any(w.lower() in t for w in SPEED_WORDS) or any(w in text for w in SPEED_WORDS): return "speed"
    if any(w.lower() in t for w in BAL_WORDS)  or any(w in text for w in BAL_WORDS):  return "balanced"
    return "balanced"

def enforce_semantics(payload: Dict, data: Dict) -> Dict:
    """LLM 출력에 사용자 의도를 '결정론적으로' 반영/보정."""
    kind = _infer_intent_kind(payload.get("user_intent_text",""))
    eta = float(data.get("η_perf", 0.5))

    # 1) η 범위 보정
    if kind == "cost" and eta > 0.4:
        data["η_perf"] = 0.25
    elif kind == "speed" and eta < 0.6:
        data["η_perf"] = 0.9
    elif kind == "balanced" and not (0.45 <= eta <= 0.55):
        data["η_perf"] = 0.5

    # 2) 비용 의도면 pref_g 상한 = 2
    if kind == "cost":
        for c in data.get("clusters", []):
            if c["pref_g"] > 2:
                c["pref_g"] = 2
            # g_rank도 1,2를 앞으로
            rank = c["g_rank"]
            front = [x for x in rank if x in [1,2]]
            back  = [x for x in rank if x in [3,4]]
            c["g_rank"] = front + back

    # 3) rationale placeholder 정리
    if not data["rationale"] or any("<=120" in r for r in data["rationale"]):
        if kind == "cost":
            bullets = [
                "Cost intent → η≈0.25; prefer smaller g.",
                "Applied guards for queue/energy and ranking.",
                "Per-cluster hints use sps and cost signals."
            ]
        elif kind == "speed":
            bullets = [
                "Speed intent → η≈0.9; sps-first.",
                "Applied guards for queue/energy and ranking.",
                "Per-cluster hints use sps and cost signals."
            ]
        else:
            bullets = [
                "Balanced intent → η≈0.5.",
                "Applied guards for queue/energy and ranking.",
                "Per-cluster hints use sps and cost signals."
            ]
        data["rationale"] = bullets
    return data

