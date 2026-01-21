from __future__ import annotations
from typing import Dict, Any, List
import json, os, re
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# OpenAI SDK (2024+)
_USE_LLM = True  # 강제 ICL 사용
_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def _kst_now_str() -> str:
    if ZoneInfo:
        tz = ZoneInfo("Asia/Seoul")
        now = datetime.now(tz)
    else:
        now = datetime.now()
    return "job-" + now.strftime("%Y%m%d-%H%M%S")

def _cluster_line(row: Dict[str, Any]) -> str:
    g = row.get("g", [1,2,3,4])
    sps = row.get("sps", [0,0,0,0])
    cost = row.get("cost", [1,1,1,1])
    return f"{row.get('cluster','clusterA')} | g={g} | sps={sps} | cost={cost}"

def _csp_line(c: Dict[str, Any]) -> str:
    return f"{c.get('cluster')} | p_fair={c.get('p_fair',0.0)} | U_t={c.get('U_t',0.0)} | E_t={c.get('E_t',0.0)}"

def _extract_json(txt: str) -> Dict[str, Any]:
    # 가장 큰 JSON 블록 추출
    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        raise ValueError("no JSON block")
    return json.loads(m.group(0))

def _heuristic_fallback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """LLM 실패 시 최소 동작 보장(간단 휴리스틱)."""
    job_id = payload.get("job_id") or _kst_now_str()
    prof = payload.get("profiling_compact", []) or []
    out_clusters = []
    for row in prof:
        # sps 최대 g 우선
        sps = row.get("sps", [0,0,0,0])
        if sps:
            pref = (max(enumerate(sps), key=lambda x: x[1])[0]) + 1
        else:
            pref = 1
        out_clusters.append({"cluster": row.get("cluster","clusterA"),
                             "pref_g": int(pref),
                             "g_rank": [pref, 1, 2, 3][:len(row.get("g",[1,2,3,4]))]})
    return {"job_id": job_id, "η_perf": 0.6, "clusters": out_clusters,
            "rationale": ["fallback heuristic"]}

def _compose_prompt(payload: Dict[str, Any]) -> str:
    user_req = payload.get("user_request", "")
    prof = payload.get("profiling_compact", [])
    csp = payload.get("csp", [])

    # Few-shot ICL
    example1 = """User request: "비용 최적화해줘"
Profiling (2 lines):
clusterA | g=[1,2,3,4] | sps=[13.59,19.08,15.17,17.68] | cost=[0.96,0.85,0.79,0.78]
clusterB | g=[1,2,3,4] | sps=[12.2,18.1,14.5,17.1] | cost=[0.92,0.86,0.80,0.81]
CSP:
clusterA | p_fair=0.25 | U_t=0.75 | E_t=0.55
clusterB | p_fair=0.10 | U_t=0.70 | E_t=0.65
Output:
{"job_id":"job-20251023-120000","η_perf":0.20,
 "clusters":[
   {"cluster":"clusterA","pref_g":2,"g_rank":[2,1,4,3]},
   {"cluster":"clusterB","pref_g":2,"g_rank":[2,1,4,3]}
 ],
 "rationale":["Cost-oriented (η≈0.2).","g=2 yields strong sps/cost trade-off.","No overriding fairness/energy pressure."]}"""

    example2 = """User request: "최대한 빨리 끝내줘"
Profiling (2 lines):
clusterA | g=[1,2,3,4] | sps=[14.0,20.1,16.2,18.0] | cost=[0.95,0.86,0.81,0.80]
clusterB | g=[1,2,3,4] | sps=[13.2,18.6,15.0,17.5] | cost=[0.93,0.87,0.82,0.82]
CSP:
clusterA | p_fair=0.70 | U_t=0.85 | E_t=0.88
clusterB | p_fair=0.20 | U_t=0.65 | E_t=0.50
Output:
{"job_id":"job-20251023-130000","η_perf":0.90,
 "clusters":[
   {"cluster":"clusterA","pref_g":3,"g_rank":[3,4,2,1]},
   {"cluster":"clusterB","pref_g":4,"g_rank":[4,3,2,1]}
 ],
 "rationale":["Speed-oriented (η≈0.9).","A guarded by fairness/energy → 3.","B favors max throughput → 4."]}"""

    clines = "\n".join(_cluster_line(r) for r in prof)
    csplines = "\n".join(_csp_line(c) for c in csp)

    prompt = f"""You are a scheduling policy generator for multi-cluster GPU training.

Output STRICT JSON only with keys: {{"job_id","η_perf","clusters","rationale"}}.
- clusters: array of {{"cluster","pref_g","g_rank"}}.
- η_perf in [0,1].
- pref_g ∈ {{1,2,3,4}}, g_rank is a permutation of [1,2,3,4] for each cluster.
- Do not include any additional keys or text outside JSON.

Instructions:
1) Interpret the user's request into η_perf:
   - cost/energy oriented → ~0.2
   - speed/ASAP oriented → ~0.9
   - balanced → ~0.5
2) For each cluster, given profiling (sps, cost for g=1..4),
   compute score(g) ≈ η*sps_norm − (1−η)*cost_norm, rank g by score.
3) Apply guards using CSP:
   - If p_fair ≥ 0.60 and pref_g > 2: decrease pref_g by 1 unless sps(g)−sps(g−1) ≥ 0.12·max(sps).
   - If E_t ≥ 0.80 and pref_g = 4: set pref_g=3 unless sps(4) ≥ 1.10·sps(3) and cost(4) not worst.
4) Rationale: up to 3 concise reasons.

Example #1:
{example1}

Example #2:
{example2}

Now generate the JSON for the following input.

Input
User request: "{user_req}"
Profiling (2 lines):
{clines}
CSP:
{csplines}

Output JSON only:
"""
    return prompt

def generate_policy_card(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    ICL로 LLM 호출하여 정책 카드 생성. 실패 시 휴리스틱 폴백.
    job_id가 없으면 서버에서 한국시간으로 생성하여 포함.
    """
    # job_id 생성(없으면)
    if not payload.get("job_id"):
        payload["job_id"] = _kst_now_str()

    if not _USE_LLM or not _OPENAI_API_KEY:
        return _heuristic_fallback(payload)

    prompt = _compose_prompt(payload)

    try:
        # OpenAI SDK 호출
        from openai import OpenAI
        client = OpenAI(api_key=_OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are a helpful scheduling policy generator."},
                {"role":"user","content": prompt}
            ],
            temperature=0.2,
            max_tokens=500,
        )
        txt = resp.choices[0].message.content
        data = _extract_json(txt)

        # 최소 유효성
        if "clusters" not in data or "η_perf" not in data:
            raise ValueError("missing keys")
        # g_rank 보정(안전)
        for c in data["clusters"]:
            if "g_rank" not in c or not c["g_rank"]:
                c["g_rank"] = [c.get("pref_g", 2), 1, 3, 4][:4]
        # job_id 없으면 생성한 값 사용
        if not data.get("job_id"):
            data["job_id"] = payload["job_id"]
        return data

    except Exception as e:
        # 폴백
        return _heuristic_fallback(payload)

