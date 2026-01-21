from __future__ import annotations

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
import re

from openai import OpenAI
from app.guardrail import normalize_policy_values

logger = logging.getLogger(__name__)

OPENAI_API_KEY = "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"


@dataclass
class PolicyVector:
    lambda_time: float
    lambda_cost: float
    lambda_energy: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "lambda_time": self.lambda_time,
            "lambda_cost": self.lambda_cost,
            "lambda_energy": self.lambda_energy,
        }


SYSTEM_PROMPT = """
You are a requirement interpreter for a multi-objective job scheduler.
Your task is to convert the user's natural-language requirements into
a normalized policy vector (λ_time, λ_cost, λ_energy).

Definitions:
- λ_time   : Preference for shorter job completion time / higher throughput.
- λ_cost   : Preference for lower **monetary cost** (GPU-hour price).
- λ_energy : Preference for lower **energy / power / carbon footprint**.

Rules:
- Use only the user's written requirements.
- Do not infer system metrics or cluster states.
- All values must be non-negative and sum to 1.
- Output JSON only. No explanation. No text outside JSON.
- Keys: lambda_time, lambda_cost, lambda_energy

In-context examples:

User Request:
"Please finish as fast as possible."
→ Output:
{
  "lambda_time": 0.80,
  "lambda_cost": 0.10,
  "lambda_energy": 0.10
}

User Request:
"Minimizing monetary cost is the highest priority."
→ Output:
{
  "lambda_time": 0.10,
  "lambda_cost": 0.80,
  "lambda_energy": 0.10
}

User Request:
"I need an energy-saving / eco-friendly mode."
→ Output:
{
  "lambda_time": 0.15,
  "lambda_cost": 0.10,
  "lambda_energy": 0.75
}

User Request:
"Balance speed, monetary cost, and energy consumption."
→ Output:
{
  "lambda_time": 0.33,
  "lambda_cost": 0.33,
  "lambda_energy": 0.34
}

Now, given a new user request, output only JSON for the policy vector.
"""

def _normalize_policy_vector(raw: Dict[str, float]) -> PolicyVector:
    lt_raw = float(raw.get("lambda_time", 0.0) or 0.0)
    lc_raw = float(raw.get("lambda_cost", 0.0) or 0.0)
    le_raw = float(raw.get("lambda_energy", 0.0) or 0.0)

    lt, lc, le = normalize_policy_values(lt_raw, lc_raw, le_raw)

    return PolicyVector(
        lambda_time=round(lt, 3),
        lambda_cost=round(lc, 3),
        lambda_energy=round(le, 3),
    )

def _build_messages(user_request: str, context: Optional[Dict[str, Any]] = None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        # context는 나중에 model/dataset, user_id, 과거 실행 패턴 등을 넣을 수 있음.
        ctx_str = json.dumps(context, ensure_ascii=False)
        user_content = f"User request:\n{user_request}\n\nJob/context metadata (JSON):\n{ctx_str}"
    else:
        user_content = user_request

    messages.append({"role": "user", "content": user_content})
    return messages

def _query_llm_policy(user_request: str, context: Optional[Dict[str, Any]] = None) -> PolicyVector:
    if OPENAI_API_KEY.startswith("sk-") is False or len(OPENAI_API_KEY) < 10:
        raise RuntimeError(
            "OPENAI_API_KEY not set correctly. Edit app/llm.py and set your key."
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    messages = _build_messages(user_request=user_request, context=context)

    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=messages,
    )

    content = completion.choices[0].message.content.strip()
    logger.info("[LLM RAW OUTPUT] %s", content)

    # ---------- 여기부터: 코드블럭/잡소리 제거 로직 ----------
    raw = content.strip()

    # 1) ```json ... ``` 또는 ``` ... ``` 감싸져 있으면 벗기기
    if raw.startswith("```"):
        # 첫 줄의 ```json 또는 ``` 제거
        raw = re.sub(r"^```[a-zA-Z0-9_-]*", "", raw).strip()
        # 끝의 ``` 제거
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    # 2) 그대로 파싱 시도
    try:
        obj = json.loads(raw)
        return _normalize_policy_vector(obj)
    except Exception as e:
        logger.warning("Primary JSON parse failed, try extracting {...}: %s", e)

    # 3) 혹시 앞뒤에 설명 텍스트가 붙은 경우: 첫 '{' ~ 마지막 '}'만 다시 잘라서 재시도
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            sliced = raw[start:end+1]
            obj = json.loads(sliced)
            return _normalize_policy_vector(obj)
    except Exception as e2:
        logger.error("Failed to parse LLM JSON even after slicing: %s", e2)

    # 4) 여기까지 오면 그냥 실패로 처리 → 상위에서 fallback 정책 사용
    raise RuntimeError(f"LLM returned non-JSON output:\n{content}")

def get_policy_from_user_request(
    user_request: str,
    context: Optional[Dict[str, Any]] = None,
) -> PolicyVector:
    try:
        return _query_llm_policy(user_request=user_request, context=context)
    except Exception:
        logger.exception(
            "[LLM] policy generation failed for request=%r, fallback to balanced policy",
            user_request,
        )
        # 균형형 디폴트: (0.33, 0.33, 0.34) → guardrail을 한 번 더 거쳐서 안전하게 사용
        lt, lc, le = normalize_policy_values(0.33, 0.33, 0.34)
        return PolicyVector(
            lambda_time=round(lt, 3),
            lambda_cost=round(lc, 3),
            lambda_energy=round(le, 3),
        )