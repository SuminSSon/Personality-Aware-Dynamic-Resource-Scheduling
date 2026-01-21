import os, json
from typing import Any, Dict
from openai import OpenAI

# Env:
#   OPENAI_API_KEY (필수)
#   OPENAI_MODEL   (기본: gpt-4o-mini)

_client = None

def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI(api_key=key)
    return _client

def call_llm_json(system_prompt: str, developer_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = _client_singleton()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "developer", "content": developer_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        top_p=1.0,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception as e:
        raise RuntimeError(f"LLM response not valid JSON: {e}; raw={content[:1200]}")
    return data

