# app/executor.py
from __future__ import annotations

import os
import json
import time
import uuid
import logging
from typing import Any, Dict, Optional, List
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# Config
GLOBAL_SERVER_BASE_URL = (os.getenv("GLOBAL_SERVER_BASE_URL", "http://localhost:8000") or "").rstrip("/")

# Global Server endpoints
GS_STATUS_PATH  = os.getenv("EXECUTOR_GS_STATUS_PATH", "/status")
GS_LAUNCH_PATH  = os.getenv("EXECUTOR_GS_LAUNCH_PATH", "/launch_job")
GS_PREEMPT_PATH = os.getenv("EXECUTOR_GS_PREEMPT_PATH", "/preempt_job")
GS_RESIZE_PATH  = os.getenv("EXECUTOR_GS_RESIZE_PATH", "/resize_job")

DEFAULT_TIMEOUT_SEC = float(os.getenv("EXECUTOR_HTTP_TIMEOUT_SEC", "20"))
DEFAULT_RETRIES = int(os.getenv("EXECUTOR_HTTP_RETRIES", "3"))
DEFAULT_RETRY_BACKOFF_SEC = float(os.getenv("EXECUTOR_HTTP_RETRY_BACKOFF_SEC", "0.8"))

DEFAULT_STOP_TIMEOUT_SEC = float(os.getenv("EXECUTOR_STOP_TIMEOUT_SEC", "220"))
DEFAULT_LAUNCH_TIMEOUT_SEC = float(os.getenv("EXECUTOR_LAUNCH_TIMEOUT_SEC", "60"))
DEFAULT_RESIZE_TIMEOUT_SEC = float(os.getenv("EXECUTOR_RESIZE_TIMEOUT_SEC", "240"))

# Helpers
def _now() -> float:
    return time.time()

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

def _truncate(s: str, n: int = 1200) -> str:
    s = s or ""
    return s if len(s) <= n else (s[:n] + "...(truncated)")

def _make_client(timeout_sec: float) -> httpx.Client:
    timeout = httpx.Timeout(connect=5.0, read=timeout_sec, write=timeout_sec, pool=timeout_sec)
    return httpx.Client(timeout=timeout, follow_redirects=True)

def _emit_http_event(
    *,
    event: str,
    job_id: str,
    cluster_id: str,
    op: str,
    method: str,
    url: str,
    http_status: int,
    ok: bool,
    req_id: str = "",
    latency_ms: float = 0.0,
    req: Optional[Dict[str, Any]] = None,
    resp: Optional[Dict[str, Any]] = None,
    ts: Optional[float] = None,
) -> None:
    """
    executor 레벨 HTTP boundary 이벤트를 http_events.csv에 남김.
    get_run_logger가 없거나 실패하면 조용히 무시.
    """
    try:
        from app.logger import get_run_logger  # local import to avoid circular
        rl = get_run_logger()
        if not rl:
            return
        rl.http_event(
            event=str(event),
            job_id=str(job_id),
            cluster=str(cluster_id),
            op=str(op),
            method=str(method),
            url=str(url),
            http_status=int(http_status),
            ok=bool(ok),
            req_id=str(req_id or ""),
            latency_ms=float(latency_ms),
            req=req if isinstance(req, dict) else {"req_raw": str(req)},
            resp=resp if isinstance(resp, dict) else {"resp_raw": str(resp)},
            ts=float(ts if ts is not None else _now()),
        )
    except Exception:
        return

def _get_json(url: str, *, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Dict[str, Any]:
    try:
        with _make_client(timeout_sec) as client:
            resp = client.get(url)

        if 200 <= resp.status_code < 300:
            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}
            if isinstance(data, dict):
                data.setdefault("ok", True)
                data.setdefault("http_status", int(resp.status_code))
                data.setdefault("status_code", int(resp.status_code))
                data.setdefault("_http_status", int(resp.status_code))
                data.setdefault("url", str(resp.request.url))
                data.setdefault("_url", str(resp.request.url))
                return data
            return {
                "ok": True,
                "data": data,
                "http_status": int(resp.status_code),
                "status_code": int(resp.status_code),
                "_http_status": int(resp.status_code),
                "url": str(resp.request.url),
                "_url": str(resp.request.url),
            }

        return {
            "ok": False,
            "reason": f"HTTP {resp.status_code}: {resp.text[:300]}",
            "http_status": int(resp.status_code),
            "status_code": int(resp.status_code),
            "url": str(resp.request.url),
        }
    except Exception as e:
        return {
            "ok": False,
            "reason": f"EXC {type(e).__name__}: {e!r}",
            "http_status": 0,
            "status_code": 0,
            "url": url,
        }

def _base_url() -> str:
    if not GLOBAL_SERVER_BASE_URL:
        raise RuntimeError("GLOBAL_SERVER_BASE_URL is empty")

    u = urlparse(GLOBAL_SERVER_BASE_URL)
    if u.scheme not in ("http", "https"):
        raise RuntimeError(f"GLOBAL_SERVER_BASE_URL must include http/https scheme: {GLOBAL_SERVER_BASE_URL!r}")
    if not u.netloc:
        raise RuntimeError(f"GLOBAL_SERVER_BASE_URL has no netloc(host:port): {GLOBAL_SERVER_BASE_URL!r}")

    if u.port == 29500:
        raise RuntimeError(
            f"GLOBAL_SERVER_BASE_URL port is 29500 (torch master port). "
            f"HTTP server port(예:8000)로 설정해야 합니다: {GLOBAL_SERVER_BASE_URL!r}"
        )

    if u.path and u.path != "/":
        logger.warning("[executor][GS] base_url contains path=%r. base_url should be host:port only.", u.path)

    return GLOBAL_SERVER_BASE_URL

# Public APIs
def ping_cluster(cluster_id: str) -> Dict[str, Any]:
    url = f"{_base_url()}{GS_STATUS_PATH}"
    return _get_json(url, timeout_sec=10.0)

def launch_or_reuse(
    *,
    cluster_id: str,
    job_id: str,
    model: str,
    dataset: str,
    world_size: int,
    epochs: int,
    batch_size_per_gpu: int,
    user_request: Optional[str] = None,
    nodes: Optional[List[str]] = None,
    is_backfill: bool = False,
    env: Optional[Dict[str, str]] = None,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = DEFAULT_LAUNCH_TIMEOUT_SEC,
) -> Dict[str, Any]:
    nodes_list = [str(x) for x in (nodes or []) if str(x)]
    if not job_id:
        return {"ok": False, "reason": "missing_job_id", "http_status": 0, "status_code": 0}
    if not cluster_id:
        return {"ok": False, "reason": "missing_cluster_id", "http_status": 0, "status_code": 0}
    if not nodes_list:
        return {"ok": False, "reason": "missing_nodes_for_global_server_launch", "http_status": 0, "status_code": 0}
    if world_size is None or int(world_size) <= 0:
        return {"ok": False, "reason": "invalid_world_size", "http_status": 0, "status_code": 0}
    if len(nodes_list) != int(world_size):
        return {
            "ok": False,
            "reason": "nodes_world_size_mismatch",
            "http_status": 0,
            "status_code": 0,
            "detail": f"len(nodes)={len(nodes_list)} world_size={int(world_size)}",
        }

    url = f"{_base_url()}{GS_LAUNCH_PATH}"

    # tracing용 request_id (랜덤 OK)
    request_id = f"{job_id}-{uuid.uuid4().hex[:10]}"

    # ----------------------------
    # SSOT: run_id / attempt / resume / reason
    # ----------------------------
    run_id_top: Optional[str] = None
    job_attempt_top: Optional[int] = None
    resume_top: Optional[str] = None
    reason_top: Optional[str] = None

    if isinstance(extra, dict) and extra:
        if extra.get("run_id"):
            run_id_top = str(extra.get("run_id")).strip() or None

        if "attempt" in extra:
            try:
                job_attempt_top = int(extra.get("attempt") or 0)
                if job_attempt_top <= 0:
                    job_attempt_top = None
            except Exception:
                job_attempt_top = None

        if extra.get("resume_from_checkpoint"):
            resume_top = str(extra.get("resume_from_checkpoint")).strip() or None

        if extra.get("reason"):
            reason_top = str(extra.get("reason")).strip() or None

    # ✅ run_id는 필수 (SSOT)
    if not run_id_top:
        return {
            "ok": False,
            "reason": "missing_run_id",
            "http_status": 0,
            "status_code": 0,
            "detail": "run_id must be provided via extra['run_id'] (SSOT).",
        }

    # attempt가 없으면 1
    if job_attempt_top is None:
        job_attempt_top = 1

    # 멱등 키는 안정적으로 고정 (job_id, run_id, job_attempt)
    idempotency_key = f"{job_id}:{run_id_top}:{int(job_attempt_top)}"

    payload: Dict[str, Any] = {
        "job_id": str(job_id),
        "cluster_id": str(cluster_id),

        # tracing/idempotency
        "request_id": request_id,
        "idempotency_key": idempotency_key,

        # 실행 파라미터
        "nodes": nodes_list,
        "model": str(model),
        "dataset": str(dataset),
        "epochs": int(epochs),
        "world_size": int(world_size),

        # per-GPU batch로 통일
        "batch_size": int(batch_size_per_gpu),
        "batch_size_per_gpu": int(batch_size_per_gpu),

        "is_backfill": bool(is_backfill),

        # ✅ SSOT run identity
        "run_id": str(run_id_top),
        "attempt": int(job_attempt_top),
    }

    if user_request is not None:
        payload["user_request"] = str(user_request)

    if env:
        payload["env"] = env

    if resume_top:
        payload["resume_from_checkpoint"] = resume_top

    payload["reason"] = reason_top if reason_top else ("BACKFILL" if is_backfill else "LAUNCH")

    # 기존 호환 유지: extra 원문도 남기되, JSON-safe 처리
    if isinstance(extra, dict) and extra:
        payload["extra"] = extra

    # ---------------------------------------------------------
    # ✅ payload JSON-safe 강제 변환
    # ---------------------------------------------------------
    try:
        payload = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        # 최후 fallback
        payload = {
            "job_id": str(job_id),
            "cluster_id": str(cluster_id),
            "request_id": str(request_id),
            "idempotency_key": str(idempotency_key),
            "nodes": [str(x) for x in nodes_list],
            "model": str(model),
            "dataset": str(dataset),
            "epochs": int(epochs),
            "world_size": int(world_size),
            "batch_size": int(batch_size_per_gpu),
            "batch_size_per_gpu": int(batch_size_per_gpu),
            "is_backfill": bool(is_backfill),
            "run_id": str(run_id_top),
            "attempt": int(job_attempt_top),
            "reason": "BACKFILL" if is_backfill else "LAUNCH",
        }
        if resume_top:
            payload["resume_from_checkpoint"] = str(resume_top)

    logger.info(
        "[executor][GS][launch_requested] url=%s job_id=%s cluster=%s run_id=%s job_attempt=%s g=%s nodes=%s backfill=%s request_id=%s idem=%s payload=%s",
        url,
        job_id,
        cluster_id,
        payload.get("run_id"),
        payload.get("attempt"),
        int(world_size),
        nodes_list,
        bool(is_backfill),
        request_id,
        idempotency_key,
        _safe_json(payload),
    )

    _emit_http_event(
        event="launch_requested",
        job_id=str(job_id),
        cluster_id=str(cluster_id),
        op="launch_job",
        method="POST",
        url=str(GS_LAUNCH_PATH),
        http_status=0,
        ok=True,
        req_id=str(request_id),
        latency_ms=0.0,
        req={"payload": payload, "idempotency_key": idempotency_key},
        resp=None,
        ts=_now(),
    )

    last_err: Optional[str] = None
    last_detail: str = ""
    last_http: int = 0
    last_body: Any = None
    retries_used = 0

    headers = {
        "X-Request-ID": request_id,
        "X-Idempotency-Key": idempotency_key,
    }

    # ✅ retry 가능한 HTTP만 허용
    def _is_retryable_http(code: int) -> bool:
        if code == 0:
            return True
        if code in (408, 429):
            return True
        if 500 <= code < 600:
            return True
        return False

    for retry_i in range(1, DEFAULT_RETRIES + 1):
        retries_used = retry_i
        t0 = _now()

        try:
            with _make_client(float(timeout_sec)) as client:
                resp = client.post(url, json=payload, headers=headers)

            dt = _now() - t0
            last_http = int(resp.status_code)

            try:
                body: Any = resp.json()
            except Exception:
                body = {"text": (resp.text or "")[:500]}
            last_body = body

            http_ok = 200 <= last_http < 300
            body_ok = True
            if isinstance(body, dict) and "ok" in body:
                body_ok = bool(body.get("ok"))

            ok_final = bool(http_ok and body_ok)

            logger.info(
                "[executor][GS][launch_response] url=%s job_id=%s run_id=%s job_attempt=%s request_id=%s idem=%s retry=%d/%d http=%d ok=%s latency=%.3fs body=%s",
                url,
                job_id,
                payload.get("run_id"),
                payload.get("attempt"),
                request_id,
                idempotency_key,
                retry_i,
                DEFAULT_RETRIES,
                last_http,
                ok_final,
                dt,
                _safe_json(body),
            )

            _emit_http_event(
                event="launch_response",
                job_id=str(job_id),
                cluster_id=str(cluster_id),
                op="launch_job",
                method="POST",
                url=str(GS_LAUNCH_PATH),
                http_status=int(last_http),
                ok=bool(ok_final),
                req_id=str(request_id),
                latency_ms=float(dt * 1000.0),
                req={
                    "payload": payload,
                    "retry_attempt": retry_i,
                    "retries_total": DEFAULT_RETRIES,
                    "idempotency_key": idempotency_key,
                },
                resp=body if isinstance(body, dict) else {"body": body},
                ts=_now(),
            )

            # ✅ HARD RULE: 409 Conflict(자원 충돌)은 절대 retry 하지 않는다.
            # 스케줄러가 다른 클러스터/노드로 재선택할 수 있게 즉시 실패 반환.
            if last_http == 409:
                reason_409 = ""
                if isinstance(body, dict):
                    reason_409 = str(body.get("reason") or "")
                return {
                    "ok": False,
                    "http_status": last_http,
                    "status_code": last_http,
                    "reason": reason_409 or "conflict_409",
                    "detail": (resp.text or "")[:500],
                    "body": body,
                    "url": url,
                    "request_id": request_id,
                    "idempotency_key": idempotency_key,
                    "retries": retry_i,
                    "run_id": run_id_top,
                    "attempt": int(job_attempt_top),
                }

            if ok_final:
                out: Dict[str, Any] = {}
                if isinstance(body, dict):
                    out.update(body)
                else:
                    out["body"] = body

                out.setdefault("ok", True)
                out.setdefault("http_status", last_http)
                out.setdefault("status_code", last_http)
                out.setdefault("latency_sec", float(dt))
                out.setdefault("url", url)
                out.setdefault("request_id", request_id)
                out.setdefault("idempotency_key", idempotency_key)
                out.setdefault("retries", retries_used)
                out.setdefault("run_id", payload.get("run_id"))
                out.setdefault("attempt", payload.get("attempt"))
                return out

            # 실패 reason 정리
            last_err = "launch_failed"
            if not http_ok:
                last_err = f"http_{last_http}"
            elif isinstance(body, dict) and body.get("reason"):
                last_err = str(body.get("reason"))
            last_detail = (resp.text or "")[:500]

            # ✅ retry 조건을 강하게 제한
            if not _is_retryable_http(last_http):
                return {
                    "ok": False,
                    "http_status": last_http,
                    "status_code": last_http,
                    "reason": last_err or f"http_{last_http}",
                    "detail": last_detail,
                    "body": last_body,
                    "url": url,
                    "request_id": request_id,
                    "idempotency_key": idempotency_key,
                    "retries": retry_i,
                    "run_id": run_id_top,
                    "attempt": int(job_attempt_top),
                }

        except Exception as e:
            dt = _now() - t0
            last_http = 0
            last_err = f"EXC {type(e).__name__}"
            last_detail = str(e)[:500]

            logger.warning(
                "[executor][GS][launch_exception] url=%s job_id=%s run_id=%s job_attempt=%s request_id=%s idem=%s retry=%d/%d latency=%.3fs err=%s detail=%s",
                url,
                job_id,
                payload.get("run_id"),
                payload.get("attempt"),
                request_id,
                idempotency_key,
                retry_i,
                DEFAULT_RETRIES,
                dt,
                last_err,
                last_detail,
            )

            _emit_http_event(
                event="launch_exception",
                job_id=str(job_id),
                cluster_id=str(cluster_id),
                op="launch_job",
                method="POST",
                url=str(GS_LAUNCH_PATH),
                http_status=0,
                ok=False,
                req_id=str(request_id),
                latency_ms=float(dt * 1000.0),
                req={
                    "payload": payload,
                    "retry_attempt": retry_i,
                    "retries_total": DEFAULT_RETRIES,
                    "idempotency_key": idempotency_key,
                },
                resp={"err": last_err, "detail": last_detail},
                ts=_now(),
            )

            # exception은 retryable 취급(0)
            if not _is_retryable_http(0):
                break

        # retry backoff (retryable일 때만 도달)
        if retry_i < DEFAULT_RETRIES:
            time.sleep(DEFAULT_RETRY_BACKOFF_SEC * retry_i)

    return {
        "ok": False,
        "http_status": last_http,
        "status_code": last_http,
        "reason": last_err or "unknown_error",
        "detail": last_detail,
        "body": last_body,
        "url": url,
        "request_id": request_id,
        "idempotency_key": idempotency_key,
        "retries": retries_used,
        "run_id": run_id_top,
        "attempt": int(job_attempt_top),
    }

def stop_job(
    *,
    cluster_id: str,
    job_id: str,
    reason: str = "preempt",
    force: bool = False,
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = DEFAULT_STOP_TIMEOUT_SEC,
    **kwargs: Any,
) -> Dict[str, Any]:
    import uuid

    if not job_id:
        return {"ok": False, "reason": "missing job_id", "http_status": 0, "status_code": 0}

    url = f"{_base_url()}{GS_PREEMPT_PATH}"
    request_id = f"{job_id}-stop-{uuid.uuid4().hex[:10]}"

    payload: Dict[str, Any] = {
        "job_id": str(job_id),
        "cluster_id": str(cluster_id),
        "reason": str(reason),
        "force": bool(force),
        "request_id": request_id,
        "idempotency_key": request_id,
    }
    if extra:
        payload["extra"] = extra

    logger.info(
        "[executor][GS][preempt_requested] url=%s job_id=%s cluster=%s reason=%s force=%s request_id=%s payload=%s",
        url, job_id, cluster_id, reason, bool(force), request_id, _safe_json(payload)
    )

    _emit_http_event(
        event="preempt_requested",
        job_id=str(job_id),
        cluster_id=str(cluster_id),
        op="preempt_job",
        method="POST",
        url=str(GS_PREEMPT_PATH),
        http_status=0,
        ok=True,
        req_id=str(request_id),
        latency_ms=0.0,
        req={"payload": payload},
        resp=None,
        ts=_now(),
    )

    t0 = _now()
    try:
        with _make_client(float(timeout_sec)) as client:
            resp = client.post(url, json=payload, headers={"X-Request-ID": request_id})
        dt = _now() - t0
        http_status = int(resp.status_code)

        try:
            body: Any = resp.json()
        except Exception:
            body = {"text": (resp.text or "")[:800]}

        ok_http = 200 <= http_status < 300
        ok_body = True
        if isinstance(body, dict) and "ok" in body:
            ok_body = bool(body.get("ok"))

        ok_final = bool(ok_http and ok_body)

        _emit_http_event(
            event="preempt_response",
            job_id=str(job_id),
            cluster_id=str(cluster_id),
            op="preempt_job",
            method="POST",
            url=str(GS_PREEMPT_PATH),
            http_status=int(http_status),
            ok=bool(ok_final),
            req_id=str(request_id),
            latency_ms=float(dt * 1000.0),
            req={"payload": payload},
            resp=body if isinstance(body, dict) else {"body": body},
            ts=_now(),
        )

        out: Dict[str, Any] = {}
        if isinstance(body, dict):
            out.update(body)
        else:
            out["body"] = body

        out.setdefault("ok", bool(ok_final))
        out.setdefault("http_status", int(http_status))
        out.setdefault("status_code", int(http_status))
        out.setdefault("latency_sec", float(dt))
        out.setdefault("url", url)
        out.setdefault("request_id", request_id)
        return out

    except Exception as e:
        dt = _now() - t0
        err = f"EXC {type(e).__name__}: {e!r}"

        _emit_http_event(
            event="preempt_exception",
            job_id=str(job_id),
            cluster_id=str(cluster_id),
            op="preempt_job",
            method="POST",
            url=str(GS_PREEMPT_PATH),
            http_status=0,
            ok=False,
            req_id=str(request_id),
            latency_ms=float(dt * 1000.0),
            req={"payload": payload},
            resp={"err": err},
            ts=_now(),
        )

        return {
            "ok": False,
            "reason": err,
            "http_status": 0,
            "status_code": 0,
            "url": url,
            "request_id": request_id,
        }

def resize_job(
    *,
    cluster_id: str,
    job_id: str,
    new_world_size: int,
    nodes: Optional[List[str]] = None,
    reason: str = "elastic_resize",
    extra: Optional[Dict[str, Any]] = None,
    timeout_sec: float = DEFAULT_RESIZE_TIMEOUT_SEC,
) -> Dict[str, Any]:
    import uuid

    if not job_id:
        return {"ok": False, "reason": "missing job_id", "http_status": 0, "status_code": 0}
    if new_world_size is None or int(new_world_size) <= 0:
        return {"ok": False, "reason": "invalid new_world_size", "http_status": 0, "status_code": 0}

    new_nodes = list(nodes or [])
    if not new_nodes:
        return {"ok": False, "reason": "missing_nodes_for_global_server_resize", "http_status": 0, "status_code": 0}
    if len(new_nodes) != int(new_world_size):
        return {
            "ok": False,
            "reason": "nodes_world_size_mismatch",
            "http_status": 0,
            "status_code": 0,
            "detail": f"len(nodes)={len(new_nodes)} new_world_size={int(new_world_size)}",
        }

    url = f"{_base_url()}{GS_RESIZE_PATH}"
    request_id = f"{job_id}-resize-{uuid.uuid4().hex[:10]}"

    payload: Dict[str, Any] = {
        "job_id": str(job_id),
        "cluster_id": str(cluster_id),
        "new_world_size": int(new_world_size),
        "new_nodes": new_nodes,
        "reason": str(reason),
        "request_id": request_id,
        "idempotency_key": request_id,
    }
    if extra:
        payload["extra"] = extra

    logger.info(
        "[executor][GS][resize_requested] url=%s job_id=%s cluster=%s new_g=%s new_nodes=%s reason=%s request_id=%s payload=%s",
        url, job_id, cluster_id, int(new_world_size), new_nodes, reason, request_id, _safe_json(payload)
    )

    _emit_http_event(
        event="resize_requested",
        job_id=str(job_id),
        cluster_id=str(cluster_id),
        op="resize_job",
        method="POST",
        url=str(GS_RESIZE_PATH),
        http_status=0,
        ok=True,
        req_id=str(request_id),
        latency_ms=0.0,
        req={"payload": payload},
        resp=None,
        ts=_now(),
    )

    t0 = _now()
    try:
        with _make_client(float(timeout_sec)) as client:
            resp = client.post(url, json=payload, headers={"X-Request-ID": request_id})
        dt = _now() - t0
        http_status = int(resp.status_code)

        try:
            body: Any = resp.json()
        except Exception:
            body = {"text": (resp.text or "")[:800]}

        ok_http = 200 <= http_status < 300
        ok_body = True
        if isinstance(body, dict) and "ok" in body:
            ok_body = bool(body.get("ok"))

        ok_final = bool(ok_http and ok_body)

        _emit_http_event(
            event="resize_response",
            job_id=str(job_id),
            cluster_id=str(cluster_id),
            op="resize_job",
            method="POST",
            url=str(GS_RESIZE_PATH),
            http_status=int(http_status),
            ok=bool(ok_final),
            req_id=str(request_id),
            latency_ms=float(dt * 1000.0),
            req={"payload": payload},
            resp=body if isinstance(body, dict) else {"body": body},
            ts=_now(),
        )

        out: Dict[str, Any] = {}
        if isinstance(body, dict):
            out.update(body)
        else:
            out["body"] = body

        out.setdefault("ok", bool(ok_final))
        out.setdefault("http_status", int(http_status))
        out.setdefault("status_code", int(http_status))
        out.setdefault("latency_sec", float(dt))
        out.setdefault("url", url)
        out.setdefault("request_id", request_id)
        return out

    except Exception as e:
        dt = _now() - t0
        err = f"EXC {type(e).__name__}: {e!r}"

        _emit_http_event(
            event="resize_exception",
            job_id=str(job_id),
            cluster_id=str(cluster_id),
            op="resize_job",
            method="POST",
            url=str(GS_RESIZE_PATH),
            http_status=0,
            ok=False,
            req_id=str(request_id),
            latency_ms=float(dt * 1000.0),
            req={"payload": payload},
            resp={"err": err},
            ts=_now(),
        )

        return {
            "ok": False,
            "reason": err,
            "http_status": 0,
            "status_code": 0,
            "url": url,
            "request_id": request_id,
        }