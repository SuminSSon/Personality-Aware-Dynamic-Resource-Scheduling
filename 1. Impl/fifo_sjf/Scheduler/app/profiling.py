from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import math
import psycopg
from psycopg.rows import dict_row

# =========================
# DB 설정 (clusterA 전용)
# =========================
_DSN = os.getenv("PROFILING_DSN", "postgresql://prof:profpw@localhost:5432/profdb")
_TABLE = os.getenv("PROFILING_TABLE", "minimal_profiling")  # 필요 시 "profiling_jobs"

# 행 컬럼 유연 매핑 키 후보
_G_KEYS = ("g", "num_gpus", "gpu_count", "ngpus")
_SPS_KEYS = ("throughput", "sps", "images_per_sec", "samples_per_sec")
_UTIL_KEYS = ("util_avg", "gpu_util", "avg_util", "utilization")


# =========================
# DB 유틸
# =========================
def _connect():
    return psycopg.connect(_DSN, autocommit=True)

def _columns(conn, table: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
        """, (table,))
        return [r[0] for r in cur.fetchall()]

def _pick_first_key(d: Dict[str, Any], keys: tuple, cast=float) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return cast(d[k])
            except Exception:
                pass
    return None

def _fetch_rows(model: str, dataset: str, limit: int = 64) -> List[Dict[str, Any]]:
    """
    clusterA의 프로파일 원시 행을 최신순으로 가져온다.
    - model_name/dataset 컬럼이 없으면 필터 없이 최신 limit만 반환
    - ts 컬럼이 있으면 ts DESC, 없으면 id DESC
    """
    try:
        with _connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cols = set(_columns(conn, _TABLE))
            has_model = "model_name" in cols
            has_dataset = "dataset" in cols
            order_col = "ts" if "ts" in cols else ("id" if "id" in cols else None)

            where_parts: List[str] = []
            params: List[Any] = []

            if has_model:
                where_parts.append("model_name = %s")
                params.append(model)
            if has_dataset:
                where_parts.append("dataset = %s")
                params.append(dataset)

            base_cols = ", ".join(sorted(cols))
            where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
            order_sql = f"ORDER BY {order_col} DESC" if order_col else ""
            limit_sql = "LIMIT %s"
            params.append(limit)

            sql = f"SELECT {base_cols} FROM {_TABLE} {where_sql} {order_sql} {limit_sql}"
            cur.execute(sql, params)
            rows = cur.fetchall() or []

            # 타입 정리
            for r in rows:
                # ts ISO 문자열화
                if "ts" in r and r["ts"] is not None:
                    try:
                        r["ts"] = r["ts"].isoformat()
                    except Exception:
                        pass
            return rows
    except Exception as e:
        print(f"[profiling] DB error: {e}")
        return []


# =========================
# compact 생성 (g:[1..4], sps[], cost[])
# =========================
def _fill_missing_monotone(base: List[Optional[float]], rule: str) -> List[float]:
    """
    None 값을 단조 증가(sps) / 단조 감소(cost) 가정으로 보정.
    rule: "sps" | "cost"
    - sps: g 증가 시 완만한 증가(감소도 허용하되 보간)
    - cost: g 증가 시 소폭 감소 경향
    """
    xs = base[:]
    # 1) 선형 보간
    # 앞뒤 값으로 보간, 양끝단은 이웃값을 이용해 완만히 증/감
    # sps는 최소 0, cost는 [0,1] 클립
    for i in range(4):
        if xs[i] is None:
            # 가장 가까운 좌우 찾기
            l = next((j for j in range(i-1, -1, -1) if xs[j] is not None), None)
            r = next((j for j in range(i+1, 4) if xs[j] is not None), None)
            if l is not None and r is not None:
                # 선형 보간
                val = xs[l] + (xs[r]-xs[l]) * ((i-l)/(r-l))
            elif l is not None:
                # 좌측 추정: 완만한 추세
                val = xs[l] * (1.12 if rule=="sps" else 0.95)
            elif r is not None:
                # 우측 추정
                val = xs[r] * (0.90 if rule=="sps" else 1.05)
            else:
                # 전부 None이면 기본 값
                val = 10.0 if rule=="sps" else 0.85
            xs[i] = float(val)

    # 2) 단조성 소정리(너무 틀어지지 않게)
    if rule == "sps":
        for i in range(1,4):
            if xs[i] < xs[i-1]*0.85:  # 통신 오버헤드로 감소는 가능, 과도 감소 방지
                xs[i] = xs[i-1]*0.85
    else:  # cost는 완만히 감소 경향
        for i in range(1,4):
            if xs[i] > xs[i-1]:      # 증가하면 살짝 눌러줌
                xs[i] = xs[i-1]*0.98
    # 클립
    if rule == "sps":
        xs = [max(0.0, float(v)) for v in xs]
    else:
        xs = [min(1.0, max(0.0, float(v))) for v in xs]
    return xs

def _rows_to_compact(rows: List[Dict[str, Any]]) -> Dict[str, List[float]]:
    """
    최신행 우선으로 g∈{1,2,3,4}에 대한 sps, cost(util_avg)를 채운다.
    - sps: throughput 계열
    - cost: util_avg 계열 (낮을수록 좋음으로 그대로 사용)
    """
    # 최신순으로 이미 정렬돼 있다고 가정
    seen = {1: False, 2: False, 3: False, 4: False}
    sps: List[Optional[float]]  = [None, None, None, None]
    cost: List[Optional[float]] = [None, None, None, None]

    for r in rows:
        g = _pick_first_key(r, _G_KEYS, cast=int)
        if g not in (1,2,3,4): 
            continue
        idx = g - 1
        if seen[g]:
            continue  # 최신 우선 한 번만
        s = _pick_first_key(r, _SPS_KEYS, cast=float)
        u = _pick_first_key(r, _UTIL_KEYS, cast=float)
        if s is not None and sps[idx] is None:
            sps[idx] = float(s)
        if u is not None and cost[idx] is None:
            # cost는 util_avg를 그대로 사용 (낮을수록 좋음)
            cost[idx] = float(u)
        seen[g] = True
        if all(seen.values()):
            break

    # 누락 채우기
    sps_filled  = _fill_missing_monotone(sps,  "sps")
    cost_filled = _fill_missing_monotone(cost, "cost")
    return {
        "g":    [1,2,3,4],
        "sps":  sps_filled,
        "cost": cost_filled,
    }


# =========================
# clusterB 더미(튜너블)
# =========================
def _clusterB_fallback(model: str, dataset: str) -> Dict[str, List[float]]:
    """
    간단 더미. ENV로 튜닝 가능:
      B_SPS_BASE=10.0, B_SPS_SCALE=1.45, B_COST1=0.92, B_COST_DECAY=0.97
    """
    base   = float(os.getenv("B_SPS_BASE",  "10.0"))
    scale  = float(os.getenv("B_SPS_SCALE", "1.45"))
    c1     = float(os.getenv("B_COST1",     "0.92"))
    cdecay = float(os.getenv("B_COST_DECAY","0.97"))

    s1 = base
    s2 = s1 * scale
    s3 = s2 * 1.15
    s4 = s3 * 1.03

    c = [c1, c1*cdecay, c1*(cdecay**2), c1*(cdecay**3)]
    return {
        "g": [1,2,3,4],
        "sps": [s1, s2, s3, s4],
        "cost": c,
    }


# =========================
# 공개 API: 2줄 compact 반환
# =========================
def build_compact_profiles(model: str, dataset: str, limit: int = 64) -> List[Dict]:
    """
    멀티클러스터 2줄 compact:
      [
        {"cluster":"clusterA","g":[1,2,3,4],"sps":[...],"cost":[...]},
        {"cluster":"clusterB","g":[1,2,3,4],"sps":[...],"cost":[...]}
      ]
    - clusterA: DB에서 최신 행을 모아 g별 하나씩 매핑(없으면 보간/추정)
    - clusterB: 더미
    """
    # clusterA
    rowsA = _fetch_rows(model, dataset, limit=limit)
    if rowsA:
        packA = _rows_to_compact(rowsA)
    else:
        # DB가 비어있으면 clusterA도 간단 fallback
        packA = {
            "g": [1,2,3,4],
            "sps": [10.0, 15.0, 17.0, 17.5],
            "cost": [0.90, 0.85, 0.83, 0.82],
        }

    # clusterB (데이터 없음 → 더미)
    packB = _clusterB_fallback(model, dataset)

    return [
        {"cluster": "clusterA", **packA},
        {"cluster": "clusterB", **packB},
    ]

