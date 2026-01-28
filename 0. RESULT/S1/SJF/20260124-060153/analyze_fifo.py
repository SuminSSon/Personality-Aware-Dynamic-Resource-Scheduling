#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIFO_300_2 같은 job_metrics.csv(중복/러닝 row 섞임) + (선택) GPU telemetry로
아래 지표를 계산합니다.

요구 지표:
- Makespan
- avg JCT (s)
- avg_queue (s)
- avg_Ut_total
- Throughput
- avg_power
- avg_cost($)

정의(기본):
1) job_metrics.csv는 같은 job_id가 여러 번 찍힐 수 있으므로 "마지막 row"만 최종으로 사용.
2) end_ts가 있는 job(= completed)을 대상으로 지표 계산.
3) Makespan = max(end_ts) - min(submitted_ts)  (completed들 기준)
4) avg JCT (s) = mean(end_ts - submitted_ts)
5) avg_queue (s) = mean(queued_sec)
6) Throughput = completed_jobs / (makespan_hours)  [jobs/hour]
7) Cost(기본) = runtime_hours * rate(cluster) * world_size
   - clusterA $10/GPU-hour, clusterB $15/GPU-hour
8) avg_power, avg_Ut_total:
   - telemetry 파일이 있으면(기본 gpu_telemetry.csv):
       avg_power = mean(power_w)
       avg_Ut_total = mean(gpu_util)/100  (0~1)
   - telemetry가 없으면(대체):
       avg_Ut_total ≈ Σ(world_size*run_sec) / (total_gpus * makespan_sec)
       avg_power = NA

사용:
  python3 analyze_fifo.py --job_metrics job_metrics.csv --telemetry gpu_telemetry.csv

telemetry 없는 경우:
  python3 analyze_fifo.py --job_metrics job_metrics.csv --no_telemetry --clusterA_gpus 4 --clusterB_gpus 4
"""

from __future__ import annotations

import argparse
import math
import pandas as pd


DEFAULT_RATES = {"clusterA": 10.0, "clusterB": 15.0}


def _safe_mean(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    if x.dropna().empty:
        return float("nan")
    return float(x.mean())


def _sec_to_hms(sec: float) -> str:
    if sec is None or (isinstance(sec, float) and math.isnan(sec)):
        return "NA"
    sec = float(sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - (h * 3600 + m * 60)
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


def load_job_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    need = {"job_id", "cluster", "world_size", "submitted_ts", "started_ts", "end_ts", "queued_sec"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"job_metrics.csv missing columns: {sorted(missing)}")

    for c in ["world_size", "submitted_ts", "started_ts", "end_ts", "queued_sec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # ✅ 같은 job_id는 마지막 row가 최종
    df_final = df.drop_duplicates(subset=["job_id"], keep="last").copy()
    df_final = df_final.sort_values(["submitted_ts", "job_id"], ascending=[True, True]).reset_index(drop=True)

    return df_final


def compute_job_costs(df: pd.DataFrame, rates: dict, rate_is_per_gpu_hour: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["run_sec"] = out["end_ts"] - out["started_ts"]
    out["run_hours"] = out["run_sec"] / 3600.0
    out["rate_per_hour"] = out["cluster"].map(rates)

    if rate_is_per_gpu_hour:
        out["cost_usd"] = out["run_hours"] * out["rate_per_hour"] * out["world_size"]
    else:
        out["cost_usd"] = out["run_hours"] * out["rate_per_hour"]

    return out


def load_telemetry(path: str) -> pd.DataFrame:
    t = pd.read_csv(path)
    # 컬럼 유연 처리
    for c in ["gpu_util", "power_w", "mem_used_mb", "mem_total_mb"]:
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_metrics", default="job_metrics.csv")
    ap.add_argument("--telemetry", default="gpu_telemetry.csv")
    ap.add_argument("--no_telemetry", action="store_true")
    ap.add_argument("--rateA", type=float, default=DEFAULT_RATES["clusterA"])
    ap.add_argument("--rateB", type=float, default=DEFAULT_RATES["clusterB"])
    ap.add_argument("--rate_is_per_gpu_hour", action="store_true",
                    help="요금이 $/GPU-hour이면 켜세요(기본 가정). $/job-hour이면 끄세요.")
    ap.add_argument("--clusterA_gpus", type=int, default=4, help="telemetry 없을 때만 사용")
    ap.add_argument("--clusterB_gpus", type=int, default=4, help="telemetry 없을 때만 사용")
    ap.add_argument("--out_jobs_csv", default="job_costs_completed.csv")
    args = ap.parse_args()

    rates = {"clusterA": float(args.rateA), "clusterB": float(args.rateB)}

    # 1) job_metrics 로드 + dedup(last)
    df_final = load_job_metrics(args.job_metrics)

    # 2) completed만 (end_ts 있는 것)
    df_comp = df_final[df_final["end_ts"].notna()].copy()
    df_run = df_final[df_final["end_ts"].isna()].copy()

    # 3) 기본 시간 지표 계산(완료 job 기준)
    n_comp = len(df_comp)
    if n_comp == 0:
        raise RuntimeError("completed(end_ts 존재) job이 0개입니다. makespan/JCT 계산 불가.")

    makespan_sec = float(df_comp["end_ts"].max() - df_comp["submitted_ts"].min())
    df_comp["jct_sec_calc"] = df_comp["end_ts"] - df_comp["submitted_ts"]
    avg_jct_sec = float(df_comp["jct_sec_calc"].mean())
    avg_queue_sec = float(df_comp["queued_sec"].mean())

    makespan_hours = makespan_sec / 3600.0
    throughput_jobs_per_hour = (n_comp / makespan_hours) if makespan_hours > 0 else float("nan")

    # 4) 비용 계산(완료 job 기준)
    df_cost = compute_job_costs(df_comp, rates=rates, rate_is_per_gpu_hour=(args.rate_is_per_gpu_hour or True))
    total_cost = float(df_cost["cost_usd"].sum())
    avg_cost = float(df_cost["cost_usd"].mean())

    # 5) Util/Power 계산
    avg_power = float("nan")
    avg_ut_total = float("nan")

    if not args.no_telemetry:
        try:
            t = load_telemetry(args.telemetry)
            if "gpu_util" in t.columns:
                avg_ut_total = _safe_mean(t["gpu_util"]) / 100.0  # 0~1
            if "power_w" in t.columns:
                avg_power = _safe_mean(t["power_w"])
        except FileNotFoundError:
            print(f"[WARN] telemetry 파일 없음: {args.telemetry}  -> allocation 기반 util로 대체합니다.")
            args.no_telemetry = True

    if args.no_telemetry:
        # allocation 기반 근사(util) (0~1)
        # avg_Ut_total ≈ Σ(world_size*run_sec) / (total_gpus * makespan_sec)
        total_gpu = int(args.clusterA_gpus) + int(args.clusterB_gpus)
        df_cost["run_sec"] = df_cost["end_ts"] - df_cost["started_ts"]
        gpu_sec = float((df_cost["world_size"] * df_cost["run_sec"]).sum())
        denom = float(total_gpu * makespan_sec) if makespan_sec > 0 else float("nan")
        avg_ut_total = (gpu_sec / denom) if denom and denom > 0 else float("nan")
        avg_power = float("nan")

    # 6) 결과 출력
    print("=== Summary (completed jobs only, dedup-last) ===")
    print(f"Completed jobs      : {n_comp}")
    print(f"Running jobs (seen) : {len(df_run)}")
    print(f"Makespan            : {_sec_to_hms(makespan_sec)} ({makespan_sec:.3f} s)")
    print(f"avg JCT (s)         : {avg_jct_sec:.3f}")
    print(f"avg_queue (s)       : {avg_queue_sec:.3f}")
    print(f"avg_Ut_total        : {avg_ut_total:.6f}  (0~1)")
    print(f"Throughput          : {throughput_jobs_per_hour:.6f} jobs/hour")
    print(f"avg_power           : {avg_power:.6f}" if not math.isnan(avg_power) else "avg_power           : NA")
    print(f"avg_cost($)         : {avg_cost:.6f}")
    print(f"total_cost($)       : {total_cost:.6f}")

    # 7) 파일 저장(완료 job별 cost 포함)
    keep_cols = [c for c in [
        "job_id", "cluster", "model", "dataset", "world_size",
        "submitted_ts", "started_ts", "end_ts",
        "queued_sec", "jct_sec_calc", "run_hours", "cost_usd", "status"
    ] if c in df_cost.columns]
    df_cost[keep_cols].to_csv(args.out_jobs_csv, index=False)
    print(f"\n[OK] wrote per-job completed costs -> {args.out_jobs_csv}")

    # 클러스터별 비용 요약도 같이
    byc = (
        df_cost.groupby("cluster")
              .agg(
                  jobs=("job_id", "count"),
                  total_cost_usd=("cost_usd", "sum"),
                  avg_cost_usd=("cost_usd", "mean"),
                  total_run_hours=("run_hours", "sum"),
                  total_gpu_hours=("run_hours", lambda s: float((s * df_cost.loc[s.index, "world_size"]).sum())),
              )
              .reset_index()
    )
    print("\n=== By cluster (completed) ===")
    with pd.option_context("display.max_columns", 50, "display.width", 140):
        print(byc.to_string(index=False))


if __name__ == "__main__":
    main()

