#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze job_metrics.csv and print a summary in the same style,
but ONLY for jobs with status == FAILED.

Supports timestamps as:
- epoch seconds (float/int)
- or datetime strings (YYYY-MM-DD HH:MM:SS ...)

Usage:
  python analyze_failed.py --csv job_metrics.csv
  cat job_metrics.csv | python analyze_failed.py

Optional cluster GPU capacities:
  python analyze_failed.py --csv job_metrics.csv --gpus clusterA=4 --gpus clusterB=4
"""

import sys
import argparse
from dataclasses import dataclass
from typing import Dict

import pandas as pd


def _to_datetime_series(s: pd.Series) -> pd.Series:
    """
    If values look numeric -> treat as epoch seconds.
    Else -> parse as datetime strings.
    """
    if s is None:
        return pd.Series([pd.NaT] * 0)

    s2 = pd.to_numeric(s, errors="coerce")
    numeric_ratio = s2.notna().mean() if len(s2) else 0.0

    if numeric_ratio >= 0.8:
        # epoch seconds (float)
        return pd.to_datetime(s2, unit="s", errors="coerce")
    else:
        return pd.to_datetime(s, errors="coerce")


def _fmt_seconds(x: float) -> str:
    return f"{x:.2f}"


def _fmt_hours_from_sec(x: float) -> str:
    return f"{x/3600.0:.3f}"


@dataclass
class Summary:
    num_total: int
    makespan_sec: float
    throughput_jobs_per_hr: float
    avg_jct_sec: float
    avg_queue_sec: float
    util_by_cluster: Dict[str, float]
    util_global: float


def parse_cluster_gpus(items) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in items or []:
        s = str(s).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"--gpus expects form cluster=NUM, got: {s}")
        k, v = s.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def compute_summary_failed_only(df: pd.DataFrame, cluster_gpus: Dict[str, int]) -> Summary:
    df = df.copy()

    # Filter FAILED only
    if "status" not in df.columns:
        raise ValueError("CSV must contain 'status' column.")
    df = df[df["status"].astype(str).str.upper().str.strip() == "FAILED"].copy()

    # Parse timestamps (epoch or string)
    df["submitted_ts_dt"] = _to_datetime_series(df.get("submitted_ts"))
    df["started_ts_dt"] = _to_datetime_series(df.get("started_ts"))
    df["end_ts_dt"] = _to_datetime_series(df.get("end_ts"))

    # Coerce numerics
    for c in ["world_size", "queued_sec", "jct_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    num_total = int(len(df))

    # Makespan over FAILED subset: max(end) - min(submitted)
    t0 = df["submitted_ts_dt"].min()
    t1 = df["end_ts_dt"].max()
    if pd.isna(t0) or pd.isna(t1) or num_total == 0:
        makespan_sec = float("nan")
    else:
        makespan_sec = float((t1 - t0).total_seconds())

    # Throughput: "failed jobs/hour" (count of FAILED per makespan hour)
    if makespan_sec and makespan_sec > 0 and num_total > 0:
        throughput = float(num_total) / (makespan_sec / 3600.0)
    else:
        throughput = float("nan")

    avg_jct_sec = float(df["jct_sec"].dropna().mean()) if "jct_sec" in df.columns and num_total > 0 else float("nan")
    avg_queue_sec = float(df["queued_sec"].dropna().mean()) if "queued_sec" in df.columns and num_total > 0 else float("nan")

    # Utilization on FAILED subset
    runtime_sec = (df["end_ts_dt"] - df["started_ts_dt"]).dt.total_seconds()
    runtime_sec = pd.to_numeric(runtime_sec, errors="coerce").where(lambda x: x >= 0)

    df["runtime_sec"] = runtime_sec
    df["gpu_sec"] = df["runtime_sec"] * df["world_size"]

    util_by_cluster: Dict[str, float] = {}
    total_gpu_sec_all = float(df["gpu_sec"].dropna().sum())

    total_gpus_all = 0
    for cid, g in cluster_gpus.items():
        total_gpus_all += int(g)
        mask = df.get("cluster").astype(str) == cid if "cluster" in df.columns else pd.Series([False] * len(df))
        gpu_sec_c = float(df.loc[mask, "gpu_sec"].dropna().sum())
        if makespan_sec and makespan_sec > 0 and g > 0:
            util_by_cluster[cid] = gpu_sec_c / (makespan_sec * float(g))
        else:
            util_by_cluster[cid] = float("nan")

    if makespan_sec and makespan_sec > 0 and total_gpus_all > 0:
        util_global = total_gpu_sec_all / (makespan_sec * float(total_gpus_all))
    else:
        util_global = float("nan")

    return Summary(
        num_total=num_total,
        makespan_sec=makespan_sec,
        throughput_jobs_per_hr=throughput,
        avg_jct_sec=avg_jct_sec,
        avg_queue_sec=avg_queue_sec,
        util_by_cluster=util_by_cluster,
        util_global=util_global,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="Path to job_metrics.csv. If omitted, read from stdin.")
    ap.add_argument("--gpus", action="append", default=[], help="Cluster GPU capacity, e.g., --gpus clusterA=4")
    args = ap.parse_args()

    cluster_gpus = parse_cluster_gpus(args.gpus)
    if not cluster_gpus:
        # your typical setup default
        cluster_gpus = {"clusterA": 4, "clusterB": 4}

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        data = sys.stdin.read()
        if not data.strip():
            print("ERROR: No input. Provide --csv or pipe CSV into stdin.", file=sys.stderr)
            sys.exit(2)
        from io import StringIO
        df = pd.read_csv(StringIO(data))

    summ = compute_summary_failed_only(df, cluster_gpus)

    print("\n=== Experiment Summary (FAILED ONLY) ===")
    print(f"num_jobs_failed: {summ.num_total}")
    print(f"Makespan (sec): {_fmt_seconds(summ.makespan_sec)}")
    print(f"Makespan (hr):  {_fmt_hours_from_sec(summ.makespan_sec)} hr")
    print(f"Throughput:      {summ.throughput_jobs_per_hr:.3f} failed_jobs/hour")

    print("\n=== JCT (FAILED only) ===")
    print(f"avg_jct_sec: {summ.avg_jct_sec:.2f}")
    print(f"avg_jct_min: {summ.avg_jct_sec/60.0:.2f}")

    print("\n=== Queueing Time (FAILED only) ===")
    print(f"avg_queue_sec: {summ.avg_queue_sec:.2f}")
    print(f"avg_queue_min: {summ.avg_queue_sec/60.0:.2f}")

    print("\n=== Utilization (FAILED only) ===")
    ordered = []
    for k in ["clusterA", "clusterB"]:
        if k in summ.util_by_cluster:
            ordered.append(k)
    for k in sorted(summ.util_by_cluster.keys()):
        if k not in ordered:
            ordered.append(k)

    for cid in ordered:
        print(f"{cid}: {summ.util_by_cluster[cid]:.4f}")
    print(f"global_avg: {summ.util_global:.4f}")


if __name__ == "__main__":
    main()


