#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze job_metrics.csv and print an experiment summary similar to your sample output.

- Averages (JCT / Queueing Time) are computed over **ALL jobs** (not only FINISHED).
- Also prints FINISHED/FAILED counts.
- Utilization is computed from (end_ts - started_ts) * world_size, normalized by makespan * cluster_total_gpus.

Usage examples:
  python analyze.py --csv job_metrics.csv
  cat job_metrics.csv | python analyze.py
  python analyze.py --csv job_metrics.csv --gpus clusterA=4 --gpus clusterB=4
"""

import sys
import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


def _parse_ts(s: pd.Series) -> pd.Series:
    # Handles "YYYY-MM-DD HH:MM:SS" and blanks
    return pd.to_datetime(s, errors="coerce")


def _fmt_seconds(x: float) -> str:
    return f"{x:.2f}"


def _fmt_hours(x: float) -> str:
    return f"{x/3600.0:.3f}"


@dataclass
class Summary:
    num_total: int
    num_finished: int
    num_failed: int
    makespan_sec: float
    throughput_jobs_per_hr: float
    avg_jct_sec_all: float
    avg_queue_sec_all: float
    util_by_cluster: Dict[str, float]
    util_global: float


def compute_summary(df: pd.DataFrame, cluster_gpus: Dict[str, int]) -> Summary:
    # Normalize columns
    df = df.copy()

    # Parse timestamps
    df["submitted_ts_dt"] = _parse_ts(df.get("submitted_ts"))
    df["started_ts_dt"] = _parse_ts(df.get("started_ts"))
    df["end_ts_dt"] = _parse_ts(df.get("end_ts"))

    # Coerce numeric
    for c in ["world_size", "queued_sec", "jct_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Basic counts
    num_total = int(len(df))
    num_finished = int((df.get("status") == "FINISHED").sum()) if "status" in df.columns else 0
    num_failed = int((df.get("status") == "FAILED").sum()) if "status" in df.columns else 0

    # Makespan: max(end_ts) - min(submitted_ts) across ALL rows with valid timestamps
    t0 = df["submitted_ts_dt"].min()
    t1 = df["end_ts_dt"].max()
    if pd.isna(t0) or pd.isna(t1):
        makespan_sec = float("nan")
    else:
        makespan_sec = float((t1 - t0).total_seconds())

    # Throughput: use FINISHED count / makespan hours (so it matches your example behavior)
    if makespan_sec and makespan_sec > 0:
        throughput = float(num_finished) / (makespan_sec / 3600.0)
    else:
        throughput = float("nan")

    # Averages over ALL jobs (your request)
    avg_jct_sec_all = float(df["jct_sec"].dropna().mean()) if "jct_sec" in df.columns else float("nan")
    avg_queue_sec_all = float(df["queued_sec"].dropna().mean()) if "queued_sec" in df.columns else float("nan")

    # Utilization
    # runtime_sec = end - start (ignore rows where either is missing or end<start)
    runtime_sec = (df["end_ts_dt"] - df["started_ts_dt"]).dt.total_seconds()
    runtime_sec = pd.to_numeric(runtime_sec, errors="coerce")

    # Negative runtimes -> NaN
    runtime_sec = runtime_sec.where(runtime_sec >= 0)

    df["runtime_sec"] = runtime_sec
    df["gpu_sec"] = df["runtime_sec"] * df["world_size"]

    util_by_cluster: Dict[str, float] = {}
    total_gpu_sec_all = float(df["gpu_sec"].dropna().sum())

    total_gpus_all = 0
    for cid, g in cluster_gpus.items():
        total_gpus_all += int(g)

        mask = (df.get("cluster") == cid) if "cluster" in df.columns else pd.Series([False] * len(df))
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
        num_finished=num_finished,
        num_failed=num_failed,
        makespan_sec=makespan_sec,
        throughput_jobs_per_hr=throughput,
        avg_jct_sec_all=avg_jct_sec_all,
        avg_queue_sec_all=avg_queue_sec_all,
        util_by_cluster=util_by_cluster,
        util_global=util_global,
    )


def parse_cluster_gpus(items) -> Dict[str, int]:
    # Accept repeated --gpus clusterA=4
    out: Dict[str, int] = {}
    for s in items or []:
        s = str(s).strip()
        if not s:
            continue
        if "=" not in s:
            raise ValueError(f"--gpus expects form cluster=NUM, got: {s}")
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        out[k] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="", help="Path to job_metrics.csv. If omitted, read from stdin.")
    ap.add_argument(
        "--gpus",
        action="append",
        default=[],
        help="Cluster GPU capacity, e.g., --gpus clusterA=4 --gpus clusterB=4",
    )
    args = ap.parse_args()

    cluster_gpus = parse_cluster_gpus(args.gpus)
    # sensible defaults for your setup
    if not cluster_gpus:
        cluster_gpus = {"clusterA": 4, "clusterB": 4}

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        # read from stdin
        data = sys.stdin.read()
        if not data.strip():
            print("ERROR: No input. Provide --csv or pipe CSV into stdin.", file=sys.stderr)
            sys.exit(2)
        from io import StringIO
        df = pd.read_csv(StringIO(data))

    summ = compute_summary(df, cluster_gpus)

    # Print in your style
    print("\n=== Experiment Summary ===")
    print(f"num_jobs_total:    {summ.num_total}")
    print(f"num_jobs_finished: {summ.num_finished}")
    print(f"num_jobs_failed:   {summ.num_failed}")
    print(f"Makespan (sec): {_fmt_seconds(summ.makespan_sec)}")
    print(f"Makespan (hr):  {_fmt_hours(summ.makespan_sec)} hr")
    print(f"Throughput:      {summ.throughput_jobs_per_hr:.3f} jobs/hour")

    print("\n=== JCT (ALL jobs) ===")
    print(f"avg_jct_sec: {summ.avg_jct_sec_all:.2f}")
    print(f"avg_jct_min: {summ.avg_jct_sec_all/60.0:.2f}")

    print("\n=== Queueing Time (ALL jobs) ===")
    print(f"avg_queue_sec: {summ.avg_queue_sec_all:.2f}")
    print(f"avg_queue_min: {summ.avg_queue_sec_all/60.0:.2f}")

    print("\n=== Utilization ===")
    # Keep stable ordering: clusterA, clusterB, then others
    ordered_keys = []
    for k in ["clusterA", "clusterB"]:
        if k in summ.util_by_cluster:
            ordered_keys.append(k)
    for k in sorted(summ.util_by_cluster.keys()):
        if k not in ordered_keys:
            ordered_keys.append(k)

    for cid in ordered_keys:
        print(f"{cid}: {summ.util_by_cluster[cid]:.4f}")
    print(f"global_avg: {summ.util_global:.4f}")


if __name__ == "__main__":
    main()

