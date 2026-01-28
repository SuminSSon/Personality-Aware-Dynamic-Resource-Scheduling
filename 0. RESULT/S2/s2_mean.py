# s2_viz.py
# Usage: python s2_viz.py
# Inputs: *_job_metrics.csv in current directory
# Outputs:
#  - prints cluster-wise pivot tables (time/acc/n)
#  - writes:
#     * acc_safe_delta_time_by_cluster.csv
#     * pareto_points_by_cluster.csv
#     * pareto_<cluster>.png (matplotlib default colors)

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILES_GLOB = "*_job_metrics.csv"
STATUS_FILTER = "FINISHED"   # set "" to disable
ACC_DELTA = 0.5              # accuracy-safe threshold (percentage points)
SCHED_ORDER = ["Pollux", "Sia", "Lucid", "Ours"]

# ----------------------------
# Helpers
# ----------------------------
def infer_scheduler(path: str) -> str:
    b = os.path.basename(path).lower()
    if "pollux" in b: return "Pollux"
    if "sia" in b:    return "Sia"
    if "lucid" in b:  return "Lucid"
    if "ours" in b or "skuld" in b: return "Ours"
    return os.path.splitext(os.path.basename(path))[0]

def to_epoch(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().any():
        return num
    dt = pd.to_datetime(series, errors="coerce")
    return dt.astype("int64") / 1e9

def ensure_order(cols):
    present = [c for c in SCHED_ORDER if c in cols]
    rest = [c for c in cols if c not in present]
    return present + rest

def fmt_time(x):
    return "NaN" if pd.isna(x) else f"{x:,.0f}"

def fmt_acc(x):
    return "NaN" if pd.isna(x) else f"{x:.2f}"

def pct_reduction(new, base):
    # positive => reduced time
    if pd.isna(new) or pd.isna(base) or base == 0:
        return np.nan
    return (base - new) / base * 100.0

# ----------------------------
# Load
# ----------------------------
def load_all() -> pd.DataFrame:
    paths = sorted(glob.glob(FILES_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files match glob: {FILES_GLOB}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["scheduler"] = infer_scheduler(p)

        for c in ["final_accuracy", "world_size", "queued_sec", "jct_sec"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        st = to_epoch(df["started_ts"])
        et = to_epoch(df["end_ts"])
        df["training_time_sec"] = et - st
        df.loc[df["training_time_sec"] < 0, "training_time_sec"] = np.nan

        if STATUS_FILTER:
            df = df[df["status"].astype(str).str.upper() == STATUS_FILTER]

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    need = ["cluster", "model", "scheduler", "training_time_sec", "final_accuracy"]
    for k in need:
        if k not in out.columns:
            raise ValueError(f"Missing required column: {k}")
    return out

# ----------------------------
# Cluster tables + acc-safe ΔTime%
# ----------------------------
def cluster_pivots(df: pd.DataFrame, cluster: str):
    sub = df[df["cluster"] == cluster].copy()
    if sub.empty:
        return None

    agg = (sub.groupby(["model", "scheduler"])
              .agg(mean_time=("training_time_sec", "mean"),
                   mean_acc=("final_accuracy", "mean"),
                   n=("training_time_sec", "count"))
              .reset_index())

    t_time = agg.pivot(index="model", columns="scheduler", values="mean_time")
    t_acc  = agg.pivot(index="model", columns="scheduler", values="mean_acc")
    t_n    = agg.pivot(index="model", columns="scheduler", values="n")

    t_time = t_time.reindex(columns=ensure_order(list(t_time.columns)))
    t_acc  = t_acc.reindex(columns=ensure_order(list(t_acc.columns)))
    t_n    = t_n.reindex(columns=ensure_order(list(t_n.columns)))

    return agg, t_time, t_acc, t_n

def print_cluster_tables(cluster, t_time, t_acc, t_n):
    t_time_fmt = t_time.apply(lambda col: col.map(fmt_time))
    t_acc_fmt  = t_acc.apply(lambda col: col.map(fmt_acc))
    t_n_fmt    = t_n.fillna(0).astype(int)

    print("\n" + "=" * 140)
    print(f"[{cluster}] Job(model)-wise mean tables by scheduler  (world_size/dataset ignored)")
    print("=" * 140)

    print("\n--- Mean Training Time (sec) ---")
    with pd.option_context("display.width", 260, "display.max_columns", None):
        print(t_time_fmt.to_string())

    print("\n--- Mean Accuracy ---")
    with pd.option_context("display.width", 260, "display.max_columns", None):
        print(t_acc_fmt.to_string())

    print("\n--- Count (n jobs) ---")
    with pd.option_context("display.width", 260, "display.max_columns", None):
        print(t_n_fmt.to_string())

def acc_safe_delta_time(df: pd.DataFrame, cluster: str) -> pd.DataFrame:
    piv = cluster_pivots(df, cluster)
    if piv is None:
        return pd.DataFrame()
    _, t_time, t_acc, t_n = piv

    # Baselines = all schedulers except Ours
    baseline_cols = [c for c in t_time.columns if c != "Ours"]

    rows = []
    for model in t_time.index:
        # reference acc: best accuracy among all schedulers for this model (avoids "moving target")
        acc_row = t_acc.loc[model]
        ref_acc = acc_row.dropna().max() if acc_row.notna().any() else np.nan

        ours_time = t_time.loc[model].get("Ours", np.nan)
        ours_acc  = t_acc.loc[model].get("Ours", np.nan)
        ours_n    = int(t_n.loc[model].get("Ours", 0) if not pd.isna(t_n.loc[model].get("Ours", np.nan)) else 0)

        # find best baseline time among baselines that are acc-safe too
        best_base_time = np.nan
        best_base_name = "N/A"

        for s in baseline_cols:
            s_time = t_time.loc[model].get(s, np.nan)
            s_acc  = t_acc.loc[model].get(s, np.nan)
            if pd.isna(s_time) or pd.isna(s_acc) or pd.isna(ref_acc):
                continue
            if abs(s_acc - ref_acc) <= ACC_DELTA:
                if pd.isna(best_base_time) or s_time < best_base_time:
                    best_base_time = float(s_time)
                    best_base_name = s

        # acc-safe for Ours?
        ours_acc_safe = (not pd.isna(ours_acc)) and (not pd.isna(ref_acc)) and (abs(ours_acc - ref_acc) <= ACC_DELTA)

        # time reduction vs best acc-safe baseline
        delta = pct_reduction(ours_time, best_base_time) if ours_acc_safe else np.nan

        rows.append({
            "cluster": cluster,
            "model": model,
            "ref_acc(best)": ref_acc,
            "Ours_n": ours_n,
            "Ours_mean_time": ours_time,
            "Ours_mean_acc": ours_acc,
            "Ours_acc_safe": bool(ours_acc_safe),
            "best_acc_safe_baseline": best_base_name,
            "best_acc_safe_baseline_time": best_base_time,
            "ΔTime%_Ours_vs_best_acc_safe_baseline": delta,
        })

    out = pd.DataFrame(rows)
    # 보기 좋게 정렬: acc-safe True 먼저, ΔTime% 큰 순
    out = out.sort_values(
        by=["Ours_acc_safe", "ΔTime%_Ours_vs_best_acc_safe_baseline"],
        ascending=[False, False],
        na_position="last"
    )
    return out

# ----------------------------
# Main
# ----------------------------
def main():
    df = load_all()
    clusters = sorted(df["cluster"].dropna().unique())
    if not clusters:
        print("No clusters found.")
        return

    all_delta = []
    all_points = []

    for c in clusters:
        piv = cluster_pivots(df, c)
        if piv is None:
            continue
        agg, t_time, t_acc, t_n = piv

        # 1) print the full tables (no cherry-picking)
        print_cluster_tables(c, t_time, t_acc, t_n)

        # 2) compute acc-safe delta time summary (still shows all jobs)
        delta = acc_safe_delta_time(df, c)
        print("\n--- Acc-safe (±{:.2f}) Ours time reduction vs best acc-safe baseline (per job) ---".format(ACC_DELTA))
        show = delta.copy()
        show["Ours_mean_time"] = show["Ours_mean_time"].map(lambda x: fmt_time(x))
        show["best_acc_safe_baseline_time"] = show["best_acc_safe_baseline_time"].map(lambda x: fmt_time(x))
        show["Ours_mean_acc"] = show["Ours_mean_acc"].map(lambda x: fmt_acc(x))
        show["ref_acc(best)"] = show["ref_acc(best)"].map(lambda x: fmt_acc(x))
        show["ΔTime%_Ours_vs_best_acc_safe_baseline"] = show["ΔTime%_Ours_vs_best_acc_safe_baseline"].map(
            lambda x: "NaN" if pd.isna(x) else f"{x:+.1f}%"
        )

        with pd.option_context("display.width", 260, "display.max_columns", None):
            print(show.to_string(index=False))

        all_delta.append(delta)

if __name__ == "__main__":
    main()
