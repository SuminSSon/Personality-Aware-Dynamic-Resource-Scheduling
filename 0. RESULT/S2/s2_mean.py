# s2.py
# Usage: python s2.py
# Purpose: Pretty print cluster-wise, model-wise comparison
# Focus: Is Ours beneficial? (acc-safe + training_time reduction)

import os
import glob
import numpy as np
import pandas as pd

FILES_GLOB = "*_job_metrics.csv"
STATUS_FILTER = "FINISHED"
ACC_DELTA = 0.5  # acc-safe threshold (±0.5%)

def infer_scheduler(path: str) -> str:
    b = os.path.basename(path).lower()
    if "pollux" in b: return "Pollux"
    if "sia" in b:    return "Sia"
    if "lucid" in b:  return "Lucid"
    if "ours" in b or "skuld" in b: return "Ours"
    return os.path.splitext(os.path.basename(path))[0]

def to_epoch(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        return num
    dt = pd.to_datetime(s, errors="coerce")
    return dt.astype("int64") / 1e9

def load_all() -> pd.DataFrame:
    frames = []
    for p in glob.glob(FILES_GLOB):
        df = pd.read_csv(p)
        df["scheduler"] = infer_scheduler(p)

        for c in ["queued_sec", "jct_sec", "final_accuracy"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        st = to_epoch(df["started_ts"])
        et = to_epoch(df["end_ts"])
        df["training_time_sec"] = et - st
        df.loc[df["training_time_sec"] < 0, "training_time_sec"] = np.nan

        if STATUS_FILTER:
            df = df[df["status"].astype(str).str.upper() == STATUS_FILTER]

        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def print_cluster_model_table(df: pd.DataFrame, cluster: str):
    sub = df[df["cluster"] == cluster]

    g = sub.groupby(["model", "scheduler"]).agg(
        mean_time=("training_time_sec", "mean"),
        mean_acc=("final_accuracy", "mean")
    ).reset_index()

    print("\n" + "=" * 120)
    print(f"[{cluster}] Model-wise comparison (Ours vs others)")
    print("=" * 120)

    models = sorted(g["model"].unique())

    rows = []
    for m in models:
        gm = g[g["model"] == m].set_index("scheduler")

        if "Ours" not in gm.index:
            continue

        ours_t = gm.loc["Ours", "mean_time"]
        ours_a = gm.loc["Ours", "mean_acc"]

        best_other_t = np.inf
        ref_acc = []

        for s in gm.index:
            if s == "Ours":
                continue
            best_other_t = min(best_other_t, gm.loc[s, "mean_time"])
            if not np.isnan(gm.loc[s, "mean_acc"]):
                ref_acc.append(gm.loc[s, "mean_acc"])

        acc_safe = (
            len(ref_acc) > 0 and
            not np.isnan(ours_a) and
            abs(ours_a - np.mean(ref_acc)) <= ACC_DELTA
        )

        time_gain = ours_t < best_other_t

        if acc_safe and time_gain:
            verdict = "GAIN ▲"
        elif acc_safe and not time_gain:
            verdict = "LOSS ▼ (time)"
        elif not acc_safe and time_gain:
            verdict = "LOSS ▼ (acc)"
        else:
            verdict = "LOSS ▼"

        rows.append([
            m,
            f"{ours_t:,.0f}",
            f"{best_other_t:,.0f}",
            f"{ours_a:.2f}" if not np.isnan(ours_a) else "NaN",
            f"{np.mean(ref_acc):.2f}" if ref_acc else "NaN",
            "YES" if acc_safe else "NO",
            verdict
        ])

    out = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Ours Time(s)",
            "Best Other Time(s)",
            "Ours Acc",
            "Other Avg Acc",
            "Acc-Safe?",
            "Verdict"
        ]
    )

    with pd.option_context(
        "display.width", 200,
        "display.max_columns", None,
        "display.float_format", lambda x: f"{x:.2f}"
    ):
        print(out.to_string(index=False))

def main():
    df = load_all()

    for c in sorted(df["cluster"].unique()):
        print_cluster_model_table(df, c)

if __name__ == "__main__":
    main()