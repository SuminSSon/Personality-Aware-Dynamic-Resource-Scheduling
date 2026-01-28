import pandas as pd
import numpy as np

CSV_PATH = "job_metrics.csv"
OUT_CSV  = "failed_jobs_top50.csv"

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(CSV_PATH)

num_cols = [
    "submitted_ts", "started_ts", "end_ts",
    "queued_sec", "jct_sec", "final_accuracy"
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

failed = df[df["status"].astype(str).str.upper() == "FAILED"].copy()

# ----------------------------
# Recompute times
# ----------------------------
failed["train_time_sec"] = failed["end_ts"] - failed["started_ts"]
failed["queued_calc"]    = failed["started_ts"] - failed["submitted_ts"]
failed["tct_calc"]       = failed["end_ts"] - failed["submitted_ts"]

failed["queued_diff"] = (failed["queued_calc"] - failed["queued_sec"]).abs()
failed["jct_diff"]    = (failed["tct_calc"] - failed["jct_sec"]).abs()

# ----------------------------
# Anomaly flags
# ----------------------------
failed["f_missing_ts"] = failed[["submitted_ts","started_ts","end_ts"]].isna().any(axis=1)
failed["f_time_order"] = (
    (failed["started_ts"] < failed["submitted_ts"]) |
    (failed["end_ts"] < failed["started_ts"])
)
failed["f_train_bad"] = (
    (failed["train_time_sec"] <= 0) |
    (failed["train_time_sec"] < 30) |
    (failed["train_time_sec"] > 6 * 3600)
)
failed["f_queue_mis"] = failed["queued_diff"] > 2.0
failed["f_jct_mis"]   = failed["jct_diff"] > 2.0
failed["f_acc_zero"]  = failed["final_accuracy"].isna() | (failed["final_accuracy"] <= 0)

# ----------------------------
# Suspicion score (높을수록 문제)
# ----------------------------
failed["suspicion_score"] = (
    failed["f_missing_ts"] * 5 +
    failed["f_time_order"] * 5 +
    failed["f_train_bad"]  * 3 +
    failed["f_jct_mis"]    * 3 +
    failed["f_queue_mis"]  * 2 +
    failed["f_acc_zero"]   * 1
)

# ----------------------------
# Top-50
# ----------------------------
top50 = (
    failed
    .sort_values(
        by=["suspicion_score", "jct_diff", "queued_diff"],
        ascending=False
    )
    .head(50)
)

cols = [
    "job_id","cluster","model","dataset","world_size",
    "train_time_sec","queued_sec","queued_calc",
    "jct_sec","tct_calc",
    "queued_diff","jct_diff",
    "final_accuracy",
    "suspicion_score",
    "f_missing_ts","f_time_order","f_train_bad",
    "f_queue_mis","f_jct_mis","f_acc_zero"
]

top50 = top50[cols]
top50.to_csv(OUT_CSV, index=False)

print(f"Saved: {OUT_CSV}")
print(top50.to_string(index=False))

