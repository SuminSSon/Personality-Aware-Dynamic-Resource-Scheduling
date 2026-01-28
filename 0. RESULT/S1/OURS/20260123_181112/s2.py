import pandas as pd

CSV_PATH = "job_metrics.csv"   # 같은 폴더에 있다고 했으니 그대로
SCHEDULER_NAME = "Ours"

# =========================
# Helpers
# =========================
def coerce_float(series):
    return pd.to_numeric(series, errors="coerce")

def compute_training_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    training_time_sec = (end_ts - started_ts)
    숫자 epoch ts 버전이므로 그대로 뺍니다.
    """
    df = df.copy()
    df["submitted_ts"] = coerce_float(df["submitted_ts"])
    df["started_ts"]   = coerce_float(df["started_ts"])
    df["end_ts"]       = coerce_float(df["end_ts"])
    df["queued_sec"]   = coerce_float(df.get("queued_sec"))
    df["jct_sec"]      = coerce_float(df.get("jct_sec"))

    # 가장 신뢰할 수 있는 건 end-start
    df["train_time_sec"] = df["end_ts"] - df["started_ts"]

    # 혹시 started/end가 비어있으면 jct-queued로 보정 (옵션)
    mask_bad = df["train_time_sec"].isna()
    if "jct_sec" in df.columns and "queued_sec" in df.columns:
        df.loc[mask_bad, "train_time_sec"] = df.loc[mask_bad, "jct_sec"] - df.loc[mask_bad, "queued_sec"]

    return df

def keep_last_record_per_job(df: pd.DataFrame) -> pd.DataFrame:
    """
    job_id가 여러 번 찍히면(end_ts 최대) 마지막 레코드만 유지.
    """
    df = df.copy()
    df = df.sort_values(["job_id", "end_ts"], ascending=[True, True])
    df = df.drop_duplicates(subset=["job_id"], keep="last")
    return df

def pick_representatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    (model, cluster)별
      - min_training_time: train_time_sec 최소
      - max_accuracy: final_accuracy 최대 (없으면 스킵)
    동점이면 train_time_sec 더 짧은 쪽 우선, 그래도 동점이면 end_ts 더 늦은 쪽.
    """
    rows = []

    # 정렬용 컬럼
    df = df.copy()
    df["final_accuracy"] = coerce_float(df.get("final_accuracy"))

    grp_cols = ["model", "cluster"]
    for (model, cluster), g in df.groupby(grp_cols, dropna=False):
        if g.empty:
            continue

        # min_training_time
        g_min = g.sort_values(
            ["train_time_sec", "end_ts"],
            ascending=[True, False],
            na_position="last"
        ).iloc[0]
        rows.append({
            "model": model,
            "cluster": cluster,
            "criterion": "min_training_time",
            "job_id": g_min["job_id"],
            "train_time_sec": float(g_min["train_time_sec"]),
            "acc": float(g_min["final_accuracy"]) if pd.notna(g_min["final_accuracy"]) else None,
        })

        # max_accuracy (accuracy가 하나도 없으면 스킵)
        g_acc = g[pd.notna(g["final_accuracy"])]
        if not g_acc.empty:
            g_max = g_acc.sort_values(
                ["final_accuracy", "train_time_sec", "end_ts"],
                ascending=[False, True, False]
            ).iloc[0]
            rows.append({
                "model": model,
                "cluster": cluster,
                "criterion": "max_accuracy",
                "job_id": g_max["job_id"],
                "train_time_sec": float(g_max["train_time_sec"]),
                "acc": float(g_max["final_accuracy"]),
            })

    out = pd.DataFrame(rows)
    return out

def print_block(df_rep: pd.DataFrame):
    # 보기 좋은 정렬: model 알파벳, clusterA 먼저, criterion min -> max
    cluster_order = {"clusterA": 0, "clusterB": 1}
    crit_order = {"min_training_time": 0, "max_accuracy": 1}

    df_rep = df_rep.copy()
    df_rep["cluster_rank"] = df_rep["cluster"].map(cluster_order).fillna(99).astype(int)
    df_rep["crit_rank"] = df_rep["criterion"].map(crit_order).fillna(99).astype(int)

    df_rep = df_rep.sort_values(["model", "cluster_rank", "crit_rank", "train_time_sec"])

    print("=== Representative Jobs (Model × Cluster) ===")
    for _, r in df_rep.iterrows():
        acc_str = "" if r["acc"] is None else f"{r['acc']:.2f}".rstrip("0").rstrip(".")
        # model 폭 맞춤(선택)
        model_str = f"{str(r['model']):<14}"
        cluster_str = f"{str(r['cluster']):<8}"
        crit_str = f"{str(r['criterion']):<17}"
        jid = str(r["job_id"])
        tsec = float(r["train_time_sec"])
        print(f"{model_str} | {cluster_str} | {crit_str} | {jid} | train_time = {tsec:8.2f}s | acc = {acc_str}")

# =========================
# Main
# =========================
df = pd.read_csv(CSV_PATH)

# 컬럼명이 뒤집혀 있을 수도 있으니 안전하게 매핑
# (너가 준 포맷은 ... final_accuracy,status 였다가 ... status,final_accuracy 였다가 섞임)
# 여기선 있는 걸 우선 사용.
required_cols = ["job_id", "cluster", "model", "started_ts", "end_ts"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"CSV에 필수 컬럼이 없습니다: {missing}")

df = compute_training_time(df)

# train_time이 음수/0이면 이상치로 제외(원하면 주석 처리)
df = df[pd.notna(df["train_time_sec"]) & (df["train_time_sec"] > 0)].copy()

# job_id 중복 정리 (핵심)
df = keep_last_record_per_job(df)

# 대표 뽑기
rep = pick_representatives(df)

# 출력
print_block(rep)