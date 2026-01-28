import pandas as pd

df = pd.read_csv("job_metrics.csv")

# FAILED 제거
df = df[df["status"] == "FINISHED"].copy()

# training time 계산
df["training_time"] = df["jct_sec"] - df["queued_sec"]

models = sorted(df["model"].unique())
clusters = sorted(df["cluster"].unique())

results = []

for model in models:
    for cluster in clusters:
        sub = df[(df["model"] == model) & (df["cluster"] == cluster)]
        if sub.empty:
            continue

        # training time 최소
        min_time_row = sub.loc[sub["training_time"].idxmin()]
        results.append({
            "model": model,
            "cluster": cluster,
            "criterion": "min_training_time",
            "job_id": min_time_row["job_id"],
            "training_time": min_time_row["training_time"],
            "accuracy": min_time_row["final_accuracy"],
        })

        # accuracy 최대
        max_acc_row = sub.loc[sub["final_accuracy"].idxmax()]
        results.append({
            "model": model,
            "cluster": cluster,
            "criterion": "max_accuracy",
            "job_id": max_acc_row["job_id"],
            "training_time": max_acc_row["training_time"],
            "accuracy": max_acc_row["final_accuracy"],
        })

# -----------------------
# 출력
# -----------------------
print("=== Representative Jobs (Model × Cluster) ===")
for r in results:
    print(
        f"{r['model']:16s} | {r['cluster']:8s} | {r['criterion']:18s} | "
        f"{r['job_id']:12s} | "
        f"train_time = {r['training_time']:8.2f}s | acc = {r['accuracy']:.2f}"
    )

print(f"\nTotal representatives: {len(results)}")