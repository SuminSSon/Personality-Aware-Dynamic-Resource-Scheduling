import csv

csv_path = "job_metrics.csv"

completed_jobs = []

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] == "completed":
            completed_jobs.append({
                "submitted_ts": float(row["submitted_ts"]),
                "end_ts": float(row["end_ts"]),
                "queued_sec": float(row["queued_sec"]),
                "jct_sec": float(row["jct_sec"]),
            })

if not completed_jobs:
    raise RuntimeError("completed job이 없습니다.")

# makespan: earliest submit ~ latest end
makespan = max(j["end_ts"] for j in completed_jobs) - min(
    j["submitted_ts"] for j in completed_jobs
)

avg_jct = sum(j["jct_sec"] for j in completed_jobs) / len(completed_jobs)
avg_queue = sum(j["queued_sec"] for j in completed_jobs) / len(completed_jobs)

print(f"Completed jobs: {len(completed_jobs)}")
print(f"Makespan (sec): {makespan:.3f}")
print(f"Average JCT (sec): {avg_jct:.3f}")
print(f"Average Queueing Time (sec): {avg_queue:.3f}")

