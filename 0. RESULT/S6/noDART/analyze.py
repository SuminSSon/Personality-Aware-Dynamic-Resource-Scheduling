import csv

csv_path = "job_metrics.csv"

completed_jobs = []

with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["status"] == "FAILED":
            submitted_ts = float(row["submitted_ts"])
            end_ts = float(row["end_ts"])

            completed_jobs.append({
                "submitted_ts": submitted_ts,
                "end_ts": end_ts,
                "queued_sec": float(row["queued_sec"]),
                "jct_sec": float(row["jct_sec"]),
                # ✅ 진짜 E2E 완료시간
                "e2e_sec": end_ts - submitted_ts,
            })

if not completed_jobs:
    raise RuntimeError("completed job이 없습니다.")

makespan = max(j["end_ts"] for j in completed_jobs) - min(
    j["submitted_ts"] for j in completed_jobs
)

avg_jct = sum(j["jct_sec"] for j in completed_jobs) / len(completed_jobs)
avg_queue = sum(j["queued_sec"] for j in completed_jobs) / len(completed_jobs)
avg_e2e = sum(j["e2e_sec"] for j in completed_jobs) / len(completed_jobs)

print(f"Completed jobs: {len(completed_jobs)}")
print(f"Makespan (sec): {makespan:.3f}")
print(f"Average JCT (sec): {avg_e2e:.3f}")
print(f"Average Queueing Time (sec): {avg_queue:.3f}")

