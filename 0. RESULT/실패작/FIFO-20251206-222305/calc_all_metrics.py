#!/usr/bin/env python3
import csv
import json
from datetime import datetime
from statistics import mean
from collections import defaultdict

JOB_EVENTS = "job_events.csv"
CSP_METRICS = "csp_metrics.csv"

# ------------------------
# timestamp parser
# ------------------------
def parse_ts(s: str):
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {s}")

# ------------------------
# 1) job_events.csv → submit / start / finish / JCT
# ------------------------
submit_ts_map = {}   # job_id → submit_ts
start_ts_map  = {}   # job_id → start_ts
finish_ts_map = {}   # job_id → finish_ts
jct_map       = {}   # job_id → jct_sec

with open(JOB_EVENTS, newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ev  = row["event"].strip()
        ts  = parse_ts(row["ts"])
        job = row["job_id"].strip()

        if ev == "submitted":
            submit_ts_map[job] = ts

        elif ev == "started":
            start_ts_map[job] = ts

        elif ev == "finished":
            finish_ts_map[job] = ts
            # metadata_json 안에서 jct_sec 파싱
            meta_raw = (row.get("metadata_json") or "").strip()
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                    if "jct_sec" in meta:
                        jct_map[job] = float(meta["jct_sec"])
                except json.JSONDecodeError:
                    pass

if not submit_ts_map or not finish_ts_map:
    raise RuntimeError("submit_ts 또는 finish_ts를 충분히 찾지 못했습니다.")

# 실험 기간
first_submit = min(submit_ts_map.values())
last_finish  = max(finish_ts_map.values())

makespan_sec = (last_finish - first_submit).total_seconds()
makespan_hr  = makespan_sec / 3600.0

# ------------------------
# 2) 평균 JCT
# ------------------------
if not jct_map:
    raise RuntimeError("metadata_json에서 jct_sec를 찾지 못했습니다.")

avg_jct_sec = mean(jct_map.values())
avg_jct_min = avg_jct_sec / 60.0
num_jobs    = len(jct_map)  # finished 된 job 수

# ------------------------
# 3) 평균 Queueing Time (start_ts - submit_ts)
# ------------------------
queue_times_sec = []
for job, sub_ts in submit_ts_map.items():
    if job in start_ts_map:
        q = (start_ts_map[job] - sub_ts).total_seconds()
        queue_times_sec.append(q)

avg_queue_sec = mean(queue_times_sec)
avg_queue_min = avg_queue_sec / 60.0

# ------------------------
# 4) Utilization (clusterA, clusterB, global avg)
# ------------------------
cluster_utils = defaultdict(list)

with open(CSP_METRICS, newline="", encoding="utf-8", errors="replace") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ts = parse_ts(row["ts"])
        # 실험 기간 안에 있는 샘플만
        if not (first_submit <= ts <= last_finish):
            continue

        clu = row["cluster"]
        u   = float(row["U_t"])
        cluster_utils[clu].append(u)

cluster_avg = {}
for clu, vals in cluster_utils.items():
    if vals:
        cluster_avg[clu] = mean(vals)
    else:
        cluster_avg[clu] = 0.0

if cluster_avg:
    global_avg_util = mean(cluster_avg.values())  # 클러스터당 GPU 수가 같다는 가정
else:
    global_avg_util = 0.0

# ------------------------
# 5) Throughput (jobs/hour)
# ------------------------
throughput_jobs_per_hour = num_jobs / makespan_hr if makespan_hr > 0 else 0.0

# ------------------------
# 출력
# ------------------------
print("=== Experiment Summary ===")
print(f"num_jobs_finished: {num_jobs}")
print(f"Makespan (sec): {makespan_sec:.2f}")
print(f"Makespan (hr):  {makespan_hr:.3f} hr")
print(f"Throughput:      {throughput_jobs_per_hour:.3f} jobs/hour")

print("\n=== JCT ===")
print(f"avg_jct_sec: {avg_jct_sec:.2f}")
print(f"avg_jct_min: {avg_jct_min:.2f}")

print("\n=== Queueing Time ===")
print(f"avg_queue_sec: {avg_queue_sec:.2f}")
print(f"avg_queue_min: {avg_queue_min:.2f}")

print("\n=== Utilization ===")
for clu, val in cluster_avg.items():
    print(f"{clu}: {val:.4f}")
print(f"global_avg: {global_avg_util:.4f}")

