#!/usr/bin/env python3
import csv
from datetime import datetime
from statistics import mean
from collections import defaultdict

JOB_EVENTS = "job_events.csv"
CSP_METRICS = "csp_metrics.csv"

# ------------------------
# ts parser
# ------------------------
def parse_ts(s: str):
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp: {s}")

# ------------------------
# 1) job_events.csv → 실험 기간: 첫 submit ~ 마지막 finish
# ------------------------
first_submit = None
last_finish = None

with open(JOB_EVENTS, newline="", encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f)
    for row in r:
        ev = row.get("event", "").strip()
        ts = parse_ts(row["ts"])

        if ev == "submitted":
            if first_submit is None or ts < first_submit:
                first_submit = ts

        elif ev == "finished":
            if last_finish is None or ts > last_finish:
                last_finish = ts

print("[INFO] first_submit =", first_submit)
print("[INFO] last_finish  =", last_finish)

if first_submit is None or last_finish is None:
    raise RuntimeError("submit / finished 기간을 job_events.csv에서 찾을 수 없습니다.")

# ------------------------
# 2) csp_metrics.csv → 실험 기간 내 U_t 추출
# ------------------------
cluster_utils = defaultdict(list)

with open(CSP_METRICS, newline="", encoding="utf-8", errors="replace") as f:
    r = csv.DictReader(f)
    for row in r:
        ts = parse_ts(row["ts"])
        if not (first_submit <= ts <= last_finish):
            continue

        cluster = row["cluster"]
        u = float(row["U_t"])
        cluster_utils[cluster].append(u)

# ------------------------
# 3) 클러스터별 평균 + 전체 평균 계산
# ------------------------
print("\n=== Average Utilization per Cluster (experiment interval) ===")

cluster_avg = {}  # cluster → avg U

for clu, vals in cluster_utils.items():
    if vals:
        avg_u = mean(vals)
        cluster_avg[clu] = avg_u
        print(f"{clu}: avg_U_t = {avg_u:.4f}")
    else:
        print(f"{clu}: no samples in interval")

# 전체 평균 (클러스터 GPU 수 동일 가정)
if cluster_avg:
    global_avg = mean(cluster_avg.values())
    print(f"\nGlobal avg utilization = {global_avg:.4f}")
else:
    print("\nNo cluster utilization samples found.")

