import pandas as pd
import matplotlib
# GUI 창이 뜨지 않도록 백엔드 설정
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 파일 설정
files = {
    "Pollux": "pollux_job_metrics.csv",
    "Sia": "sia_job_metrics.csv",
    "Lucid": "Lucid_job_metrics.csv",
    "Skuld (Ours)": "ours_job_metrics.csv"
}

# 2. 데이터 로드 및 전처리 함수
def load_and_process_data(files):
    data_makespan = {}  # Makespan용 (모든 작업 포함)
    data_stats = {}     # JCT/Queueing용 (성공한 작업만 포함)

    for name, filepath in files.items():
        try:
            df = pd.read_csv(filepath)

            # 전처리
            if name == "Skuld (Ours)":
                # Timestamp 변환 (seconds -> datetime)
                df['submit_time'] = pd.to_datetime(df['submitted_ts'], unit='s')
                df['end_time'] = pd.to_datetime(df['end_ts'], unit='s')

                # 중복 job_id 제거 (가장 마지막 기록 사용)
                df = df.sort_values('end_time').drop_duplicates('job_id', keep='last')

                # 성공 기준 (Accuracy > 0 and jct_sec > 0)
                is_success = (df['final_accuracy'] > 0) & (df['jct_sec'] > 0)

            else:  # Baselines
                # Timestamp 변환
                df['submit_time'] = pd.to_datetime(df['submitted_ts'])
                df['end_time'] = pd.to_datetime(df['end_ts'])

                # 중복 job_id 제거 (가장 마지막 기록 사용)
                df = df.sort_values('end_time').drop_duplicates('job_id', keep='last')

                # 성공 기준 (Status == FINISHED)
                is_success = df['status'] == 'FINISHED'

            # ✅ 메트릭 계산: seconds -> hours
            df['jct'] = df['jct_sec'] / 3600.0
            df['queueing'] = df['queued_sec'] / 3600.0

            # Makespan용 상대 시간 (실험 시작 시각 기준, hours)
            t0 = df['submit_time'].min()
            df['completion_time_relative_hours'] = (df['end_time'] - t0).dt.total_seconds() / 3600.0

            # 1. Makespan용 데이터셋: 실패 포함 모든 작업 저장
            data_makespan[name] = df.copy()

            # 2. 통계용 데이터셋: 성공한 작업만 저장
            data_stats[name] = df[is_success].copy()

            print(f"[{name}] Total (Makespan): {len(df)}, Success (Stats): {len(df[is_success])}")

        except Exception as e:
            print(f"Error loading {name}: {e}")

    return data_makespan, data_stats

# 데이터 로드
data_makespan, data_stats = load_and_process_data(files)

# 3. 그래프 스타일 설정
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

# ---------------------------------
# 공통 스타일: 색/마커/선굵기/점선/마커간격
# ---------------------------------
color_map = {
    "Pollux": "tab:blue",
    "Sia": "tab:orange",
    "Lucid": "tab:green",
    "Skuld (Ours)": "tab:red",
}
marker_map = {
    "Pollux": "o",        # 동그라미
    "Sia": "s",           # 네모
    "Lucid": "^",         # 세모
    "Skuld (Ours)": "*",  # 별
}

LS_ALL = "--"          # ✅ 전부 점선 유지
LW_BASE = 1.3          # ✅ 얇은 선
LW_OURS = 1.8          # ✅ ours도 얇게 (baseline보다 약간 강조만)
MARK_EVERY = (4, 5)    # ✅ 5개마다 마커: 5,10,15,...번째(0-based start=4)
MEW = 1.3              # marker edge width


# ---------------------------------------------------------
# Figure 1: CDF of JCT (성공한 작업만 사용) - hours
#  - 점선 + 빈 마커 + 5개마다 마커 + 얇은 선
#  - 범례에도 마커 포함
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
for name, df in data_stats.items():
    if len(df) == 0:
        continue

    x = np.asarray(df['jct'].values)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        continue

    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    lw = LW_OURS if "Skuld" in name else LW_BASE
    ms = 9 if "Skuld" in name else 7
    c = color_map.get(name, "k")
    m = marker_map.get(name, "o")

    ax.step(
        x_sorted, y, where='post',
        label=name,
        color=c,
        linestyle=LS_ALL,
        linewidth=lw,
        marker=m,
        markevery=MARK_EVERY,
        markerfacecolor='none',
        markeredgecolor=c,
        markeredgewidth=MEW,
        markersize=ms
    )

ax.set_xlabel("(a) Job Completion Time (h)")
ax.set_ylabel("CDF")
ax.set_title("CDF of JCT")
ax.legend()
fig.tight_layout()
fig.savefig("s1_jct.png", dpi=300)
plt.close(fig)


# ---------------------------------------------------------
# Figure 2: CDF of Queueing Time (성공한 작업만 사용) - hours
#  - 점선 + 빈 마커 + 5개마다 마커 + 얇은 선
#  - 범례에도 마커 포함
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
for name, df in data_stats.items():
    if len(df) == 0:
        continue

    x = np.asarray(df['queueing'].values)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        continue

    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    lw = LW_OURS if "Skuld" in name else LW_BASE
    ms = 9 if "Skuld" in name else 7
    c = color_map.get(name, "k")
    m = marker_map.get(name, "o")

    ax.step(
        x_sorted, y, where='post',
        label=name,
        color=c,
        linestyle=LS_ALL,
        linewidth=lw,
        marker=m,
        markevery=MARK_EVERY,
        markerfacecolor='none',
        markeredgecolor=c,
        markeredgewidth=MEW,
        markersize=ms
    )

ax.set_xlabel("(b) Queueing Time (h)")
ax.set_ylabel("CDF")
ax.set_title("CDF of Queueing Time")
ax.legend()
fig.tight_layout()
fig.savefig("s1_queueing.png", dpi=300)
plt.close(fig)


# ---------------------------------------------------------
# Figure 3: Cumulative Completed Jobs (모든 작업 사용)
#  - 점선(step) + 빈 마커 + 5 job마다 마커 + 얇은 선
#  - 범례에도 마커 포함
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for name, df in data_makespan.items():
    if len(df) == 0:
        continue

    sorted_times = np.sort(df['completion_time_relative_hours'].values)
    job_counts = np.arange(1, len(sorted_times) + 1)

    lw = LW_OURS if "Skuld" in name else LW_BASE
    ms = 9 if "Skuld" in name else 7
    c = color_map.get(name, "k")
    m = marker_map.get(name, "o")

    ax.step(
        sorted_times, job_counts, where='post',
        label=name,
        color=c,
        linestyle=LS_ALL,
        linewidth=lw,
        marker=m,
        markevery=MARK_EVERY,  # ✅ 5,10,15,... job 위치
        markerfacecolor='none',
        markeredgecolor=c,
        markeredgewidth=MEW,
        markersize=ms
    )

ax.set_xlabel("Wall-clock time (h)")
ax.set_ylabel("Completed jobs")
ax.set_title("Cumulative completed jobs over time")
ax.legend(loc='lower right')
ax.grid(True, which='both', linestyle='--', alpha=0.7)
fig.tight_layout()
fig.savefig("s1_makespan.png", dpi=300)
plt.close(fig)

print("All plots generated and saved successfully!")
