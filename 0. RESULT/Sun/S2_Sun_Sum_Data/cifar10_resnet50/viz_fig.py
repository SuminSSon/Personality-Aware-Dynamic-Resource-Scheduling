import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. 파일 설정
files = {
    "Pollux": "pollux_1nodes_DDP_CIFAR10_Resnet50_163.180.160.62_20008_rank0.csv",
    "Lucid": "Lucid_1nodes_DDP_CIFAR10_Resnet50_163.180.160.62_20017_rank0.csv",
    "Sia": "sia_1nodes_DDP_CIFAR10_Resnet50_163.180.160.62_20036_rank0.csv",
    "Skuld (Ours)": "ours_1nodes_CIFAR10_Resnet50_163.180.117.216_29529_rank0.csv"
}

# 2. 스타일 설정
sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.6})
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.0,
    'figure.dpi': 300
})

styles = {
    "Skuld (Ours)": {"color": "#D62728", "marker": "*", "label": "Skuld (Ours)"},
    "Lucid":        {"color": "#2CA02C", "marker": "^", "label": "Lucid"},
    "Sia":          {"color": "#FF7F0E", "marker": "s", "label": "Sia"},
    "Pollux":       {"color": "#1F77B4", "marker": "o", "label": "Pollux"}
}

# -----------------------------------------------------------------------------
# [Step 3 & 4] 기존 Line Plot 생성 (유지)
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 6))

# 히트맵용 데이터를 모으기 위한 딕셔너리 (재로딩 방지)
processed_data = {} 
max_time_global = 0

for name, filepath in files.items():
    try:
        df = pd.read_csv(filepath)
        df['elapsed_time'] = df['total_time'].cumsum()
        
        # 히트맵을 위해 데이터 저장
        processed_data[name] = df[['elapsed_time', 'accuracy']].copy()
        max_time_global = max(max_time_global, df['elapsed_time'].max())

        style = styles.get(name, {"color": "black", "marker": "."})
        
        # Skuld만 실선(-), 나머지는 점선(--)
        line_style = '-' if "Skuld" in name else '--'
        
        plt.plot(df['elapsed_time'], df['accuracy'], 
                 label=name, 
                 color=style['color'], 
                 marker=style['marker'], 
                 linestyle=line_style, 
                 linewidth=1.8, 
                 markersize=9, 
                 markerfacecolor='none',  # 내부 비우기
                 markeredgecolor=style['color'], 
                 markeredgewidth=1.5)
                 
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Line Plot 데코레이션 및 저장
plt.xlabel("Training Time (s)", fontweight='bold')
plt.ylabel("Validation Accuracy (%)", fontweight='bold')
plt.legend(frameon=True, edgecolor='black', framealpha=0.9, fancybox=False, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

for spine in plt.gca().spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig("s2_accuracy_curve.png", bbox_inches='tight')
plt.close()
print("Generated: s2_accuracy_curve.png")

# -----------------------------------------------------------------------------
# [Step 5] Heatmap 생성 (수정된 버전)
# -----------------------------------------------------------------------------
# 히트맵은 연속된 시간을 표현해야 하므로 보간(Interpolation)이 필요합니다.

plt.figure(figsize=(10, 3.5)) # 가로로 긴 형태

# 1. 공통 시간축 생성 (0초 ~ 최대시간, 100개 구간)
common_time_grid = np.linspace(0, max_time_global, 100)
heatmap_matrix = []
scheduler_names = ["Skuld (Ours)", "Lucid", "Sia", "Pollux"] # 순서 지정 (Skuld 맨 위)

for name in scheduler_names:
    if name not in processed_data: continue
    
    df = processed_data[name]
    
    # 시간과 정확도 데이터 추출 (0,0 지점 추가하여 보간 정확도 향상)
    times = np.concatenate(([0], df['elapsed_time'].values))
    accs = np.concatenate(([0], df['accuracy'].values))
    
    # 공통 시간축에 맞춰 정확도 보간 (Linear Interpolation)
    interp_acc = np.interp(common_time_grid, times, accs)
    heatmap_matrix.append(interp_acc)

# 2. 데이터프레임 변환
# 컬럼명을 정수형 시간(초)으로 변환
df_hm = pd.DataFrame(heatmap_matrix, index=scheduler_names, columns=np.round(common_time_grid, 0).astype(int))

# 3. 히트맵 그리기
ax = sns.heatmap(df_hm, cmap="YlOrRd", vmin=0, vmax=90, 
                 cbar_kws={'label': 'Validation Accuracy (%)'},
                 linewidths=0.5, linecolor='white')

# 4. 축 설정
plt.xlabel("Training Time (s)", fontweight='bold')
plt.ylabel("") 
plt.title("Convergence Speed Heatmap", pad=10, fontsize=14, fontweight='bold')

# [핵심 수정 부분] ---------------------------------------------------------
# 틱 위치(Ticks)를 데이터 길이(100개)를 기준으로 10개 간격으로 명시적 생성
# np.arange(0, 100, 10) -> [0, 10, 20, ... 90] 인덱스 생성
tick_step = 10
tick_indices = np.arange(0, len(df_hm.columns), tick_step)
tick_labels = df_hm.columns[tick_indices]

# 위치(Ticks)와 라벨(Labels)을 동시에 쌍으로 설정해야 에러가 안 남
ax.set_xticks(tick_indices)
ax.set_xticklabels(tick_labels, rotation=0)
# -------------------------------------------------------------------------

plt.tight_layout()
plt.savefig("s2_accuracy_heatmap.png", bbox_inches='tight', dpi=300)
plt.close()

print("Generated: s2_accuracy_heatmap.png (Fixed Error)")