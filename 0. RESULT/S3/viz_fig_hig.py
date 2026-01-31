import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 정의 (라벨을 논문 본문의 Policy Name과 통일)
data = [
    {"Cluster": "A", "GPUs": 1, "JCT": 5640, "Cost": 2.21, "Label": "Green-Computing"}, # 수정됨
    {"Cluster": "A", "GPUs": 2, "JCT": 3423, "Cost": 2.68, "Label": ""},
    {"Cluster": "A", "GPUs": 4, "JCT": 2058, "Cost": 3.22, "Label": ""},
    {"Cluster": "B", "GPUs": 1, "JCT": 6382, "Cost": 1.77, "Label": "Cost-Efficient"},  # 수정됨
    {"Cluster": "B", "GPUs": 2, "JCT": 3310, "Cost": 1.84, "Label": ""},
    {"Cluster": "B", "GPUs": 4, "JCT": 1703, "Cost": 1.89, "Label": "Time-Critical"}    # 수정됨
]
df = pd.DataFrame(data)

# 2. 스타일 설정
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})
plt.figure(figsize=(8, 6))

# 3. 기본 스캐터 플롯
markers = {"A": "o", "B": "s"}
colors = {"A": "#1f77b4", "B": "#ff7f0e"} # Blue, Orange

for idx, row in df.iterrows():
    # 선택된 점인지 확인
    is_selected = row['Label'] != ""
    
    # 마커 스타일 설정
    size = 300 if is_selected else 150
    edge_color = 'red' if is_selected else 'black'
    edge_width = 2.5 if is_selected else 1.0
    alpha = 1.0 if is_selected else 0.6
    
    plt.scatter(
        row['JCT'], 
        row['Cost'], 
        s=size, 
        c=colors[row['Cluster']], 
        marker=markers[row['Cluster']], 
        edgecolor=edge_color,
        linewidth=edge_width,
        alpha=alpha,
        zorder=5 if is_selected else 3
    )
    
    # [GPU 개수] 텍스트 (점 바로 위)
    plt.text(
        row['JCT'], 
        row['Cost'] + 0.12,  # 간격 약간 조정
        f"{row['GPUs']} GPU", 
        ha='center', va='bottom', fontsize=11, fontweight='bold', color='black'
    )
    
    # [Policy Name] 텍스트 (점 아래, 빨간색 강조)
    if is_selected:
        plt.text(
            row['JCT'], 
            row['Cost'] - 0.15, 
            f"[{row['Label']}]", # .split() 제거하여 전체 이름 표시
            ha='center', va='top', fontsize=12, fontweight='bold', color='#D62728'
        )

# 4. 범례 생성
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Cluster A', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#ff7f0e', label='Cluster B', markersize=10, markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none', label='Selected Config', markersize=15, markeredgecolor='red', markeredgewidth=2)
]
plt.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.9, edgecolor='black')

# 5. 축 및 기타 설정
plt.xlabel("Job Completion Time (s)", fontweight='bold')
plt.ylabel("Monetary Cost ($)", fontweight='bold')
plt.ylim(1.5, 3.8)
plt.xlim(1000, 7000)
plt.grid(True, linestyle='--', alpha=0.6)

# 테두리 강화
for spine in plt.gca().spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
plt.savefig("s3_slo_pareto_highlighted.png", dpi=300)
plt.close()

print("Generated s3_slo_pareto_highlighted.png with updated labels")