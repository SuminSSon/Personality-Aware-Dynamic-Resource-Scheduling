import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 정의 (사용자 Table 기반)
data = [
    {"Cluster": "A", "GPUs": 1, "JCT": 5640, "Cost": 2.21, "Energy": 9.41e5},
    {"Cluster": "A", "GPUs": 2, "JCT": 3423, "Cost": 2.68, "Energy": 1.11e6},
    {"Cluster": "A", "GPUs": 4, "JCT": 2058, "Cost": 3.22, "Energy": 1.20e6},
    {"Cluster": "B", "GPUs": 1, "JCT": 6382, "Cost": 1.77, "Energy": 1.35e6},
    {"Cluster": "B", "GPUs": 2, "JCT": 3310, "Cost": 1.84, "Energy": 1.38e6},
    {"Cluster": "B", "GPUs": 4, "JCT": 1703, "Cost": 1.89, "Energy": 1.40e6}
]
df = pd.DataFrame(data)

# 2. 스타일 설정
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

plt.figure(figsize=(7, 5))

# 3. 스캐터 플롯 그리기
# Cluster별로 마커 모양 다르게, GPU 개수는 텍스트로 표시
markers = {"A": "o", "B": "s"}
colors = {"A": "#1f77b4", "B": "#ff7f0e"} # Blue, Orange

for idx, row in df.iterrows():
    plt.scatter(row['JCT'], row['Cost'], 
                s=200, # 점 크기
                c=colors[row['Cluster']], 
                marker=markers[row['Cluster']], 
                edgecolor='black', 
                label=f"Cluster {row['Cluster']}" if row['GPUs']==1 else "", # 범례 중복 방지
                zorder=3)
    
    # 텍스트 주석 (GPU 개수)
    plt.text(row['JCT'], row['Cost']+0.08, f"{row['GPUs']} GPU", 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. 축 및 범례 설정
plt.xlabel("Job Completion Time (s)", fontweight='bold')
plt.ylabel("Monetary Cost ($)", fontweight='bold')
plt.title("Trade-off Space: Cost vs. Time", fontsize=14, pad=15)

# 축 범위 여유 있게
plt.ylim(1.5, 3.6)
plt.xlim(1000, 7000)

plt.legend(loc='upper right', frameon=True, framealpha=0.9, edgecolor='black')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("s3_slo_pareto.png", dpi=300)
plt.close()

print("Generated s3_slo_pareto.png")