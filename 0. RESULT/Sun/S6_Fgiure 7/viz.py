import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================

def load_data():
    files = glob.glob("**/details_*.csv", recursive=True)
    
    if not files:
        print("❌ No 'details_*.csv' files found! Run simulation first.")
        return None
        
    print(f"📂 Found {len(files)} detailed CSV files. Loading...")
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # 파일명 파싱: details_{nodes}_{scheduler}.csv
            basename = os.path.basename(f)
            name_body = basename.replace('.csv', '')
            parts = name_body.split('_')
            
            # parts 구조 예상:
            # ['details', '64', 'FIFO']
            # ['details', '64', 'Ours', 'Cost'] -> 'Ours_Cost'
            # ['details', '64', 'Ours', '(mixed)'] -> 'Ours (mixed)' (파일명에 공백이 있다면 split이 다르게 동작하지 않음, '_' 기준임)
            # 만약 run.py가 "details_64_Ours (mixed).csv"로 저장했다면, '_'로 자르면
            # ['details', '64', 'Ours (mixed)'] 가 됨.
            
            if len(parts) < 3:
                continue
                
            nodes = int(parts[1])
            scheduler = "_".join(parts[2:]) # 나머지 부분을 다시 합침
            
            df['Nodes'] = nodes
            df['Scheduler'] = scheduler
            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")
            
    if not all_data:
        return None

    full_df = pd.concat(all_data, ignore_index=True)
    
    # 컬럼 매핑 (simulation.py 출력 -> 시각화용 이름)
    col_map = {
        'Cost_$': 'Cost', 
        'Energy_kWh': 'Energy', 
        'JCT': 'JCT', 
        'Alloc_GPUs': 'GPUs',
        'Alloc_Cluster': 'Cluster'
    }
    full_df.rename(columns=col_map, inplace=True)
    
    # 누락된 컬럼 기본값 처리
    if 'Slowdown' not in full_df.columns: full_df['Slowdown'] = 0 
    if 'WaitTime' not in full_df.columns: full_df['WaitTime'] = 0 
    
    return full_df

# ==========================================
# 2. Aggregation
# ==========================================

def aggregate_metrics(full_df):
    # 기본 집계
    job_stats = full_df.groupby(['Nodes', 'Scheduler']).agg({
        'Cost': 'sum',
        'Energy': 'sum',
        'JCT': 'mean'
    }).reset_index()
    
    # run.py가 만든 summary.csv (Makespan, Overhead 정보 포함) 로드
    summary_files = glob.glob("**/summary.csv", recursive=True)
    
    if summary_files:
        print(f"📂 Loading Overhead & Makespan from {summary_files[0]}...")
        try:
            meta_df = pd.read_csv(summary_files[0])
            
            # run.py의 컬럼명 매핑
            meta_map = {
                'Overhead(ms)': 'Overhead',
                'Makespan(s)': 'Makespan'
            }
            meta_df.rename(columns=meta_map, inplace=True)
            
            # 병합
            final_summary = pd.merge(job_stats, meta_df[['Nodes', 'Scheduler', 'Overhead', 'Makespan']], 
                                     on=['Nodes', 'Scheduler'], how='left')
            
            final_summary.fillna(0, inplace=True)
            return final_summary
        except Exception as e:
            print(f"⚠️ Error reading summary.csv: {e}")
            
    job_stats['Overhead'] = 0
    job_stats['Makespan'] = 0
    return job_stats

# ==========================================
# 3. Plotting Functions
# ==========================================

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def save_fig(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved Figure: {filename}")
    plt.close()

def plot_all(full_df, summary_df):
    set_style()
    
    schedulers = sorted(summary_df['Scheduler'].unique())
    print(f"📊 Detected Schedulers: {schedulers}")
    
    # [수정] 색상 매핑에 'Ours (mixed)' 추가
    colors = {
        'FIFO': 'grey', 'SFJ': 'orange', 
        'Pollux': '#9467BD', 'Sia': '#2CA02C', 'Lucid': '#D62728',
        'Ours': '#1F77B4',
        'Ours (mixed)': '#1F77B4', # Ours와 같은 색 (Blue)
        'Ours_Time': '#17BECF',  
        'Ours_Cost': '#00008B'   
    }
    # 매핑되지 않은 스케줄러를 위한 안전장치
    def get_color(s): return colors.get(s, '#333333')
    
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
    
    # 1. Scalability (Overhead)
    plt.figure(figsize=(8, 5))
    for i, s in enumerate(schedulers):
        data = summary_df[summary_df['Scheduler'] == s]
        if not data.empty:
            plt.plot(data['Nodes'], data['Overhead'], marker=markers[i % len(markers)], label=s, 
                     color=get_color(s), linewidth=2)
    plt.xscale('log', base=2)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Scheduling Overhead (ms)')
    plt.title('Scalability Analysis (Overhead)')
    plt.legend()
    save_fig('Fig1_Scalability_Overhead.png')

    # 2. Cost
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x='Nodes', y='Cost', hue='Scheduler', palette=colors)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Total Cost ($)')
    plt.title('Cost Efficiency')
    save_fig('Fig2_Total_Cost.png')

    # 3. Energy
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary_df, x='Nodes', y='Energy', hue='Scheduler', palette=colors)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Total Energy (kWh)')
    plt.title('Energy Efficiency')
    save_fig('Fig3_Total_Energy.png')

    # 4. Makespan
    plt.figure(figsize=(8, 5))
    for i, s in enumerate(schedulers):
        data = summary_df[summary_df['Scheduler'] == s]
        if not data.empty:
            plt.plot(data['Nodes'], data['Makespan'], marker=markers[i % len(markers)], label=s, 
                     color=get_color(s), linewidth=2)
    plt.xscale('log', base=2)
    plt.xlabel('Number of Nodes')
    plt.ylabel('Makespan (s)')
    plt.title('Performance (Makespan)')
    plt.legend()
    save_fig('Fig4_Makespan.png')

    # 5. JCT Distribution (CDF) - 128 Nodes
    plt.figure(figsize=(7, 5))
    target_node = 128
    subset = full_df[full_df['Nodes'] == target_node]
    if not subset.empty:
        for s in schedulers:
            s_data = subset[subset['Scheduler'] == s]
            if not s_data.empty:
                data = np.sort(s_data['JCT'])
                yvals = np.arange(len(data)) / float(len(data) - 1)
                plt.plot(data, yvals, label=s, color=get_color(s), linewidth=2)
        plt.xlabel('Job Completion Time (s)')
        plt.ylabel('CDF')
        plt.title(f'JCT Distribution ({target_node} Nodes)')
        plt.legend()
        save_fig('Fig6_JCT_CDF.png')

# ==========================================
# 4. Table Generation
# ==========================================

def create_table1(summary_df):
    try:
        df_4096 = summary_df[summary_df['Nodes'] == 4096].set_index('Scheduler')
        
        baseline = 'FIFO' if 'FIFO' in df_4096.index else df_4096.index[0]
        # Ours (mixed)를 우선 찾고, 없으면 Ours를 찾음
        ours = 'Ours (mixed)' if 'Ours (mixed)' in df_4096.index else ('Ours' if 'Ours' in df_4096.index else baseline)
        
        if ours not in df_4096.index: return

        metrics = ['Cost', 'Energy', 'Overhead', 'JCT']
        
        display_data = []
        for m in metrics:
            base_val = df_4096.loc[baseline, m]
            ours_val = df_4096.loc[ours, m]
            if base_val > 0:
                imp = (base_val - ours_val) / base_val * 100
                imp_str = f"{imp:.1f}% Saved" if imp > 0 else f"{abs(imp):.1f}% Increased"
            else:
                imp_str = "-"
            
            display_data.append([m, f"{base_val:.2f}", f"{ours_val:.2f}", imp_str])

        cols = ['Metric', f'Baseline ({baseline})', f'Proposed ({ours})', 'Improvement']
        
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('off')
        table = ax.table(cellText=display_data, colLabels=cols, loc='center', cellLoc='center')
        table.scale(1, 1.5)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        plt.title('Table 1. Performance Summary (4096 Nodes)', fontweight='bold', y=1.1)
        save_fig('Table1_KPI_Summary.png')
        
    except Exception as e:
        print(f"⚠️ Could not generate Table 1: {e}")

if __name__ == "__main__":
    full_df = load_data()
    if full_df is not None:
        summary_df = aggregate_metrics(full_df)
        
        print("\n🎨 Generating Figures...")
        plot_all(full_df, summary_df)
        
        print("\n📊 Generating Tables...")
        create_table1(summary_df)
        
        print("\n✨ All done! Check the generated PNG files.")