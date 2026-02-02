import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Output Directory Creation
output_dir = "viz_paper"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Data Loading
try:
    df = pd.read_csv('summary.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: summary.csv not found.")
    df = pd.DataFrame() 

# 3. Data Filtering (Exclude FIFO, SFJ)
exclude_list = ['FIFO', 'SFJ']
df_filtered = df[~df['Scheduler'].isin(exclude_list)].copy()

# Style Settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Colors & Markers for Trend Plots
colors = {
    'Pollux': '#9467BD',   # Purple
    'Sia': '#2CA02C',      # Green
    'Lucid': '#D62728',    # Red
    'Ours (mixed)': '#1F77B4', # Blue (Main)
    'Ours_Cost': '#00008B',    # Dark Blue
    'Ours_Time': '#17BECF',    # Cyan
    'Ours_Energy': '#2ca02c'   # Green for Energy
}
markers = {'Pollux': 'v', 'Sia': 'P', 'Lucid': '^', 'Ours (mixed)': 'D', 'Ours_Cost': 'X', 'Ours_Time': '*', 'Ours_Energy': 's'}
linestyles = {'Pollux': '--', 'Sia': '-.', 'Lucid': ':', 'Ours (mixed)': '-', 'Ours_Cost': '--', 'Ours_Time': '-.', 'Ours_Energy': ':'}

# Helper function for Trend Plots
def plot_trend(metric_col, ylabel, title, filename, yscale='linear'):
    plt.figure(figsize=(8, 5))
    schedulers = df_filtered['Scheduler'].unique()
    
    # Sort: Ours last for visibility
    sorted_scheds = sorted(schedulers, key=lambda x: 'Ours' in x)
    
    for sched in sorted_scheds:
        subset = df_filtered[df_filtered['Scheduler'] == sched]
        c = colors.get(sched, 'gray')
        m = markers.get(sched, 'o')
        l = linestyles.get(sched, '-')
        
        plt.plot(subset['Nodes'], subset[metric_col], 
                 marker=m, label=sched, color=c, linestyle=l,
                 linewidth=2.5 if 'Ours' in sched else 1.5)
    
    plt.xscale('log', base=2)
    
    if not df_filtered.empty:
        unique_nodes = sorted(df_filtered['Nodes'].unique())
        plt.xticks(unique_nodes, unique_nodes)
        plt.minorticks_off()
        
    if yscale == 'log':
        plt.yscale('log')
        
    plt.xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(title='Scheduler', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# --- Generate Trend Plots ---
if not df_filtered.empty:
    plot_trend('Cost($)', 'Total Cost ($)', 'Cost Efficiency Comparison', 'Fig_Cost_Trend.png')
    plot_trend('Energy(kWh)', 'Total Energy (kWh)', 'Energy Efficiency Comparison', 'Fig_Energy_Trend.png')
    plot_trend('AvgJCT(s)', 'Average JCT (s)', 'Performance (Speed) Comparison', 'Fig_JCT_Trend.png')
    plot_trend('Overhead(ms)', 'Scheduling Overhead (ms)', 'Scalability Analysis', 'Fig_Overhead_Trend.png', yscale='log')

# --- Generate Ablation Study Plot (Impact of Intent) including Energy ---
def plot_ablation_with_energy():
    target_node = 1024
    df_1024 = df[df['Nodes'] == target_node]
    
    # Check for Ours_Energy availability
    ours_variants = ['Ours_Cost', 'Ours (mixed)', 'Ours_Time']
    if 'Ours_Energy' in df_1024['Scheduler'].values:
        ours_variants.append('Ours_Energy')
        
    # Filter only if they exist in the data
    existing_variants = [v for v in ours_variants if v in df_1024['Scheduler'].values]
    
    if not existing_variants:
        print("No Ours variants found for ablation plot.")
        return

    df_ablation = df_1024[df_1024['Scheduler'].isin(existing_variants)].copy()
    
    # Normalize based on 'Ours (mixed)' if available, else first one
    base_sched = 'Ours (mixed)' if 'Ours (mixed)' in existing_variants else existing_variants[0]
    base_vals = df_ablation[df_ablation['Scheduler'] == base_sched].iloc[0]
    
    df_ablation['Norm_Cost'] = df_ablation['Cost($)'] / base_vals['Cost($)']
    df_ablation['Norm_JCT'] = df_ablation['AvgJCT(s)'] / base_vals['AvgJCT(s)']
    df_ablation['Norm_Energy'] = df_ablation['Energy(kWh)'] / base_vals['Energy(kWh)']
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6)) # Wider figure for 4 variants
    bar_width = 0.2
    x = np.arange(len(existing_variants))
    
    metrics = [('Norm_Cost', 'Cost'), ('Norm_JCT', 'Speed (JCT)'), ('Norm_Energy', 'Energy')]
    
    bar_colors = {
        'Cost': 'C0',       # Blue
        'Speed (JCT)': 'C1', # Orange
        'Energy': 'C2'      # Green
    }

    for i, (col, label) in enumerate(metrics):
        vals = []
        for sched in existing_variants:
            val = df_ablation[df_ablation['Scheduler'] == sched][col].values
            vals.append(val[0] if len(val) > 0 else 0)
            
        # Adjust offset for 3 bars per group
        offset = (i - 1) * bar_width
        rects = ax.bar(x + offset, vals, bar_width, label=label, 
                        color=bar_colors[label], alpha=0.9, edgecolor='black')
        
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    f'{height:.2f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(existing_variants, fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Normalized Ratio (vs {base_sched})', fontsize=12, fontweight='bold')
    ax.set_title(f'Impact of User Intent (at {target_node} Nodes)', fontsize=14, fontweight='bold')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5)
    ax.legend(title='Metric', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    save_path = os.path.join(output_dir, 'Fig_Ablation_Intent_With_Energy.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

if not df.empty:
    plot_ablation_with_energy()