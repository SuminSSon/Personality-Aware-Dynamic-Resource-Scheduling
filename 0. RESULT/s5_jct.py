import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =========================
# Data
# =========================
labels = ["64", "128", "256", "512", "1024"]
x = np.arange(len(labels))

data = {
    "Pollux": [19823.8, 11619.3, 6281.6, 3365.3, 1793.1],
    "Lucid":  [12748.3, 6566.9, 3477.5, 2012.1, 1453.6],
    "Sia":    [14748.8, 8064.2, 4642.6, 2486.8, 1308.5],
    "Ours":   [12390.3, 6229.2, 3133.8, 1639.2, 1123.7],
}

# =========================
# IEEE-style rcParams
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.dpi": 200,
    "savefig.dpi": 600,
})

# =========================
# Figure
# =========================
fig, ax = plt.subplots(figsize=(3.3, 2.1))

marker_map = {
    "Pollux": "o",
    "Lucid":  "s",
    "Sia":    "D",
    "Ours":   "^",
}

order = ["Pollux", "Lucid", "Sia", "Ours"]

all_values = []
for name in order:
    y = data[name]
    all_values.extend(y)
    ax.plot(x, y, marker=marker_map[name], label=name)

# =========================
# Axes
# =========================
ax.set_xlabel("Cluster Size (Nodes)")
ax.set_ylabel("Makespan (s)")
ax.set_xticks(x)
ax.set_xticklabels(labels)

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda v, _: f"{int(v):,}")
)

# ✅ 핵심: y축 범위 재설계
y_min = min(all_values) * 0.95
y_max = max(all_values) * 1.05
ax.set_ylim(y_min, y_max)

# =========================
# Grid & Spines
# =========================
ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax.grid(False, axis="x")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# =========================
# Legend
# =========================
leg = ax.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    borderpad=0.25,
    handlelength=1.4,
)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout(pad=0.3)
plt.show()