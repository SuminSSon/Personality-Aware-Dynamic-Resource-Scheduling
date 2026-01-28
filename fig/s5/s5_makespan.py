import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# =========================
# Data
# =========================
labels = ["64", "128", "256", "512", "1024"]
x = np.arange(len(labels))  # ✅ 균등 간격을 위한 categorical x

data = {
    "Pollux": [41606.0, 25811.0, 15776.0, 10378.0, 7591.0],
    "Lucid":  [32142.0, 16850.0, 9897.0, 7073.0, 5723.0],
    "Sia":    [31172.0, 17804.0, 11155.0, 7213.0, 4920.0],
    "Ours":   [26077.0, 13753.0, 8218.0, 5103.0, 4087.0],
}

# =========================
# IEEE-style rcParams (작게!)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
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
# Figure (IEEE single-column)
# =========================
fig, ax = plt.subplots(figsize=(3.3, 2.1))

marker_map = {
    "Pollux": "o",
    "Lucid":  "s",
    "Sia":    "D",
    "Ours":   "^",
}

order = ["Pollux", "Lucid", "Sia", "Ours"]

for name in order:
    ax.plot(
        x,
        data[name],
        marker=marker_map[name],
        label=name,
    )

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

# =========================
# Grid & Spines
# =========================
ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax.grid(False, axis="x")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# =========================
# Legend (논문용 compact)
# =========================
leg = ax.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    borderpad=0.25,
    handlelength=1.4,
    handletextpad=0.4,
)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout(pad=0.3)

# =========================
# Save / Show
# =========================
# fig.savefig("makespan_vs_nodes_ieee.pdf", bbox_inches="tight")
plt.show()