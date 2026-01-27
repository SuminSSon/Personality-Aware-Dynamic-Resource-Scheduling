# s1_cdf.py
# CDF-style "work completed over time" figure for S1.
# - X axis: elapsed time since *each scheduler run's* first submission (per-run normalization)
# - Y axis: cumulative completed jobs
# - No file saving (shows plot only)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -----------------------------
# Config: place this script in the SAME folder as the csv files,
# or set BASE_DIR to that folder.
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "Pollux": os.path.join(BASE_DIR, "pollux_job_metrics.csv"),
    "Sia":    os.path.join(BASE_DIR, "sia_job_metrics.csv"),
    "Lucid":  os.path.join(BASE_DIR, "lucid_job_metrics.csv"),
    "Skuld":   os.path.join(BASE_DIR, "ours_job_metrics.csv"),
}

# Order & style (keep readable in IEEE/ACM two-column)
PLOT_ORDER = ["Pollux", "Sia", "Lucid", "Skuld"]

def _to_numeric_ts(series: pd.Series) -> pd.Series:
    """
    Convert a timestamp column to numeric seconds.
    Handles:
      - float/int seconds (already)
      - string numeric
      - datetime-like strings (rare here, but safe)
    Returns float seconds, with NaN for invalid rows.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        return s.astype(float)

    # fallback: parse as datetime and convert to unix seconds
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    # use astype('int64') to avoid .view deprecation
    return (dt.astype("int64") / 1e9).astype(float)

def load_metrics(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.upper().str.strip()
        # ✅ terminal 상태만 카운트 (makespan 용)
        df = df[df["status"].isin(["FINISHED", "FAILED"])].copy()


    # timestamp 변환은 기존대로
    df["submitted_ts"] = _to_numeric_ts(df["submitted_ts"])
    df["end_ts"] = _to_numeric_ts(df["end_ts"])
    df = df.dropna(subset=["submitted_ts", "end_ts"]).copy()
    df = df[df["end_ts"] >= df["submitted_ts"]].copy()

    # ✅ preemption/재시도 중복 제거: 같은 job_id면 가장 마지막 end_ts만
    if "job_id" in df.columns:
        df["job_id"] = df["job_id"].astype(str)
        df = df[df["job_id"].notna() & (df["job_id"].str.len() > 0)].copy()
        df = df.sort_values("end_ts").drop_duplicates(subset=["job_id"], keep="last")


    if df.empty:
        raise RuntimeError(f"{os.path.basename(path)} has no valid rows after cleaning.")

    return df

def build_jct_cdf(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    CDF of per-job JCT (seconds).
    X: JCT (sec)
    Y: fraction of jobs with JCT <= x
    """
    if "jct_sec" not in df.columns:
        raise ValueError("Missing column: jct_sec")

    jct = pd.to_numeric(df["jct_sec"], errors="coerce").dropna().astype(float).to_numpy()
    if len(jct) == 0:
        raise RuntimeError("No valid jct_sec rows.")

    x = np.sort(jct)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y

def nice_time_axis_seconds(max_sec: float) -> tuple[np.ndarray, list[str]]:
    """
    Create nicer tick labels for seconds axis:
    - If long enough, show hours; otherwise minutes/seconds.
    """
    if max_sec <= 0:
        return np.array([0.0]), ["0"]

    # choose unit
    if max_sec >= 3600:
        # hours
        step_h = 1
        if max_sec / 3600 > 12:
            step_h = 2
        if max_sec / 3600 > 24:
            step_h = 4
        ticks = np.arange(0, (max_sec / 3600) + 1e-9, step_h) * 3600
        labels = [f"{int(t/3600)}h" for t in ticks]
        return ticks, labels

    if max_sec >= 300:
        # minutes
        step_m = 5
        if max_sec / 60 < 20:
            step_m = 2
        ticks = np.arange(0, (max_sec / 60) + 1e-9, step_m) * 60
        labels = [f"{int(t/60)}m" for t in ticks]
        return ticks, labels

    # seconds
    step_s = 50 if max_sec > 200 else 20 if max_sec > 100 else 10
    ticks = np.arange(0, max_sec + 1e-9, step_s)
    labels = [f"{int(t)}s" for t in ticks]
    return ticks, labels

def main():
    dfs = {}
    for name, path in FILES.items():
        dfs[name] = load_metrics(path)

    curves = {}
    max_x = 0.0
    max_y = 0
    for name in PLOT_ORDER:
        x, y = build_jct_cdf(dfs[name])
        curves[name] = (x, y)
        max_x = max(max_x, float(np.nanmax(x)))
        max_y = max(max_y, int(np.nanmax(y)))

    # IEEE-ish readable defaults (without requiring external fonts)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
    })


    fig, ax = plt.subplots(figsize=(4.6, 4.0))  # wide but still paper-friendly

    # Plot: step curves, distinct line styles (no markers to reduce clutter)
    line_styles = {
        "Pollux": ("--", 1.6),
        "Sia":    (":",  1.6),
        "Lucid":  ("-.", 1.6),
        "Skuld":  ("-",  2.4),   # ✅ OURS 강조 (두께 + 실선)
    }

    marker_styles = {
        "Pollux": None,
        "Sia":    None,
        "Lucid":  None,
        "Skuld":  None,          # ✅ 마커 제거: CDF는 곡선 비교가 핵심
    }

    for name in PLOT_ORDER:
        x, y = curves[name]          # x: seconds
        ls, lw = line_styles.get(name, ("-", 1.8))

        # seconds -> hours
        xh = x / 3600.0

        ax.step(
            xh, y,
            where="post",
            linestyle=ls,
            linewidth=lw,
            label=name,
        )

    ax.set_xlabel("JCT (h)")
    ax.set_ylabel("CDF")

    max_h = max_x / 3600.0
    xmax = np.ceil(max_h)
    ax.set_xlim(-0.2, xmax)
    ax.set_xticks(np.arange(0, xmax + 1e-9, 1.0))

    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.78),
        frameon=True,
        fancybox=False,
        framealpha=0.9,
        edgecolor="black"
    )

    # Tight layout for paper cropping
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()