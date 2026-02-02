import os
import re
import glob
import math
import argparse
import pandas as pd
import numpy as np

# -------------------------
# Utilities
# -------------------------
def infer_cluster_from_path(path: str) -> str:
    p = path.lower().replace("\\", "/")
    if "clustera" in p:
        return "clusterA"
    if "clusterb" in p:
        return "clusterB"
    return "unknown"

def infer_gpu_count_from_filename(fname: str) -> int:
    base = os.path.basename(fname)
    m = re.match(r"^(\d+)\s*nodes_", base)
    if m:
        return int(m.group(1))
    m2 = re.search(r"(?:ws|world_size)[=_-]?(\d+)", base, re.IGNORECASE)
    if m2:
        return int(m2.group(1))
    return 1

def short_path(p: str) -> str:
    p = str(p).replace("\\", "/")
    parts = [x for x in p.split("/") if x and x != "."]
    return "/".join(parts[-2:]) if len(parts) >= 2 else p

def read_csv_auto(path: str) -> pd.DataFrame:
    # comma/tsv auto
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, delimiter="\t")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def pick_duration_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"duration col not found. have={list(df.columns)}")

# -------------------------
# Energy computation with fill strategy
# -------------------------
def compute_energy_wh_per_gpu_with_fill(
    df: pd.DataFrame,
    epoch_min: int,
    epoch_max: int,
    warmup_epochs: int = 3,
    fill_mode: str = "warmup_mean",
) -> tuple[float, bool, float, int, int]:
    """
    Returns:
      (energy_wh_per_gpu, power_any, fill_power_w, measured_epochs, missing_epochs)

    energy_wh_per_gpu = Σ(dt_sec * power_w) / 3600 over epochs [epoch_min..epoch_max]
    If power missing for some epochs, fill with:
      - warmup_mean: mean power over epochs [epoch_min..epoch_min+warmup_epochs-1] where available; fallback to mean of all available
      - global_mean: mean power over all available epochs in range
      - epoch1: power of epoch_min (if available) else fallback global mean
    """
    if "epoch" not in df.columns:
        return (float("nan"), False, float("nan"), 0, 0)

    if "epoch_power_dt_sec" in df.columns:
        dt_col = "epoch_power_dt_sec"
    elif "total_time" in df.columns:
        # fallback: if someone stored per-epoch total_time as epoch duration
        dt_col = "total_time"
    else:
        return (float("nan"), False, float("nan"), 0, 0)

    if "gpu_power_w" not in df.columns:
        return (float("nan"), False, float("nan"), 0, 0)

    tmp = df.copy()
    tmp["epoch"] = to_num(tmp["epoch"])
    tmp = tmp.dropna(subset=["epoch"])
    tmp = tmp[(tmp["epoch"] >= epoch_min) & (tmp["epoch"] <= epoch_max)]
    if tmp.empty:
        return (float("nan"), False, float("nan"), 0, 0)

    tmp[dt_col] = to_num(tmp[dt_col])
    tmp["gpu_power_w"] = to_num(tmp["gpu_power_w"])

    # measured power epochs
    measured = tmp.dropna(subset=[dt_col, "gpu_power_w"]).copy()
    power_any = not measured.empty

    # If no power at all, cannot compute power-based energy
    if not power_any:
        return (float("nan"), False, float("nan"), 0, int(tmp[dt_col].notna().sum()))

    # Determine fill power
    fill_power = float("nan")

    if fill_mode == "warmup_mean":
        w_hi = epoch_min + max(warmup_epochs, 1) - 1
        warm = measured[(measured["epoch"] >= epoch_min) & (measured["epoch"] <= w_hi)]
        if not warm.empty:
            fill_power = float(warm["gpu_power_w"].mean())
        else:
            fill_power = float(measured["gpu_power_w"].mean())

    elif fill_mode == "global_mean":
        fill_power = float(measured["gpu_power_w"].mean())

    elif fill_mode == "epoch1":
        e1 = measured[measured["epoch"] == epoch_min]
        if not e1.empty:
            fill_power = float(e1["gpu_power_w"].iloc[0])
        else:
            fill_power = float(measured["gpu_power_w"].mean())
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    # Build per-epoch power series: use measured where available else fill_power
    # Only count epochs with dt available and >0
    tmp = tmp.dropna(subset=[dt_col]).copy()
    tmp[dt_col] = tmp[dt_col].fillna(0.0)
    tmp = tmp[tmp[dt_col] > 0]

    # mark missing power
    missing_mask = tmp["gpu_power_w"].isna()
    missing_epochs = int(missing_mask.sum())
    measured_epochs = int((~missing_mask).sum())

    tmp.loc[missing_mask, "gpu_power_w"] = fill_power

    energy_wh_per_gpu = float((tmp[dt_col] * tmp["gpu_power_w"]).sum() / 3600.0)
    return (energy_wh_per_gpu, True, fill_power, measured_epochs, missing_epochs)

# -------------------------
# Per-file metrics
# -------------------------
def compute_metrics_for_file(
    csv_path: str,
    epoch_min: int,
    epoch_max: int,
    duration_candidates,
    rateA: float,
    rateB: float,
    warmup_epochs: int,
    fill_mode: str,
) -> dict | None:
    df = read_csv_auto(csv_path)
    if "epoch" not in df.columns:
        return None

    df["epoch"] = to_num(df["epoch"])
    df = df.dropna(subset=["epoch"])
    df = df[(df["epoch"] >= epoch_min) & (df["epoch"] <= epoch_max)]
    if df.empty:
        return None

    duration_col = pick_duration_col(df, duration_candidates)
    df[duration_col] = to_num(df[duration_col]).fillna(0.0)

    # JCT is sum of per-epoch total_time (or chosen duration col)
    jct_sec = float(df[duration_col].sum())
    if jct_sec <= 0:
        return None

    cluster = infer_cluster_from_path(csv_path)
    gpu_count = infer_gpu_count_from_filename(os.path.basename(csv_path))

    # cost
    rate = rateA if cluster == "clusterA" else rateB if cluster == "clusterB" else float("nan")
    jct_hour = jct_sec / 3600.0
    cost_usd = float(rate * jct_hour * gpu_count) if not math.isnan(rate) else float("nan")

    # energy with fill
    e_per_gpu, power_any, fill_power, meas_ep, miss_ep = compute_energy_wh_per_gpu_with_fill(
        df, epoch_min, epoch_max, warmup_epochs=warmup_epochs, fill_mode=fill_mode
    )
    energy_wh_total = e_per_gpu * gpu_count if power_any else float("nan")

    # proxy always available
    gpu_seconds = jct_sec * gpu_count

    # efficiency metrics only if power_any
    edp = (energy_wh_total * jct_sec) if power_any and (energy_wh_total > 0) else float("nan")
    ed2p = (energy_wh_total * (jct_sec ** 2)) if power_any and (energy_wh_total > 0) else float("nan")

    # final accuracy
    acc = float(df["accuracy"].dropna().iloc[-1]) if "accuracy" in df.columns and df["accuracy"].dropna().size else float("nan")
    top5 = float(df["top5_accuracy"].dropna().iloc[-1]) if "top5_accuracy" in df.columns and df["top5_accuracy"].dropna().size else float("nan")

    return {
        "csv_path": csv_path,
        "run": short_path(csv_path),
        "cluster": cluster,
        "gpu_count": gpu_count,
        "duration_col_used": duration_col,
        "jct_sec": jct_sec,
        "jct_min": jct_sec / 60.0,
        "cost_usd": cost_usd,
        "gpu_seconds": gpu_seconds,
        "energy_wh_total": energy_wh_total,
        "energy_wh_per_gpu": e_per_gpu,
        "power_available": bool(power_any),
        "power_fill_mode": fill_mode,
        "power_fill_w": fill_power if power_any else float("nan"),
        "power_measured_epochs": meas_ep,
        "power_missing_epochs": miss_ep,
        "edp": edp,
        "ed2p": ed2p,
        "final_accuracy": acc,
        "final_top5_accuracy": top5,
    }

# -------------------------
# Best selection
# -------------------------
def pick_best(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # time best
    best_time = summary.loc[summary["jct_sec"].idxmin()].to_dict()
    rows.append({"objective": "time_min_jct", **best_time})

    # cost best
    cost_df = summary[summary["cost_usd"].notna()]
    if not cost_df.empty:
        best_cost = cost_df.loc[cost_df["cost_usd"].idxmin()].to_dict()
        rows.append({"objective": "cost_min_usd", **best_cost})

    # absolute energy best (Wh)
    e_df = summary[summary["energy_wh_total"].notna() & (summary["energy_wh_total"] > 0)]
    if not e_df.empty:
        best_e = e_df.loc[e_df["energy_wh_total"].idxmin()].to_dict()
        best_e["energy_objective_used"] = "energy_wh_total"
        rows.append({"objective": "energy_abs_min_wh", **best_e})
    else:
        # fallback proxy
        best_e = summary.loc[summary["gpu_seconds"].idxmin()].to_dict()
        best_e["energy_objective_used"] = "gpu_seconds_fallback"
        rows.append({"objective": "energy_abs_min_proxy_gpu_seconds", **best_e})

    # energy-efficient EDP
    edp_df = summary[summary["edp"].notna() & (summary["edp"] > 0)]
    if not edp_df.empty:
        best_edp = edp_df.loc[edp_df["edp"].idxmin()].to_dict()
        best_edp["energy_objective_used"] = "EDP=E*T"
        rows.append({"objective": "energy_eff_min_edp", **best_edp})

    # energy-efficient ED2P
    ed2p_df = summary[summary["ed2p"].notna() & (summary["ed2p"] > 0)]
    if not ed2p_df.empty:
        best_ed2p = ed2p_df.loc[ed2p_df["ed2p"].idxmin()].to_dict()
        best_ed2p["energy_objective_used"] = "ED2P=E*T^2"
        rows.append({"objective": "energy_eff_min_ed2p", **best_ed2p})

    return pd.DataFrame(rows)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="top directory containing cluster folders")
    ap.add_argument("--epoch-min", type=int, default=1)
    ap.add_argument("--epoch-max", type=int, default=20)
    ap.add_argument("--only-rank0", action="store_true", help="use only files containing rank0 in filename")
    ap.add_argument("--out-summary", type=str, default="summary_runs.csv")
    ap.add_argument("--out-best", type=str, default="best_by_objective.csv")
    ap.add_argument("--rateA", type=float, default=1.0)
    ap.add_argument("--rateB", type=float, default=1.5)

    # fill behavior
    ap.add_argument("--fill-mode", type=str, default="warmup_mean",
                    choices=["warmup_mean", "global_mean", "epoch1"],
                    help="how to fill missing gpu_power_w epochs")
    ap.add_argument("--warmup-epochs", type=int, default=3,
                    help="used when fill-mode=warmup_mean (e.g., 3 means epochs 1..3)")

    args = ap.parse_args()

    duration_candidates = ["total_time", "train_time", "epoch_power_dt_sec"]

    csv_files = glob.glob(os.path.join(args.root, "**", "*.csv"), recursive=True)

    rows = []
    for p in csv_files:
        if args.only_rank0 and ("rank0" not in os.path.basename(p)):
            continue
        m = compute_metrics_for_file(
            p,
            args.epoch_min,
            args.epoch_max,
            duration_candidates,
            args.rateA,
            args.rateB,
            warmup_epochs=args.warmup_epochs,
            fill_mode=args.fill_mode,
        )
        if m:
            rows.append(m)

    if not rows:
        raise SystemExit("No valid CSV found (need epoch column and selected epoch range).")

    summary = pd.DataFrame(rows)

    # Sort & save
    summary = summary.sort_values(["cluster", "gpu_count", "jct_sec"], ascending=[True, True, True]).reset_index(drop=True)

    # Round key numeric columns for readability
    for c in ["jct_sec","jct_min","cost_usd","gpu_seconds","energy_wh_total","energy_wh_per_gpu","edp","ed2p","power_fill_w"]:
        if c in summary.columns:
            summary[c] = pd.to_numeric(summary[c], errors="coerce")

    summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    best = pick_best(summary)
    for c in ["jct_sec","jct_min","cost_usd","gpu_seconds","energy_wh_total","energy_wh_per_gpu","edp","ed2p","power_fill_w"]:
        if c in best.columns:
            best[c] = pd.to_numeric(best[c], errors="coerce").round(6)

    best.to_csv(args.out_best, index=False, encoding="utf-8-sig")

    # Console diagnostics
    power_ok = int(summary["power_available"].sum())
    print(f"[OK] wrote summary: {args.out_summary} (rows={len(summary)})")
    print(f"[OK] wrote best:    {args.out_best}")
    print(f"[INFO] power_available rows: {power_ok}/{len(summary)}")
    if power_ok > 0:
        miss_total = int(summary["power_missing_epochs"].fillna(0).sum())
        meas_total = int(summary["power_measured_epochs"].fillna(0).sum())
        print(f"[INFO] total measured epochs: {meas_total}, total missing epochs: {miss_total}")
        print(f"[INFO] fill_mode={args.fill_mode}, warmup_epochs={args.warmup_epochs}")

if __name__ == "__main__":
    main()

# python s3_result.py --root . --only-rank0 --fill-mode warmup_mean --warmup-epochs 20