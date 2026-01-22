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
    return "clusterB" if "clusterb" in p else ("clusterA" if "clustera" in p else "unknown")

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
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, delimiter="\t")

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def pick_dt_col(df: pd.DataFrame) -> str | None:
    # power 측정 시간 우선
    for c in ["epoch_power_dt_sec", "total_time"]:
        if c in df.columns:
            return c
    return None

def get_power_series(df: pd.DataFrame) -> pd.Series | None:
    if "gpu_power_w" not in df.columns:
        return None
    return to_num(df["gpu_power_w"])

# -------------------------
# Fill power per epoch
# -------------------------
def fill_power_per_epoch(tmp: pd.DataFrame, epoch_min: int, warmup_epochs: int, fill_mode: str) -> tuple[pd.DataFrame, bool, float, int, int]:
    """
    tmp must already be filtered to epoch range and have 'epoch', dt_col, 'gpu_power_w'
    Returns: (tmp_filled, power_any, fill_power, measured_epochs, missing_epochs)
    """
    tmp = tmp.copy()
    tmp["gpu_power_w"] = to_num(tmp["gpu_power_w"])
    measured = tmp.dropna(subset=["gpu_power_w"])
    power_any = not measured.empty
    if not power_any:
        return (tmp, False, float("nan"), 0, int(tmp["gpu_power_w"].isna().sum()))

    if fill_mode == "warmup_mean":
        w_hi = epoch_min + max(warmup_epochs, 1) - 1
        warm = measured[(measured["epoch"] >= epoch_min) & (measured["epoch"] <= w_hi)]
        fill_power = float(warm["gpu_power_w"].mean()) if not warm.empty else float(measured["gpu_power_w"].mean())
    elif fill_mode == "global_mean":
        fill_power = float(measured["gpu_power_w"].mean())
    elif fill_mode == "epoch1":
        e1 = measured[measured["epoch"] == epoch_min]
        fill_power = float(e1["gpu_power_w"].iloc[0]) if not e1.empty else float(measured["gpu_power_w"].mean())
    else:
        raise ValueError(f"Unknown fill_mode={fill_mode}")

    missing_mask = tmp["gpu_power_w"].isna()
    missing_epochs = int(missing_mask.sum())
    measured_epochs = int((~missing_mask).sum())

    tmp.loc[missing_mask, "gpu_power_w"] = fill_power
    return (tmp, True, fill_power, measured_epochs, missing_epochs)

# -------------------------
# Infer power mode per cluster (per_gpu vs node_sum)
# -------------------------
def infer_power_mode(summary_power: pd.DataFrame) -> dict:
    """
    summary_power columns must include: cluster, gpu_count, mean_power_w
    Heuristic:
      - compare mean_power(2GPU)/mean_power(1GPU)
      - if ratio ~ 2 => node_sum
      - if ratio ~ 1 => per_gpu
    """
    out = {}
    for cluster, g in summary_power.groupby("cluster"):
        one = g[g["gpu_count"] == 1]["mean_power_w"].dropna()
        two = g[g["gpu_count"] == 2]["mean_power_w"].dropna()
        if one.empty or two.empty:
            out[cluster] = "unknown"
            continue
        r = float(two.median() / one.median())
        # thresholds: allow noise
        if r >= 1.6:
            out[cluster] = "node_sum"
        elif r <= 1.2:
            out[cluster] = "per_gpu"
        else:
            out[cluster] = "unknown"
    return out

# -------------------------
# Compute metrics for one file (energy integrated)
# -------------------------
def compute_metrics_for_file(
    csv_path: str,
    epoch_min: int,
    epoch_max: int,
    rateA: float,
    rateB: float,
    warmup_epochs: int,
    fill_mode: str,
    power_mode: str,   # "per_gpu" | "node_sum" | "unknown"
) -> dict | None:

    df = read_csv_auto(csv_path)
    if "epoch" not in df.columns:
        return None

    dt_col = pick_dt_col(df)
    if dt_col is None:
        return None

    df = df.copy()
    df["epoch"] = to_num(df["epoch"])
    df = df.dropna(subset=["epoch"])
    df = df[(df["epoch"] >= epoch_min) & (df["epoch"] <= epoch_max)]
    if df.empty:
        return None

    df[dt_col] = to_num(df[dt_col])
    df = df.dropna(subset=[dt_col])
    df = df[df[dt_col] > 0]
    if df.empty:
        return None

    cluster = infer_cluster_from_path(csv_path)
    gpu_count = infer_gpu_count_from_filename(os.path.basename(csv_path))

    # JCT: sum of epoch durations (dt_col)
    jct_sec = float(df[dt_col].sum())
    if jct_sec <= 0:
        return None

    # cost
    rate = rateA if cluster == "clusterA" else rateB if cluster == "clusterB" else float("nan")
    cost_usd = float((jct_sec / 3600.0) * gpu_count * rate) if not math.isnan(rate) else float("nan")

    # energy integration
    power_any = False
    energy_wh_total = float("nan")
    energy_wh_per_gpu = float("nan")
    fill_power = float("nan")
    meas_ep = 0
    miss_ep = 0

    if "gpu_power_w" in df.columns:
        tmp = df[["epoch", dt_col, "gpu_power_w"]].copy()
        tmp["epoch"] = to_num(tmp["epoch"])
        tmp = tmp.dropna(subset=["epoch"])
        tmp["gpu_power_w"] = to_num(tmp["gpu_power_w"])

        tmp_f, power_any, fill_power, meas_ep, miss_ep = fill_power_per_epoch(
            tmp.rename(columns={dt_col: "dt_sec"}), epoch_min, warmup_epochs, fill_mode
        )
        if power_any:
            # Σ(P * dt) gives:
            #  - node_sum mode: total node GPU power already summed -> that's total energy
            #  - per_gpu mode: per-GPU average -> multiply by gpu_count to get total energy
            pdot = float((tmp_f["gpu_power_w"] * tmp_f["dt_sec"]).sum())  # W*sec
            base_wh = pdot / 3600.0  # Wh
            if power_mode == "node_sum":
                energy_wh_total = base_wh
                energy_wh_per_gpu = base_wh / max(gpu_count, 1)
            elif power_mode == "per_gpu":
                energy_wh_per_gpu = base_wh
                energy_wh_total = base_wh * gpu_count
            else:
                # unknown: store both estimates (conservative choice: assume node_sum to avoid overcount)
                # but also record alt for debugging
                energy_wh_total = base_wh
                energy_wh_per_gpu = base_wh / max(gpu_count, 1)

    gpu_seconds = jct_sec * gpu_count

    edp = (energy_wh_total * jct_sec) if power_any and (energy_wh_total > 0) else float("nan")
    ed2p = (energy_wh_total * (jct_sec ** 2)) if power_any and (energy_wh_total > 0) else float("nan")

    # final metrics
    acc = float(df["accuracy"].dropna().iloc[-1]) if "accuracy" in df.columns and df["accuracy"].dropna().size else float("nan")
    top5 = float(df["top5_accuracy"].dropna().iloc[-1]) if "top5_accuracy" in df.columns and df["top5_accuracy"].dropna().size else float("nan")

    return {
        "csv_path": csv_path,
        "run": short_path(csv_path),
        "cluster": cluster,
        "gpu_count": gpu_count,
        "dt_col_used": dt_col,
        "jct_sec": jct_sec,
        "jct_min": jct_sec / 60.0,
        "cost_usd": cost_usd,
        "gpu_seconds": gpu_seconds,
        "power_mode_used": power_mode,
        "power_available": bool(power_any),
        "power_fill_mode": fill_mode,
        "power_fill_w": fill_power if power_any else float("nan"),
        "power_measured_epochs": meas_ep,
        "power_missing_epochs": miss_ep,
        "energy_wh_total": energy_wh_total,
        "energy_wh_per_gpu": energy_wh_per_gpu,
        "edp": edp,
        "ed2p": ed2p,
        "final_accuracy": acc,
        "final_top5_accuracy": top5,
    }

def pick_best(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({"objective": "time_min_jct", **summary.loc[summary["jct_sec"].idxmin()].to_dict()})

    cost_df = summary[summary["cost_usd"].notna()]
    if not cost_df.empty:
        rows.append({"objective": "cost_min_usd", **cost_df.loc[cost_df["cost_usd"].idxmin()].to_dict()})

    e_df = summary[summary["energy_wh_total"].notna() & (summary["energy_wh_total"] > 0)]
    if not e_df.empty:
        best_e = e_df.loc[e_df["energy_wh_total"].idxmin()].to_dict()
        best_e["energy_objective_used"] = "energy_wh_total"
        rows.append({"objective": "energy_abs_min_wh", **best_e})

    edp_df = summary[summary["edp"].notna() & (summary["edp"] > 0)]
    if not edp_df.empty:
        best_edp = edp_df.loc[edp_df["edp"].idxmin()].to_dict()
        best_edp["energy_objective_used"] = "EDP=E*T"
        rows.append({"objective": "energy_eff_min_edp", **best_edp})

    ed2p_df = summary[summary["ed2p"].notna() & (summary["ed2p"] > 0)]
    if not ed2p_df.empty:
        best_ed2p = ed2p_df.loc[ed2p_df["ed2p"].idxmin()].to_dict()
        best_ed2p["energy_objective_used"] = "ED2P=E*T^2"
        rows.append({"objective": "energy_eff_min_ed2p", **best_ed2p})

    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--epoch-min", type=int, default=1)
    ap.add_argument("--epoch-max", type=int, default=20)
    ap.add_argument("--only-rank0", action="store_true")
    ap.add_argument("--max-gpu", type=int, default=2)
    ap.add_argument("--rateA", type=float, default=1.0)
    ap.add_argument("--rateB", type=float, default=1.5)
    ap.add_argument("--fill-mode", type=str, default="warmup_mean", choices=["warmup_mean","global_mean","epoch1"])
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--out-summary", type=str, default="summary_maxgpu.csv")
    ap.add_argument("--out-best", type=str, default="best_maxgpu.csv")
    args = ap.parse_args()

    # 1) 먼저 power_mode 추정용으로 mean power를 뽑는다 (max_gpu까지)
    csv_files = glob.glob(os.path.join(args.root, "**", "*.csv"), recursive=True)
    power_rows = []
    for p in csv_files:
        if args.only_rank0 and ("rank0" not in os.path.basename(p)):
            continue
        g = infer_gpu_count_from_filename(os.path.basename(p))
        if g > args.max_gpu:
            continue

        df = read_csv_auto(p)
        if "epoch" not in df.columns or "gpu_power_w" not in df.columns:
            continue
        df["epoch"] = to_num(df["epoch"])
        dt_col = pick_dt_col(df)
        if dt_col is None:
            continue
        df = df.dropna(subset=["epoch"])
        df = df[(df["epoch"] >= args.epoch_min) & (df["epoch"] <= args.epoch_max)]
        if df.empty:
            continue
        pw = to_num(df["gpu_power_w"]).dropna()
        if pw.empty:
            continue
        power_rows.append({
            "cluster": infer_cluster_from_path(p),
            "gpu_count": g,
            "mean_power_w": float(pw.mean()),
        })

    power_mode_map = infer_power_mode(pd.DataFrame(power_rows)) if power_rows else {}
    # 기본은 unknown
    def get_mode(cluster: str) -> str:
        return power_mode_map.get(cluster, "unknown")

    # 2) 본 계산
    rows = []
    skipped_gpu = 0
    for p in csv_files:
        if args.only_rank0 and ("rank0" not in os.path.basename(p)):
            continue
        g = infer_gpu_count_from_filename(os.path.basename(p))
        if g > args.max_gpu:
            skipped_gpu += 1
            continue

        cluster = infer_cluster_from_path(p)
        m = compute_metrics_for_file(
            p,
            args.epoch_min,
            args.epoch_max,
            args.rateA,
            args.rateB,
            args.warmup_epochs,
            args.fill_mode,
            power_mode=get_mode(cluster),
        )
        if m:
            rows.append(m)

    if not rows:
        raise SystemExit("No valid CSV after filtering.")

    summary = pd.DataFrame(rows).sort_values(["cluster","gpu_count","jct_sec"]).reset_index(drop=True)
    summary.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    best = pick_best(summary)
    best.to_csv(args.out_best, index=False, encoding="utf-8-sig")

    print(f"[INFO] power_mode_map={power_mode_map}")
    print(f"[OK] wrote {args.out_summary} rows={len(summary)}")
    print(f"[OK] wrote {args.out_best} rows={len(best)}")
    print(f"[INFO] max_gpu={args.max_gpu}, skipped_files={skipped_gpu}")

if __name__ == "__main__":
    main()