import time
import random
import csv
import json
from datetime import datetime
import argparse
import requests
from statistics import mean, pstdev
from math import floor

# Defaults (explicit, no env)
DEFAULT_POLLUX_URL = "http://localhost:8000/submit_job"
DEFAULT_JOBS     = 50
DEFAULT_SEED     = 20251106
DEFAULT_EPOCHS   = 20            # shorter epochs like prior work
DEFAULT_MEAN_GAP = 300.0         # seconds, Poisson arrivals (exp inter-arrival)
DEFAULT_TIMEOUT  = 10.0          # HTTP seconds

# Target world-size distribution (will be renormalized per model’s allowed scales)
WORLD_SIZE_MIX = {1: 0.50, 2: 0.35, 4: 0.15}

# Target job-type ratio (Vision:NLP:Speech = 5:3:2)
TYPE_RATIO = {"vision": 5, "nlp": 3, "speech": 2}

STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

# ==============================
# Job catalog (J1–J6)
#   - batch_choices are PER-GPU choices; if g>1, we simply keep per-GPU batch the same
#     (global effective batch = per_gpu_batch * g; adjust here if you prefer scaling).
# ==============================
CATALOG = {
    # Vision (4)
    "J1": {
        "job_type": "vision",
        "model": "ResNet-50",
        "dataset": "CIFAR-10",
        "scales": [1, 2, 4],
        "batch_choices": [32, 64, 128]
    },
    "J2": {
        "job_type": "vision",
        "model": "EfficientNetV2-S",
        "dataset": "MNIST",
        "scales": [1, 2],
        "batch_choices": [64]
    },
    "J3": {
        "job_type": "vision",
        "model": "DenseNet-121",
        "dataset": "TinyImageNet",
        "scales": [4],
        "batch_choices": [64]
    },
    "J4": {
        "job_type": "vision",
        "model": "ResNet-18",
        "dataset": "CIFAR-100",
        "scales": [1, 2, 4],
        "batch_choices": [32, 64, 128]
    },
    # NLP (1)
    "J5": {
        "job_type": "nlp",
        "model": "DistilBERT",
        "dataset": "SST-2",
        "scales": [1, 2],
        "batch_choices": [16, 32, 64]
    },
    # Speech (1)
    "J6": {
        "job_type": "speech",
        "model": "DeepSpeech2",
        "dataset": "ARCTIC",
        "scales": [1, 2, 4],
        "batch_choices": [8, 16, 32]
    },
}

# Group keys by type for ratio sampling
TYPE_TO_KEYS = {"vision": [], "nlp": [], "speech": []}
for k, v in CATALOG.items():
    TYPE_TO_KEYS[v["job_type"]].append(k)

# ==============================
# Example user intents (free-form, reusable later)
# ==============================
USER_INTENTS = [
    "I need this training to finish as fast as possible. Prioritize performance and speed over everything else.",
    "Please keep the cost reasonable while still maintaining decent speed.",
    "Try to share resources fairly with other users, but don’t delay my job too much.",
    "This job is for testing only — minimal cost and stable runtime are more important than raw speed.",
    "Focus on efficiency. Use GPUs effectively but avoid unnecessary power consumption.",
    "Balance between speed and cost. I don’t need the fastest result, just good efficiency overall.",
    "Prefer reliability and consistency. I’d rather not have the job preempted or rescheduled.",
    "Maximize GPU utilization for this run — I’m fine if it uses more energy temporarily.",
    "Please minimize queue waiting time. Start as soon as a slot is available.",
    "This is a short experiment, so give it quick access to GPUs, even small ones.",
]

def _sample_world_size(allowed: list[int], rng: random.Random) -> int:
    """Sample a world_size respecting the global WORLD_SIZE_MIX but renormalized to allowed."""
    weights = []
    for g in allowed:
        weights.append(max(0.0, float(WORLD_SIZE_MIX.get(g, 0.0))))
    s = sum(weights)
    if s <= 0:
        # fallback uniform over allowed
        return rng.choice(allowed)
    probs = [w / s for w in weights]
    r = rng.random()
    acc = 0.0
    for g, p in zip(allowed, probs):
        acc += p
        if r <= acc:
            return g
    return allowed[-1]

def _sample_batch_size(batch_choices: list[int], world_size: int, rng: random.Random) -> int:
    """
    Pick a per-GPU batch size from choices. By default, we keep per-GPU batch
    the same even if world_size>1 (so global batch scales linearly).
    If you prefer to cap global batch, modify here.
    """
    b = rng.choice(batch_choices)
    return b  # per-GPU batch; global_effective = b * world_size

def _quota_counts(n_jobs: int, type_ratio: dict[str, int]) -> dict[str, int]:
    """
    Compute integer counts per job_type according to the given ratio.
    Ensures the sum equals n_jobs by distributing remainders deterministically.
    """
    total_ratio = sum(type_ratio.values())
    raw = {t: n_jobs * r / total_ratio for t, r in type_ratio.items()}
    base = {t: floor(x) for t, x in raw.items()}
    assigned = sum(base.values())
    remainder = n_jobs - assigned
    # Distribute remaining jobs by largest fractional part (stable order by key)
    fracs = sorted(((t, raw[t] - base[t]) for t in type_ratio.keys()),
                   key=lambda x: (-x[1], x[0]))
    for i in range(remainder):
        base[fracs[i % len(fracs)][0]] += 1
    return base

def build_job_plan(n_jobs: int, epochs: int, rng: random.Random):
    """
    Generate a list of jobs that satisfies the Vision:NLP:Speech ratio (5:3:2 by default),
    with varied intents, GPU scales, and per-GPU batch sizes. Order is shuffled at the end.
    """
    quotas = _quota_counts(n_jobs, TYPE_RATIO)

    jobs = []
    for job_type, count in quotas.items():
        keys = TYPE_TO_KEYS[job_type]
        if not keys:
            continue
        for _ in range(count):
            jkey = rng.choice(keys)
            meta = CATALOG[jkey]
            model, dataset, scales, batch_choices = meta["model"], meta["dataset"], meta["scales"], meta["batch_choices"]
            world_size = _sample_world_size(scales, rng)
            batch_size = _sample_batch_size(batch_choices, world_size, rng)
            user_prompt = rng.choice(USER_INTENTS)

            jobs.append({
                "job_tag": jkey,
                "job_type": job_type,
                "model": model,
                "dataset": dataset,
                "world_size": world_size,
                "batch_size": batch_size,     # per-GPU batch
                "epochs": epochs,
                "exact_g": False,             # can flip if you later need pinned g
                "user_request": user_prompt,
            })

    # Shuffle to mix types while keeping reproducibility
    rng.shuffle(jobs)
    return jobs

import uuid

def submit_job(fifo_url: str, job: dict, timeout_s: float = DEFAULT_TIMEOUT):
    # Pollux 쪽은 job_id를 클라이언트가 넣어줘야 함
    job_id = f"job-{uuid.uuid4().hex[:8]}"

    payload = {
        "job_id": job_id,
        "model_name": job["model"],
        "dataset": job["dataset"],
        "epochs": job["epochs"],
        "batch_size_per_gpu": job["batch_size"],  # per-GPU batch 그대로 사용
        "learning_rate": 1e-3,                   # 고정값 or 타입별로 바꾸고 싶으면 여기서 로직 추가
    }

    t0 = time.time()
    try:
        resp = requests.post(fifo_url, json=payload, timeout=timeout_s)
        t1 = time.time()
    except Exception as e:
        t1 = time.time()
        return {
            "ok": False,
            "elapsed_ms": int((t1 - t0) * 1000),
            "status_code": -1,
            "error": f"{type(e).__name__}: {e}",
        }

    if resp.status_code != 200:
        return {
            "ok": False,
            "elapsed_ms": int((t1 - t0) * 1000),
            "status_code": resp.status_code,
            "error": resp.text,
        }

    data = resp.json()
    return {
        "ok": True,
        "elapsed_ms": int((t1 - t0) * 1000),
        "job_id": data.get("job_id", job_id),
        "cluster": None,      
        "position": None,      
        "free_slots": None,
    }

def main():
    parser = argparse.ArgumentParser(description="S1: POLLUX mixed workload generator (Poisson; ratio-controlled types).")
    parser.add_argument("--url",        type=str, default=DEFAULT_POLLUX_URL, help="POLLUX submit URL")
    parser.add_argument("--jobs",       type=int, default=DEFAULT_JOBS,     help="number of jobs to submit")
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED,     help="random seed")
    parser.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS,   help="epochs for each job")
    parser.add_argument("--mean-gap",   type=float, default=DEFAULT_MEAN_GAP, help="mean inter-arrival gap (s) for exponential distribution")
    parser.add_argument("--timeout",    type=float, default=DEFAULT_TIMEOUT,  help="HTTP timeout seconds")
    # Optional burst for transient-overload probing
    parser.add_argument("--burst-first-k", type=int, default=0, help="send first K jobs as a short burst (gap=burst-gap)")
    parser.add_argument("--burst-gap",     type=float, default=5.0, help="gap (sec) between burst submissions")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    plan = build_job_plan(args.jobs, args.epochs, rng)

    # Output files (tag with key params)
    tag = f"{STAMP}_N{args.jobs}_seed{args.seed}_gap{int(args.mean_gap)}"
    out_csv   = f"S1_POLLUX_submissions_{tag}.csv"
    plan_json = f"S1_POLLUX_plan_{tag}.json"

    # Save exact plan (reproducible)
    meta = {
        "url": args.url,
        "jobs": args.jobs,
        "seed": args.seed,
        "epochs": args.epochs,
        "arrival": {"process": "poisson", "mean_gap_s": args.mean_gap,
                    "burst_first_k": args.burst_first_k, "burst_gap_s": args.burst_gap},
        "world_size_mix": WORLD_SIZE_MIX,
        "type_ratio": TYPE_RATIO,
        "catalog": CATALOG,
        "generated_at": STAMP,
    }
    with open(plan_json, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "jobs": plan}, f, ensure_ascii=False, indent=2)

    # CSV header
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "submit_time",
            "index",
            "job_tag",
            "job_type",
            "model",
            "dataset",
            "world_size",
            "batch_size_per_gpu",
            "exact_g",
            "job_id",
            "cluster",
            "queue_position",
            "submit_elapsed_ms",
            "status",
            "error",
            "user_request",
            "inter_arrival_s",
        ])

    print(f"[S1] URL={args.url} | jobs={args.jobs} | seed={args.seed} | epochs={args.epochs} | mean_gap={args.mean_gap:.1f}s")
    inter_arrivals = []

    # Submission loop (sleep before each submission to maintain stationarity)
    for i, job in enumerate(plan, 1):
        if args.burst_first_k > 0 and i <= args.burst_first_k:
            gap = max(0.0, float(args.burst_gap))
        else:
            rate = 1.0 / max(1e-9, args.mean_gap)
            gap = rng.expovariate(rate)

        time.sleep(gap)
        inter_arrivals.append(gap)

        result = submit_job(args.url, job, timeout_s=args.timeout)

        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                i,
                job["job_tag"],
                job["job_type"],
                job["model"],
                job["dataset"],
                job["world_size"],
                job["batch_size"],
                job["exact_g"],
                result.get("job_id"),
                result.get("cluster"),
                result.get("position"),
                result.get("elapsed_ms"),
                "ok" if result.get("ok") else "fail",
                result.get("error", ""),
                job["user_request"],
                f"{gap:.3f}",
            ])

        if result.get("ok"):
            print(f"[{i:02d}/{args.jobs}] submitted {result['job_id']} "
                  f"| {job['job_type']} | {job['model']} / {job['dataset']} "
                  f"| g={job['world_size']} | perGPU batch={job['batch_size']} "
                  f"| cluster={result.get('cluster')} | queue pos={result.get('position')} "
                  f"| Δt={gap:.2f}s")
        else:
            print(f"[{i:02d}/{args.jobs}] submission FAILED "
                  f"| {job['job_type']} | {job['model']} / {job['dataset']} "
                  f"| g={job['world_size']} | perGPU batch={job['batch_size']} "
                  f"| Δt={gap:.2f}s | err={result.get('error')}")

    # Inter-arrival summary (sanity check)
    if inter_arrivals:
        ia_mean = mean(inter_arrivals)
        ia_std  = pstdev(inter_arrivals) if len(inter_arrivals) > 1 else 0.0
        ia_min  = min(inter_arrivals)
        ia_max  = max(inter_arrivals)
        print("\n[S1] Inter-arrival (s) summary:")
        print(f"  mean={ia_mean:.3f}, std={ia_std:.3f}, min={ia_min:.3f}, max={ia_max:.3f}")

    print(f"\n[S1] Done.\n  Plan saved: {plan_json}\n  Log saved:  {out_csv}")

if __name__ == "__main__":
    main()

