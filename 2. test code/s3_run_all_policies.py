#!/usr/bin/env python3
# s3_run_all_policies.py
#
# S3 – SLO-aware Scheduling (FIXED λ 실험용)
# - 정책(λ)을 미리 고정해서 /submit_fixed로 job 제출
# - 각 정책별로 num_jobs개를 "1개씩 순차 실행"
# - 각 job은 intent 텍스트를 달고 들어가고,
#   policy_text에 [mode_name] prefix를 붙여서 나중에 분석 시 구분 가능

import time
import argparse
import requests
import sys

# S3 시나리오에서 사용할 user_request 텍스트들
INTENTS = [
    "최대한 빨리 끝나면 좋겠습니다.",          # TIME_PREF
    "비용을 줄이는 게 더 중요합니다.",        # COST_PREF
    "전력/에너지 부담을 줄이고 싶습니다.",    # ENERGY_PREF
    "속도/비용/에너지를 적당히 균형 맞추고 싶습니다.",  # BALANCE_PREF
]

# ★ 진짜로 끝난 상태만 terminal로 취급
TERMINAL_STATES = {"FINISHED", "FAILED", "CANCELLED", "COMPLETED"}


# ------------------------------------------------
# 공통 helper
# ------------------------------------------------
def submit_one_job(
    base_url: str,
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    lambda_time: float,
    lambda_cost: float,
    lambda_energy: float,
    cluster_id: str | None,
    user_request_text: str,
    mode_tag: str,
) -> str:
    """하나의 job을 /submit_fixed로 제출하고 job_id를 반환."""
    url = f"{base_url}/submit_fixed"

    payload = {
        "model_name": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "batch_size_per_gpu": batch_size,
        "lambda_time": lambda_time,
        "lambda_cost": lambda_cost,
        "lambda_energy": lambda_energy,
        # 나중에 분석 시 policy_text로 policy 모드 + intent 둘 다 구분
        "policy_text": f"[{mode_tag}] {user_request_text}",
    }

    if cluster_id:
        payload["cluster_id"] = cluster_id

    print(f"[SUBMIT] POST {url}")
    print(f"         payload = {payload}")
    try:
        resp = requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] submit failed: {e}")
        sys.exit(1)

    if resp.status_code != 200:
        print(f"[ERROR] submit HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    job_id = data.get("job_id")
    if not job_id:
        print(f"[ERROR] submit response has no job_id: {data}")
        sys.exit(1)

    print(
        f"[SUBMIT] job_id={job_id}, cluster={data.get('cluster_id')}, "
        f"g_target={data.get('g_target')}, status={data.get('status')}"
    )
    return job_id


def wait_for_job(
    base_url: str,
    job_id: str,
    poll_interval: int = 15,
    max_wait_sec: int = 4 * 3600,
) -> str:
    url = f"{base_url}/job_status/{job_id}"
    print(f"[WAIT] job_id={job_id} → GET {url}")

    start = time.time()
    while True:
        if time.time() - start > max_wait_sec:
            print(f"[WARN] job_id={job_id} wait timeout (> {max_wait_sec} sec), stop waiting")
            return "TIMEOUT"

        try:
            resp = requests.get(url, timeout=5)
        except Exception as e:
            print(f"[WARN] job_status request failed: {e}")
            time.sleep(poll_interval)
            continue

        if resp.status_code != 200:
            print(f"[WARN] job_status HTTP {resp.status_code}: {resp.text}")
            time.sleep(poll_interval)
            continue

        data = resp.json()
        raw_status = str(data.get("status", "UNKNOWN"))
        status = raw_status.upper()
        extra = {k: v for k, v in data.items() if k not in ("status",)}
        print(f"[STATUS] job_id={job_id}, status={status}, extra={extra}")

        # 1) COMPLETED / FINISHED / FAILED / CANCELLED → 종료
        if status in TERMINAL_STATES:
            print(f"[DONE] job_id={job_id} finished with status={status}")
            return status

        # 2) NOT_FOUND 인데 어느 정도 기다린 뒤면, 이미 정리된 것으로 보고 종료
        if status == "NOT_FOUND" and (time.time() - start) > 60:
            print(f"[DONE] job_id={job_id} NOT_FOUND after 60s, treat as completed")
            return "NOT_FOUND"

        # 그 외(RUNNING 등)는 계속 기다림
        time.sleep(poll_interval)

def run_policy_round(
    base_url: str,
    cluster_id: str,
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    lam_t: float,
    lam_c: float,
    lam_e: float,
    num_jobs: int,
    sleep_sec: int,
    poll_interval: int,
    mode_name: str,
):
    """
    특정 λ 설정으로 num_jobs개를 '1개씩 순차 실행'하는 라운드.
    - INTENTS를 라운드로 돌리면서 TIME_PREF / COST_PREF / ENERGY_PREF / BALANCE_PREF 섞어서 진행
    """
    print("\n" + "=" * 60)
    print(f"[MODE] {mode_name}  λ = (time={lam_t}, cost={lam_c}, energy={lam_e})")
    print("=" * 60 + "\n")

    for i in range(num_jobs):
        intent = INTENTS[i % len(INTENTS)]
        print(f"\n==== [MODE {mode_name}] RUN {i+1}/{num_jobs} intent='{intent}' ====")

        job_id = submit_one_job(
            base_url=base_url,
            model_name=model_name,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lambda_time=lam_t,
            lambda_cost=lam_c,
            lambda_energy=lam_e,
            cluster_id=cluster_id,
            user_request_text=intent,
            mode_tag=mode_name,
        )

        status = wait_for_job(
            base_url=base_url,
            job_id=job_id,
            poll_interval=poll_interval,
        )

        print(f"[MODE {mode_name}] RUN {i+1} job_id={job_id} finished with status={status}")
        if i != num_jobs - 1:
            print(f"[SLEEP] {sleep_sec} seconds before next job (same mode)...")
            time.sleep(sleep_sec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8082", help="Scheduler base URL")
    p.add_argument("--cluster_id", default="clusterB", help="실험에 사용할 클러스터 (예: clusterA, clusterB)")
    p.add_argument("--num_jobs", type=int, default=12, help="각 정책별 job 개수 (기본: 12)")
    p.add_argument("--model_name", default="ResNet-50")
    p.add_argument("--dataset", default="CIFAR-10")
    p.add_argument("--epochs", type=int, default=10, help="S3용 epoch 수 (기본: 10)")
    p.add_argument("--batch_size", type=int, default=32, help="batch_size_per_gpu (기본: 32)")
    p.add_argument("--sleep_sec", type=int, default=10, help="job 사이 쉬는 시간 (기본: 10s)")
    p.add_argument("--poll_interval", type=int, default=15, help="job_status 폴링 주기 (기본: 15s)")
    args = p.parse_args()

    base_url = args.host.rstrip("/")

    print("========== S3 Fixed Policies Runner (ALL MODES) ==========")
    print(f"base_url   = {base_url}")
    print(f"cluster_id = {args.cluster_id}")
    print(f"model      = {args.model_name}, dataset={args.dataset}")
    print(f"epochs     = {args.epochs}, batch_size={args.batch_size}")
    print(f"num_jobs   (per mode) = {args.num_jobs}")
    print(f"sleep_sec  = {args.sleep_sec}, poll_interval = {args.poll_interval}")
    print("==========================================================\n")

    # 5개 정책 정의
    policies = [
        ("time_only",   1.0, 0.0, 0.0),
        ("cost_only",   0.0, 1.0, 0.0),
        ("energy_only", 0.0, 0.0, 1.0),
        ("time_cost",   0.5, 0.5, 0.0),
        ("cost_energy", 0.0, 0.5, 0.5),
    ]

    for mode_name, lam_t, lam_c, lam_e in policies:
        run_policy_round(
            base_url=base_url,
            cluster_id=args.cluster_id,
            model_name=args.model_name,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lam_t=lam_t,
            lam_c=lam_c,
            lam_e=lam_e,
            num_jobs=args.num_jobs,
            sleep_sec=args.sleep_sec,
            poll_interval=args.poll_interval,
            mode_name=mode_name,
        )

    print("\n=== All modes done (time-only, cost-only, energy-only, time+cost, cost+energy) ===")


if __name__ == "__main__":
    main()

