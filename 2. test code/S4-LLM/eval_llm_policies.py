#!/usr/bin/env python3

import json
import time
import statistics
import csv
import os
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import requests
from openai import OpenAI


# ===================== 설정 =====================

DATASET_PATH = "policy_requests.json"
REPEATS_PER_PROMPT = 3
TOLERANCE_BANDS = [0.1, 0.2]

openai_client = OpenAI(api_key="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")

MODELS = {
    "codellama-7b": {
        "kind": "ollama",
        "name": "codellama:7b",
        "endpoint": "http://localhost:11434/api/chat",
    },
    "gpt-4o": {
        "kind": "openai",
        "name": "gpt-4o",
    },
    "gpt-5.1": {
        "kind": "openai",
        "name": "gpt-5.1",
    },
}


# ===================== 데이터 구조 =====================

@dataclass
class LambdaVec:
    time: float
    cost: float
    energy: float

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LambdaVec":
        return cls(float(d["time"]), float(d["cost"]), float(d["energy"]))

    def to_dict(self):
        return {"time": self.time, "cost": self.cost, "energy": self.energy}


@dataclass
class Sample:
    id: int
    request: str
    gt_lambda: LambdaVec


@dataclass
class TimingInfo:
    response_time: float
    ttfb: float


@dataclass
class ModelResult:
    predictions: Dict[int, List[LambdaVec]] = field(default_factory=dict)
    timings: Dict[int, List[TimingInfo]] = field(default_factory=dict)


# ===================== 유틸 =====================

def load_dataset(path: str) -> List[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        samples.append(Sample(
            id=int(item["id"]),
            request=item["request"],
            gt_lambda=LambdaVec.from_dict(item["lambda"])
        ))
    return samples


def build_prompt(user_request: str) -> str:
    return (
        "You are a policy generator for a multi-objective ML job scheduler.\n"
        "Return ONLY a JSON object with fields: time, cost, energy (0~1, sum=1).\n\n"
        f"User request: {user_request}"
    )


def parse_lambda_from_text(text: str) -> LambdaVec:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    start = text.find("{")
    end = text.rfind("}")
    text = text[start:end+1]
    return LambdaVec.from_dict(json.loads(text))


# ===================== 모델 호출 =====================

def call_openai_model(model: str, prompt: str):
    t_start = time.perf_counter()
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise policy generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
    )
    t_end = time.perf_counter()
    ttfb = t_end - t_start
    lam = parse_lambda_from_text(resp.choices[0].message.content)
    return lam, TimingInfo(t_end - t_start, ttfb)


def call_ollama_model(endpoint: str, model: str, prompt: str):
    t_start = time.perf_counter()
    resp = requests.post(endpoint, json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise policy generator."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    })
    t_end = time.perf_counter()
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    lam = parse_lambda_from_text(content)
    return lam, TimingInfo(t_end - t_start, t_end - t_start)


def query_model(model_key: str, prompt: str):
    cfg = MODELS[model_key]
    if cfg["kind"] == "openai":
        return call_openai_model(cfg["name"], prompt)
    else:
        return call_ollama_model(cfg["endpoint"], cfg["name"], prompt)


# ===================== 메트릭 계산 =====================

def argmax_key(lam: LambdaVec):
    d = lam.to_dict()
    return max(d, key=d.get)


def compute_directional_correctness(samples, result: ModelResult):
    total = correct = 0
    for s in samples:
        preds = result.predictions.get(s.id, [])
        if not preds:
            continue
        if argmax_key(preds[0]) == argmax_key(s.gt_lambda):
            correct += 1
        total += 1
    return correct / total if total else 0


def compute_tolerance_accuracy(samples, result, eps):
    hits = total = 0
    for s in samples:
        preds = result.predictions.get(s.id, [])
        if not preds:
            continue
        p = preds[0]
        g = s.gt_lambda
        for key in ["time", "cost", "energy"]:
            if abs(getattr(p, key) - getattr(g, key)) <= eps:
                hits += 1
            total += 1
    return hits / total if total else 0


def compute_output_consistency(samples, result):
    tstd = cstd = estd = []
    tstd = []
    cstd = []
    estd = []
    for s in samples:
        preds = result.predictions.get(s.id, [])
        if len(preds) <= 1:
            continue
        tstd.append(statistics.pstdev([p.time for p in preds]))
        cstd.append(statistics.pstdev([p.cost for p in preds]))
        estd.append(statistics.pstdev([p.energy for p in preds]))

    def avg(lst):
        return sum(lst)/len(lst) if lst else 0.0

    return {
        "time": avg(tstd),
        "cost": avg(cstd),
        "energy": avg(estd),
    }


def compute_latency_stats(result):
    rt = []
    ttfb = []
    for arr in result.timings.values():
        for t in arr:
            rt.append(t.response_time)
            ttfb.append(t.ttfb)

    if not rt:
        return {"avg":0,"p95":0,"ttfb":0}

    rt_sorted = sorted(rt)
    p95 = rt_sorted[int(len(rt_sorted)*0.95)-1]

    return {
        "avg": sum(rt)/len(rt),
        "p95": p95,
        "ttfb": sum(ttfb)/len(ttfb),
    }


# ===================== CSV 저장 =====================

def save_summary_csv(filename, samples, results):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "dir_correct",
            *(f"tol_acc_pm{eps}" for eps in TOLERANCE_BANDS),
            "cons_time", "cons_cost", "cons_energy",
            "lat_avg", "lat_p95", "lat_ttfb"
        ])
        for m, r in results.items():
            dir_acc = compute_directional_correctness(samples, r)
            tol = [compute_tolerance_accuracy(samples, r, eps) for eps in TOLERANCE_BANDS]
            cons = compute_output_consistency(samples, r)
            lat = compute_latency_stats(r)

            w.writerow([
                m,
                dir_acc,
                *tol,
                cons["time"], cons["cost"], cons["energy"],
                lat["avg"], lat["p95"], lat["ttfb"]
            ])


def save_predictions_csv(filename, results):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","sample_id","repeat","time","cost","energy"])
        for m, r in results.items():
            for sid, preds in r.predictions.items():
                for idx, p in enumerate(preds):
                    w.writerow([m, sid, idx, p.time, p.cost, p.energy])


def save_timings_csv(filename, results):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","sample_id","repeat","response_time","ttfb"])
        for m, r in results.items():
            for sid, arr in r.timings.items():
                for idx, t in enumerate(arr):
                    w.writerow([m, sid, idx, t.response_time, t.ttfb])


# ===================== 메인 =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="codellama-7b, gpt-4o, gpt-5.1, or all"
    )
    args = parser.parse_args()

    samples = load_dataset(DATASET_PATH)
    print(f"Loaded {len(samples)} samples.")

    # 실행할 모델 목록 선택
    if args.model == "all":
        target_models = list(MODELS.keys())
    else:
        if args.model not in MODELS:
            print(f"Unknown model: {args.model}")
            return
        target_models = [args.model]

    print("Models to evaluate:", target_models)

    results = {m: ModelResult() for m in target_models}

    for m in target_models:
        cfg = MODELS[m]
        print(f"\n=== Running {m} ===")
        mr = results[m]

        for s in samples:
            prompt = build_prompt(s.request)
            mr.predictions.setdefault(s.id, [])
            mr.timings.setdefault(s.id, [])

            for r in range(REPEATS_PER_PROMPT):
                lam, timing = query_model(m, prompt)
                mr.predictions[s.id].append(lam)
                mr.timings[s.id].append(timing)

                print(f"[{m}] id={s.id} rep={r} λ={lam.to_dict()} ({timing.response_time:.3f}s)")

    # CSV 저장
    save_summary_csv("llm_policy_metrics_summary.csv", samples, results)
    save_predictions_csv("llm_policy_predictions.csv", results)
    save_timings_csv("llm_policy_timings.csv", results)

    print("\n>> Results saved to CSV files.")


if __name__ == "__main__":
    main()

