from __future__ import annotations
import os, json, time

LOG_DIR = os.getenv("SCHED_LOG_DIR", "logs")
LOG_PATH = os.path.join(LOG_DIR, "scheduler.log")

os.makedirs(LOG_DIR, exist_ok=True)

def log_event(tag: str, payload):
    rec = {"ts": time.time(), "tag": tag, "payload": payload}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

