# global_server.py (422 Unprocessable Entity 최종 수정)

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import httpx  # httpx 임포트
import time
import threading
import itertools # 포트 번호 생성을 위해 import
import logging
from typing import Optional, Dict, List, Any
from collections import defaultdict
import argparse
import asyncio
import os

# [POLLUX 통합] Pollux 스케줄러 및 헬퍼 클래스 임포트
from pollux_scheduler import PolluxScheduler, JobInfo, NodeInfo, GoodputFunction, SpeedupFunction

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (GlobalServer) %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

app = FastAPI()

# --- 1. 글로벌 상태 관리 ---

GLOBAL_SERVER_ADDRESS = None # 서버 자신의 주소 (main에서 설정됨)
MASTER_PORT_COUNTER = itertools.count(20000) # 고유 포트 할당용

NODE_REGISTRY = {
    "gpu_node_0": {"ip": "127.0.0.1", "agent_port": 8001, "gpu_id": 0, "status": "idle", "current_job_id": None},
    "gpu_node_1": {"ip": "127.0.0.1", "agent_port": 8002, "gpu_id": 1, "status": "idle", "current_job_id": None},
    "gpu_node_2": {"ip": "127.0.0.1", "agent_port": 8003, "gpu_id": 2, "status": "idle", "current_job_id": None},
    "gpu_node_3": {"ip": "127.0.0.1", "agent_port": 8004, "gpu_id": 3, "status": "idle", "current_job_id": None},
}
NODE_REGISTRY_LOCK = threading.Lock()

JOB_QUEUE = [
    {"job_id": "densenet_job_1", "script_path": "/home/ubuntu216/SUN/DDP_MNIST_EfficientNetV2.py", "model_name": "densenet121"},
    {"job_id": "resnet_job_1", "script_path": "/home/ubuntu216/SUN/DDP_MNIST_MobileNetV2.py", "model_name": "resnet50"},
    {"job_id": "resnet_job_1", "script_path": "/home/ubuntu216/SUN/DDP_MNIST_MobileNetV2.py", "model_name": "resnet50"},
    {"job_id": "resnet_job_1", "script_path": "/home/ubuntu216/SUN/DDP_MNIST_MobileNetV2.py", "model_name": "resnet50"},
]
JOB_QUEUE_LOCK = threading.Lock()

ACTIVE_JOBS = {}
ACTIVE_JOBS_LOCK = threading.Lock()

JOB_PROFILING_DB = {
    "densenet121": {
        "t_grad": 0.15,
        "t_sync": 0.08,
        "gns_slope": 0.001
    },
    "resnet50": {
        "t_grad": 0.10,
        "t_sync": 0.05,
        "gns_slope": 0.002
    },
    "default": { # DB에 없는 잡은 기본값 사용
        "t_grad": 0.2,
        "t_sync": 0.1,
        "gns_slope": 0.001
    }
}

POLLUX_SCHEDULER = PolluxScheduler()


# --- 2. API 데이터 모델 ---

# [수정] 422 오류 해결을 위해 Pydantic 모델의 키 이름을
# 님의 학습 스크립트(CIFAR100_Densenet121.py)가 보내는 키 이름과 일치시킵니다.

class JobProgressReport(BaseModel):
    job_id: str
    current_step: int
    epoch: int  # 'current_epoch' -> 'epoch'
    total_epochs: int
    loss: float # 'latest_loss' -> 'loss'
    accuracy: float # 'latest_accuracy' -> 'accuracy'

class JobCheckpointReport(BaseModel):
    job_id: str
    epoch: int  # 'current_epoch' -> 'epoch'
    total_epochs: int
    loss: float # 'latest_loss' -> 'loss'
    accuracy: float # 'latest_accuracy' -> 'accuracy'
    path: str # 'checkpoint_path' -> 'path'

class JobStopReport(BaseModel):
    job_id: str

class JobSubmitRequest(BaseModel):
    job_id: str
    script_path: str 
    model_name: str  


# --- 3. 핵심 스케줄링 로직 (Pollux 통합) ---

def _get_available_nodes() -> List[str]:
    with NODE_REGISTRY_LOCK:
        return [node_id for node_id, info in NODE_REGISTRY.items() if info["status"] == "idle"]

def _set_node_status(node_ids: List[str], status: str, job_id: Optional[str]):
    with NODE_REGISTRY_LOCK:
        for node_id in node_ids:
            if node_id in NODE_REGISTRY:
                NODE_REGISTRY[node_id]["status"] = status
                NODE_REGISTRY[node_id]["current_job_id"] = job_id
            else:
                log.warning(f"Attempted to set status for unknown node: {node_id}")

async def _launch_job_on_nodes(job_config: Dict, assigned_nodes: List[str]):
    job_id = job_config["job_id"]
    world_size = len(assigned_nodes)
    
    if world_size == 0:
        log.warning(f"[{job_id}] No nodes assigned, cannot launch.")
        return
        
    log.info(f"[{job_id}] Launching on {world_size} nodes: {assigned_nodes}")

    master_node_id = assigned_nodes[0]
    with NODE_REGISTRY_LOCK:
        master_info = NODE_REGISTRY[master_node_id]
        master_addr = master_info["ip"]
    
    # [수정] 고유 포트 할당
    master_port = next(MASTER_PORT_COUNTER)
    log.info(f"[{job_id}] Assigning unique master port: {master_port}")

    tasks = []
    
    for rank, node_id in enumerate(assigned_nodes):
        with NODE_REGISTRY_LOCK:
            node_info = NODE_REGISTRY[node_id]
            agent_url = f"http://{node_info['ip']}:{node_info['agent_port']}/launch_task"
            
        task_payload = {
            "job_id": job_id,
            "script_path": job_config["script_path"],
            "master_addr": master_addr,
            "master_port": master_port, # 고유 포트 전달
            "world_size": world_size,
            "rank": rank,
            "local_rank": 0, 
            "gpu_id": node_info["gpu_id"], 
            "epochs": 100, 
            "checkpoint_dir": f"./checkpoints/{job_id}", 
            "global_server_addr": GLOBAL_SERVER_ADDRESS,
            # "dataset": job_config.get("model_name"), # 스크립트가 안 받는 인자
            # "model_name": job_config.get("model_name"), # 스크립트가 안 받는 인자
            "resume_from_checkpoint": job_config.get("latest_checkpoint_path")
        }
        
        async def send_request(url, json):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=json, timeout=10)
                    response.raise_for_status()
                    log.info(f"[{job_id}] Successfully launched task on {node_id} (Rank {rank})")
            except Exception as e:
                log.error(f"[{job_id}] FAILED to launch task on {node_id} (Rank {rank}): {e}")

        tasks.append(send_request(agent_url, task_payload))

    await asyncio.gather(*tasks)

    with ACTIVE_JOBS_LOCK:
        ACTIVE_JOBS[job_id] = {
            "config": job_config,
            "status": "running",
            "nodes": assigned_nodes,
            "stop_event": asyncio.Event(), 
            "latest_progress": 0.0 
        }
    _set_node_status(assigned_nodes, "busy", job_id)


async def _stop_job(job_id: str, timeout: int = 60) -> bool:
    log.info(f"[{job_id}] Attempting to stop job...")
    
    with ACTIVE_JOBS_LOCK:
        if job_id not in ACTIVE_JOBS:
            log.warning(f"[{job_id}] Job not in ACTIVE_JOBS, cannot stop.")
            return True
        job_info = ACTIVE_JOBS[job_id]
        job_info["status"] = "stopping"
        assigned_nodes = job_info["nodes"]
        stop_event = job_info["stop_event"]

    tasks = []
    
    for node_id in assigned_nodes:
        with NODE_REGISTRY_LOCK:
            if node_id not in NODE_REGISTRY:
                log.warning(f"[{job_id}] Node {node_id} not in registry, skipping stop.")
                continue
            node_info = NODE_REGISTRY[node_id]
            agent_url = f"http://{node_info['ip']}:{node_info['agent_port']}/stop_task"
            
        async def send_stop_request(url, json):
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(url, json=json, timeout=5)
                    log.info(f"[{job_id}] Sent stop signal to {node_id}")
            except Exception as e:
                log.error(f"[{job_id}] FAILED to send stop signal to {node_id}: {e}")

        tasks.append(send_stop_request(agent_url, {"job_id": job_id}))

    await asyncio.gather(*tasks)

    try:
        log.info(f"[{job_id}] Waiting for stop confirmation from worker (timeout: {timeout}s)...")
        await asyncio.wait_for(stop_event.wait(), timeout=timeout)
        log.info(f"[{job_id}] Stop confirmation received. Job stopped successfully.")
        
        with ACTIVE_JOBS_LOCK:
            if job_id in ACTIVE_JOBS:
                del ACTIVE_JOBS[job_id]
        _set_node_status(assigned_nodes, "idle", None)
        return True
        
    except asyncio.TimeoutError:
        log.error(f"[{job_id}] Timeout waiting for stop confirmation. Forcing cleanup.")
        with ACTIVE_JOBS_LOCK:
            if job_id in ACTIVE_JOBS:
                del ACTIVE_JOBS[job_id]
        _set_node_status(assigned_nodes, "idle", None) 
        return False


async def schedule_and_dispatch_jobs():
    # UnboundLocalError 해결
    global JOB_QUEUE 
    
    log.info("Starting Pollux scheduling loop...")
    while True:
        try:
            with NODE_REGISTRY_LOCK:
                all_node_ids = list(NODE_REGISTRY.keys())
                pollux_nodes = [
                    NodeInfo(node_id, resources={"gpu": 1, "cpu": 4}) 
                    for node_id in all_node_ids
                ]

            current_jobs_to_schedule = []
            
            with JOB_QUEUE_LOCK:
                for job_config in JOB_QUEUE:
                    current_jobs_to_schedule.append(job_config)
            
            with ACTIVE_JOBS_LOCK:
                for job_id, job_info in ACTIVE_JOBS.items():
                    job_info["config"]["latest_checkpoint_path"] = job_info.get("latest_checkpoint_path")
                    current_jobs_to_schedule.append(job_info["config"])

            if not current_jobs_to_schedule:
                await asyncio.sleep(10) 
                continue

            pollux_jobs = []
            job_config_map = {} 
            
            for job_config in current_jobs_to_schedule:
                job_id = job_config["job_id"]
                job_config_map[job_id] = job_config
                
                model_name = job_config.get("model_name", "default")
                profiling_data = JOB_PROFILING_DB.get(model_name, JOB_PROFILING_DB["default"])
                
                with ACTIVE_JOBS_LOCK:
                    current_progress = ACTIVE_JOBS.get(job_id, {}).get("latest_progress", 0.0)

                goodput_fn = GoodputFunction(profiling_data, current_progress)
                speedup_fn = SpeedupFunction(goodput_fn)
                
                pollux_jobs.append(
                    JobInfo(
                        job_id=job_id,
                        speedup_fn=speedup_fn,
                        min_replicas=1, 
                        max_replicas=len(pollux_nodes) 
                    )
                )

            desired_allocations = POLLUX_SCHEDULER.optimize(pollux_jobs, pollux_nodes)

            current_allocations = {}
            with ACTIVE_JOBS_LOCK:
                for job_id, job_info in ACTIVE_JOBS.items():
                    current_allocations[job_id] = len(job_info["nodes"])
            
            log.info(f"Pollux desired state: {desired_allocations}")
            log.info(f"Current state: {current_allocations}")

            jobs_to_stop = []
            jobs_to_start_or_modify = [] 

            for job_id, current_gpus in current_allocations.items():
                desired_gpus = desired_allocations.get(job_id, 0)
                
                if current_gpus != desired_gpus:
                    log.info(f"[{job_id}] Re-allocation planned: {current_gpus} -> {desired_gpus} GPUs")
                    jobs_to_stop.append(job_id) 
                    if desired_gpus > 0:
                        jobs_to_start_or_modify.append((job_id, desired_gpus))
                
            for job_id, desired_gpus in desired_allocations.items():
                if job_id not in current_allocations and desired_gpus > 0:
                    log.info(f"[{job_id}] New job start planned: {desired_gpus} GPUs")
                    jobs_to_start_or_modify.append((job_id, desired_gpus))

            if jobs_to_stop:
                log.info(f"Stopping {len(jobs_to_stop)} jobs for re-allocation...")
                stop_tasks = [asyncio.create_task(_stop_job(job_id)) for job_id in jobs_to_stop]
                await asyncio.gather(*stop_tasks)
                log.info("All necessary jobs stopped. Nodes are now free.")

            await asyncio.sleep(2) 

            if jobs_to_start_or_modify:
                log.info(f"Launching {len(jobs_to_start_or_modify)} jobs...")
                available_node_ids = _get_available_nodes() 
                
                launch_tasks = []
                
                jobs_to_start_or_modify.sort(key=lambda x: x[1], reverse=True)
                
                for job_id, desired_gpus in jobs_to_start_or_modify:
                    if len(available_node_ids) < desired_gpus:
                        log.warning(f"[{job_id}] Not enough idle nodes ({len(available_node_ids)}) to satisfy {desired_gpus} GPUs. Skipping launch.")
                        continue
                        
                    nodes_for_this_job = available_node_ids[:desired_gpus]
                    available_node_ids = available_node_ids[desired_gpus:] 
                    
                    job_config = job_config_map.get(job_id)
                    if job_config is None:
                        log.error(f"[{job_id}] Config not found?! Skipping.")
                        continue
                        
                    with JOB_QUEUE_LOCK:
                        JOB_QUEUE = [j for j in JOB_QUEUE if j["job_id"] != job_id]
                    
                    launch_tasks.append(
                        asyncio.create_task(_launch_job_on_nodes(job_config, nodes_for_this_job))
                    )
                
                await asyncio.gather(*launch_tasks)
                log.info("Job launch/modify phase complete.")

        except Exception as e:
            log.error(f"Error in scheduling loop: {e}", exc_info=True)
            
        await asyncio.sleep(30) # 30초마다 스케줄링 루프 실행


# --- 4. 워커 통신 API ---

@app.on_event("startup")
async def on_startup():
    log.info("Starting background scheduling loop...")
    asyncio.create_task(schedule_and_dispatch_jobs())

@app.post("/submit_job")
async def submit_job(request: JobSubmitRequest):
    global JOB_QUEUE 
    
    log.info(f"Received new job submission: {request.job_id}")
    with JOB_QUEUE_LOCK:
        if any(j["job_id"] == request.job_id for j in JOB_QUEUE):
            raise HTTPException(status_code=400, detail="Job ID already in queue")
        with ACTIVE_JOBS_LOCK:
             if request.job_id in ACTIVE_JOBS:
                 raise HTTPException(status_code=400, detail="Job ID is already running")
                 
        JOB_QUEUE.append(request.dict()) 
        
    return {"status": "job_submitted_to_queue", "job_id": request.job_id}

@app.post("/report_progress")
async def report_progress(report: JobProgressReport):
    job_id = report.job_id
    with ACTIVE_JOBS_LOCK:
        if job_id in ACTIVE_JOBS:
            if report.total_epochs > 0:
                 # [수정] 422 오류 해결: 'epoch' 키로 읽음
                 ACTIVE_JOBS[job_id]["latest_progress"] = report.epoch / report.total_epochs
            return {"status": "progress_acked"}
    return HTTPException(status_code=404, detail="Job not found")


@app.post("/report_checkpoint")
async def report_checkpoint(report: JobCheckpointReport):
    job_id = report.job_id
    with ACTIVE_JOBS_LOCK:
        if job_id in ACTIVE_JOBS:
            # [수정] 422 오류 해결: 'path' 키로 읽음
            ACTIVE_JOBS[job_id]["latest_checkpoint_path"] = report.path
            log.info(f"[{job_id}] Checkpoint saved: {report.path} (Epoch {report.epoch})")
            return {"status": "checkpoint_acked"}
    
    raise HTTPException(status_code=404, detail="Job not found")

@app.post("/report_job_stopped")
async def report_job_stopped(report: JobStopReport):
    job_id = report.job_id
    with ACTIVE_JOBS_LOCK:
        if job_id in ACTIVE_JOBS and ACTIVE_JOBS[job_id].get("stop_event"):
            log.info(f"[{job_id}] Received stop confirmation from worker.")
            ACTIVE_JOBS[job_id]["stop_event"].set()
            return {"status": "stop_acked"}
            
    log.warning(f"[{job_id}] Received stop report, but job not found or no stop event.")
    raise HTTPException(status_code=404, detail="Job or stop event not found")


# argparse를 사용하여 실행
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global Server Manager with Pollux Scheduler")
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, 
                        help='Port to bind the server to (default: 8000)')
    parser.add_argument('--public_ip', type=str, required=True,
                        help='Publicly reachable IP or hostname of this server (e.g., 127.0.0.1)')
    
    args = parser.parse_args()
    
    GLOBAL_SERVER_ADDRESS = f"http://{args.public_ip}:{args.port}"
    log.info(f"Global Server Address set to: {GLOBAL_SERVER_ADDRESS}")
    
    if args.public_ip == "127.0.0.1":
        log.info("Updating NODE_REGISTRY IPs to 127.0.0.1 for local testing.")
        with NODE_REGISTRY_LOCK:
            for node_id in NODE_REGISTRY:
                NODE_REGISTRY[node_id]["ip"] = "127.0.0.1"

    uvicorn.run(app, host=args.host, port=args.port)