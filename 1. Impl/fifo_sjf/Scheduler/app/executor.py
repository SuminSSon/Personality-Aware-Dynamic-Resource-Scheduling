import os
import requests
from typing import Tuple, Dict, Any, Optional, List

GLOBAL_SERVER = os.getenv("GLOBAL_SERVER", "http://163.180.117.216:8000")

CLUSTER_NODES = {
    "clusterA": ["node_a", "node_b", "node_c", "node_d"],
    "clusterB": ["node_e", "node_f", "node_g", "node_h"],
}

def _pick_nodes(cluster: str, world_size: int):
    """
    아주 단순하게 '앞에서부터 world_size개' 노드를 고릅니다.
    - 글로벌 서버(global_server.py)가 자체적으로 노드 busy/idle을 관리하므로
      여기서는 선정만 해서 /launch_job 에 전달하면 됩니다.
    """
    nodes = CLUSTER_NODES.get(cluster, [])
    if not nodes or world_size <= 0:
        return []
    return nodes[:world_size]

def launch_or_reuse(
    job_id: str,
    cluster: str,
    world_size: int,
    dataset: str,
    model: str,
    epochs: int = 100,
    preferred_nodes: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
):
    """
    - preferred_nodes가 주어지면 그 노드들로 바로 /launch_job 호출
    - 없으면 기존 로직(라운드로빈 등)으로 노드 선택
    """
    # 1) 사용할 노드 목록 결정
    if preferred_nodes:
        nodes = preferred_nodes
    else:
        # 기존에 쓰던 선택 로직 (예: A->a,b,c,d / B->e,f,g,h 중에서 world_size개 선택)
        nodes_pool = {
            "clusterA": ["node_a","node_b","node_c","node_d"],
            "clusterB": ["node_e","node_f","node_g","node_h"],
        }[cluster]
        nodes = nodes_pool[:world_size]

    # 2) 글로벌 서버로 런치 호출
    payload = {"job_id": job_id, "nodes": nodes, "model": model, "dataset": dataset, "epochs": epochs, "batch_size": batch_size,}
    resp = requests.post("http://127.0.0.1:8000/launch_job", json=payload, timeout=5)
    if resp.status_code != 200:
        return False, {"status": "error", "detail": resp.text}

    data = resp.json()
    # 표준화된 리턴
    return True, {
        "status": "started",
        "nodes": nodes,
        "job_id": data.get("job_id"),
    }
def job_status(job_id: str) -> Dict[str, Any]:
    """
    글로벌 서버의 /job_status/{job_id} 를 프록시해서 반환.
    - FIFO의 /maint/sweep_running 이 이 값을 참고해서 청소합니다.
    """
    try:
        r = requests.get(f"{GLOBAL_SERVER}/job_status/{job_id}", timeout=5)
        if r.status_code == 200:
            return r.json()  # {"job_id":..., "status_details": {...}} 등
        return {"status": "unknown", "http": r.status_code, "text": r.text}
    except requests.RequestException as e:
        return {"status": "unknown", "error": str(e)}

