from __future__ import annotations
from typing import Dict, List

# 클러스터 ↔ 노드
CLUSTERS: Dict[str, List[str]] = {
    "clusterA": ["node_a", "node_b", "node_c", "node_d"],
    "clusterB": ["node_e", "node_f", "node_g", "node_h"],
}

## 노드 에이전트
#NODE_REGISTRY: Dict[str, Dict] = {
    # 216
#    "node_a": {"ip": "163.180.117.216", "agent_port": 8001},
#    "node_b": {"ip": "163.180.117.216", "agent_port": 8002},
#    "node_c": {"ip": "163.180.117.216", "agent_port": 8003},
#    "node_d": {"ip": "163.180.117.216", "agent_port": 8004},
    # 세라프
#    "node_e": {"ip": "163.180.160.56", "agent_port": 8005},
#    "node_f": {"ip": "163.180.160.56", "agent_port": 8006},
#    "node_g": {"ip": "163.180.160.56", "agent_port": 8007},
#    "node_h": {"ip": "163.180.160.56", "agent_port": 8008},
#}

#def node_url(node_id: str) -> str:
#    n = NODE_REGISTRY[node_id]
#    return f"http://{n['ip']}:{n['agent_port']}"

