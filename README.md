# Personality-Aware Dynamic Resource Scheduling

이기종 GPU 클러스터에서 작업별 요구사항과 GPU 성능을 고려하여 **GPU 배치와 할당량을 결정하고, 실행 중 자원 상태에 따라 할당을 조정하는 분산 학습 스케줄러**입니다.

작업마다 완료 시간, 비용, 에너지 등 중요하게 고려하는 조건을 다르게 설정할 수 있으며, GPU별 학습 성능과 현재 사용 가능한 자원을 함께 고려하여 스케줄링합니다.

## How It Works

```text
Job 제출
   │
   ├── 작업 정보
   ├── 자원 요구사항
   └── 사용자 요구사항
           │
           ▼
    Scheduling Policy 생성
           │
           ▼
  GPU 성능 Profile + Cluster 상태
           │
           ▼
        Scheduler
           │
   ┌───────┼─────────┐
   ▼       ▼         ▼
Placement Scaling  Backfilling
   │
   ▼
Distributed Training
   │
   ▼
Monitoring / Rebalancing
```

Scheduler는 작업이 제출되면 사용 가능한 GPU와 각 GPU에서의 예상 학습 성능을 확인하여 초기 자원을 할당합니다.

작업 실행 중에도 GPU가 반환되거나 새로운 작업이 들어오는 등 클러스터 상태가 변경되면 기존 할당을 다시 계산할 수 있습니다.

## Usage

### 1. GPU Cluster 준비

여러 GPU 노드에서 분산 학습이 가능한 환경이 필요합니다.

현재 구현은 Slurm 기반 실행 환경을 사용하며, 각 worker는 `sbatch`를 통해 실행됩니다.

Worker 실행 설정은 다음 경로에서 확인할 수 있습니다.

```text
1. Impl/ours/DARTManagement/
├── worker_e.sbatch
├── worker_f.sbatch
├── worker_g.sbatch
└── worker_h.sbatch
```

사용할 클러스터의 node, GPU, Python 환경 등에 맞게 각 `sbatch` 파일을 수정합니다.

### 2. 학습 코드 준비

스케줄링 대상 학습 코드는 GPU 수가 변경될 수 있는 분산 학습 환경에서 실행 가능해야 합니다.

예제 학습 코드는 다음 경로에 있습니다.

```text
1. Impl/ours/TrainingCode(DART)/
```

자체 학습 작업을 사용하는 경우 해당 부분을 실제 training script로 교체하여 사용할 수 있습니다.

### 3. GPU 성능 Profiling

Scheduler는 GPU 종류별 예상 학습 성능을 이용하여 placement와 resource allocation을 결정합니다.

따라서 사용하는 모델과 GPU 조합에 대한 profiling 정보가 필요합니다.

관련 코드는 다음 파일에 있습니다.

```text
1. Impl/ours/app/profiling.py
```

새로운 GPU 또는 새로운 학습 workload를 사용하는 경우 해당 환경에서 측정한 profiling 정보를 추가해야 합니다.

### 4. Worker 실행

각 GPU node에서 Worker Agent를 실행합니다.

```text
1. Impl/ours/DARTManagement/worker_agent.py
```

Worker Agent는 각 노드의 작업 실행을 담당하며 Global Server의 scheduling 결과에 따라 training process를 관리합니다.

Slurm 환경에서는 제공된 `worker_*.sbatch` 파일을 이용해 worker를 실행할 수 있습니다.

### 5. Global Server 실행

전체 worker와 Job을 관리하는 Global Server를 실행합니다.

```bash
python "1. Impl/ours/DARTManagement/global_server_ours.py"
```

Global Server는 worker 상태와 작업 실행 상태를 관리하고 scheduler와 실제 클러스터 실행 환경을 연결합니다.

### 6. Scheduler 실행

제안 스케줄러의 entry point는 다음과 같습니다.

```bash
python "1. Impl/ours/main.py"
```

주요 scheduling 로직은 다음 디렉터리에 있습니다.

```text
1. Impl/ours/app/
```

Scheduler는 입력된 Job 정보, GPU profiling 정보, 현재 cluster state를 이용하여 각 작업의 GPU placement와 allocation을 결정합니다.

## Scheduling

작업이 들어오면 다음 정보를 기준으로 초기 배치를 결정합니다.

```text
Job Requirements
      +
Scheduling Policy
      +
GPU Performance Profile
      +
Available GPU Resources
      ↓
GPU Type / GPU Count 결정
```

예를 들어 동일한 작업이라도 빠른 완료가 중요한 경우와 자원 사용량을 줄이는 것이 중요한 경우 서로 다른 GPU 구성에 배치될 수 있습니다.

실행 중 클러스터 상태가 변경되면 `rebalance.py`에서 기존 allocation을 다시 평가합니다.

```text
GPU 반환 / Job 완료 / 신규 Job 도착
              │
              ▼
         Rebalancing
              │
       ┌──────┴──────┐
       ▼             ▼
    Scale Up      Scale Down
```

사용되지 않는 GPU가 있는 경우 `backfill_policy.py`를 통해 대기 중인 작업을 실행할 수 있습니다.

## Required Inputs

새로운 환경에서 사용하려면 다음 정보가 필요합니다.

- 실행할 학습 코드 및 모델
- GPU 종류 및 GPU 수
- GPU별 학습 성능 profiling 결과
- Job별 최소/최대 GPU 요구량
- Job의 사용자 요구사항
- Worker node 정보
- Slurm 실행 환경 및 `sbatch` 설정

새로운 GPU나 모델을 추가하는 경우 해당 조합에 대한 profiling 정보를 먼저 준비해야 합니다.

## Project Structure

```text
.
├── 0. RESULT/                 # Experiment results
│
├── 1. Impl/
│   │
│   ├── ours/                  # Proposed scheduler
│   │   ├── main.py
│   │   │
│   │   ├── app/
│   │   │   ├── llm.py
│   │   │   ├── profiling.py
│   │   │   ├── scheduler_state.py
│   │   │   ├── backfill_policy.py
│   │   │   ├── rebalance.py
│   │   │   ├── executor.py
│   │   │   ├── guardrail.py
│   │   │   ├── metrics.py
│   │   │   └── ours_core.py
│   │   │
│   │   ├── DARTManagement/
│   │   │   ├── global_server_ours.py
│   │   │   ├── worker_agent.py
│   │   │   └── worker_*.sbatch
│   │   │
│   │   └── TrainingCode(DART)/
│   │
│   ├── fifo_sjf/
│   ├── pollux/
│   ├── sia/
│   └── Lucid/
│
├── 2. test code/
└── fig/
```

## Main Components

| Component | Description |
|---|---|
| `main.py` | Scheduler 실행 진입점 |
| `ours_core.py` | 주요 scheduling 로직 |
| `profiling.py` | GPU 및 workload 성능 profile 관리 |
| `scheduler_state.py` | Job 및 cluster resource 상태 관리 |
| `rebalance.py` | 실행 중 GPU allocation 재조정 |
| `backfill_policy.py` | 유휴 GPU를 이용한 backfilling |
| `executor.py` | Scheduling 결과 실행 |
| `llm.py` | 사용자 요구사항 기반 scheduling policy 처리 |
| `global_server_ours.py` | 전체 worker 및 작업 관리 |
| `worker_agent.py` | 각 worker node의 작업 실행 관리 |

## Baselines

비교 실험을 위해 다음 스케줄러 구현이 함께 포함되어 있습니다.

```text
FIFO / SJF
Pollux
Sia
Lucid
```

제안 스케줄러는 `1. Impl/ours/`에서 확인할 수 있으며, 다른 디렉터리는 baseline 및 실험 비교에 사용됩니다.

## Notes

현재 코드는 연구 및 실험 환경에서 사용한 구현을 기준으로 구성되어 있습니다.

다른 클러스터에서 사용하려면 **GPU/모델 profiling 정보, worker node 설정, Slurm 환경 및 학습 코드 경로를 해당 환경에 맞게 수정해야 합니다.**
