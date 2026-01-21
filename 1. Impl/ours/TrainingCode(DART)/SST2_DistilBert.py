import os
import argparse
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler
# [삭제] import torchvision
# [삭제] import torchvision.transforms as transforms

# [NEW] NLP(Hugging Face)를 위한 Import
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding
from torch.optim import AdamW # [수정] AdamW는 torch.optim에서 가져옵니다. # Adam -> AdamW
from datasets import load_dataset

from metrics_monitor import MetricsMonitor
from allocator_interface import ResourceAllocator
from log_manager import LogManager
# [삭제] from torchvision import datasets, transforms, models
import math
import json
import psutil
from sklearn.metrics import f1_score

# [ADDED] 서버 통신 및 로깅을 위한 import
import requests 
import logging

from typing import Optional
import httpx

def send_progress(
    global_server_addr: str,
    job_id: str,
    world_size: int,
    local_batch: int,
    grad_accum: int,
    steps_per_sec: float,
    loss: Optional[float] = None,
    accuracy: Optional[float] = None,
    gns: Optional[float] = None,
    stat_eff: Optional[float] = None,
):
    payload = {
        "job_id": job_id,
        "world_size": world_size,
        "local_batch": local_batch,
        "grad_accum": grad_accum,
        "steps_per_sec": steps_per_sec,
        "loss": loss,
        "accuracy": accuracy,
        "gns": gns,
        "stat_eff": stat_eff,
    }

    try:
        httpx.post(
            f"{global_server_addr}/report_progress",
            json=payload,
            timeout=2.0,
        )
    except Exception:
        # 서버 잠깐 죽어도 학습은 계속
        pass

def report_attained_service(global_server_addr: str, job_id: str, attained: float):
    try:
        httpx.post(
            f"{global_server_addr}/report_job_metrics",
            json={"job_id": job_id, "attained_service": attained},
            timeout=1.0,
        )
    except Exception:
        # 서버 잠깐 죽어도 학습은 계속
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, default=1, help="Number of processes per node")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--init-lr', type=float, default=0.00002)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--nnodes", type=int, default=1, help="Total number of nodes")
    parser.add_argument("--node_rank", type=int, default=0, help="Rank of the node")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master_port", type=int, default=29500, help="Master node port")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_from', type=str, default=None, 
                        help='Path to checkpoint file to resume training')

    # [ADDED] 글로벌 서버 통신을 위한 인자
    parser.add_argument('--job_id', type=str, default='local_test', 
                        help='Job ID for reporting and stop signal')
    parser.add_argument('--global_server_addr', type=str, default=None, 
                        help='Address of the global server (e.g., http://127.0.0.1:8000)')

    # [NEW] Hugging Face 모델 이름 인자
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='Name of the Hugging Face model to use')

    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()

    master_addr = args.master_addr
    master_port = args.master_port

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    current_idx = torch.cuda.current_device()
    print(f"Current CUDA device index: {current_idx}")

    device_name = torch.cuda.get_device_name(current_idx)
    print(f"Current CUDA device name: {device_name}")


    print(f"Master address: {master_addr}, port: {master_port}", flush=True)
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    world_size = args.nnodes * args.nproc_per_node
    rank = args.node_rank * args.nproc_per_node + local_rank
    
    
    # [ADDED] 헬퍼 함수 (서버 통신 및 로깅)
    log = logging.getLogger(f"DDP_Script [Rank {rank}]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    def _get_stop_flag_path(job_id: str) -> str:
        """
        WorkerAgent가 env로 내려주는 STOP_FLAG_PATH를 우선 사용.
        (job_id는 형식 맞추기용으로만 남겨둠)
        """
        env_path = os.environ.get("STOP_FLAG_PATH")
        if env_path:
            return env_path
        # 혹시 env가 없을 때만 fallback (worker_agent 패턴과 맞춤)
        attempt = int(os.environ.get("ATTEMPT", "1"))
        return f"/tmp/{job_id}-attempt-{attempt}.flag"

    def _report_checkpoint(server_addr: str, job_id: str, epoch: int, total_epochs: int,
                            acc: float, loss: float, path: str):
        if rank != 0 or not server_addr:
            return
        url = f"{server_addr}/report_checkpoint"
        payload = {
            "job_id": job_id,
            "current_epoch": epoch,
            "total_epochs": total_epochs,
            "latest_accuracy": acc,
            "latest_eval_loss": loss,
            "checkpoint_path": os.path.abspath(path)
        }
        try:
            requests.post(url, json=payload, timeout=5)
            log.info(f"Successfully reported epoch {epoch}/{total_epochs} (Acc: {acc:.2f}%) to global server.")
        except requests.RequestException as e:
            log.warning(f"Failed to report checkpoint to global server: {e}")

    def _report_job_completed(
        server_addr: str,
        job_id: str,
        exit_code: int = 0,
        final_accuracy: Optional[float] = None,
    ):
        return
    
    def _report_job_stopped(server_addr: str, job_id: str):
        """ (Rank 0 전용) 우아한 종료(Graceful Shutdown) 직전 서버에 보고 """
        if rank != 0 or not server_addr:
            return
            
        url = f"{server_addr}/report_job_stopped"
        payload = {"job_id": job_id}
        try:
            requests.post(url, json=payload, timeout=5)
            log.info("Successfully reported graceful shutdown to global server.")
        except requests.RequestException as e:
            log.warning(f"Failed to report job stop to global server: {e}")
    # [END 헬퍼 함수]
    
    attained_service = 0.0
    last_global_acc = None

    log.info(f"Node rank: {rank}, world size: {world_size}")
    
    # Initialize DDP
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

    log.info(f"Process group initialized: {master_addr}:{master_port}")
    log.info(f"Rank {rank}/{world_size} initialized")

    # 1. Load Tokenizer (DDP-safe)
    if rank == 0:
        log.info(f"Rank 0 downloading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dist.barrier()
    if rank != 0:
        log.info(f"Rank {rank} loading tokenizer from cache: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 2. Load Dataset (GLUE, SST-2) (DDP-safe)
    if rank == 0:
        log.info("Rank 0 downloading dataset: glue/sst2")
        raw_datasets = load_dataset("glue", "sst2")
    dist.barrier()
    if rank != 0:
        log.info(f"Rank {rank} loading dataset from cache: glue/sst2")
        raw_datasets = load_dataset("glue", "sst2")

    # 3. Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding=False) # DataCollator가 패딩
    
    # 4. Apply preprocessing
    log.info("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
    # 5. Format dataset for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["validation"] # SST-2는 'validation' set을 test용으로 씀

    # 6. Create Data Collator (Dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    log.info("SST-2 Dataset and Tokenizer loaded.")
    
    # [기존] Image transforms (삭제)
    # transform_train = ...
    # transform_val = ...

    # [기존] CIFAR-100 (삭제)
    # train_dataset = datasets.CIFAR100(...)
    # test_dataset  = datasets.CIFAR100(...)

    # [기존] train_sampler, train_loader (삭제)
    # (이 파일은 루프 안에서 train_loader를 생성함)
    
    # [MODIFIED] val_loader: collate_fn 추가
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        sampler=test_sampler, 
        num_workers=4,
        collate_fn=data_collator # [NEW]
    )

    batch_size = args.batch_size

    # =================================================================
    # [MODIFIED] Model, DDP, criterion, optimizer
    # =================================================================
    
    # (Rank 0만 다운로드하고 나머지는 대기)
    if rank == 0:
        log.info(f"Rank 0 downloading model: {args.model_name}")
        model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    dist.barrier()
    if rank != 0:
        log.info(f"Rank {rank} loading model from cache: {args.model_name}")
        model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    
    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # [MODIFIED] Criterion (불필요)
    # criterion = torch.nn.CrossEntropyLoss().cuda(local_rank) # (삭제)
    
    # [MODIFIED] Optimizer (Adam -> AdamW)
    optimizer = AdamW(model.parameters(), lr=args.init_lr)

    log.info("Model (DistilBert), DDP, optimizer (AdamW) initialized")

    # Modules
    monitor = MetricsMonitor(interval=5)
    monitor.start()
    
    # [NO CHANGE] ResourceAllocator는 그대로 유지
    allocator = ResourceAllocator(
        min_batch_size=16, max_batch_size=64,
        min_lr=1e-4, max_lr=0.1,
        gamma=0.1, target_accuracy=90.0,
        ckf_params={
            'state_dim': 8,
            'obs_dim':   8,
            'process_noise_cov':    np.eye(8) * 0.05,
            'measurement_noise_cov': np.eye(8) * 0.1,
        },
        dyn_ratio_params={'lower':0.1,'upper':0.5,'k':10,'threshold':0.5} # 데이터 샤딩 비율
    )

    # [NO CHANGE] LogManager는 그대로 유지
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    csv_name = f"{world_size}nodes_{script_name}_{master_addr}_{master_port}_rank{rank}.csv"
    fieldnames = [
        'epoch','rank',
        'train_time','eval_time','total_time',
        'cpu_usage','gpu_usage','memory_usage','cpu_temp','gpu_temp','gpu_mem_percent',
        'accuracy','top5_accuracy','f1_score','ckf_dim',
        'shard_pct','batch_size','num_workers','lr','all_allocs', 
        'global_train_loss','global_eval_loss',
        'local_train_loss','local_eval_loss',
    ]
    logger = LogManager(csv_name, fieldnames)

    device = torch.device('cuda', local_rank)

    log.info("Modules initialized")

    dataset_size = len(train_dataset)

    
    # [NO CHANGE] 체크포인트 디렉토리 생성
    if rank == 0 and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log.info(f"Checkpoint directory created: {args.checkpoint_dir}")

    start_epoch = 1

    # [NO CHANGE] 체크포인트 로드
    if args.resume_from and os.path.exists(args.resume_from):
        log.info(f"Loading checkpoint from {args.resume_from} ...")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.resume_from, map_location=map_location)
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        log.info(f"Resumed from checkpoint. Starting epoch: {start_epoch}")
    else:
        log.info(f"Starting training from scratch (epoch 1).")

    dist.barrier()

    base_seed = 42

    for epoch in range(start_epoch, args.epochs + 1):

        log.info(f"epoch {epoch} 시작")

        epoch_start = time.time()

        # ============================================================
        # ❶ [NO CHANGE] 현재 할당값으로 DataLoader 생성 (동적)
        #    - (collate_fn만 추가)
        # ============================================================
        if hasattr(allocator, "last_alloc") and allocator.last_alloc[rank]:
            prev_alloc = allocator.last_alloc[rank]
            shard_pct   = prev_alloc["shard_pct"]
            batch_size  = prev_alloc["batch_size"]
            num_workers = prev_alloc["num_workers"]
        else:  # 첫 epoch
            shard_pct   = 1.0 / world_size
            batch_size  = args.batch_size
            num_workers = 4

        shard_size = max(1, math.floor(dataset_size * shard_pct))

        g = torch.Generator()
        g.manual_seed(epoch)                 # deterministic shuffle
        perm = torch.randperm(dataset_size, generator=g).tolist()

        all_indices = torch.randperm(dataset_size, generator=g).tolist()
        start = rank * shard_size
        end = start + shard_size
        if rank == world_size - 1:
            end = dataset_size
        shard_indices = all_indices[start:end]

        # [MODIFIED] collate_fn 추가
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(shard_indices),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=data_collator # [NEW]
        )

        log.info(f"DataLoader created with batch_size={batch_size}, num_workers={num_workers}")
        
        # ============================================================
        # ❷ [MODIFIED] TRAIN — NLP에 맞게 수정
        # ============================================================
        model.train()

        # [NO CHANGE] 스텝 패딩 로직
        local_steps = math.ceil(len(shard_indices) / batch_size)
        steps_tensor = torch.tensor([local_steps], device=device)
        dist.all_reduce(steps_tensor, op=dist.ReduceOp.MAX)
        max_steps = steps_tensor.item()
        if dist.get_rank() == 0:
            log.info(f"모든 노드 중 max_steps = {max_steps}")
        log.info(f"local_steps={local_steps}, global_steps={max_steps}")

        step = 0
        local_train_loss = torch.tensor(0.0, device=device)
        local_train_samples = torch.tensor(0, device=device)
        t0 = time.time()

        # [MODIFIED] lastinput, lasttarget -> last_batch
        last_batch = None

        # [MODIFIED] Train loop: (inputs, targets) -> batch
        for batch in train_loader:
            # batch는 {'input_ids': ..., 'attention_mask': ..., 'labels': ...} 딕셔너리
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            optimizer.zero_grad()
            
            # [MODIFIED] 모델 입력 및 손실 계산
            outputs = model(**batch)
            loss = outputs.loss # 모델이 반환한 loss 사용

            # [MODIFIED] loss 및 샘플 수 계산
            local_train_loss += loss.detach() * batch['input_ids'].size(0)
            local_train_samples += batch['input_ids'].size(0)
            
            loss.backward()
            optimizer.step()

            # [MODIFIED] 마지막 배치 저장 (패딩 루프용)
            last_batch = batch

            step += 1

        model.eval() 

        # [MODIFIED] Padding loop (NLP에 맞게 수정)
        while step < max_steps:
            if last_batch is None: # 혹시 local_steps == 0 인 경우
                log.warning("Padding loop skipped: last_batch is None (empty dataloader)")
                break
                
            optimizer.zero_grad()
            
            # [MODIFIED] 마지막 배치로 재학습
            outputs = model(**last_batch)
            loss = outputs.loss
            
            loss.backward()
            # [NO CHANGE] optimizer.step()은 원래도 주석처리/생략되어 있었음 (스텝 수만 맞춤)
            step += 1

        train_time = time.time() - t0
        
        # [수정] local_train_samples가 0일 경우 대비
        if local_train_samples.item() > 0:
            local_train_loss_value = local_train_loss.item() / local_train_samples.item()
        else:
            local_train_loss_value = 0.0

        log.info(f"epoch {epoch} train step time: {train_time:.2f} seconds")

        dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_train_samples, op=dist.ReduceOp.SUM)
        
        if local_train_samples.item() > 0:
            global_train_loss = local_train_loss.item() / local_train_samples.item()
        else:
            global_train_loss = 0.0

        dist.barrier()

        log.info(f"epoch {epoch} 끝 (local_steps={local_steps}, global_steps={max_steps})")

        if rank == 0 and args.global_server_addr:
            # 이번 epoch에서 처리한 샘플 수 (모든 rank 합)
            samples_this_epoch = float(local_train_samples.item())

            # 누적 서비스량 (단순히 샘플 수 누적 예시)
            attained_service += samples_this_epoch
            report_attained_service(
                global_server_addr=args.global_server_addr,
                job_id=args.job_id,
                attained=attained_service,
            )

            # step/s (epoch 전체 기준) – step은 max_steps, batch는 local batch
            steps_per_sec = max_steps / max(train_time, 1e-6)

            send_progress(
                global_server_addr=args.global_server_addr,
                job_id=args.job_id,
                world_size=world_size,
                local_batch=batch_size,
                grad_accum=1,               # 지금은 grad_accum 안 쓰고 있음
                steps_per_sec=steps_per_sec,
                loss=float(global_train_loss),
                accuracy=None,              # 정확도는 val 끝난 후에 따로 보낼 수도 있음
                gns=None,
                stat_eff=None,
            )
            
        # ============================================================
        # ❸ [MODIFIED] VALIDATION — NLP에 맞게 수정
        # ============================================================
        t1 = time.time()
        local_stats = monitor.latest.copy()
        model.eval()

        # [MODIFIED] local_top5_correct 제거
        # local_top5_correct = 0 
        local_preds = []
        local_labels = []

        total_loss = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0, device=device)
        correct = torch.tensor(0, device=device)
        
        with torch.no_grad():
            # [MODIFIED] Validation loop: (images, labels) -> batch
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # [MODIFIED] 모델 출력 및 손실
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                # [MODIFIED] 예측 (logits 사용)
                preds1 = logits.argmax(dim=-1)
                
                # [MODIFIED] Top-5 계산 제거
                # _, top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                # local_top5_correct += ...
                
                # [MODIFIED] F1 스코어 계산을 위한 레이블 저장 (batch['labels'] 사용)
                local_preds.extend(preds1.cpu().tolist())
                local_labels.extend(batch['labels'].cpu().tolist())
                
                # [MODIFIED] 집계 (batch['input_ids'] 및 batch['labels'] 사용)
                total_loss += loss * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
                correct += (preds1 == batch['labels']).sum()

        eval_time = time.time() - t1

        if total_samples.item() > 0:
            local_eval_loss_value = total_loss.item() / total_samples.item()
        else:
            local_eval_loss_value = 0.0

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

        # [MODIFIED] Top-5 all_reduce 제거
        # top5_tensor = torch.tensor(local_top5_correct, device=device)
        # dist.all_reduce(top5_tensor, op=dist.ReduceOp.SUM)

        if total_samples.item() > 0:
            global_eval_loss = total_loss.item() / total_samples.item()
            global_acc = 100.0 * correct.item() / total_samples.item()
            global_top5_acc = 0.0 # [MODIFIED]
        else:
            global_eval_loss = 0.0
            global_acc = 0.0
            global_top5_acc = 0.0

        last_global_acc = global_acc
        local_stats["accuracy"] = global_acc
        log.info(f"val acc {global_acc:.2f}%")

        if rank == 0:
            preds_list = [None] * world_size
            labels_list = [None] * world_size
            dist.gather_object(local_preds,  preds_list, dst=0)
            dist.gather_object(local_labels, labels_list, dst=0)
        else:
            dist.gather_object(local_preds,  dst=0)
            dist.gather_object(local_labels, dst=0)
        dist.barrier()

        if rank == 0:
            all_preds = sum(preds_list, [])   # flatten
            all_labels = sum(labels_list, [])
            if len(all_labels) > 0:
                f1 = f1_score(all_labels, all_preds, average='macro')
            else:
                f1 = 0.0
        else:
            f1 = 0.0

        f1_tensor = torch.tensor(f1, device=device)
        dist.broadcast(f1_tensor, src=0)
        global_f1 = f1_tensor.item()

        # ============================================================
        # ❹ [NO CHANGE] 메트릭 gather → Rank-0 자원 재할당
        #    - (이 로직은 모델 타입과 무관하게 동작)
        # ============================================================
        if rank == 0:
            gathered = [None for _ in range(world_size)]
            dist.gather_object(local_stats, gathered, dst=0)
        else:
            dist.gather_object(local_stats, dst=0)
        dist.barrier()
        
        if rank == 0:
            # [수정] 'cpu_cores'가 local_stats에 없을 수 있으므로, 있는 키만 수집
            available_keys = gathered[0].keys()
            metric_keys = ['cpu_usage','gpu_usage','memory_usage','cpu_temp',
                           'gpu_temp','gpu_mem_percent','accuracy']
            # 'cpu_cores'가 있으면 추가
            if 'cpu_cores' in available_keys:
                metric_keys.append('cpu_cores')
                
            metrics = {k:[g[k] for g in gathered] for k in metric_keys}
            
            # 'cpu_cores'가 없는 경우 allocator가 기본값을 사용하도록 처리
            if 'cpu_cores' not in metrics:
                log.warning("cpu_cores not found in local_stats. Allocator might use default.")
                # CKF가 'cpu_cores'를 8번째 차원으로 기대한다면, 
                # allocator.preprocess_ckf 내부에서 처리해야 함.
                # 여기서는 'ops', 'mu'만 추가
                metrics.update({'ops':[0.0]*world_size,'mu':[0.0]*world_size})
            else:
                 metrics.update({'ops':[0.0]*world_size,'mu':[0.0]*world_size})


            x_filt, _ = allocator.preprocess_ckf(metrics)
            
            try:
                log.info(f"CKF x_filt vector: {[x_filt[i] for i in range(x_filt.shape[0])]}")
            except Exception as e:
                log.warning(f"Could not print x_filt: {e}")

            allocs = [
                allocator.allocate(i, metrics, x_filt, batch_size, args.init_lr)
                for i in range(world_size)
            ]
            log.info(f"allocation results: {allocs}")

            total_pct = sum(a["shard_pct"] for a in allocs)
            if total_pct > 0:
                for a in allocs:
                    a["shard_pct"] /= total_pct

            
        else:
            allocs = [None] * world_size

        output = [None]
        dist.scatter_object_list(output, allocs, src=0)
        local_alloc = output[0]

        for g in optimizer.param_groups:
            g["lr"] = local_alloc["lr"]

        if not hasattr(allocator, "last_alloc"):
            allocator.last_alloc = [None] * world_size
        log.info(f"local_alloc: {local_alloc}")
        allocator.last_alloc[rank] = local_alloc

        total_time = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "rank":  rank,
            "train_time": train_time,
            "eval_time":  eval_time,
            "total_time": total_time,
            "cpu_usage":  local_stats["cpu_usage"],
            "gpu_usage":  local_stats["gpu_usage"],
            "memory_usage": local_stats["memory_usage"],
            "cpu_temp":   local_stats["cpu_temp"],
            "gpu_temp":   local_stats["gpu_temp"],
            "gpu_mem_percent": local_stats["gpu_mem_percent"],
            "accuracy":   global_acc,
            "top5_accuracy": global_top5_acc, # [MODIFIED] 0.0이 됨
            "f1_score":      global_f1
        }

        if rank == 0:
            ckf_vec = x_filt if x_filt.ndim == 1 else x_filt[rank]
            row["ckf_dim"] = ckf_vec
        else:
            row["ckf_dim"] = None

        row.update({
            "shard_pct":  local_alloc["shard_pct"],
            "batch_size": local_alloc["batch_size"],
            "num_workers": local_alloc["num_workers"],
            "lr":         local_alloc["lr"],
            "all_allocs": json.dumps(allocs) if rank == 0 else "",
            "global_train_loss" : global_train_loss,
            "global_eval_loss": global_eval_loss,
            'local_train_loss': local_train_loss_value,
            'local_eval_loss': local_eval_loss_value,
        })
        logger.log(row)

        
        # [NO CHANGE] 체크포인트 저장
        if rank == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
            last_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
            log.info(f"Saving checkpoint to {last_checkpoint_path} ...")
            
            save_obj = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_acc': global_acc,
            }
            torch.save(save_obj, last_checkpoint_path)
            
            # [NO CHANGE] 글로벌 서버에 보고
            _report_checkpoint(
                server_addr=args.global_server_addr,
                job_id=args.job_id,
                epoch=epoch,
                total_epochs=args.epochs,
                acc=global_acc,
                loss=global_eval_loss,
                path=last_checkpoint_path
            )

        dist.barrier()
        
        # [NO CHANGE] 중지 신호 확인
        stop_flag_path = _get_stop_flag_path(args.job_id)
        
        stop_signal = torch.tensor(0.0, device=device)
        if rank == 0 and os.path.exists(stop_flag_path):
            log.info(f"Stop signal file detected: {stop_flag_path}")
            stop_signal = torch.tensor(1.0, device=device)
            _report_job_stopped(args.global_server_addr, args.job_id)
            os.remove(stop_flag_path)

        dist.broadcast(stop_signal, src=0)

        if stop_signal.item() == 1.0:
            log.info(f"Received stop signal from Rank 0. Gracefully shutting down...")
            dist.barrier()
            break 
        # [END Stop Signal Detection]

    if rank == 0 and epoch == args.epochs:
        log.info("Training completed successfully.")
        _report_job_completed(
            args.global_server_addr,
            args.job_id,
            exit_code=0,
            final_accuracy=last_global_acc,
        )


    log.info("Training loop finished.")
    monitor.stop()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()