import os
import argparse
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler
# [삭제] torchvision import (대부분 불필요)
# import torchvision
# import torchvision.transforms as transforms

# [NEW] NLP(Hugging Face)를 위한 Import
from transformers import AutoTokenizer, DistilBertForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from torch.optim import AdamW # Adam -> AdamW

from metrics_monitor import MetricsMonitor
from log_manager import LogManager
# [삭제] import torchvision.models
import math
import json
import psutil
from sklearn.metrics import f1_score

# [ADDED] 서버 통신 및 로깅을 위한 import
import requests 
import logging


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

    args = parser.parse_args()

    hf_cache_root = "/local_datasets/hf_cache"
    os.makedirs(hf_cache_root, exist_ok=True)

    os.environ.setdefault("HF_HOME", hf_cache_root)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_cache_root, "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_cache_root, "datasets"))

    master_addr = args.master_addr
    master_port = args.master_port

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    current_idx = torch.cuda.current_device()
    print(f"Current CUDA device index: {current_idx}")

    device_name = torch.cuda.get_device_name(current_idx)
    print(f"Current CUDA device name: {device_name}")


    print(f"Master address: {master_addr}, port: {master_port}", flush=True)
    
    world_size = args.nnodes
    rank = args.node_rank
    
    
    # [ADDED] 헬퍼 함수 (서버 통신 및 로깅)
    log = logging.getLogger(f"DDP_Script [Rank {rank}]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    def _get_stop_flag_path(job_id: str) -> str:
        """ 워커 에이전트가 생성하는 중지 신호 파일 경로 """
        return f"/tmp/{job_id}_stop.flag"

    def _report_job_completed(server_addr: str, job_id: str, exit_code: int = 0):
        if rank != 0 or not server_addr:
            return
        try:
            requests.post(f"{server_addr}/report_job_completed",
                        json={"job_id": job_id, "exit_code": exit_code}, timeout=5)
            log.info("Reported job completion to global server.")
        except requests.RequestException as e:
            log.warning(f"Failed to report completion: {e}")

    # [수정] 함수 정의: epoch, total_epochs, acc, loss 등 모든 메트릭을 인자로 받음
    def _report_checkpoint(server_addr: str, job_id: str, epoch: int, total_epochs: int, 
                           acc: float, loss: float, path: str):
        """ (Rank 0 전용) 체크포인트 및 학습 상태를 글로벌 서버에 보고 """
        if rank != 0 or not server_addr:
            return
        
        url = f"{server_addr}/report_checkpoint"
        
        # [수정] payload: 서버의 Pydantic 모델과 일치하는 키로 모든 메트릭 전송
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
            # 로그 메시지 개선
            log.info(f"Successfully reported epoch {epoch}/{total_epochs} (Acc: {acc:.2f}%) to global server.")
        except requests.RequestException as e:
            log.warning(f"Failed to report checkpoint to global server: {e}")

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

    # =================================================================
    # [MODIFIED] Data preparation (CIFAR-100 -> SST-2)
    # =================================================================
    
    # 1. Load Tokenizer
    # (Rank 0만 다운로드하고 나머지는 대기)
    if rank == 0:
        log.info(f"Rank 0 downloading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dist.barrier()
    if rank != 0:
        log.info(f"Rank {rank} loading tokenizer from cache: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 2. Load Dataset (GLUE, SST-2)
    # (Rank 0만 다운로드하고 나머지는 대기)
    if rank == 0:
        log.info("Rank 0 downloading dataset: glue/sst2")
        raw_datasets = load_dataset("glue", "sst2")
    dist.barrier()
    if rank != 0:
        log.info(f"Rank {rank} loading dataset from cache: glue/sst2")
        raw_datasets = load_dataset("glue", "sst2")

    # 3. Preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding=False) # 패딩은 DataCollator가
    
    # 4. Apply preprocessing
    log.info("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
    # 5. Format dataset for PyTorch
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    # 'label' 컬럼은 SST-2에 이미 있으므로 'labels'로 rename할 필요 없음 (모델이 알아서 인식)
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["validation"] # SST-2는 'validation' set을 test용으로 씀

    # 6. Create Data Collator (Dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    log.info("SST-2 Dataset and Tokenizer loaded.")

    # [기존] 이미지 Transform (삭제)
    # transform_train = ...
    # transform_val = ...
    # train_dataset = datasets.CIFAR100(...)
    # test_dataset  = datasets.CIFAR100(...)

    # [MODIFIED] val_loader: collate_fn 추가
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        test_dataset, 
        batch_size=64, 
        sampler=test_sampler, 
        num_workers=4,
        collate_fn=data_collator # [NEW]
    )

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
    # Hugging Face 모델은 'labels'가 제공되면 내부적으로 loss를 계산 (outputs.loss)
    # criterion = torch.nn.CrossEntropyLoss().cuda(local_rank) # (삭제)
    
    # [MODIFIED] Optimizer (Adam -> AdamW)
    optimizer = AdamW(model.parameters(), lr=args.init_lr)

    log.info("Model (DistilBert), DDP, optimizer (AdamW) initialized")

    # Modules
    monitor = MetricsMonitor(interval=5)
    monitor.start()

    # LogManager for per-node logging with dynamic filename
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

    
    # 체크포인트 저장 디렉토리 생성 (Rank 0만)
    if rank == 0 and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log.info(f"Checkpoint directory created: {args.checkpoint_dir}")

    start_epoch = 1  # 기본 시작 에포크

    # --resume_from 인자가 있으면 체크포인트 로드
    if args.resume_from and os.path.exists(args.resume_from):
        log.info(f"Loading checkpoint from {args.resume_from} ...")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.resume_from, map_location=map_location)
        
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']  # 저장된 epoch + 1 (다음에 시작할 epoch)

        log.info(f"Resumed from checkpoint. Starting epoch: {start_epoch}")
    else:
        log.info(f"Starting training from scratch (epoch 1).")

    # 모든 프로세스가 로드를 완료할 때까지 대기
    dist.barrier()

    base_seed = 42

    # --- 고정 설정 (동적 자원 할당/하이퍼파라미터 튜닝 제거) ---
    fixed_shard_pct = 1.0 / world_size
    fixed_batch_size = args.batch_size
    fixed_num_workers = 4
    fixed_lr = args.init_lr

    for epoch in range(start_epoch, args.epochs + 1):

        log.info(f"epoch {epoch} 시작")

        epoch_start = time.time()

        # ============================================================
        # ❶ 고정 자원 설정으로 DataLoader 생성
        #    - SubsetRandomSampler 로 각 rank에 균등 샤딩
        # ============================================================
        shard_pct   = fixed_shard_pct
        batch_size  = fixed_batch_size
        num_workers = fixed_num_workers

        shard_size = max(1, math.floor(dataset_size * shard_pct))

        g = torch.Generator()
        g.manual_seed(epoch)                 # deterministic shuffle
        all_indices = torch.randperm(dataset_size, generator=g).tolist()
        start = rank * shard_size
        end = start + shard_size
        if rank == world_size - 1:
            end = dataset_size
        shard_indices = all_indices[start:end]

        # [MODIFIED] DataLoader: collate_fn 추가
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
        # ❷ TRAIN — “패딩”으로 global_steps 맞추기
        # ============================================================
        model.train()

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

        # [MODIFIED] lastinput, lasttarget -> last_batch (딕셔너리)
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
            # [수정] optimizer.step()은 패딩 스텝에서 생략 (기존 코드에도 없었음)
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

        # ============================================================
        # ❸ VALIDATION — [MODIFIED] NLP에 맞게 수정
        # ============================================================
        t1 = time.time()
        local_stats = monitor.latest.copy()
        model.eval()

        # [MODIFIED] local_top5_correct 제거 (SST-2는 2개 클래스)
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
                # local_top5_correct += top5.eq(labels.view(-1,1).expand_as(top5)).any(dim=1).sum().item()
                
                # [MODIFIED] F1 스코어 계산을 위한 레이블 저장 (batch['labels'] 사용)
                local_preds.extend(preds1.cpu().tolist())
                local_labels.extend(batch['labels'].cpu().tolist())
                
                # [MODIFIED] 집계 (batch['input_ids'] 및 batch['labels'] 사용)
                total_loss += loss * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
                correct += (preds1 == batch['labels']).sum()

        eval_time = time.time() - t1

        # [수정] total_samples가 0일 경우 대비
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
            # [MODIFIED] global_top5_acc 0으로 고정
            global_top5_acc = 0.0
            # global_top5_acc = 100.0 * top5_tensor.item() / total_samples.item()
        else:
            global_eval_loss = 0.0
            global_acc = 0.0
            global_top5_acc = 0.0

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
        # ❹ (삭제됨) CKF 및 동적 자원할당/하이퍼파라미터 튜닝
        # ============================================================
        for g in optimizer.param_groups:
            g["lr"] = fixed_lr  # 안전하게 고정

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
            "top5_accuracy": global_top5_acc, # [MODIFIED]
            "f1_score":      global_f1,
            "ckf_dim": None,  # CKF 제거
            "shard_pct":  shard_pct,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "lr":         fixed_lr,
            "all_allocs": "",
            "global_train_loss" : global_train_loss,
            "global_eval_loss": global_eval_loss,
            'local_train_loss': local_train_loss_value,
            'local_eval_loss': local_eval_loss_value,
        }
        logger.log(row)

        
        # --- 체크포인트 저장 (Rank 0만 수행) ---
        if rank == 0:
            # [MODIFIED] worker_agent가 전달한 checkpoint_dir 사용
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
            last_checkpoint_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
            log.info(f"Saving checkpoint to {last_checkpoint_path} ...")
            
            save_obj = {
                'epoch': epoch + 1,  # 다음 시작할 에포크 번호
                'model_state_dict': model.module.state_dict(), # DDP 래핑 해제 후 저장
                'optimizer_state_dict': optimizer.state_dict(),
                'global_acc': global_acc, # 기타 메트릭
            }
            torch.save(save_obj, last_checkpoint_path)
            
            # [MODIFIED] 글로벌 서버에 체크포인트 보고
            _report_checkpoint(
                server_addr=args.global_server_addr,
                job_id=args.job_id,
                epoch=epoch,                 # 현재 에포크
                total_epochs=args.epochs,    # 총 에포크
                acc=global_acc,              # 현재 정확도
                loss=global_eval_loss,       # 현재 손실
                path=last_checkpoint_path
            )


        # 모든 프로세스가 다음 에포크로 넘어가기 전, Rank 0의 저장이 완료될 때까지 대기
        dist.barrier()
        
        # [ADDED] 중지 신호(Stop Signal) 확인
        stop_flag_path = _get_stop_flag_path(args.job_id)
        
        # Rank 0이 신호 파일을 확인하고 모든 노드에 전파
        stop_signal = torch.tensor(0.0, device=device)
        if rank == 0 and os.path.exists(stop_flag_path):
            log.info(f"Stop signal file detected: {stop_flag_path}")
            stop_signal = torch.tensor(1.0, device=device)
            
            # (중요) 글로벌 서버에 "이제 중지하겠다"고 보고
            _report_job_stopped(args.global_server_addr, args.job_id)
            
            # (중요) 신호 파일 삭제 (다음 잡 실행에 영향 없도록)
            os.remove(stop_flag_path)

        # 모든 노드가 Rank 0의 결정(stop_signal)을 공유받음
        dist.broadcast(stop_signal, src=0)

        if stop_signal.item() == 1.0:
            log.info(f"Received stop signal from Rank 0. Gracefully shutting down...")
            dist.barrier() # 모든 노드가 메시지를 확인할 때까지 대기
            break # epoch loop 탈출
        # [END Stop Signal Detection]

    if rank == 0 and epoch == args.epochs:
        log.info(f"Training completed successfully.")
        _report_job_completed(args.global_server_addr, args.job_id, exit_code=0)

    log.info("Training loop finished.")
    monitor.stop()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
