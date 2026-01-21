from typing import Optional

import os
import argparse
import time
import numpy as np
import torch
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
        # 서버가 잠깐 죽어도 학습은 계속
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
    
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms

from metrics_monitor import MetricsMonitor
from log_manager import LogManager
from torchvision import datasets, transforms, models
import math
import json
import psutil
from sklearn.metrics import f1_score

# [ADDED] 서버 통신 및 로깅을 위한 import
import requests 
import logging
import os
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetVal(Dataset):
    def __init__(self, root, wnids_txt, transform=None):
        self.root      = root                     # ← root 저장
        self.transform = transform

        # 1) val_annotations.txt 파싱
        ann_path = os.path.join(self.root, 'val_annotations.txt')
        self.items = [line.split('\t')[:2]
                      for line in open(ann_path).read().splitlines()]

        # 2) wnids.txt 읽어서 클래스 목록·매핑 생성
        wnids = open(wnids_txt).read().splitlines()
        self.classes  = wnids                   # ← ImageFolder 호환용
        self.cls2idx  = {c: i for i, c in enumerate(wnids)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, cls = self.items[idx]
        img_path = os.path.join(self.root, 'images', img_name)  # ← self.root 사용
        img      = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.cls2idx[cls]
        return img, label

def main():
    parser = argparse.ArgumentParser()
    # ==== Global 서버 / WorkerAgent 인터페이스 ====
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--global_server_addr", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)

    # ==== 학습 하이퍼파라미터 ====
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size_per_gpu", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_accum", type=int, default=1)

    # ==== 데이터셋 / 모델 / 재시작 ====
    parser.add_argument("--dataset", type=str, default="TinyImageNet")
    parser.add_argument("--model_name", type=str, default="DenseNet-121")
    parser.add_argument("--resume_from", type=str, default=None)

    # ==== 분산(DDP) 관련 ====
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=int, default=29500)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=1)

    args = parser.parse_args()

    master_addr = args.master_addr
    master_port = args.master_port

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    current_idx = torch.cuda.current_device()
    print(f"Current CUDA device index: {current_idx}")

    device_name = torch.cuda.get_device_name(current_idx)
    print(f"Current CUDA device name: {device_name}")


    print(f"Master address: {master_addr}, port: {master_port}", flush=True)
    
    world_size = args.nnodes * args.nproc_per_node
    rank = args.node_rank * args.nproc_per_node + local_rank
    
    
    # [ADDED] 헬퍼 함수 (서버 통신 및 로깅)
    log = logging.getLogger(f"DDP_Script [Rank {rank}]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    def _get_stop_flag_path(job_id: str, attempt: int) -> str:
        return f"/tmp/{job_id}-attempt-{attempt}.flag"

    def _report_job_completed(
        server_addr: str,
        job_id: str,
        exit_code: int = 0,
        final_accuracy: Optional[float] = None,
    ):
        if rank != 0 or not server_addr:
            return
        try:
            payload = {
                "job_id": job_id,
                "status": ("FINISHED" if exit_code == 0 else "FAILED"),
                "exit_code": exit_code,
            }
            # accuracy가 있으면 같이 보냄
            if final_accuracy is not None:
                payload["final_accuracy"] = float(final_accuracy)

            requests.post(
                f"{server_addr}/report_job_status",
                json=payload,
                timeout=5,
            )
            log.info(
                "Reported job completion to global server."
                + (f" (final_accuracy={final_accuracy:.2f})" if final_accuracy is not None else "")
            )
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

    # Data preparation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),     # 랜덤 크롭 후 224×224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],  # TinyImageNet 통계
                             std=[0.2302, 0.2265, 0.2262]),
    ])

    # 검증용 Transform (추가)
    transform_val = transforms.Compose([
         transforms.Resize(256),                # 짧은 쪽을 256으로 리사이즈
        transforms.CenterCrop(224),            # 중앙 224×224 크롭
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2302, 0.2265, 0.2262]),
    ])

    train_dataset = datasets.ImageFolder(
        root='/local_datasets/tiny-imagenet-200/train',
        transform=transform_train)
    
    wnids = open('/local_datasets/tiny-imagenet-200/wnids.txt').read().splitlines()
    train_dataset.classes      = wnids
    train_dataset.class_to_idx = {c: i for i, c in enumerate(wnids)}

    new_samples = []
    for img_path, _ in train_dataset.samples:
        # 경로: …/train/<wnid>/images/<file>
        # -1: file, -2: 'images', -3: '<wnid>'
        wnid = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        new_label = train_dataset.class_to_idx[wnid]
        new_samples.append((img_path, new_label))

    train_dataset.samples = new_samples
    train_dataset.imgs    = new_samples

    # ↓ val은 ImageFolder가 아니라 TinyImageNetVal 사용
    val_dataset = TinyImageNetVal(
        root='/local_datasets/tiny-imagenet-200/val',
        wnids_txt='/local_datasets/tiny-imagenet-200/wnids.txt',
        transform=transform_val)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader  = DataLoader(train_dataset, batch_size=32,
                            sampler=train_sampler, num_workers=4,pin_memory=True)
    val_loader    = DataLoader(val_dataset,   batch_size=32,
                            sampler=val_sampler,   num_workers=4,pin_memory=True)
    
    imgs, labels = next(iter(val_loader))
    print("Val sample labels:", labels.unique(), flush=True)
    print("Train samples:", len(train_dataset), flush=True)  # 100,000이어야 함
    print("Val   samples:", len(val_dataset), flush=True)    # 10,000이어야 함

    # Model, DDP, criterion, optimizer
    # Swap ResNet50 to DenseNet-121
    model = torchvision.models.densenet121(pretrained=False, num_classes=200).cuda(local_rank)
    model.cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    criterion = torch.nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    batch_size = args.batch_size_per_gpu

    log.info("Model, DDP, criterion, optimizer initialized")

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
    # [MODIFIED] args.checkpoint_dir 사용 (worker_agent가 job_id별로 전달)
    if rank == 0 and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log.info(f"Checkpoint directory created: {args.checkpoint_dir}")

    start_epoch = 1  # 기본 시작 에포크

    # --resume_from 인자가 있으면 체크포인트 로드
    if args.resume_from and os.path.exists(args.resume_from):
        log.info(f"Loading checkpoint from {args.resume_from} ...")
        # 모든 장치가 동일한 위치에서 로드하도록 map_location 설정
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.resume_from, map_location=map_location)
        
        # DDP로 래핑된 모델은 .module을 통해 실제 모델의 state_dict에 접근
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
    fixed_batch_size = args.batch_size_per_gpu
    fixed_num_workers = 4
    fixed_lr = args.learning_rate
    last_global_acc: float = 0.0
    global_step: float = 0.0
    stopped_early: bool = False

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

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(shard_indices),
            num_workers=num_workers,
            pin_memory=True
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

        lastinput = None
        lasttarget = None

        for inputs, targets in train_loader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            local_train_loss += loss.detach() * inputs.size(0)
            local_train_samples += inputs.size(0)
            
            loss.backward()
            optimizer.step()

            lastinput = inputs
            lasttarget = targets

            step += 1

        model.eval() 

        while step < max_steps:
            optimizer.zero_grad()
            outputs = model(lastinput)
            loss = criterion(outputs, lasttarget)
            loss.backward()
            step += 1

        train_time = time.time() - t0
        local_train_loss_value = local_train_loss.item() / local_train_samples.item()

        log.info(f"epoch {epoch} train step time: {train_time:.2f} seconds")

        dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_train_samples, op=dist.ReduceOp.SUM)
        global_train_loss = local_train_loss.item() / local_train_samples.item()

        dist.barrier()

        log.info(f"epoch {epoch} 끝 (local_steps={local_steps}, global_steps={max_steps})")

        # ★ Rank 0에서 누적 step 기반 attained_service 리포트
        if rank == 0:
            global_step += max_steps  # 에폭별 max_steps 누적
            report_attained_service(
                args.global_server_addr,
                args.job_id,
                float(global_step),
            )

        # ============================================================
        # ❸ VALIDATION — 기존 코드 그대로
        # ============================================================
        t1 = time.time()
        local_stats = monitor.latest.copy()
        model.eval()

        local_top5_correct = 0
        local_preds = []
        local_labels = []

        total_loss = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0, device=device)
        correct = torch.tensor(0, device=device)
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds1 = outputs.argmax(dim=1)
                _, top5 = outputs.topk(5, dim=1, largest=True, sorted=True)
                local_top5_correct += top5.eq(labels.view(-1,1).expand_as(top5)).any(dim=1).sum().item()
                local_preds.extend(preds1.cpu().tolist())
                local_labels.extend(labels.cpu().tolist())
                total_loss += loss * images.size(0)
                total_samples += images.size(0)
                correct += (preds1 == labels).sum()

        eval_time = time.time() - t1

        local_eval_loss_value = total_loss.item() / total_samples.item()

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

        top5_tensor = torch.tensor(local_top5_correct, device=device)
        dist.all_reduce(top5_tensor, op=dist.ReduceOp.SUM)

        global_eval_loss = total_loss.item() / total_samples.item()
        global_acc = 100.0 * correct.item() / total_samples.item()
        global_top5_acc = 100.0 * top5_tensor.item() / total_samples.item()

        local_stats["accuracy"] = global_acc
        log.info(f"val acc {global_acc:.2f}%")
        # rank 0이 마지막 global_acc를 기억해 둔다 (job 완료 시 report_job_status에 넣기 위함)
        if rank == 0:
            last_global_acc = float(global_acc)

        if rank == 0:
            # step 당 속도 (Pollux goodput용)
            steps_per_sec = max_steps / max(train_time, 1e-6)

            send_progress(
                global_server_addr=args.global_server_addr,
                job_id=args.job_id,
                world_size=world_size,
                local_batch=batch_size,
                grad_accum=args.grad_accum,
                steps_per_sec=steps_per_sec,
                loss=float(global_eval_loss),
                accuracy=float(global_acc),
                gns=None,
                stat_eff=None,
            )
            
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
            f1 = f1_score(all_labels, all_preds, average='macro')
        else:
            f1 = 0.0

        f1_tensor = torch.tensor(f1, device=device)
        dist.broadcast(f1_tensor, src=0)
        global_f1 = f1_tensor.item()

        # ============================================================
        # ❹ (삭제됨) CKF 및 동적 자원할당/하이퍼파라미터 튜닝
        #     - 고정 lr/배치/워커/샤드 비율 사용
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
            "top5_accuracy": global_top5_acc,
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
        
        # stop signal
        stop_flag_path = _get_stop_flag_path(args.job_id, args.attempt)
        stop_signal = torch.tensor(0.0, device=device)
        if rank == 0 and os.path.exists(stop_flag_path):
            log.info(f"Stop signal file detected: {stop_flag_path}")
            stop_signal = torch.tensor(1.0, device=device)
            _report_job_stopped(args.global_server_addr, args.job_id)
            # os.remove(stop_flag_path)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1.0:
            log.info("Received stop signal. Gracefully shutting down...")
            stopped_early = True
            dist.barrier()
            break

    # if rank == 0 and epoch == args.epochs:
    #     log.info(
    #         f"Training completed successfully. Reporting final_accuracy={last_global_acc:.2f} to global server."
    #     )
    #     _report_job_completed(
    #         args.global_server_addr,
    #         args.job_id,
    #         exit_code=0,
    #         final_accuracy=last_global_acc,
    #     )

    log.info("Training loop finished.")
    monitor.stop()
    dist.destroy_process_group()

    # ★ STOP으로 끊긴 경우에는 FINISHED 보고 안 함
    if rank == 0 and not stopped_early:
        log.info(
            f"Training completed successfully. Reporting final_accuracy={last_global_acc:.2f} to global server."
        )
        _report_job_completed(
            args.global_server_addr,
            args.job_id,
            exit_code=0,
            final_accuracy=last_global_acc,
        )

if __name__ == '__main__':
    main()
