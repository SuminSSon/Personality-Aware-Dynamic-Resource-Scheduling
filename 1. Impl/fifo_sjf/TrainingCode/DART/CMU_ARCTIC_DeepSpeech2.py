import os
import argparse
import time
import math
import json
import re
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset, SubsetRandomSampler

# ===== Audio =====
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# metrics / utils (schema compatibility kept)
from sklearn.metrics import f1_score  # not used for ASR, kept for schema

from metrics_monitor import MetricsMonitor
from allocator_interface import ResourceAllocator
from log_manager import LogManager

# 서버 통신 / 로깅 (kept)
import requests
import logging

###############################################################
# Vocabulary & Text Utils (DeepSpeech2-style CTC)
###############################################################
CHARS = " 'abcdefghijklmnopqrstuvwxyz"  # 1..N
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARS)}
ID_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
BLANK = 0

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z' ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def text_to_ids(s: str):
    s = normalize_text(s)
    return [CHAR_TO_ID[c] for c in s if c in CHAR_TO_ID]

def ids_to_text(ids: List[int]) -> str:
    return ''.join(ID_TO_CHAR.get(i, '') for i in ids)

###############################################################
# CMU-ARCTIC Dataset (robust, regex-based)
###############################################################
class CMUArcticDataset(Dataset):
    """
    Supports either:
      1) data_root/{train,test} with .wav and paired .txt
      2) data_root containing one or more CMU-ARCTIC speaker folders (wav/ + etc/txt.done.data)
      3) data_root itself being a single CMU-ARCTIC speaker folder
    """
    def __init__(self, data_root: str, split: str = 'train', sample_rate: int = 16000, n_mels: int = 128):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        self.examples = self._index_dataset()
        self.melspec = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.amp2db = AmplitudeToDB()

    def _parse_txt_done(self, spk_dir: str) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        txt_file = os.path.join(spk_dir, 'etc', 'txt.done.data')
        wav_dir = os.path.join(spk_dir, 'wav')
        if not (os.path.isfile(txt_file) and os.path.isdir(wav_dir)):
            return pairs
        # robust to optional spaces: ( arctic_a0001 "TEXT" ) or (arctic_a0001 "TEXT")
        pat = re.compile(r'^\(\s*(\S+)\s+"(.*)"\s*\)$')
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                m = pat.match(line.strip())
                if not m:
                    continue
                utt_id, text = m.group(1), m.group(2)
                wav_path = os.path.join(wav_dir, f"{utt_id}.wav")
                if os.path.exists(wav_path):
                    pairs.append((wav_path, text))
        return pairs

    def _index_dataset(self) -> List[Tuple[str, str]]:
        # Case 1: explicit split dirs with .wav + .txt files
        pairs: List[Tuple[str, str]] = []
        split_dir = os.path.join(self.data_root, self.split)
        if os.path.isdir(split_dir):
            for root, _, files in os.walk(split_dir):
                for f in files:
                    if f.endswith('.wav'):
                        wav_path = os.path.join(root, f)
                        txt_path = os.path.splitext(wav_path)[0] + '.txt'
                        if os.path.exists(txt_path):
                            with open(txt_path, 'r', encoding='utf-8') as t:
                                pairs.append((wav_path, t.read().strip()))
            return [(w, s) for (w, s) in pairs if len(text_to_ids(s)) > 0]

        # Case 2: data_root contains multiple speakers
        speaker_pairs: List[Tuple[str, str]] = []
        found_any = False
        for name in os.listdir(self.data_root):
            spk_dir = os.path.join(self.data_root, name)
            if not os.path.isdir(spk_dir):
                continue
            parsed = self._parse_txt_done(spk_dir)
            if parsed:
                found_any = True
                parsed.sort(key=lambda x: x[0])
                n = len(parsed); cut = int(n * 0.9)
                speaker_pairs.extend(parsed[:cut] if self.split == 'train' else parsed[cut:])
        if found_any:
            return [(w, s) for (w, s) in speaker_pairs if len(text_to_ids(s)) > 0]

        # Case 3: data_root itself is a single speaker folder
        single = self._parse_txt_done(self.data_root)
        if single:
            single.sort(key=lambda x: x[0])
            n = len(single); cut = int(n * 0.9)
            pairs = single[:cut] if self.split == 'train' else single[cut:]
            return [(w, s) for (w, s) in pairs if len(text_to_ids(s)) > 0]

        # Nothing found
        return []

    def __len__(self):
        return len(self.examples)

    def _wav_to_mel(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # mono
        mel = self.melspec(wav)
        mel = self.amp2db(mel)
        mean = mel.mean(); std = mel.std().clamp_min(1e-5)
        mel = (mel - mean) / std
        return mel.squeeze(0)  # [n_mels, T]

    def __getitem__(self, idx):
        wav_path, text = self.examples[idx]
        target = torch.tensor(text_to_ids(text), dtype=torch.long)
        features = self._wav_to_mel(wav_path)
        return features, target


def speech_collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    # drop empty-target samples just in case
    batch = [(f, t) for (f, t) in batch if t.numel() > 0]
    n_mels = batch[0][0].shape[0]
    B = len(batch)
    times = [x[0].shape[1] for x in batch]
    max_t = max(times)
    inputs = torch.zeros(B, 1, n_mels, max_t, dtype=torch.float32)
    input_lengths = torch.tensor(times, dtype=torch.long)
    targets, target_lengths = [], []
    for i, (feat, tgt) in enumerate(batch):
        T = feat.shape[1]
        inputs[i, 0, :, :T] = feat
        targets.append(tgt)
        target_lengths.append(len(tgt))
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return inputs, input_lengths, targets, target_lengths

###############################################################
# DeepSpeech2 (minimal)
###############################################################
class DeepSpeech2(nn.Module):
    def __init__(self, n_mels: int, n_classes: int, rnn_hidden: int = 512, rnn_layers: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
        )
        # freq stride total = 2 * 2 -> /4; channels=32
        self.rnn_input = (n_mels // 4) * 32
        self.rnn = nn.LSTM(input_size=self.rnn_input, hidden_size=512, num_layers=rnn_layers,
                           dropout=0.1, bidirectional=True, batch_first=False)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.Hardswish(),
            nn.Linear(512, n_classes),
        )

    def _output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        # time stride: conv1=2, conv2=1 -> /2 overall
        t = torch.floor((input_lengths + 2*5 - 11)/2 + 1)
        t = torch.clamp(t, min=1).to(torch.long)
        return t

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor):
        B = x.size(0)
        x = self.conv(x)              # [B, 32, F', T']
        C, F, T = x.size(1), x.size(2), x.size(3)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, T, C, F]
        x = x.view(B, T, C*F).permute(1, 0, 2).contiguous()  # [T, B, C*F]
        x, _ = self.rnn(x)            # [T, B, 1024]
        x = self.classifier(x)        # [T, B, n_classes]
        out_lengths = self._output_lengths(input_lengths)
        return x, out_lengths

###############################################################
# Decoding & Error Rates
###############################################################

def ctc_greedy_decode(logits: torch.Tensor) -> List[List[int]]:
    probs = torch.argmax(logits, dim=-1)  # [T, B]
    T, B = probs.shape
    hyps = []
    for b in range(B):
        prev = -1
        seq = []
        for t in range(T):
            p = probs[t, b].item()
            if p != prev and p != BLANK:
                seq.append(p)
            prev = p
        hyps.append(seq)
    return hyps


def _edit_distance(a: List[str], b: List[str]) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[-1][-1]


def wer(ref: str, hyp: str) -> float:
    r = ref.strip().split(); h = hyp.strip().split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _edit_distance(r, h) / max(1, len(r))


def cer(ref: str, hyp: str) -> float:
    r = list(ref.strip()); h = list(hyp.strip())
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _edit_distance(r, h) / max(1, len(r))

###############################################################
# Main Training Loop (structure preserved)
###############################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--init-lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)

    # kept: global server params
    parser.add_argument('--job_id', type=str, default='local_test')
    parser.add_argument('--global_server_addr', type=str, default=None)

    # data params
    parser.add_argument('--data_root', type=str, default='/local_datasets/ARCTIC/cmu_us_bdl_arctic')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--n_mels', type=int, default=128)

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

    world_size = args.nnodes
    rank = args.node_rank

    # logging
    log = logging.getLogger(f"DDP_Script [Rank {rank}]")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")

    def _get_stop_flag_path(job_id: str) -> str:
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

    def _report_job_stopped(server_addr: str, job_id: str):
        if rank != 0 or not server_addr:
            return
        url = f"{server_addr}/report_job_stopped"
        payload = {"job_id": job_id}
        try:
            requests.post(url, json=payload, timeout=5)
            log.info("Successfully reported graceful shutdown to global server.")
        except requests.RequestException as e:
            log.warning(f"Failed to report job stop to global server: {e}")

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

    # ===================== Data =====================
    train_dataset = CMUArcticDataset(args.data_root, split='train', sample_rate=args.sample_rate, n_mels=args.n_mels)
    val_dataset = CMUArcticDataset(args.data_root, split='test', sample_rate=args.sample_rate, n_mels=args.n_mels)

    # keep original style: local shard per rank using SubsetRandomSampler for train
    dataset_size = len(train_dataset)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True, collate_fn=speech_collate)

    # ===================== Model =====================
    n_classes = len(CHARS) + 1
    model = DeepSpeech2(n_mels=args.n_mels, n_classes=n_classes).cuda(local_rank)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True).cuda(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    log.info("Model, DDP, criterion, optimizer initialized")

    # Modules
    monitor = MetricsMonitor(interval=5)
    monitor.start()
    allocator = ResourceAllocator(
        min_batch_size=4, max_batch_size=16,
        min_lr=1e-4, max_lr=1e-2,
        gamma=0.1, target_accuracy=90.0,
        ckf_params={
            'state_dim': 8,
            'obs_dim':   8,
            'process_noise_cov':    np.eye(8) * 0.05,
            'measurement_noise_cov': np.eye(8) * 0.1,
        },
        dyn_ratio_params={'lower':0.1,'upper':0.5,'k':10,'threshold':0.5}
    )

    # LogManager
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

    # checkpoint dir
    if rank == 0 and not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        log.info(f"Checkpoint directory created: {args.checkpoint_dir}")

    start_epoch = 1
    if args.resume_from and os.path.exists(args.resume_from):
        log.info(f"Loading checkpoint from {args.resume_from} ...")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(args.resume_from, map_location=map_location)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        log.info(f"Resumed from checkpoint. Starting epoch: {start_epoch}")
    else:
        log.info("Starting training from scratch (epoch 1).")

    dist.barrier()

    # --- Fixed resource settings (kept semantics from prior pattern) ---
    fixed_shard_pct = 1.0 / world_size
    fixed_batch_size = args.batch_size
    fixed_num_workers = 4
    fixed_lr = args.init_lr

    for epoch in range(start_epoch, args.epochs + 1):
        log.info(f"epoch {epoch} 시작")
        epoch_start = time.time()

        shard_pct   = fixed_shard_pct
        batch_size  = fixed_batch_size
        num_workers = fixed_num_workers

        shard_size = max(1, math.floor(dataset_size * shard_pct))
        g = torch.Generator(); g.manual_seed(epoch)
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
            pin_memory=True,
            collate_fn=speech_collate,
        )

        # ========== TRAIN with padding to max_steps ==========
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

        last_batch = None

        for batch in train_loader:
            inputs, in_lens, targets, tgt_lens = batch
            inputs = inputs.to(device, non_blocking=True)
            in_lens = in_lens.to(device)
            targets = targets.to(device)
            tgt_lens = tgt_lens.to(device)

            optimizer.zero_grad()
            logits, out_lens = model(inputs, in_lens)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, targets, out_lens, tgt_lens)
            loss.backward()
            # optional: grad clipping (can be enabled if needed)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            bs = inputs.size(0)
            local_train_loss += loss.detach() * bs
            local_train_samples += bs

            last_batch = batch
            step += 1

        model.eval()
        while step < max_steps and last_batch is not None:
            inputs, in_lens, targets, tgt_lens = last_batch
            inputs = inputs.to(device, non_blocking=True)
            in_lens = in_lens.to(device)
            targets = targets.to(device)
            tgt_lens = tgt_lens.to(device)
            optimizer.zero_grad()
            logits, out_lens = model(inputs, in_lens)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = criterion(log_probs, targets, out_lens, tgt_lens)
            loss.backward()
            step += 1

        train_time = time.time() - t0
        local_train_loss_value = (local_train_loss.item() / max(1, local_train_samples.item()))

        dist.all_reduce(local_train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_train_samples, op=dist.ReduceOp.SUM)
        global_train_loss = local_train_loss.item() / max(1, local_train_samples.item())
        dist.barrier()

        # ========== VALIDATION ==========
        t1 = time.time()
        local_stats = monitor.latest.copy()
        model.eval()

        total_loss = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0, device=device)
        wer_sum = 0.0
        cer_sum = 0.0
        utt_count = 0

        for batch in val_loader:
            with torch.no_grad():
                inputs, in_lens, targets, tgt_lens = batch
                inputs = inputs.to(device)
                in_lens = in_lens.to(device)
                targets = targets.to(device)
                tgt_lens = tgt_lens.to(device)
                logits, out_lens = model(inputs, in_lens)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                loss = criterion(log_probs, targets, out_lens, tgt_lens)

            bs = inputs.size(0)
            total_loss += loss * bs
            total_samples += bs

            hyps_ids = ctc_greedy_decode(logits)
            offset = 0
            for b in range(bs):
                L = tgt_lens[b].item()
                ref_ids = targets[offset:offset+L].tolist(); offset += L
                ref = ids_to_text(ref_ids)
                hyp = ids_to_text(hyps_ids[b])
                wer_sum += wer(ref, hyp)
                cer_sum += cer(ref, hyp)
                utt_count += 1

        eval_time = time.time() - t1
        local_eval_loss_value = total_loss.item() / max(1, total_samples.item())

        # global reduce
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        wer_sum_t = torch.tensor(wer_sum, device=device)
        cer_sum_t = torch.tensor(cer_sum, device=device)
        utt_count_t = torch.tensor(utt_count, device=device, dtype=torch.long)
        dist.all_reduce(wer_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(cer_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(utt_count_t, op=dist.ReduceOp.SUM)

        total_utts_global = int(utt_count_t.item())
        if total_utts_global == 0:
            if rank == 0:
                log.warning("Validation set is empty across all ranks; metrics are undefined.")
            avg_wer_g = float("nan"); avg_cer_g = float("nan"); global_acc = float("nan")
        else:
            avg_wer_g = float(wer_sum_t.item()) / total_utts_global
            avg_cer_g = float(cer_sum_t.item()) / total_utts_global
            global_acc = max(0.0, (1.0 - avg_wer_g) * 100.0)

        global_eval_loss = total_loss.item() / max(1, total_samples.item())
        global_top5_acc = 0.0
        global_f1 = 0.0

        local_stats["accuracy"] = global_acc
        log.info(f"val WER {avg_wer_g}, CER {avg_cer_g}, pseudo-acc {global_acc}")

        # === Resource allocator (kept) ===
        if rank == 0:
            gathered = [None for _ in range(world_size)]
            dist.gather_object(local_stats, gathered, dst=0)
        else:
            dist.gather_object(local_stats, dst=0)
        dist.barrier()

        if rank == 0:
            metrics = {k:[g[k] for g in gathered] for k in [
                'cpu_usage','gpu_usage','memory_usage','cpu_temp','gpu_temp','gpu_mem_percent','accuracy','cpu_cores']}
            metrics.update({'ops':[0.0]*world_size,'mu':[0.0]*world_size})
            x_filt, _ = allocator.preprocess_ckf(metrics)
            try:
                log.info(f"CKF x_filt vector: {[x_filt[i] for i in range(8)]}")
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
            "top5_accuracy": global_top5_acc,
            "f1_score":      global_f1,
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

        # checkpoint (rank 0)
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

        # stop signal
        stop_flag_path = _get_stop_flag_path(args.job_id)
        stop_signal = torch.tensor(0.0, device=device)
        if rank == 0 and os.path.exists(stop_flag_path):
            log.info(f"Stop signal file detected: {stop_flag_path}")
            stop_signal = torch.tensor(1.0, device=device)
            _report_job_stopped(args.global_server_addr, args.job_id)
            os.remove(stop_flag_path)
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1.0:
            log.info("Received stop signal. Gracefully shutting down...")
            dist.barrier()
            break

    if rank == 0 and epoch == args.epochs:
        log.info("Training completed successfully.")
        _report_job_completed(args.global_server_addr, args.job_id, exit_code=0)

    log.info("Training loop finished.")
    monitor.stop()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
