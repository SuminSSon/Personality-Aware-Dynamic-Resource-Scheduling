import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import densenet121
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import os
import time

# 체크포인트 파일 경로
CHECKPOINT_PATH = "elastic_checkpoint.pth"

def setup_distributed(backend='gloo'):
    """ 분산 환경 설정 """
    # torch.distributed.run이 환경 변수를 자동으로 설정해줍니다.
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # GPU 사용 시: torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    print(f"[Rank {rank}] 가 입 DDP 그룹에 (World Size: {world_size})")

def cleanup():
    """ 분산 환경 종료 """
    dist.destroy_process_group()

def save_checkpoint(epoch, step, model, optimizer):
    """ 체크포인트 저장 (rank 0만 저장) """
    if int(os.environ['RANK']) == 0:
        print(f"\n[Rank 0] 체크포인트 저장: Epoch {epoch}, Step {step}")
        state = {
            'epoch': epoch,
            'step': step,
            # DDP 모델은 .module을 통해 원본 모델 state_dict에 접근
            'model_state': model.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(state, CHECKPOINT_PATH)
    
    # 모든 프로세스가 저장 완료를 기다림
    dist.barrier()

def load_checkpoint(model, optimizer):
    """ 체크포인트 불러오기 """
    start_epoch = 0
    start_step = 0
    
    if os.path.exists(CHECKPOINT_PATH):
        # 모든 프로세스가 동일한 체크포인트를 불러와야 함
        # CPU로 먼저 로드 (DDP 로드 시 권장)
        map_location = {'cuda:%d' % 0: 'cpu'} # GPU 사용 시 'cuda:%d' % int(os.environ['LOCAL_RANK'])
        
        state = torch.load(CHECKPOINT_PATH, map_location=map_location)
        
        start_epoch = state['epoch']
        start_step = state['step']
        
        # DDP 모델은 .module을 통해 원본 모델에 로드
        model.module.load_state_dict(state['model_state'])
        optimizer.load_state_dict(state['optimizer_state'])
        
        print(f"[Rank {int(os.environ['RANK'])}] 체크포인트 로드 완료: Epoch {start_epoch}, Step {start_step}")
    else:
        print(f"[Rank {int(os.environ['RANK'])}] 체크포인트 없음. 처음부터 시작.")

    return start_epoch, start_step

def train():
    setup_distributed()
    
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # --- 1. 데이터셋 준비 (MNIST) ---
    # DenseNet은 3채널 이미지를 기대하므로 Grayscale -> RGB 변환 추가
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # MNIST (1ch) -> 3ch
        transforms.Resize(32), # DenseNet은 최소 32x32 필요
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # rank 0에서만 데이터 다운로드
    if rank == 0:
        MNIST(root='/local_datasets', train=True, download=True, transform=transform)
    dist.barrier() # 다른 노드들이 다운로드 기다림

    train_dataset = MNIST(root='/local_datasets', train=True, download=False, transform=transform)

    # *** Elastic의 핵심 ***
    # DistributedSampler는 world_size에 따라 데이터를 분배
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    
    # world_size가 1일 때는 64개, 2일 때는 32개씩 로드 (총 64개)
    batch_size = 32 # 로컬 배치 크기
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    
    # --- 2. 모델 및 옵티마이저 준비 ---
    model = densenet121(num_classes=10) # MNIST는 10개 클래스
    # device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}") # GPU
    device = torch.device("cpu") # CPU (MNIST/DenseNet은 CPU로도 데모 가능)
    model.to(device)
    model = DDP(model, device_ids=None) # CPU 사용 시 device_ids=None
                                       # GPU 사용 시 device_ids=[int(os.environ['LOCAL_RANK'])]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # --- 3. 체크포인트 로드 ---
    start_epoch, start_step = load_checkpoint(model, optimizer)
    
    # --- 4. 학습 루프 ---
    num_epochs = 5
    for epoch in range(start_epoch, num_epochs):
        
        # *** 중요: Sampler에 현재 epoch 알려주기 (데이터 셔플링)
        train_sampler.set_epoch(epoch)
        
        # 만약 step 중간에 재시작했다면, 해당 step부터 다시 시작
        if start_step > 0:
            print(f"[Rank {rank}] Epoch {epoch}의 Step {start_step}부터 재시작...")
            # 데이터로더를 해당 스텝까지 건너뛰기
            loader_iter = iter(train_loader)
            for _ in range(start_step):
                try:
                    next(loader_iter)
                except StopIteration:
                    break # 스텝 건너뛰다 epoch 종료
            start_step = 0 # 다음 epoch부터는 0부터 시작
        else:
            loader_iter = iter(train_loader)
            
        step = 0
        while True:
            try:
                images, labels = next(loader_iter)
                step += 1
                
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if (step + 1) % 10 == 0:
                    print(f"[Rank {rank}] Epoch [{epoch+1}/{num_epochs}], WorldSize [{world_size}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    
                    # *** 5. 주기적 체크포인트 저장 ***
                    # (실제로는 더 긴 간격으로 저장)
                    save_checkpoint(epoch, step + 1, model, optimizer)
                    
            except StopIteration:
                # Epoch 종료
                # Epoch가 끝났을 때도 다음 epoch를 위해 0번 스텝으로 저장
                save_checkpoint(epoch + 1, 0, model, optimizer)
                break
            
            # 데모를 위해 30스텝에서 잠시 대기 (이때 2번 노드 실행)
            if step == 30 and epoch == 0 and world_size == 1:
                if rank == 0:
                    print("\n--- [Rank 0] 시연: 10초간 대기. 이떄 두 번째 터미널을 실행하세요. ---")
                    time.sleep(10)


    cleanup()

if __name__ == '__main__':
    train()