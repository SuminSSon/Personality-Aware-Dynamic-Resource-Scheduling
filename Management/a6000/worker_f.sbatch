#!/usr/bin/env bash
#SBATCH -J worker-f
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH -t 3-00:00:00
#SBATCH -w ariel-v11
#SBATCH -o /nas2/data/dlwmznzl1/logs/worker-f-%A.out


cd /nas2/data/dlwmznzl1/CCGRID/DARTManagement

# --- 에이전트 실행 ---
python worker_agent_f.py

exit 0
