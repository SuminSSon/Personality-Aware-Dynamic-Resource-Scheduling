#!/usr/bin/env bash
#SBATCH -J worker-h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -p batch_grad
#SBATCH -t 3-00:00:00
#SBATCH -w ariel-v11
#SBATCH -o /data/breath12/logs/worker-h-%A.out

cd /data/breath12/CCGRID/DARTManagement

# --- 에이전트 실행 ---
python worker_agent_h.py

exit 0
