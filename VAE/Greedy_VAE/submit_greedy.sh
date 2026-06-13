#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --job-name=greedy_vae

export PYTHONUNBUFFERED=1

cd ~/experiments

# Explicitly load the environment for the non-interactive Slurm job
module load Python/3.12.3-GCCcore-13.3.0
source ~/py312/bin/activate

# 2 estimators x 5 seeds = 10 configurations total
EST_NAMES=("MVE" "gumbel_rao")

# array ranges from 0 to 14
CFG_IDX=$SLURM_ARRAY_TASK_ID

EST_IDX=$((CFG_IDX / 5))
SEED=$((CFG_IDX % 5))

ESTIMATOR=${EST_NAMES[$EST_IDX]}

echo "--- Array Task $SLURM_ARRAY_TASK_ID ---"
echo "Starting Estimator=$ESTIMATOR with Seed=$SEED"

# Optional: You can set a timeout or simply run the script
python VAE/Greedy_VAE/Greedy_optim.py \
    --estimator "$ESTIMATOR" \
    --seed "$SEED" \
    --nxk "(8,64)"

echo "Task $SLURM_ARRAY_TASK_ID complete!"
