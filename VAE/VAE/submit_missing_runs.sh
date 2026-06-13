#!/bin/bash
#SBATCH --partition=amdgpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --array=0
#SBATCH --output=slurm_logs/missing_%A_%a.out
#SBATCH --error=slurm_logs/missing_%A_%a.err
#SBATCH --job-name=vae_missing_runs

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Define the exact configurations for the 1 missing run
NS=(128)
KS=(256)
LRS=(0.01)
SEEDS=(0)

EMBEDDING="int"
ESTIMATOR="gumbel_rao"
SCHEDULE="lambda epoch: exp_STGS(epoch, total_epochs)"

N=${NS[$SLURM_ARRAY_TASK_ID]}
K=${KS[$SLURM_ARRAY_TASK_ID]}
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Running task $SLURM_ARRAY_TASK_ID with configuration:"
echo "LR=$LR N=$N K=$K Seed=$SEED Est=$ESTIMATOR Schedule=$SCHEDULE"

cd ~/experiments

# Explicitly load the environment for the non-interactive Slurm job
module load Python/3.12.3-GCCcore-13.3.0
source ~/py312/bin/activate

python VAE/VAE/train_VAE.py \
    --gradient_estimator "$ESTIMATOR" \
    --embedding "$EMBEDDING" \
    --temperature_schedule "$SCHEDULE" \
    --n "$N" \
    --k "$K" \
    --lr "$LR" \
    --seed "$SEED"
