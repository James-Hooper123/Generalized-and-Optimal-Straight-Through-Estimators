#!/bin/bash
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --array=0-179
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --job-name=vae_reinmax

export PYTHONUNBUFFERED=1

cd ~/experiments

# Explicitly load the environment for the non-interactive Slurm job
module load Python/3.12.3-GCCcore-13.3.0
source ~/py312/bin/activate

WORKERS_PER_GPU=10
TOTAL_CONFIGS=1800

# Each array task handles WORKERS_PER_GPU configs
START=$((SLURM_ARRAY_TASK_ID * WORKERS_PER_GPU))

echo "--- Array Task $SLURM_ARRAY_TASK_ID ---"
echo "Starting tasks $START to $((START + WORKERS_PER_GPU - 1))"

PIDS=()
for i in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    CFG_IDX=$((START + i))
    if [ $CFG_IDX -ge $TOTAL_CONFIGS ]; then
        break
    fi

    if [ $i -eq 0 ]; then
        # First worker prints fully to stdout
        python VAE/Reinmax/Train_fixed_schedule.py --idx $CFG_IDX --verbose &
    else
        # Other workers redirect output to avoid interleaving noise, but still output basic logs
        python VAE/Reinmax/Train_fixed_schedule.py --idx $CFG_IDX > /dev/null 2>&1 &
    fi

    PIDS+=($!)
done

# Wait for all workers to finish
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "All $WORKERS_PER_GPU workers complete for array task $SLURM_ARRAY_TASK_ID"
