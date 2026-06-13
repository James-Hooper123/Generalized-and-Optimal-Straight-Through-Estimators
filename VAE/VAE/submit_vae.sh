#!/bin/bash
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --array=0-74
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --job-name=vae_grid

export PYTHONUNBUFFERED=1

# --- Parameter Grid ---
# np.geomspace(1e-4, 1e-2, 5)
LRS=(0.0001 0.00031623 0.001 0.00316228 0.01)

NXK_N=(128 32 8)
NXK_K=(4 16 64)

SEEDS=(0 1 2 3 4)

EST_NAMES=("ST" "gumbel_rao" "ZGR" "MVE" "reinmax" "reinmax")
EST_SCHEDULES=(
    "lambda epoch: fixed_temp(epoch, 0.0)"
    "lambda epoch: exp_STGS(epoch, total_epochs)"
    "lambda epoch: fixed_temp(epoch, 0.0)"
    "lambda epoch: exp_MVE(epoch, total_epochs)"
    "lambda epoch: fixed_temp(epoch, 1.2)"
    "lambda epoch: fixed_temp(epoch, 1.4)"
)

EMBEDDING="OHE"
WORKERS_PER_GPU=6
TOTAL_CONFIGS=450

# --- Map config index to parameters ---
get_params() {
    local IDX=$1
    
    local BATCH_IDX=$((IDX / 6))
    local CONFIG_IN_BATCH=$((IDX % 6))
    
    local LR_IDX=$((BATCH_IDX / 15))
    local SEED_IDX=$(((BATCH_IDX / 3) % 5))
    local SUB_BATCH_IDX=$((BATCH_IDX % 3))
    
    local NXK_IDX=$((CONFIG_IN_BATCH / 2))
    local PARITY=$((CONFIG_IN_BATCH % 2))
    
    local EST_IDX=$(((2 * NXK_IDX + 2 * SUB_BATCH_IDX + PARITY) % 6))

    echo "${LRS[$LR_IDX]} ${NXK_N[$NXK_IDX]} ${NXK_K[$NXK_IDX]} ${SEEDS[$SEED_IDX]} ${EST_NAMES[$EST_IDX]} ${EST_SCHEDULES[$EST_IDX]}"
}

cd ~/experiments

# Explicitly load the environment for the non-interactive Slurm job
module load Python/3.12.3-GCCcore-13.3.0
source ~/py312/bin/activate

# Each array task handles WORKERS_PER_GPU configs
START=$((SLURM_ARRAY_TASK_ID * WORKERS_PER_GPU))

PIDS=()
for i in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    CFG_IDX=$((START + i))
    if [ $CFG_IDX -ge $TOTAL_CONFIGS ]; then
        break
    fi

    read LR N K SEED ESTIMATOR SCHEDULE <<< "$(get_params $CFG_IDX)"

    echo "[$CFG_IDX] LR=$LR N=$N K=$K Seed=$SEED Est=$ESTIMATOR Schedule=$SCHEDULE"

    if [ $i -eq 0 ]; then
        # First worker prints to stdout
        python VAE/VAE/train_VAE.py \
            --gradient_estimator "$ESTIMATOR" \
            --embedding "$EMBEDDING" \
            --temperature_schedule "$SCHEDULE" \
            --n "$N" \
            --k "$K" \
            --lr "$LR" \
            --seed "$SEED" &
    else
        # Other workers run silently
        python VAE/VAE/train_VAE.py \
            --gradient_estimator "$ESTIMATOR" \
            --embedding "$EMBEDDING" \
            --temperature_schedule "$SCHEDULE" \
            --n "$N" \
            --k "$K" \
            --lr "$LR" \
            --seed "$SEED" > /dev/null 2>&1 &
    fi

    PIDS+=($!)
done

# Wait for all workers to finish
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "All $WORKERS_PER_GPU workers complete for array task $SLURM_ARRAY_TASK_ID"
