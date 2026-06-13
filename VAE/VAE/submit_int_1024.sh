#!/bin/bash
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --array=0-24
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err
#SBATCH --job-name=int_vae_1024

export PYTHONUNBUFFERED=1

# --- Parameter Grid ---
# np.geomspace(1e-4, 1e-2, 5)
LRS=(0.0001 0.00031623 0.001 0.00316228 0.01)

NXK_N=(128)
NXK_K=(1024)

SEEDS=(0 1 2 3 4)

EST_NAMES=("ST" "gumbel_rao" "ZGR" "MVE_int" "reinmax" "reinmax" "MVE_int")
EST_SCHEDULES=(
    "lambda epoch: fixed_temp(epoch, 0.0)"
    "lambda epoch: exp_STGS(epoch, total_epochs)"
    "lambda epoch: fixed_temp(epoch, 0.0)"
    "lambda epoch: exp_MVE(epoch, total_epochs)"
    "lambda epoch: fixed_temp(epoch, 1.2)"
    "lambda epoch: fixed_temp(epoch, 1.4)"
    "lambda epoch: fixed_temp(epoch, 1e6)"
)

EMBEDDING="int"
WORKERS_PER_GPU=7
TOTAL_CONFIGS=175

# --- Map config index to parameters ---
get_params() {
    local IDX=$1
    
    local BATCH_IDX=$((IDX / 7))
    local CONFIG_IN_BATCH=$((IDX % 7))
    
    local LR_IDX=$((BATCH_IDX / 5))
    local SEED_IDX=$((BATCH_IDX % 5))
    
    local EST_IDX=$CONFIG_IN_BATCH

    echo "${LRS[$LR_IDX]} ${NXK_N[0]} ${NXK_K[0]} ${SEEDS[$SEED_IDX]} ${EST_NAMES[$EST_IDX]} ${EST_SCHEDULES[$EST_IDX]}"
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
