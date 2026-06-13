#!/bin/bash
#SBATCH --job-name=bv
#SBATCH --partition=cpuextralong
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-8
#SBATCH --output=slurm_logs/bias_variance_%A_%a.out
#SBATCH --error=slurm_logs/bias_variance_%A_%a.err

export PYTHONUNBUFFERED=1

# Navigate to your working directory
cd ~/experiments

# Load the environment for the non-interactive Slurm job
module load Python/3.12.3-GCCcore-13.3.0
source ~/py312/bin/activate

# Define parameter ranges matching your Python script
SEEDS=(0 1 2)
DIMS=(4 8 12)

# Calculate array bounds dynamically 
NUM_DIMS=${#DIMS[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

if [ -z "$TASK_ID" ]; then
    echo "Error: This script must be run as a Slurm job array."
    echo "Usage: sbatch submit_bias_variance.sh"
    exit 1
fi

# Ensure the logs directory exists
mkdir -p slurm_logs

# Map Task ID to specific seed and dim
SEED_IDX=$((TASK_ID / NUM_DIMS))
DIM_IDX=$((TASK_ID % NUM_DIMS))

SEED=${SEEDS[$SEED_IDX]}
DIM=${DIMS[$DIM_IDX]}

echo "Starting Bias_Variance.py for Standard Run (dim=$DIM, seed=$SEED)..."
python Bias_Variance/Bias_Variance.py --dim "$DIM" --seed "$SEED"

echo "Finished task $TASK_ID!"