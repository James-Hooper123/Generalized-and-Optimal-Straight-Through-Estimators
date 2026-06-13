#!/bin/bash
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/%A_reinmax_missing.out
#SBATCH --error=slurm_logs/%A_reinmax_missing.err
#SBATCH --job-name=vae_reinmax_missing

export PYTHONUNBUFFERED=1

cd ~/experiments

module load Python/3.12.3-GCCcore-13.3.0
source ~/py312/bin/activate

# Dynamically find which config indices are missing by checking for their output files
MISSING_INDICES=$(python - <<'EOF'
import os, numpy as np
from pathlib import Path

data_dir = os.path.expanduser("~/experiments/Results/Reinmax")

ESTIMATORS = ['MVE', 'reinmax']
TEMPS = np.round(np.linspace(0.01, 1, 30), 2)
LRS = np.geomspace(1e-4, 1e-2, 30)

tasks = []
for est in ESTIMATORS:
    for temp in TEMPS:
        for lr in LRS:
            tasks.append({'estimator': est, 'temp': temp, 'lr': lr})

missing = []
for idx, cfg in enumerate(tasks):
    f_name = f"{cfg['estimator']}_temp{str(cfg['temp']).replace('.', '_')}_lr{str(cfg['lr']).replace('.', '_')}.pkl"
    if not os.path.exists(os.path.join(data_dir, f_name)):
        missing.append(str(idx))

print(' '.join(missing))
EOF
)

echo "Missing indices: $MISSING_INDICES"

if [ -z "$MISSING_INDICES" ]; then
    echo "No missing configs found. Nothing to do."
    exit 0
fi

# Run all missing configs in parallel, with full stderr logging (not /dev/null)
PIDS=()
for IDX in $MISSING_INDICES; do
    echo "Launching missing config idx=$IDX"
    python VAE/Reinmax/Train_fixed_schedule.py --idx $IDX \
        > slurm_logs/missing_${IDX}.out 2>&1 &
    PIDS+=($!)
done

# Wait for all to finish
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "All missing configs complete."
