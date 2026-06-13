#think this is good to go - nned to check though - should detail setup for each experiment - sort out outputs

import argparse
import os
import sys
import time
import queue
import numpy as np
import torch.multiprocessing as mp
import subprocess

# --- Configuration ---
NUM_WORKERS_PER_GPU = 4 
TOTAL_GPUS = 8

# --- Combinations ---
# 1e-4 to 1e-2 geometrically spaced learning rates
LRS = np.geomspace(1e-4, 1e-2, 5)

# Latent spaces
NXK_LIST = [(128, 4), (32, 16), (8, 64)]

# Seeds
SEEDS = [0, 1, 2]

# Estimators and Temp Schedules (Paired)
# ST, GRMC-20 exp temp, ZGR, MVE exp temp, Reinmax 1.2 fixed temp, Reinmax 1.4 fixed temp
# We now point to the explicit schedule functions based on your request.
ESTIMATOR_CONFIGS = [
    {"estimator": "ST", "schedule": "lambda epoch: fixed_temp(epoch, 0.0)"},
    {"estimator": "gumbel_rao", "schedule": "lambda epoch: exp_STGS(epoch, total_epochs)"},
    {"estimator": "ZGR", "schedule": "lambda epoch: fixed_temp(epoch, 0.0)"},
    {"estimator": "MVE", "schedule": "lambda epoch: exp_MVE(epoch, total_epochs)"},
    {"estimator": "reinmax", "schedule": "lambda epoch: fixed_temp(epoch, 1.2)"},
    {"estimator": "reinmax", "schedule": "lambda epoch: fixed_temp(epoch, 1.4)"},
    {"estimator": "MVE", "schedule": "lambda epoch: fixed_temp(epoch, 1e6)"},
]

def generate_task_list(embedding):
    """Generates the full list of hyperparameters to search over."""
    tasks = []
    
    if embedding == "int":
        nxk_list = [(128, 4), (128, 16), (128, 64), (128, 256)]
    else:
        nxk_list = [(128, 4), (32, 16), (8, 64)]
        
    for lr in LRS:
        for (n, k) in nxk_list:
            for seed in SEEDS:
                for est_conf in ESTIMATOR_CONFIGS:
                    
                    est = est_conf['estimator']
                    if embedding == 'int' and est == 'MVE':
                        est = 'MVE_int'
                    
                    tasks.append({
                        'lr': lr,
                        'n': n,
                        'k': k,
                        'seed': seed,
                        'estimator': est,
                        'schedule': est_conf['schedule'],
                        'embedding': embedding
                    })
    return tasks

def worker_process(gpu_id, task_queue, worker_id, script_path):
    """
    Persistent Worker executing sub-processes for train_VAE.py
    """
    print(f"[Worker {worker_id} on GPU {gpu_id}] Started.")
    
    # Hide all GPUs except the one assigned to this worker
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        try:
            task = task_queue.get(timeout=3)
            
            cmd = [
                sys.executable, script_path,
                "--gradient_estimator", task['estimator'],
                "--embedding", task['embedding'],
                "--temperature_schedule", task['schedule'],
                "--n", str(task['n']),
                "--k", str(task['k']),
                "--lr", str(task['lr']),
                "--seed", str(task['seed'])
            ]

            print(f"[GPU {gpu_id} | W{worker_id}] Running: Est={task['estimator']}({task['schedule']}), NxK={task['n']}x{task['k']}, LR={task['lr']:.5f}, Seed={task['seed']}")
            subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL)
            
        except queue.Empty:
            break
        except subprocess.CalledProcessError as e:
            print(f"[Worker {worker_id} on GPU {gpu_id}] Task Failed: {e}")
        except Exception as e:
            print(f"[Worker {worker_id} on GPU {gpu_id}] Stopped with error: {e}")
            break
            
    print(f"[Worker {worker_id} on GPU {gpu_id}] Shutting down.")

def main():
    parser = argparse.ArgumentParser(description="Multiprocessing Run Script for VAE grid search")
    parser.add_argument("embedding", type=str, choices=["OHE", "int"], help="Embedding space to use (OHE or int)")
    args = parser.parse_args()

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    all_tasks = generate_task_list(args.embedding)
    print(f"--- Starting Global Run ({args.embedding}) ---")
    print(f"Total Combinations: {len(all_tasks)}")
    print(f"Total GPUs: {TOTAL_GPUS}")
    print(f"Workers per GPU: {NUM_WORKERS_PER_GPU}")
    print("-" * 30)

    processes = []
    start_time = time.time()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "train_VAE.py")

    for gpu_id in range(TOTAL_GPUS):
        gpu_tasks = all_tasks[gpu_id::TOTAL_GPUS]
        task_queue = mp.Queue()
        for t in gpu_tasks:
            task_queue.put(t)
            
        for w_id in range(NUM_WORKERS_PER_GPU):
            p = mp.Process(target=worker_process, args=(gpu_id, task_queue, w_id, script_path))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
        
    duration = (time.time() - start_time) / 60
    print(f"\n--- All Grid Search Tasks Finished in {duration:.2f} minutes ---")

if __name__ == "__main__":
    main()
