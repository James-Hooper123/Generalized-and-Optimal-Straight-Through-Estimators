#run over 3 GPUs

import argparse
import subprocess
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Run Greedy VAE across estimators for a specific seed/GPU.")
    parser.add_argument("gpu_id", type=int, help="ID of the GPU to use (e.g., 0, 1, or 2)")
    args = parser.parse_args()

    # Map GPU ID directly to a seed (e.g., GPU 0 gets Seed 0, GPU 1 gets Seed 1)
    # This ensures 1 seed per GPU as requested.
    seed = args.gpu_id 
    
    estimators = ["MVE", "gumbel_rao", "reinmax"]
    
    print(f"--- Starting Greedy VAE Runs on GPU {args.gpu_id} ---")
    print(f"Assigned Seed: {seed}")
    print(f"Estimators to run: {estimators}")
    print("-" * 40)
    
    # Set CUDA_VISIBLE_DEVICES so Greedy_optim.py only sees this specific GPU
    # This ensures that when Greedy_optim.py spawns subprocesses for temperature search, 
    # they are all constrained to the correct GPU.
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "Greedy_optim.py")
    
    for est in estimators:
        print(f"\n>>> Running Estimator: {est} on GPU {args.gpu_id} (Seed {seed}) <<<")
        
        cmd = [
            sys.executable, script_path,
            "--estimator", est,
            "--seed", str(seed)
        ]
        
        try:
            # subprocess.run will block until the training cycle finishes for this estimator
            subprocess.run(cmd, env=env, check=True)
            print(f"Finished {est} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error running {est}: {e}")
            # Stop the loop if one estimator crashes
            break 
            
    print(f"\n--- All estimators finished for GPU {args.gpu_id} (Seed {seed}) ---")

if __name__ == "__main__":
    main()
