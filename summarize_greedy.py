import os
import pickle
import numpy as np
import re
from collections import defaultdict

def summarize_greedy_results():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "Greedy_VAE")
    if not os.path.exists(results_dir):
        print(f"Directory not found: {results_dir}")
        return

    # Files follow the pattern: {estimator}_{seed}_{nxk}_{embedding}_epoch_{epoch}.pkl
    # Example: MVE_0_8x64_OHE_epoch_500.pkl
    pattern = re.compile(r"(.+)_(\d+)_(\d+x\d+)_([a-zA-Z]+)_epoch_(\d+)\.pkl")

    # Group final ELBOs by (estimator, nxk, embedding)
    grouped_results = defaultdict(list)

    all_files = os.listdir(results_dir)
    print(f"Found {len(all_files)} files in total.")
    
    final_files = [f for f in all_files if "epoch_500" in f]
    print(f"Found {len(final_files)} epoch_500 files.")

    if not final_files:
        print("No epoch_500 files found, searching for max epoch per group...")
        groups = defaultdict(list)
        for f in all_files:
            match = pattern.match(f)
            if match:
                estimator, seed, nxk, embedding, epoch = match.groups()
                groups[(estimator, nxk, embedding, seed)].append((int(epoch), f))
        
        for key, epoch_list in groups.items():
            # Get the one with the maximum epoch
            max_epoch_file = max(epoch_list, key=lambda x: x[0])[1]
            final_files.append(max_epoch_file)

    for f in final_files:
        match = pattern.match(f)
        if not match:
            print(f"Skipping {f} - doesn't match pattern")
            continue
        
        estimator, seed, nxk, embedding, epoch = match.groups()
        group_key = (estimator, nxk, embedding)
        # print(f"Processing {f} -> group {group_key}")
        
        file_path = os.path.join(results_dir, f)
        try:
            with open(file_path, "rb") as pf:
                data = pickle.load(pf)
                # print(f"  Loaded data type: {type(data)}")
                # Greedy_optim.py saves (ELBO, TEMP)
                if isinstance(data, tuple) and len(data) >= 1:
                    elbo_list = data[0]
                    # print(f"  ELBO list type: {type(elbo_list)}")
                    if isinstance(elbo_list, list) and len(elbo_list) > 0:
                        final_elbo = elbo_list[-1]
                        grouped_results[group_key].append(final_elbo)
                    else:
                        print(f"  Skipping {f} - ELBO list is not a non-empty list. Type: {type(elbo_list)}")
                else:
                    print(f"  Skipping {f} - data is not a valid tuple. Type: {type(data)}")
        except Exception as e:
            print(f"  Error loading {f}: {e}")

    print(f"Grouped results keys: {list(grouped_results.keys())}")

    print("\n" + "="*85)
    print(f"{'Estimator':<15} | {'Latent Dims':<12} | {'Embedding':<10} | {'Mean ELBO':<12} | {'Std Dev':<10} | {'N'}")
    print("-" * 85)
    
    for group_key, elbos in sorted(grouped_results.items()):
        estimator, nxk, embedding = group_key
        mean_elbo = np.mean(elbos)
        std_elbo = np.std(elbos)
        n = len(elbos)
        print(f"{estimator:<15} | {nxk:<12} | {embedding:<10} | {mean_elbo:<12.4f} | {std_elbo:<10.4f} | {n}")
    
    print("="*85 + "\n")

if __name__ == "__main__":
    summarize_greedy_results()
