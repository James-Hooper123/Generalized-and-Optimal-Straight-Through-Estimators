import os
import sys
import time
import pickle
import torch
import argparse
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# --- Fix PYTHONPATH for VAE and Gradient_estimators modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))       # Adds Experiments/VAE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Adds Experiments/

# --- Custom Modules ---
from VAE import VAE
from Data import train_loader
from Gradient_estimators import MVE, reinmax

# --- Estimator Map ---
ESTIMATOR_MAP = {
    "MVE": MVE,
    "reinmax": reinmax,
}

def run_training_task(config):
    """
    Runs one experiment configuration.
    """
    est_name = config['estimator']
    temp_raw = config['temp']
    lr = config['lr']
    device = config['device']

    # Apply temperature scaling outside the gradient estimator
    if est_name == 'reinmax':
        temp = temp_raw + 1
    elif est_name == 'MVE':
        temp = 1.5 / temp_raw
    else:
        temp = temp_raw
    verbose = config.get('verbose', False)
    
    # Extract configs with default fallbacks
    total_epochs = config.get('epochs', 50)
    n_latents = config.get('n_latents', 128)
    k_categories = config.get('k_categories', 4)
    seed = config.get('seed', 0)
    experiments_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    output_dir = config.get('output_dir', os.path.join(experiments_dir, "Results", "Reinmax"))

    torch.manual_seed(seed)
    estimator_class = ESTIMATOR_MAP[est_name]

    net = VAE(
        n=n_latents, 
        K=k_categories, 
        embedding="OHE",
        temperature_schedule=lambda epoch: temp,
        step_size=lr, 
        gradient_estimator=estimator_class, 
        device=device, 
        epochs=total_epochs
    )
    net.to(device)

    elbo_log = [] 

    f_name = f"{est_name}_temp{str(temp_raw).replace('.', '_')}_lr{str(lr).replace('.', '_')}.pkl"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f_name)

    if os.path.exists(save_path):
        return f"Skipped: {f_name}"

    # --- Main Training Loop ---
    for epoch in range(total_epochs):
        net.temp = temp
        net.train()
        epoch_train_loss = 0.0
        num_batches = 0
        
        # Verbose inner loop: See the actual batches moving
        batch_iterator = train_loader
        if verbose:
            batch_iterator = tqdm(
                train_loader, 
                desc=f"Task {est_name} | Ep {epoch+1}/{total_epochs}", 
                leave=False,
                unit="batch"
            )

        epoch_start = time.time()

        for (x_batch,) in batch_iterator:
            # Re-assign device to ensure tensor is on the worker's specified device
            x_batch = x_batch.view(x_batch.size(0), -1).to(device)

            net.optimiser.zero_grad()
            loss = net.learn_step(x_batch)
            net.optimiser.step()

            epoch_train_loss += loss.item()
            num_batches += 1

        net.scheduler.step()
        epoch_elbo = epoch_train_loss / num_batches
        elbo_log.append(epoch_elbo)
        
        epoch_end = time.time()

        # PRINT SUMMARY: Only the 'verbose' worker (Worker 0) prints this
        if verbose:
            current_lr = net.optimiser.param_groups[0]['lr']
            print(f">>> [Task {est_name}] Epoch {epoch+1} Complete | Time: {epoch_end - epoch_start:.2f}s | ELBO: {epoch_elbo:.2f} | LR: {current_lr:g}")

    # --- Save Results ---
    save_data = {"elbo_history": elbo_log, "final_elbo": elbo_log[-1], "config": config}
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    return f"Finished: {f_name}"


def get_config(idx):
    """
    Generates a deterministic parameter grid and returns the configuration
    for the specified index.
    """
    ESTIMATORS = ['MVE', 'reinmax']
    TEMPS = np.round(np.linspace(0.01, 0.3, 30), 2)
    LRS = np.geomspace(1e-4, 1e-2, 30)
    
    tasks = []
    for est in ESTIMATORS:
        for temp in TEMPS:
            for lr in LRS:
                tasks.append({
                    'estimator': est, 
                    'temp': temp, 
                    'lr': lr
                })
    return tasks[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True, help="Task index from 0 to 1799")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results to")
    args = parser.parse_args()
    
    config = get_config(args.idx)
    config['device'] = torch.device(args.device)
    config['verbose'] = args.verbose
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    print(f"[{args.idx}] Starting task with Config: Estimator={config['estimator']}, Temp={config['temp']}, LR={config['lr']}")
    status = run_training_task(config)
    print(f"[{args.idx}] Status: {status}")