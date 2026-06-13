## need to pick fixed learning rates


import torch
import pickle
import argparse
import sys
import os

# --- Fix PYTHONPATH for VAE and Gradient_estimators modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))       # Adds Experiments/VAE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Adds Experiments/

from VAE import VAE
from Data import train_loader
from Gradient_estimators import gumbel_rao, MVE, reinmax, fixed_temp
import ast
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp


def temperature(prev_temp, next_temp, zoom_level, estimator):
    NUM_GRID_POINTS = 25
    NUM_TEMP_CANDIDATES = 7

    if estimator == "MVE":
        global_grid = np.geomspace(0.1, 1e6, NUM_GRID_POINTS)
    elif estimator == "gumbel_rao":
        global_grid = np.linspace(0.1, 1.0, NUM_GRID_POINTS)

    temps = []
    if zoom_level == 0:
        indices = np.round(np.linspace(0, NUM_GRID_POINTS - 1, num=NUM_TEMP_CANDIDATES)).astype(int)
        temps = global_grid[indices]
    else:
        start_index = np.argmin(np.abs(global_grid - prev_temp))
        end_index = np.argmin(np.abs(global_grid - next_temp))
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        indices = np.round(np.linspace(start_index, end_index, num=NUM_TEMP_CANDIDATES)).astype(int)
        temps = global_grid[indices]

    return temps

def get_cpu_optim_state(optimiser):
    import copy
    state_dict = optimiser.state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        if key == 'state':
            cpu_state = {}
            for param_id, param_state in value.items():
                cpu_param_state = {}
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        cpu_param_state[k] = v.cpu()
                    else:
                        cpu_param_state[k] = v
                cpu_state[param_id] = cpu_param_state
            cpu_state_dict['state'] = cpu_state
        else:
            cpu_state_dict[key] = copy.deepcopy(value)
    return cpu_state_dict


def load_optim_state_to_device(optimiser, state_dict, device):
    optimiser.load_state_dict(state_dict)
    for state in optimiser.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def train_and_evaluate_for_temp(args_bundle):
    """
    Runs training for a given temperature candidate.
    """
    (temp, initial_state, initial_optim_state, initial_scheduler_state, n, k, embedding, estimator_class, device_str,
     total_epochs, search_epochs, seed, start_epoch) = args_bundle

    device = torch.device(device_str)
    torch.manual_seed(seed)
    np.random.seed(seed)


    net = VAE(
        n=n, K=k, embedding=embedding, step_size=0.00316228,
        gradient_estimator=estimator_class, device=device,
        epochs=total_epochs, temperature_schedule=lambda epoch: fixed_temp(epoch, temp)
    )
    net.load_state_dict(initial_state)
    net.to(device)
    load_optim_state_to_device(net.optimiser, initial_optim_state, device)
    net.scheduler.load_state_dict(initial_scheduler_state)


    elbo_history = []

    for epoch in range(search_epochs):
        net.cur_epoch = start_epoch + epoch
        net.train()

        for (x_batch,) in train_loader:
            x_batch = x_batch.view(x_batch.size(0), -1).to(device)
            # learn_step handles zero_grad, forward, backward, and optimizer.step
            net.learn_step(x_batch)

        # Step the VAE's internal scheduler
        net.scheduler.step()

        # Evaluate ELBO at the end of this epoch
        epoch_elbo = net.evaluate_elbo(train_loader, device)
        elbo_history.append(epoch_elbo)

    final_elbo = elbo_history[-1]
    final_state_cpu = {k: v.cpu() for k, v in net.state_dict().items()}
    final_optim_state = get_cpu_optim_state(net.optimiser)
    final_scheduler_state = net.scheduler.state_dict()

    return (final_elbo, elbo_history, final_state_cpu, final_optim_state, final_scheduler_state)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str,
                        choices=["MVE", "gumbel_rao"],
                        required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nxk", type=str, default="(128,4)")
    parser.add_argument("--embedding", type=str, default="OHE", choices=["OHE", "int"],
                        help="Embedding type for the decoder.")
    args = parser.parse_args()

    
    if args.estimator == "MVE":     estimator = MVE
    elif args.estimator == "gumbel_rao": estimator = gumbel_rao
    else: raise ValueError(f"Unknown estimator: {args.estimator}")

    torch.manual_seed(args.seed)

    total_epochs = 500
    search_epochs = 25
    num_cycles = total_epochs // search_epochs

    (n, k) = ast.literal_eval(args.nxk)

    # Create the main VAE with a dummy temperature schedule (will be overridden per cycle)
    net = VAE(
        n=n, K=k, embedding=args.embedding, step_size=0.00316228,
        gradient_estimator=estimator, device=device,
        epochs=total_epochs, temperature_schedule=lambda epoch: fixed_temp(epoch, 1.0)
    )
    net.train()

    ELBO = []
    TEMP = []
    total_epochs_completed = 0

    base_filename = f"{args.estimator}_{args.seed}_{n}x{k}_{args.embedding}"

    ctx = mp.get_context('spawn')

    for cycle in range(num_cycles):
        print(f"--- Cycle {cycle + 1}/{num_cycles} | Total Epochs {total_epochs_completed} to {total_epochs_completed + search_epochs - 1} ---")
        print(f"Searching for best temperature by training each candidate for {search_epochs} epochs...")

        prev_temp = None
        next_temp = None
        initial_state = {key: val.cpu() for key, val in net.state_dict().items()}
        initial_optim_state = get_cpu_optim_state(net.optimiser)
        initial_scheduler_state = net.scheduler.state_dict()

        best_temp_iter = None
        best_model_state_iter = None
        best_optim_state_iter = None
        best_scheduler_state_iter = None
        best_elbo_history_iter = None

        for zoom_level in range(2):
            temps = temperature(prev_temp, next_temp, zoom_level, args.estimator)
            temps_to_search = temps[1:-1]

            if temps_to_search.size == 0:
                print(f"Zoom level {zoom_level + 1}: No new temperatures to search. Proceeding with previous best.")
                continue

            print(f"Zoom level {zoom_level + 1}: Searching {len(temps_to_search)} temperatures.")

            worker_seed = args.seed + total_epochs_completed + zoom_level

            args_for_pool = [
                (temp, initial_state, initial_optim_state, initial_scheduler_state, n, k, args.embedding, estimator, device,
                 total_epochs, search_epochs, worker_seed + i, total_epochs_completed)
                for i, temp in enumerate(temps_to_search)
            ]

            with ProcessPoolExecutor(max_workers=len(temps_to_search), mp_context=ctx) as executor:
                results = list(executor.map(train_and_evaluate_for_temp, args_for_pool))

            final_losses, all_elbo_histories, all_final_states, all_final_optim_states, all_final_scheduler_states = zip(*results)

            best_loss_index = np.argmin(final_losses)

            best_temp_iter = temps_to_search[best_loss_index]
            best_model_state_iter = all_final_states[best_loss_index]
            best_optim_state_iter = all_final_optim_states[best_loss_index]
            best_scheduler_state_iter = all_final_scheduler_states[best_loss_index]
            best_elbo_history_iter = all_elbo_histories[best_loss_index]

            original_temps_index = np.where(temps == best_temp_iter)[0][0]

            prev_temp_index = max(0, original_temps_index - 1)
            next_temp_index = min(len(temps) - 1, original_temps_index + 1)

            prev_temp = temps[prev_temp_index]
            next_temp = temps[next_temp_index]

        best_temp = best_temp_iter
        best_elbo_history = best_elbo_history_iter
        best_model_state = best_model_state_iter
        best_optim_state = best_optim_state_iter
        best_scheduler_state = best_scheduler_state_iter

        ELBO.extend(best_elbo_history)
        TEMP.extend([best_temp] * search_epochs)

        net.load_state_dict(best_model_state)
        load_optim_state_to_device(net.optimiser, best_optim_state, device)
        net.scheduler.load_state_dict(best_scheduler_state)

        print(f"Cycle Complete. Best Temp was {best_temp:.4f} with a final ELBO of {best_elbo_history[-1]:.4f}")
        total_epochs_completed += search_epochs
        print(f"Total epochs completed: {total_epochs_completed}")

        results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "Results", "Greedy_VAE")
        os.makedirs(results_dir, exist_ok=True)
        
        checkpoint_filename = os.path.join(results_dir, f"{base_filename}_epoch_{total_epochs_completed}.pkl")
        print(f"Saving checkpoint: {checkpoint_filename}")
        with open(checkpoint_filename, 'wb') as f:
            pickle.dump((ELBO, TEMP), f)

        print("-" * 20)

    print("Training complete.")