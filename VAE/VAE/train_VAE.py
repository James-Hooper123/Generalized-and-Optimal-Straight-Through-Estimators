#need to run over 1e-4 1e-2 gemoetrically spaced learning rates
#run over latent spaces 128x4, 32x16, 8x64
#run over 3 seeds
#run over estimators: ST, GRMC-20 exp temp, ZGR, MVE exp temp, Reinmax 1.2 fixed temp, Reinmax 1.4 fixed temp


import argparse
import torch
import numpy as np
import os
import sys
import random
import pickle

# Add parent directories to path so imports resolve when running directly
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '..'))       # VAE/ for VAE.py, Data.py
sys.path.insert(0, os.path.join(_script_dir, '..', '..')) # Experiments/ for Gradient_estimators.py

from VAE import VAE
from Data import train_loader
import Gradient_estimators


def train(args):

    total_epochs = 500

    step_size = args.lr  # Learning rate passed from CLI
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    n = args.n  # Latent variables
    K = args.k  # Categories per variable

    gradient_estimator = getattr(Gradient_estimators, args.gradient_estimator)

    # Eval schedule string (e.g. "lambda epoch: exp_MVE(epoch, total_epochs)")
    temperature_schedule = eval(args.temperature_schedule, {**{k: v for k, v in vars(Gradient_estimators).items()}, 'total_epochs': total_epochs})

    # Initialize VAE
    vae = VAE(n=n, K=K, embedding=args.embedding, step_size=step_size, 
              gradient_estimator=gradient_estimator, 
              device=device, epochs=total_epochs,
              temperature_schedule=temperature_schedule)
    
    print(f"Training for {total_epochs} epochs on {device}...")
    print(f"Config: n={n}, K={K}, Embedding={args.embedding}, Estimator={args.gradient_estimator}, Schedule={args.temperature_schedule}")


    loss_history = []
    temp_history = []
    lr_history = []

    #warmup
    x = next(iter(train_loader))[0]
    x = x.to(device)
    loss = vae.learn_step(x)
    
    for epoch in range(total_epochs):
        vae.cur_epoch = epoch
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, (x,) in enumerate(train_loader):
            x = x.to(device)
            loss = vae.learn_step(x, compile = False)
            epoch_loss += loss.item()
            num_batches += 1
            
        avg_loss = epoch_loss / num_batches
        temp = temperature_schedule(epoch)
        lr = vae.optimiser.param_groups[0]['lr']
        vae.scheduler.step()
        loss_history.append(avg_loss)
        temp_history.append(temp)
        lr_history.append(lr)
        print(f"Epoch {epoch+1}/{total_epochs} | Loss: {avg_loss:.4f} | Temp: {temp:.4f} | LR: {lr:.6f}")
        
    final_elbo = vae.evaluate_elbo(train_loader, device)


    experiments_dir = os.path.join(_script_dir, '..', '..')
    results_dir = os.path.join(experiments_dir, "Results", f"VAE_{args.embedding}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Deterministic filename from hyperparameters to avoid race conditions
    import hashlib
    sched_hash = hashlib.md5(args.temperature_schedule.encode()).hexdigest()[:8]
    save_filename = f"{args.gradient_estimator}_{sched_hash}_n{args.n}_k{args.k}_lr{args.lr}_seed{args.seed}.pkl"
    save_path = os.path.join(results_dir, save_filename)
    
    save_data = {
        "loss_history": loss_history,
        "temp_history": temp_history,
        "lr_history": lr_history,
        "final_elbo": final_elbo,
        "arguments": vars(args)
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"Training Complete. Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE with custom gradient estimators.")
    
    # Define command line arguments
    parser.add_argument("--gradient_estimator", type=str, default="ZGR", 
                        help="Name of the gradient estimator in Gradient_estimators.py (e.g., ZGR)")
    parser.add_argument("--embedding", type=str, default="int", 
                        help="Type of embedding to use (e.g., int, ohe)")
    parser.add_argument("--temperature_schedule", type=str, default="lambda epoch: exp_MVE(epoch, total_epochs)",
                        help="Lambda string for temperature schedule")
    parser.add_argument("--n", type=int, default=200, 
                        help="Number of latent variables")
    parser.add_argument("--k", type=int, default=4, 
                        help="Number of categories per variable")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate (step_size)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    
    args = parser.parse_args()
    
    train(args)