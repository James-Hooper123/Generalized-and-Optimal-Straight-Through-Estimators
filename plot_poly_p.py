import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
results_dir_ohe = os.path.join(script_dir, "Results", "polynomial_programming")
results_dir_int = os.path.join(script_dir, "Results", "polynomial_programming_int")
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

estimators = ["ST", "reinmax", "STGS", "GRMC-20", "MVE"]
p_values = [1.5, 2.0, 3.0]
seed = 1

colors = {
    'ST': '#7f7f7f',
    'reinmax': '#FF8C00',
    'STGS': '#7700ff',
    'GRMC-20': '#0044ff',
    'MVE': '#e71010'
}

labels = {
    'ST': 'ST',
    'reinmax': 'ReinMax',
    'STGS': 'STGS',
    'GRMC-20': 'GRMC-20',
    'MVE': 'MVE'
}

def load_pkl(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    print(f"Warning: File not found: {path}")
    return None

def main():
    print("--- Plotting Polynomial Programming Results ---")
    
    for p in p_values:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plt.rcParams.update({'font.size': 12})
        
        # OHE Plot
        has_data_ohe = False
        for est in estimators:
            filename = f"poly_p_{est}_p{p}_seed{seed}.pkl"
            filepath = os.path.join(results_dir_ohe, filename)
            
            losses = load_pkl(filepath)
            if losses is not None:
                has_data_ohe = True
                epochs = np.arange(1, len(losses) + 1)
                color = colors.get(est, 'black')
                label = labels.get(est, est)
                linestyle = (0, (2, 2)) if est == 'MVE' else '-'
                axes[0].plot(epochs, losses, label=label, color=color, linewidth=2, linestyle=linestyle)
        
        if has_data_ohe:
            axes[0].set_yscale('log')  # Log scale on y-axis
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(f"OHE Continuous Encoding (p={p})")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].set_title(f"OHE Encoding - No Data (p={p})")
            
        # Integer Plot
        has_data_int = False
        for est in estimators:
            filename = f"poly_p_int_{est}_p{p}_seed{seed}.pkl"
            filepath = os.path.join(results_dir_int, filename)
            
            losses = load_pkl(filepath)
            if losses is not None:
                has_data_int = True
                epochs = np.arange(1, len(losses) + 1)
                color = colors.get(est, 'black')
                label = labels.get(est, est)
                linestyle = (0, (2, 2)) if est == 'MVE' else '-'
                axes[1].plot(epochs, losses, label=label, color=color, linewidth=2, linestyle=linestyle)
                
        if has_data_int:
            axes[1].set_yscale('log')  # Log scale on y-axis
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].set_title(f"Integer Encoding (p={p})")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_title(f"Integer Encoding - No Data (p={p})")
            
        if not has_data_ohe and not has_data_int:
            print(f"No data found for p={p}, skipping plot.")
            plt.close()
            continue
            
        plt.tight_layout()
        
        output_path = os.path.join(plots_dir, f"poly_p_curve_p{p}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved plot for p={p} to {output_path}")

if __name__ == "__main__":
    main()