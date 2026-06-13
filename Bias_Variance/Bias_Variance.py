import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import pickle
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '..'))  # Experiments/

from Gradient_estimators import MVE, gumbel_rao, ZGR, ST, reinmax, reinmax_scaled

# ==========================================
#       ARGUMENT PARSING
# ==========================================
parser = argparse.ArgumentParser(description="Run Bias/Variance calculations for gradient estimators.")
parser.add_argument('--dim', type=int, required=True, help="Dimensionality of the problem")
parser.add_argument('--seed', type=int, required=True, help="Random seed")
args = parser.parse_args()

dim = args.dim
seed = args.seed

# ==========================================
#       MAIN EXECUTION
# ==========================================
print(f"\n--- Running for Dim: {dim}, Seed: {seed} ---")
torch.manual_seed(seed)
np.random.seed(seed)

# --- Quadratic params ---
Q = torch.randn(dim, dim)
A = Q.T @ Q
b_param = torch.randn(dim)

def loss(z):
    return z @ A @ z + b_param @ z

# --- Logit Distribution ---
# Standard Normal Distribution
eta_data = torch.randn(dim)

eta = eta_data.clone().detach().requires_grad_(True)
p = F.softmax(eta, dim=0)

# Compute True Gradient
true_losses = torch.stack([loss(F.one_hot(torch.tensor(i), dim).float()) for i in range(dim)])
expected_loss = (p * true_losses).sum()
true_grad, = torch.autograd.grad(expected_loss, eta)

true_grad_norm = true_grad.norm().item()  # ||true_grad||

# --- Computation Functions ---
def compute_bias_variance(estimator_fn, temps):
    biases = []
    variances = []

    for temp in temps:
        grad_values = []
        for i in range(dim):
            y_i = F.one_hot(torch.tensor(i), num_classes=dim).float()

            if eta.grad is not None:
                eta.grad.zero_()

            res = estimator_fn(eta.unsqueeze(0), y_i.unsqueeze(0), temp=temp)

            if isinstance(res, tuple):
                out = res[0].squeeze(0)
            else:
                out = res.squeeze(0)

            l = loss(out)
            l.backward()
            grad_values.append(eta.grad.clone())

        grad_values = torch.stack(grad_values)  # (dim, D)

        # E[g_est] under the true distribution p
        E_grad   = torch.sum(p.unsqueeze(1) * grad_values, dim=0)                # (D,)
        Var_grad = torch.sum(p.unsqueeze(1) * (grad_values - E_grad)**2, dim=0)  # (D,)

        # Estimator Scale-Invariant Bias & Variance
        scale = true_grad_norm / max(E_grad.norm().item(), 1e-12)
        
        squared_bias = ((E_grad * scale) - true_grad).pow(2).sum().item()
        variance     = Var_grad.sum().item() * (scale ** 2)

        biases.append(squared_bias)
        variances.append(variance)

    return biases, variances

def compute_bias_variance_mc(estimator_fn, temps, num_samples=2000):
    biases = []
    variances = []

    for temp in temps:
        grads = []
        for _ in range(num_samples):
            if eta.grad is not None:
                eta.grad.zero_()

            res = estimator_fn(eta.unsqueeze(0), temp=temp)

            if isinstance(res, tuple):
                out = res[0].squeeze(0)
            else:
                out = res.squeeze(0)

            l = loss(out)
            l.backward()
            grads.append(eta.grad.clone())

        grads  = torch.stack(grads)              # (N, D)
        E_grad = grads.mean(dim=0)               # (D,)
        Var_grad = grads.var(dim=0, unbiased=True)

        # Estimator Scale-Invariant Bias & Variance
        scale = true_grad_norm / max(E_grad.norm().item(), 1e-12)

        squared_bias = ((E_grad * scale) - true_grad).pow(2).sum().item()
        variance     = Var_grad.sum().item() * (scale ** 2)

        biases.append(squared_bias)
        variances.append(variance)

    return biases, variances

def compute_bias_variance_gr(estimator_fn, temps, grmc):
    M = 1000
    g = true_grad.double()
    squared_biases, total_variances = [], []

    for temp in temps:
        grad_matrix_list = []
        for i in range(dim):
            y_i = F.one_hot(torch.tensor(i), num_classes=dim).float()
            grads_for_class = []
            for _ in range(M):
                if eta.grad is not None:
                    eta.grad.zero_()

                res = estimator_fn(eta.unsqueeze(0), y_i.unsqueeze(0), temp=temp, k=grmc)
                if isinstance(res, tuple):
                    out = res[0].squeeze(0)
                else:
                    out = res.squeeze(0)

                loss_val = loss(out)
                loss_val.backward()
                grads_for_class.append(eta.grad.clone())
            grad_matrix_list.append(torch.stack(grads_for_class))

        grad_matrix = torch.stack(grad_matrix_list).double()  # (N_classes, M, D)
        p_double = p.double()
        N, _, D = grad_matrix.shape

        f_bar_x  = grad_matrix.mean(dim=1)               # (N, D)
        s_sq_x   = grad_matrix.var(dim=1, unbiased=True) # (N, D)
        p_reshaped   = p_double.view(N, 1)

        within_group_var        = (p_reshaped * s_sq_x).sum(dim=0)
        mu_hat                  = (p_reshaped * f_bar_x).sum(dim=0)      # E[g_est], (D,)
        between_group_var_naive = (p_reshaped * f_bar_x.square()).sum(dim=0) - mu_hat.square()
        p_sq_reshaped           = p_double.square().view(N, 1)
        sum_p_sq_s_sq           = (p_sq_reshaped * s_sq_x).sum(dim=0)
        variance_bias_correction = (1.0 / M) * (within_group_var - sum_p_sq_s_sq)
        unbiased_total_variance  = within_group_var + between_group_var_naive - variance_bias_correction

        # Estimator Scale-Invariant Bias & Variance
        scale = true_grad_norm / max(mu_hat.norm().item(), 1e-12)

        squared_bias = ((mu_hat * scale) - g).pow(2).sum().item()
        variance     = unbiased_total_variance.sum().item() * (scale ** 2)

        squared_biases.append(squared_bias)
        total_variances.append(variance)

    return squared_biases, total_variances

# --- Temps Setup ---
log_spaced_points          = np.logspace(np.log10(0.01), np.log10(1), 1000)
temps1                     = list(log_spaced_points)
log_spaced_points_reversed = np.logspace(np.log10(1e7), np.log10(1e-6), 1000)
temps2                     = list(log_spaced_points_reversed)
temps_reinmax              = np.linspace(1.0, 2.0, 1000).tolist()

# --- Execution & Data Collection ---
data = {}

print("Computing for ZGR...")
b0, v0 = compute_bias_variance(ZGR, [0])
data['ZGR'] = {'bias': b0[0], 'variance': v0[0]}

print("Computing for ST...")
b3, v3 = compute_bias_variance(ST, [0])
data['ST'] = {'bias': b3[0], 'variance': v3[0]}

print("Computing for ReinMax...")
b_rm, v_rm = compute_bias_variance(reinmax, temps_reinmax)
data['ReinMax'] = {'temps': temps_reinmax, 'bias': b_rm, 'variance': v_rm}

for grmc, name in [(1, 'STGS'), (10, 'GRMC_10'), (100, 'GRMC_100')]:
    print(f"Computing for {name}...")
    b_gr, v_gr = compute_bias_variance_gr(gumbel_rao, temps1, grmc)
    data[name] = {'temps': temps1, 'bias': b_gr, 'variance': v_gr}

print("Computing for MVE...")
b2, v2 = compute_bias_variance(MVE, temps2)
data['MVE'] = {'temps': temps2, 'bias': b2, 'variance': v2}

# ==========================================
#       SAVE TO PKL
# ==========================================
script_dir  = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(script_dir)
results_dir = os.path.join(parent_dir, "Results", "Bias_Variance")
os.makedirs(results_dir, exist_ok=True)

output_pkl = os.path.join(results_dir, f'bias_variance_data_dim_{dim}_seed_{seed}.pkl')
with open(output_pkl, 'wb') as f:
    pickle.dump(data, f)

print(f"\nData successfully generated and saved to: {output_pkl}")