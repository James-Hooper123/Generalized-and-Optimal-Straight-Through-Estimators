import torch

torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

def compute_ste_original(eta: torch.Tensor, y: torch.Tensor):
    """The original STE using the eigendecomposition of the covariance."""
    K = eta.shape[0]
    Phi = torch.eye(K, dtype=eta.dtype, device=eta.device)

    pi = torch.softmax(eta, dim=-1).detach()
    J = torch.diag(pi) - torch.outer(pi, pi)
    lam, U = torch.linalg.eigh(J)

    z = y - pi
    Utz = U.T @ z
    Utpi = U.T @ pi

    L = lam.unsqueeze(1) + lam.unsqueeze(0)
    safe = torch.abs(L) > 1e-12
    L_inv = torch.where(
        safe, 
        1.0 / torch.where(safe, L, torch.ones_like(L)), 
        torch.zeros_like(L)
    )

    M1 = L_inv * Utz.unsqueeze(0)
    c = M1 @ Utpi
    B = U @ M1.T
    A = (U - Utpi.unsqueeze(0)) * B - U * c.unsqueeze(0)
    GZ = U @ A.T

    S = Phi + GZ
    M = S @ J
    
    dx = M @ eta
    ste_y = y + (dx - dx.detach())
    
    return ste_y, S

def compute_ste_limit(eta: torch.Tensor, y: torch.Tensor):
    """The new STE using the graph Laplacian limit for tau -> infinity."""
    K = eta.shape[0]
    pi = torch.softmax(eta, dim=-1).detach()
    
    x_idx = torch.argmax(y).item()
    pi_x = pi[x_idx]

    # --- 1. Construct the Laplacian L ---
    P_i = pi.unsqueeze(1)
    P_j = pi.unsqueeze(0)
    
    W = (P_i * P_j) / (P_i + P_j + 1e-12)
    W.fill_diagonal_(0.0)
    
    D = torch.diag(W.sum(dim=1))
    L_mat = D - W

    # --- 2. Construct the Pseudoinverse L^dagger ---
    L_pinv = torch.linalg.pinv(L_mat, hermitian=True)

    # --- 3. Construct S(x) ---
    D1 = torch.diag(1.0 / (pi + pi_x))
    M2 = torch.diag(pi) - torch.outer(pi, y)
    
    term1 = D1 @ M2 @ L_pinv
    term2 = torch.outer(y, y) / pi_x
    
    S = 0.5 * (term1 + term2)

    # --- 4. Straight-Through Estimator ---
    J = torch.diag(pi) - torch.outer(pi, pi)
    M = S @ J
    
    dx = M @ eta
    ste_y = y + (dx - dx.detach())
    
    return ste_y, S

def compare_gradients(eta, L_func, ste_func):
    K = eta.shape[0]
    
    # 1. True Expected Grad
    pi_true = torch.softmax(eta, dim=-1)
    expected_loss = torch.tensor(0.0, device=eta.device)
    for i in range(K):
        y = torch.zeros(K, device=eta.device)
        y[i] = 1.0
        expected_loss = expected_loss + pi_true[i] * L_func(y)

    grad_expected = torch.autograd.grad(expected_loss, eta, retain_graph=True)[0]

    # 2. Expected STE Grad
    E_grad_ste = torch.zeros_like(eta)
    pi_det = pi_true.detach()

    for i in range(K):
        y = torch.zeros(K, device=eta.device)
        y[i] = 1.0
        
        ste_y, _ = ste_func(eta, y)
        loss_ste = L_func(ste_y)
        
        grad_ste = torch.autograd.grad(loss_ste, eta, retain_graph=True)[0]
        E_grad_ste += pi_det[i] * grad_ste

    max_dev = torch.max(torch.abs(grad_expected - E_grad_ste)).item()
    return max_dev

if __name__ == "__main__":
    torch.manual_seed(42)
    K = 5
    eta = torch.randn(K, requires_grad=True)
    
    # Loss functions
    w = torch.randn(K)
    def L_linear(v): return torch.dot(w, v)
    
    Q = torch.randn(K, K)
    def L_quad(v): return v.T @ Q @ v

    estimators = {
        "Original STE (Covariance Eigendecomposition)": compute_ste_original,
        "Limit STE (Graph Laplacian)": compute_ste_limit
    }

    # --- Run Gradient Comparisons ---
    for name, func in estimators.items():
        print(f"\n=== {name} ===")
        dev_lin = compare_gradients(eta, L_linear, func)
        print(f"Linear Loss Max Dev:    {dev_lin:.4e} -> {'MATCH' if dev_lin < 1e-4 else 'MISMATCH'}")
        
        dev_quad = compare_gradients(eta, L_quad, func)
        print(f"Quadratic Loss Max Dev: {dev_quad:.4e} -> {'MATCH' if dev_quad < 1e-4 else 'MISMATCH'}")

    # --- Print S Matrices for a Fixed Eta and y ---
    print("\n" + "="*50)
    print("=== S Matrix Comparison for a Fixed Sample ===")
    print("="*50)
    
    # Fixed inputs
    eta_fixed = torch.tensor([1.5, -0.5, 0.2, 2.0, -1.0])
    y_fixed = torch.zeros(K)
    y_fixed[0] = 1.0  # Let's say we sampled class 0

    pi_fixed = torch.softmax(eta_fixed, dim=-1)
    print(f"Fixed Logits (eta): {eta_fixed.numpy()}")
    print(f"Probabilities (pi): {pi_fixed.numpy()}")
    print(f"Sampled Class (y):  {y_fixed.numpy()}\n")

    # Compute S matrices
    _, S_orig = compute_ste_original(eta_fixed, y_fixed)
    _, S_limit = compute_ste_limit(eta_fixed, y_fixed)

    print("S Matrix (Original Covariance Method):")
    print(S_orig.numpy())

    print("\nS Matrix (Graph Laplacian Limit Method):")
    print(S_limit.numpy())
    
    diff = S_orig - S_limit
    print(f"\nDifference between the two S matrices:")
    print(diff)