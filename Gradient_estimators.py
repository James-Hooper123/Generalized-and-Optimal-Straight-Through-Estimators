

import torch
import torch.nn.functional as F


def linear_STGS(epoch, max_epochs, T_start=1.0, T_end=0.1):
    progress = min(epoch / (max_epochs-1), 1.0)
    return T_start + (T_end - T_start) * progress

def exp_STGS(epoch, max_epochs, T_start=1.0, T_end=0.1):
    decay_rate = (T_end / T_start) ** (1 / (max_epochs - 1))
    return max(T_end, T_start * (decay_rate ** epoch))

def exp_MVE(epoch, max_epochs, T_start=1, T_end=1e6):
    alpha = (T_end / T_start) ** (1.0 / (max_epochs-1))
    return T_start * (alpha ** epoch)

def fixed_temp(epoch,temp):
    return temp






def ST(eta,y,temp=0):
    p = F.softmax(eta,dim=-1)
    return y + (p - p.detach())



def ZGR(eta, y,temp=0):

    logp = F.log_softmax(eta, dim=-1)
    p = logp.exp()
    dx_ST = p

    logpx = ((logp * y).sum(dim=-1)).unsqueeze(-1)
    dx_RE = (y - p.detach()) * logpx

    dx = (dx_ST + dx_RE) / 2
    return y + (dx - dx.detach())




def MVE(eta, z, temp=1.0):
    
    #tau = 1/tau
    B, K = eta.shape

    # 1) Softmax probabilities and Jacobian J[b,i,k]
    p = F.softmax(eta, dim=-1)                  # (B, K)
    p_i = p.unsqueeze(2)                        # (B, K, 1)
    p_j = p.unsqueeze(1)                        # (B, 1, K)
    I = torch.eye(K, device=eta.device)         # (K, K)
    J = p_i * (I.unsqueeze(0) - p_j)            # (B, K, K)

    # 2) Build coefficient matrix A (B, K, K)
    denom_ij = 1 + temp * (p_i + p_j)            # (B, K, K)
    # off-diagonals
    A = -temp * p_i * p_j / denom_ij             # (B, K, K)

    # correct diagonals:
    #   A_ii = (1+tau*p_i)*sum_j[p_j/(1+tau*(p_i+p_j))] - tau*p_i^2/(1+2*tau*p_i)
    sum_term = torch.sum(p_j / denom_ij, dim=2)                    # (B, K)
    diag_vals = (1 + temp * p) * sum_term \
              - temp * p * p / (1 + 2 * temp * p)                    # (B, K)
    A.diagonal(dim1=1, dim2=2).copy_(diag_vals)

    #print(A)

    # 3) Build RHS for each k: B_mat[b,i,k] = ((1+τ p_i)/(1+2τ p_i)) * J[b,i,k]
    ratio = (1 + temp * p) / (1 + 2 * temp * p)   # (B, K)
    B_mat = ratio.unsqueeze(-1) * J            # (B, K, K)

    # 4) Solve A v = B_mat for v[b,i,k] (B, K, K)
    try:
        #solve equation using cholesky inverse
        L = torch.linalg.cholesky(A)
        v = torch.linalg.cholesky_solve(B_mat, L).solution
    except:
        #solve with least squares
        v = torch.linalg.lstsq(A, B_mat).solution

     #calculate S
    
    # 5) Calculate S
    batch_idx = torch.arange(B, device=eta.device)

    # Handle z (indices or one-hot)
    z_idx = z.argmax(dim=-1) if z.ndim > 1 else z

    # Slices for fixed row x
    p_x = p[batch_idx, z_idx].unsqueeze(1)      # (B, 1)
    v_x = v[batch_idx, z_idx, :]                # (B, K)
    J_x = J[batch_idx, z_idx, :]                # (B, K)

    # Numerator Terms
    term1 = (1 + temp * p).unsqueeze(-1) * v
    term2 = -temp * p.unsqueeze(-1) * v_x.unsqueeze(1)
    term3 = torch.zeros_like(v)
    term3[batch_idx, z_idx, :] = temp * J_x

    # Denominator
    denom = (1 + temp * (p + p_x)).unsqueeze(-1)

    S = (term1 + term2 + term3) / denom
    
    dx = torch.einsum('bij,bj->bi', S.detach(), eta)
    #print(S)
    #print(J)

    return z + (dx - dx.detach())





@torch.no_grad()
def conditional_gumbel(logits, D, k=1, eps=1e-8):
    """
    Outputs k samples of Q = StandardGumbel(), such that argmax(logits + Q) is given by D (one hot vector).
    """
    # iid. exponential
    E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample([k])

    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True) + eps  # avoid div by zero

    # log partition function (more stable than logits.exp().sum().log())
    logZ = torch.logsumexp(logits, dim=-1, keepdim=True)

    # Sampled gumbel-adjusted logits (numerically stable form)
    adjusted = (D * (-Ei.log() + logZ) +
                (1 - D) * -torch.log(E / logits.exp() + Ei / logits.exp().sum(dim=-1, keepdim=True) + eps))

    return adjusted - logits


@torch.no_grad()
def exact_conditional_gumbel(logits, D, k=1):
    """
    Same as conditional_gumbel but uses rejection sampling.
    """
    idx = D.argmax(dim=-1)
    gumbels = []
    while len(gumbels) < k:
        gumbel = torch.rand_like(logits).log().neg().log().neg()
        if logits.add(gumbel).argmax() == idx:
            gumbels.append(gumbel)
    return torch.stack(gumbels)


def replace_gradient(value, surrogate):
    """
    Returns `value` but backpropagates gradients through `surrogate`.
    """
    return surrogate + (value - surrogate).detach()


def gumbel_rao(logits, D, temp=1.0, k=20):
    """
    Returns a categorical sample from logits (over axis=-1) as a
    one-hot vector, with Rao-Blackwellized Gumbel-Softmax gradient.

    k: number of samples to use in the Rao-Blackwellization.
    """
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = F.softmax(adjusted / temp, dim=-1).mean(dim=0)
    return replace_gradient(D, surrogate)


def gumbel_rao_fixed(logits, D, temp=1.0, k=20):
    """
    Returns a categorical sample from logits (over axis=-1) as a
    one-hot vector, with Rao-Blackwellized Gumbel-Softmax gradient.

    k: number of samples to use in the Rao-Blackwellization.
    """
    temp = 0.1
    adjusted = logits + conditional_gumbel(logits, D, k=k)
    surrogate = F.softmax(adjusted / temp, dim=-1).mean(dim=0)
    return replace_gradient(D, surrogate)






class ReinMaxCore(torch.autograd.Function):
    """
    Modified ReinMax estimator that allows forcing the 'sample' to a specific target
    while maintaining the ReinMax gradient calculation logic.
    """
    
    @staticmethod
    def forward(
        ctx, 
        logits: torch.Tensor, 
        tau: torch.Tensor,
        targets: torch.Tensor = None  # <--- Expects INDICES [N,]
    ):
        y_soft = logits.softmax(dim=-1)
        
        if targets is not None:
            # 1. USE FORCED TARGETS
            if targets.dim() == 1:
                sample = targets.unsqueeze(-1)
            else:
                sample = targets
        else:
            # 2. ORIGINAL SAMPLING (Fallback)
            sample = torch.multinomial(
                y_soft,
                num_samples=1,
                replacement=True,
            )

        # Create one-hot representation
        one_hot_sample = torch.zeros_like(
            y_soft, 
            memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, sample, 1.0)

        # Save for backward
        ctx.save_for_backward(one_hot_sample, logits, y_soft, tau)
        
        return one_hot_sample, y_soft

    @staticmethod
    def backward(
        ctx, 
        grad_at_sample: torch.Tensor, 
        grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors
        
        shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        
        # ReinMax Gradient Calculation
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)
        
        # Note: grad_at_p will be None if you return only grad_sample below.
        # This is correct. ReinMax does not use the standard softmax gradient.
        grad_at_input_0 = -0.5 * grad_at_sample * y_soft
        
        if grad_at_p is not None:
             grad_at_input_0 = grad_at_input_0 + grad_at_p * y_soft

        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
        
        grad_at_input = grad_at_input_0 + grad_at_input_1
        
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None, None


def reinmax(
        logits: torch.Tensor,
        y: torch.Tensor, 
        temp: float, 
    ):
    """
    Wrapper for ReinMax with target forcing.
    
    logits: [B, K] Tensor of logits
    y: [B, K] ONE-HOT Tensor of forced targets
    temp: float temperature (tau)
    """
    # --- 1. TEMP CHECK KEPT (As Requested) ---
    if temp < 1:
        raise ValueError("ReinMax prefers to set the temperature (tau) larger or equal to 1.")
    
    shape = logits.size()
    
    # Flatten logits [Batch * Seq, Vocab]
    logits_flat = logits.view(-1, shape[-1])
    
    # Handle Targets Flattening
    targets_flat = None
    if y is not None:
        # CONVERT y (one-hot) to target_indices (long)
        targets_flat = y.view(-1, shape[-1]).argmax(dim=-1)
    
    # Apply the modified Function
    grad_sample, y_soft = ReinMaxCore.apply(
        logits_flat, 
        logits.new_empty(1).fill_(temp),
        targets_flat
    )
    
    # --- 2. FIXED RETURN STATEMENT ---
    # We return ONLY the output of the Core. 
    # The Core's backward pass IS the surrogate gradient.
    return grad_sample.view(shape)


def reinmax_scaled(
        logits: torch.Tensor,
        y: torch.Tensor, 
        temp: float, 
    ):
    """
    ReinMax with gradient scaled by tau to compensate for the 1/tau
    scaling introduced by softmax(eta/tau). Forward value is unchanged.
    
    logits: [B, K] Tensor of logits
    y: [B, K] ONE-HOT Tensor of forced targets
    temp: float temperature (tau)
    """
    if temp < 1:
        raise ValueError("ReinMax prefers to set the temperature (tau) larger or equal to 1.")
    
    shape = logits.size()
    logits_flat = logits.view(-1, shape[-1])
    
    targets_flat = None
    if y is not None:
        targets_flat = y.view(-1, shape[-1]).argmax(dim=-1)
    
    grad_sample, y_soft = ReinMaxCore.apply(
        logits_flat, 
        logits.new_empty(1).fill_(temp),
        targets_flat
    )
    
    # Scale gradient by tau without changing forward value:
    # forward: x.detach() + tau*(x - x.detach()) = x
    # backward: tau * d(x)/d(logits)
    grad_sample = grad_sample.detach() + (1/temp) * (grad_sample - grad_sample.detach())
    
    return grad_sample.view(shape)



def MVE_int(eta, y, temp):
    #temp = 0
    #print(eta.shape)
    #print(y.shape)

    K = eta.shape[1]
    p = F.softmax(eta, dim=-1) # (B, K)

    Phi = torch.arange(K, dtype=torch.float32, device=eta.device) # (K)
    m1 = torch.einsum('Bi,i->B', p, Phi) # (B)
    m2 = torch.einsum('Bi,i,i->B', p, Phi, Phi) # (B)
    v = m2 - m1**2 # (B)

    term1 = (1 - 2 * temp * m1 * y + 2 * temp * m2) # (B)
    term2 = temp * (y - m1) # (B)
    denom_scalar = 1 + 2 * temp * v # (B)
    numerator = term1.unsqueeze(-1) * Phi + term2.unsqueeze(-1) * (Phi**2)
    denominator = denom_scalar.unsqueeze(-1)

    S = numerator / denominator

    #print(S) 

    dx = torch.einsum('bi,bi->b', S.detach(), p)
    
    return y + (dx - dx.detach())


def reinmax_limit(eta, y, temp):
    B, K = eta.shape
    p = F.softmax(eta, dim=-1)                          # (B, K)

    # p(x) for each batch element
    p_x = (p * y).sum(dim=-1, keepdim=True)             # (B, 1)

    # Rank-1 term: (1/p(x)) * (e_x - p) e_x^T          -> (B, K, K)
    col = (y - p) / p_x                                 # (B, K)
    rank1 = col.unsqueeze(-1) * y.unsqueeze(-2)         # (B, K, K)

    I = torch.eye(K, device=eta.device, dtype=eta.dtype) # (K, K)

    # First term: (1/2K) [I + rank1]
    term1 = (I.unsqueeze(0) + rank1) / (2 * K)          # (B, K, K)

    # Second term: (1/2K^2) 1 1^T
    ones = torch.ones(K, K, device=eta.device, dtype=eta.dtype)
    term2 = ones.unsqueeze(0) / (2 * K * K)             # (B, K, K)

    S = term1 - term2                                    # (B, K, K)

    dx = torch.einsum('bij,bj->bi', S.detach(), eta)
    return y + (dx - dx.detach())

