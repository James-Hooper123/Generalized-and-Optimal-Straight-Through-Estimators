import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
import os
import pickle

parser = argparse.ArgumentParser(description='Poly P Integer')
parser.add_argument('--temperature', type=float, default=1.0, metavar='S',
                    help='softmax temperature')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--method', default='gumbel',
                    help='Gradient estimator method')
parser.add_argument('--pnorm', type=float, default=2,
                    help="p-norm would be used in the loss function")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

train_step_per_epoch = 200
epochs = 50
lr = 1e-3

print(args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

latent_dim = 128
batch_size = 8
categorical_dim = 4  # K = 4 for Integer representation
targets = torch.Tensor([0.45]).repeat(latent_dim).contiguous()
batched_targets = targets.unsqueeze(0).expand(batch_size, -1)

if args.cuda:
    targets = targets.cuda()
    batched_targets = batched_targets.cuda()

class Quadratic_Toy_Int(nn.Module):
    def __init__(self):
        super().__init__()        
        self.theta = nn.Parameter(torch.Tensor(latent_dim, categorical_dim))
        self.theta.data.uniform_(-0.01, 0.01)

    def forward(self, temp, batch_size):
        theta_b = self.theta.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        
        qy = F.softmax(theta_b, dim=-1)
        p_flat = qy.view(-1, categorical_dim)
        z_idx = torch.multinomial(p_flat, num_samples=1).view(batch_size, latent_dim)
        
        # MVE_int directly takes the indices, others take one-hot
        if self.method == 'MVE':
            from Gradient_estimators import MVE_int
            temp_val = 1e6
            # MVE_int expects 2D inputs (B*L), so we flatten and unflatten indices
            z_flat = MVE_int(theta_b.view(-1, categorical_dim), z_idx.view(-1), temp=temp_val)
            z = z_flat.view(batch_size, latent_dim)
            log_y = None
        else:
            y_onehot = torch.zeros_like(theta_b).scatter_(-1, z_idx.unsqueeze(-1), 1.0)
            
            if self.method == 'ST':
                from Gradient_estimators import ST
                z_hard = ST(theta_b, y_onehot)
            elif self.method == 'reinmax':
                from Gradient_estimators import reinmax
                temp_val = temp if temp is not None else 1.0
                z_hard = reinmax(theta_b, y_onehot, temp=temp_val)
            elif self.method == 'STGS':
                from Gradient_estimators import gumbel_rao
                z_hard = gumbel_rao(theta_b, y_onehot, temp=0.1, k=1)
            elif self.method == 'GRMC-20':
                from Gradient_estimators import gumbel_rao
                z_hard = gumbel_rao(theta_b, y_onehot, temp=0.1, k=20)
            else:
                raise NotImplementedError(f"Method {self.method} is not implemented.")
            
            # Integer embedding: multiply by [0, 1, ..., K-1] and scale
            Phi = torch.arange(categorical_dim, device=theta_b.device, dtype=theta_b.dtype)
            z = torch.einsum('bnk,k->bn', z_hard, Phi)
            log_y = None
            
        # Scale to [0, 1] range to match the target c=0.45
        z = z / (categorical_dim - 1)
        
        return z, qy, log_y

model = Quadratic_Toy_Int()
if args.cuda:
    model.cuda()
    
optimizer = optim.Adam(model.parameters(), lr=lr)
model.method = args.method

def loss_scale(mse):
    return mse.abs().pow(args.pnorm)
    
def loss_function(z):
    MSE = loss_scale(z - batched_targets).sum(dim = -1) / latent_dim
    MSE = MSE.sum() / z.size(0)
    return MSE


def train(epoch):
    model.train()
    train_loss = 0
    temp = args.temperature
    for batch_idx in enumerate(range(train_step_per_epoch)):
        z, qy, _ = model(temp, batch_size)
        loss = loss_function(z)
        optimizer.zero_grad()
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    avg_loss = train_loss / train_step_per_epoch
    print('=====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    return avg_loss


def run():
    losses = []
    for epoch in range(1, epochs + 1):
        epoch_loss = train(epoch)
        losses.append(epoch_loss)

    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "polynomial_programming_int")
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"poly_p_int_{args.method}_p{args.pnorm}_seed{args.seed}.pkl"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(losses, f)
    print(f"Saved integer results to {filepath}")

if __name__ == '__main__':
    run()
