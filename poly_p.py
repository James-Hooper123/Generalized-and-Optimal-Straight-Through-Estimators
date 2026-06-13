import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
import os
import pickle

parser = argparse.ArgumentParser(description='Poly P OHE')
parser.add_argument('--temperature', type=float, default=1.0, metavar='S',
                    help='softmax temperature')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed')
parser.add_argument('--method', default='ST',
                    help='ST, reinmax, STGS, GRMC-20, MVE')
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
categorical_dim = 4
targets = torch.Tensor([0.45]).repeat(latent_dim).contiguous()
batched_targets = targets.unsqueeze(0).expand(batch_size, -1)

# Fixed random embedding matrix W: maps one-hot (K,) -> scalar (1,)
# Shape: (K,) — one scalar output per category
torch.manual_seed(0)  # fix W independently of run seed so it's the same across methods
W = torch.randn(categorical_dim)
W = (W - W.min()) / (W.max() - W.min())  # normalise to [0, 1] so target scale is meaningful


class Quadratic_Toy(nn.Module):
    def __init__(self):
        super(Quadratic_Toy, self).__init__()
        self.theta = nn.Parameter(torch.Tensor(latent_dim, categorical_dim))
        self.theta.data.uniform_(-0.01, 0.01)
        self.register_buffer('W', W)  # shape: (K,)

    def forward(self, temp, batch_size):
        theta_b = self.theta.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        qy = F.softmax(theta_b, dim=-1)
        sample_idx = torch.distributions.Categorical(probs=qy).sample()
        y_onehot = torch.zeros_like(theta_b).scatter_(-1, sample_idx.unsqueeze(-1), 1.0)

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
        elif self.method == 'MVE':
            from Gradient_estimators import MVE
            temp_val = 1e6
            b, l, k = theta_b.shape
            z_flat = MVE(theta_b.view(-1, k), y_onehot.view(-1, k), temp=temp_val)
            z_hard = z_flat.view(b, l, k)
        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")

        # Map one-hot -> scalar via fixed random embedding W: (B, L, K) x (K,) -> (B, L)
        z = torch.einsum('blk,k->bl', z_hard, self.W)
        log_y = None

        return z, qy, log_y


model = Quadratic_Toy()
if args.cuda:
    model.cuda()
    targets = targets.cuda()
    batched_targets = batched_targets.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)
model.method = args.method


def loss_scale(mse):
    return mse.abs().pow(args.pnorm)


def loss_function(z):
    MSE = loss_scale(z - batched_targets).sum(dim=-1) / latent_dim
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

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results", "polynomial_programming")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"poly_p_{args.method}_p{args.pnorm}_seed{args.seed}.pkl"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "wb") as f:
        pickle.dump(losses, f)
    print(f"Saved results to {filepath}")


if __name__ == '__main__':
    run()