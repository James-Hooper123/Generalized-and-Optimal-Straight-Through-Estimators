import torch
import torch.nn as nn
import torch.distributions as dstr
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class Encoder(nn.Module):

    def __init__(self,n,K):
        super(Encoder,self).__init__()
        self.n = n
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, n * K)
        )


    def forward(self,x):
        eta = self.net(x)
        eta = eta.view(-1,self.n,self.K)
        return eta
    

class Decoder(nn.Module):
    
    def __init__(self,embedding, n, K):
        super(Decoder,self).__init__()
        self.n = n
        self.K = K
        if embedding == 'OHE':
            self.input_dim = n * K
        elif embedding == 'int':
            self.input_dim = n
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,784)
        )


    def forward(self,z):
        z = z.view(-1, self.input_dim)
        mu = self.net(z)
        return mu
    

class VAE(nn.Module):
    def __init__(self, n, K, embedding, step_size, gradient_estimator, device, epochs,
                 temperature_schedule):
        super(VAE, self).__init__()
        self.encoder = Encoder(n, K).to(device)
        self.decoder = Decoder(embedding, n, K).to(device)

        self.n = n
        self.K = K
        self.embedding = embedding
        self.optimiser = torch.optim.AdamW(self.parameters(), lr=step_size)

        if epochs == 500:
            warmup_epochs = 10
        elif epochs == 50:
            warmup_epochs = 5
        else:
            warmup_epochs = 0
            
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(self.optimiser, start_factor=1e-3, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(self.optimiser, T_max=max(1, epochs - warmup_epochs))
            self.scheduler = SequentialLR(self.optimiser, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        else:
            self.scheduler = CosineAnnealingLR(self.optimiser, T_max=epochs)

        self.gradient_estimator = gradient_estimator
        self.temperature_schedule = temperature_schedule
        self.cur_epoch = 0  

    def compute_loss(self,x):
        # ===== Forward pass =====
        eta = self.encoder(x)
        p = F.softmax(eta, dim=-1)
        qz = dstr.Categorical(probs=p)

        # Sample hard z
        p_nxK = p.view(-1, self.K)
        z_idx = torch.multinomial(p_nxK, num_samples=1).view(-1, self.n)
        eta = eta.reshape(-1, self.K)
        temp = self.temperature_schedule(self.cur_epoch)


        if self.embedding == 'int':
            if self.gradient_estimator.__name__ == 'MVE_int':
                z_idx = z_idx.reshape(-1)
                z = self.gradient_estimator(eta, z_idx, temp=temp)
                z = z.reshape(-1, self.n)
            else:
                #print("Check")
                z_hard = torch.zeros_like(p).scatter_(-1, z_idx.unsqueeze(-1), 1.0)
                z_hard = z_hard.reshape(-1, self.K)
                z = self.gradient_estimator(eta, z_hard, temp=temp)
                z = z.reshape(-1, self.n, self.K)
                Phi = torch.arange(self.K, device=z.device, dtype=z.dtype)
                z = torch.einsum('bnk,k->bn', z, Phi)

            z = z / (self.K - 1)

        else:
            z_hard = torch.zeros_like(p).scatter_(-1, z_idx.unsqueeze(-1), 1.0)
            z_hard = z_hard.reshape(-1, self.K)
            z = self.gradient_estimator(eta, z_hard, temp=temp)
            z = z.reshape(-1, self.n, self.K)

        
        x_mu = self.decoder(z)
        # Reconstruction loss (BCE)
        recon_loss = F.binary_cross_entropy_with_logits(x_mu, x, reduction='sum') / x.size(0)

        pz = dstr.Categorical(probs=torch.ones_like(p) / self.K)

        KL_div = dstr.kl_divergence(qz, pz).sum() / x.size(0)
        nelbo = recon_loss + KL_div

        return nelbo

    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def compute_loss_compiled(self, x):
        return self.compute_loss(x)

    def learn_step(self, x, compile=False):
        self.optimiser.zero_grad(set_to_none=False)
        if compile:
            nelbo = self.compute_loss_compiled(x)
        else:
            nelbo = self.compute_loss(x)
        # ===== Backward =====
        nelbo.backward()

        self.optimiser.step()

        return nelbo.detach()


    def evaluate_elbo(self, data_loader, device):
        """
        Evaluate ELBO without gradients or optimizer updates.
        """
        self.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for (x_batch,) in data_loader:
                x_batch = x_batch.view(x_batch.size(0), -1).to(device)

                eta = self.encoder(x_batch)
                p = F.softmax(eta, dim=-1)
                qz = dstr.Categorical(probs=p)

                # Sample z (same as training)
                p_flat = p.view(-1, self.K)
                z_idx = torch.multinomial(p_flat, num_samples=1).view(-1, self.n)
                z_hard = torch.zeros_like(p).scatter_(-1, z_idx.unsqueeze(-1), 1.0)

                if self.embedding == 'int':
                    Phi = torch.arange(self.K, device=device, dtype=z_hard.dtype)
                    z = torch.einsum('bnk,k->bn', z_hard, Phi)
                    z = z / (self.K - 1)
                else:
                    z = z_hard

                x_mu = self.decoder(z)
                recon_loss = F.binary_cross_entropy_with_logits(x_mu, x_batch, reduction='sum') / x_batch.size(0)
                pz = dstr.Categorical(probs=torch.ones_like(p) / self.K)
                KL_div = dstr.kl_divergence(qz, pz).sum() / x_batch.size(0)
                nelbo = recon_loss + KL_div

                total_loss += nelbo.item()
                n_batches += 1

        self.train()
        return total_loss / n_batches

