import torch
from torch import nn
import torch.nn.functional as F

class VectorQuantizer (nn.Module):
    def __init__(self, codebook_size, codebook_dim, ema_coef, commit_coef, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.ema_coef = ema_coef
        self.commit_coef = commit_coef
        self.eps = eps

        codebook = torch.randn(codebook_size, codebook_dim) * 0.01
        self.register_buffer("codebook", codebook)

        ema_count = torch.zeros(codebook_size)
        self.register_buffer("ema_count", ema_count)
        ema_sum = torch.zeros(codebook_size, codebook_dim)
        self.register_buffer("ema_sum", ema_sum)
    
    def forward(self, x):
        batch_size, codebook_dim, count_frames = x.shape
        x_permuted = x.permute(0, 2, 1).contiguous().view(-1, codebook_dim)

        distances = (x_permuted.pow(2).sum(dim=1, keepdim=True)
                    - 2 * x_permuted @ self.codebook.t()
                    + self.codebook.pow(2).sum(dim=1))
        i_nearest = torch.argmin(distances, dim=1)
        one_hot = F.one_hot(i_nearest, self.codebook_size).float()
        quantized = F.embedding(i_nearest, self.codebook).view(batch_size, count_frames, codebook_dim).permute(0, 2, 1)

        if self.training:
            with torch.no_grad():
                self.ema_count.mul_(self.ema_coef).add_(one_hot.sum(dim=0), alpha=1 - self.ema_coef)
                self.ema_sum.mul_(self.ema_coef).add_(one_hot.t() @ x_permuted, alpha=1 - self.ema_coef)
                n = self.ema_count.sum()
                counts = (self.ema_count + self.eps) / (n + self.codebook_size * self.eps) * n
                self.codebook.copy_(self.ema_sum / counts.unsqueeze(1))

        commit_loss = self.commit_coef * F.mse_loss(x, quantized.detach())
        quantized = x + (quantized - x).detach()
        indices = i_nearest.view(batch_size, count_frames)
        return quantized, commit_loss, indices

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_quantizers, codebook_size, codebook_dim, ema_coef, commit_coef, eps=1e-5):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([VectorQuantizer(codebook_size, codebook_dim, ema_coef, commit_coef, eps) for i in range(num_quantizers)])

    def forward(self, x):
        residual = x
        quantized_sum = torch.zeros_like(x)
        commit_losses = []
        indices_list = []

        for quantizer in self.quantizers:
            q_i, loss_i, idx_i = quantizer(residual)
            quantized_sum += q_i
            commit_losses.append(loss_i)
            indices_list.append(idx_i)
            residual = residual - q_i

        commit_loss = torch.stack(commit_losses).mean() # на будущее - может иначе а не усреднять (аналогично с усреднением вектора кодбука)
        indices = torch.stack(indices_list, dim=1)

        return quantized_sum, commit_loss, indices
