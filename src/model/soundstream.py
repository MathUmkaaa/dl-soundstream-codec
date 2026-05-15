import torch
from torch import nn
import torch.nn.functional as F

from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.rvq import ResidualVectorQuantizer


class SoundStream(nn.Module):
    "Full codec: encoder - RVQ - decoder.Returns reconstructed waveform, commit loss and code indices"

    def __init__(
        self,
        num_quantizers=8,
        codebook_size=1024,
        codebook_dim=256,
        ema_coef=0.99,
        commit_coef=1.0,
    ):
        super().__init__()
        self.encoder = Encoder()
        self.rvq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            ema_coef=ema_coef,
            commit_coef=commit_coef,
        )
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z_q, commit_loss, indices = self.rvq(z)
        x_hat = self.decoder(z_q)
        if x_hat.shape[-1] < x.shape[-1]:
            x_hat = F.pad(x_hat, (0, x.shape[-1] - x_hat.shape[-1]))
        x_hat = x_hat[..., :x.shape[-1]]
        return x_hat, commit_loss, indices

    @torch.no_grad()
    def encode(self, x):
        z = self.encoder(x)
        z_q, commit_loss, indices = self.rvq(z)
        return indices

    @torch.no_grad()
    def decode(self, indices):
        z_q = torch.zeros(
            indices.shape[0],
            self.rvq.quantizers[0].codebook_dim,
            indices.shape[2],
            device=indices.device,
        )
        for i, quantizer in enumerate(self.rvq.quantizers):
            q_i = quantizer.codebook[indices[:, i, :]].permute(0, 2, 1)
            z_q = z_q + q_i
        x_hat = self.decoder(z_q)
        return x_hat
