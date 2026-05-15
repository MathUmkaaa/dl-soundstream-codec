import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F


class WaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, padding=7),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20),
            nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20),
            nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20),
            nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20),
            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.Conv1d(1024, 1, kernel_size=3, padding=1),
        ])
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        feature_maps = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
            feature_maps.append(x)
        return feature_maps


class STFTResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=(2, 1), padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(2, 1))
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.conv2(self.activation(self.conv1(x))) + self.skip(x)


class STFTDiscriminator(nn.Module):
    "2D discriminator on complex STFT"

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

        self.layers = nn.ModuleList([
            nn.Conv2d(2, 32, kernel_size=7, padding=3),
            STFTResidualUnit(32, 64),
            STFTResidualUnit(64, 128),
            STFTResidualUnit(128, 256),
            STFTResidualUnit(256, 512),
            nn.Conv2d(512, 1, kernel_size=3, padding=1),
        ])
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        stft = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        spec = torch.stack([stft.real, stft.imag], dim=1)

        feature_maps = []
        for i, layer in enumerate(self.layers):
            spec = layer(spec)
            if i < len(self.layers) - 1:
                spec = self.activation(spec)
            feature_maps.append(spec)
        return feature_maps


class MultiDiscriminator(nn.Module):
    "Three wave discriminators at three scales and one STFT discriminator"

    def __init__(self):
        super().__init__()
        self.wave_discriminators = nn.ModuleList([
            WaveDiscriminator(),
            WaveDiscriminator(),
            WaveDiscriminator(),
        ])
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        self.stft_discriminator = STFTDiscriminator()

    def forward(self, x):
        outputs = []
        current = x
        for i, disc in enumerate(self.wave_discriminators):
            outputs.append(disc(current))
            if i < len(self.wave_discriminators) - 1:
                current = self.pooling(current)
        outputs.append(self.stft_discriminator(x))
        return outputs
