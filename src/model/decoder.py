from torch import nn
from torch.nn import Sequential

class ResidualUnit(nn.Module):
    "Same residual unit as in the encoder"

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.layers = Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=3*dilation),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ELU()
        )

    def forward(self, x):
        return x + self.layers(x)

class DecoderBlock(nn.Module):
    "Transposed conv that halves channels, then three residual units"

    def __init__(self, channels, stride):
        super().__init__()
        self.layers = Sequential(
            nn.ConvTranspose1d(channels * 2, channels, kernel_size=2*stride, stride=stride, padding=(stride+1)//2, output_padding=2 * ((stride + 1) // 2) - stride),
            ResidualUnit(channels, dilation=1),
            ResidualUnit(channels, dilation=3),
            ResidualUnit(channels, dilation=9),
        )
    
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    "SEANet decoder, mirrors the encoder with strides 5-5-4-2, final tanh keeps output in [-1, 1]"

    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            nn.Conv1d(256, 512, kernel_size=7, padding=3),
            DecoderBlock(256, stride=5),
            DecoderBlock(128, stride=5),
            DecoderBlock(64, stride=4),
            DecoderBlock(32, stride=2),
            nn.ELU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.layers(x)

    