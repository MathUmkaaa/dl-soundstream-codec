from torch import nn
from torch.nn import Sequential

class ResidualUnit(nn.Module):
    "SEANet residual unit: dilated conv 7 + 1 conv"

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

class EncoderBlock(nn.Module):
    "three residual units + strided conv that doubles channels"

    def __init__(self, channels, stride):
        super().__init__()
        self.layers = Sequential(
            ResidualUnit(channels, dilation=1),
            ResidualUnit(channels, dilation=3),
            ResidualUnit(channels, dilation=9),
            nn.ELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=2*stride, stride=stride, padding=(stride+1)//2),
        )
    
    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    "SEANet encoder, four blocks with strides 2-4-5-5"

    def __init__(self):
        super().__init__()
        self.layers = Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            EncoderBlock(32, stride=2),
            EncoderBlock(64, stride=4),
            EncoderBlock(128, stride=5),
            EncoderBlock(256, stride=5),
            nn.ELU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return self.layers(x)

    