import torch
from torch import nn
from attention import MultiheadAttention

class ResidualConnect(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, originalInput,subLayerOutput):
        return originalInput + subLayerOutput
    

    
