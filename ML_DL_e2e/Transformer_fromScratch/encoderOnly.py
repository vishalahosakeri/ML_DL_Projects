import torch
from torch import nn
from encoder_block import EncoderBlockPostLN

'''
This is enocoder which has noOfEncoders encoder blocks and gets output from last encoder.
'''
class EncoderTransformer(nn.Module):
    def __init__(self,d_model,heads,ffnDim,noOfEncoders):
        super().__init__()
        self.encoderModule = nn.ModuleList([EncoderBlockPostLN(d_model=d_model,heads=heads,ffnDim=ffnDim) for i in range(noOfEncoders)])

    def forward(self,x):
        for encoder in self.encoderModule:
            x = encoder(x)
        return x
