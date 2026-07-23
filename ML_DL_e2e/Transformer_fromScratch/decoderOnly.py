import torch
from torch import nn
from decoder_block import DecoderBlockPostLN

'''
This is decoder which has noOfEncoders encoder blocks and gets output from last encoder.
It does not take any input from encoder, so no cross attention.
'''
class DecoderTransformer(nn.Module):
    def __init__(self,d_model,heads,ffnDim,noOfDecoders):
        super().__init__()
        self.decoderModule = nn.ModuleList([DecoderBlockPostLN(d_model=d_model,heads=heads,ffnDim=ffnDim) for i in range(noOfDecoders)])

    def forward(self,x):
        for decoder in self.decoderModule:
            x = decoder(x)
        return x
