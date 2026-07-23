import torch
from torch import nn
from encoder_block import EncoderBlockPostLN
from decoder_block import DecoderBlockPostLNcrossAtt

class EncoderDecoderTransformer(nn.Module):
    def __init__(self,d_model,heads,ffnDim,noOfEncoders,noOfDecoders):
        super().__init__()
        self.encoders = nn.ModuleList([EncoderBlockPostLN(d_model,heads,ffnDim) for i in range(noOfEncoders)])
        self.decoders = nn.ModuleList([DecoderBlockPostLNcrossAtt(d_model,heads,ffnDim) for i in range(noOfDecoders)])
    
    def forward(self,x,y):
        #x is emedded posital encoded input
        #y is embedded positional encoded target token
        encoder_op = x
        decoder_op = y
        for encoder in self.encoders:
            encoder_op = encoder(encoder_op)
        for decoder in self.decoders:
            decoder_op = decoder(decoder_op,encoder_op)
        return decoder_op
