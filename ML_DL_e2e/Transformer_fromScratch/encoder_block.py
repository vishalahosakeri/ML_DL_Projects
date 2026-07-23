import torch
from torch import nn
from attention import MultiheadAttention
from residual_connection import ResidualConnect
from layerNormalization import LayerNormalization
from feed_forward import FeedForward

class EncoderBlockPostLN(nn.Module):
    def __init__(self,d_model,heads,ffnDim):
        super().__init__()
        self.attention = MultiheadAttention(d_model,heads)
        self.residualCon1 = ResidualConnect()
        self.layerNorm1 = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model,ffnDim)
        self.residualCon2 = ResidualConnect()
        self.layerNorm2 = LayerNormalization(d_model)

    def forward(self, x,encoder_padded_mask):
        #x is input with embedded positional encoding.
        attentionOp = self.attention(x,x,x,mask=encoder_padded_mask)
        residualOp = self.residualCon1(x,attentionOp)
        x = self.layerNorm1(residualOp)
        ffnOp = self.ffn(x)
        residualOp = self.residualCon2(x,ffnOp)
        output = self.layerNorm2(residualOp)
        return output
    
class EncoderBlockPreLN(nn.Module):
    def __init__(self,d_model,heads,ffnDim):
        super().__init__()
        self.attention = MultiheadAttention(d_model,heads)
        self.residualCon1 = ResidualConnect()
        self.layerNorm1 = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model,ffnDim)
        self.residualCon2 = ResidualConnect()
        self.layerNorm2 = LayerNormalization(d_model)

    def forward(self, x,encoder_padded_mask):
        layerOp1 = self.layerNorm1(x)
        #x is input with embedded positional encoding.
        attentionOp = self.attention(layerOp1,layerOp1,layerOp1,mask=encoder_padded_mask)
        residualOp = self.residualCon1(x,attentionOp)
        
        layerOp2 = self.layerNorm2(residualOp)
        ffnOp = self.ffn(layerOp2)
        output = self.residualCon2(x,ffnOp)
        return output