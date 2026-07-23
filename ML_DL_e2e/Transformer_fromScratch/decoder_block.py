import torch
from torch import nn
from attention import MultiheadAttention
from residual_connection import ResidualConnect
from layerNormalization import LayerNormalization
from feed_forward import FeedForward

class DecoderBlockPostLN(nn.Module):
    def __init__(self,d_model,heads,ffnDim):
        super().__init__()
        self.attention = MultiheadAttention(d_model,heads)
        self.residualCon1 = ResidualConnect()
        self.layerNorm1 = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model,ffnDim)
        self.residualCon2 = ResidualConnect()
        self.layerNorm2 = LayerNormalization(d_model)

    def forward(self, x):
        #x is input with embedded positional encoding.
        #create casual mask
        maskMatrix = casualMask(x)
        #masked self attention
        attentionOp = self.attention(x,x,x,mask=maskMatrix)
        residualOp = self.residualCon1(x,attentionOp)
        x = self.layerNorm1(residualOp)
        ffnOp = self.ffn(x)
        residualOp = self.residualCon2(x,ffnOp)
        output = self.layerNorm2(residualOp)
        return output
    
class DecoderBlockPreLN(nn.Module):
    def __init__(self,d_model,heads,ffnDim):
        super().__init__()
        self.attention = MultiheadAttention(d_model,heads)
        self.residualCon1 = ResidualConnect()
        self.layerNorm1 = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model,ffnDim)
        self.residualCon2 = ResidualConnect()
        self.layerNorm2 = LayerNormalization(d_model)

    def forward(self, x):
        layerOp1 = self.layerNorm1(x)
        #x is input with embedded positional encoding.
        maskMatrix = casualMask(x)
        attentionOp = self.attention(layerOp1,layerOp1,layerOp1,mask=maskMatrix)
        residualOp = self.residualCon1(x,attentionOp)
        
        layerOp2 = self.layerNorm2(residualOp)
        ffnOp = self.ffn(layerOp2)
        output = self.residualCon2(x,ffnOp)
        return output

class DecoderBlockPostLNcrossAtt(nn.Module):
    def __init__(self,d_model,heads,ffnDim):
        super().__init__()
        self.maskAttention = MultiheadAttention(d_model,heads)
        self.crossAttention = MultiheadAttention(d_model,heads)
        self.residualCon1 = ResidualConnect()
        self.residualCon3 = ResidualConnect()
        self.layerNorm1 = LayerNormalization(d_model)
        self.layerNorm3 = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model,ffnDim)
        self.residualCon2 = ResidualConnect()
        self.layerNorm2 = LayerNormalization(d_model)
    
    def forward(self, decoder_input,encoder_output,combined_mask,encoder_pad_mask):
        #decoder_input is input with embedded positional encoding for <SOS>Id german sentence tokenIds.
        #masked self attention
        attentionOp = self.maskAttention(decoder_input,decoder_input,decoder_input,mask=combined_mask)
        residualOp = self.residualCon1(decoder_input,attentionOp)
        x = self.layerNorm1(residualOp)
        #cross attention - query goes from decoder, key value comes from encoder
        # print(f"x:{x.shape}")
        # print(f"**encoder_output:{encoder_output.shape}")
        crossAttentionOp = self.crossAttention(x,encoder_output,encoder_output,mask=encoder_pad_mask)
        residualOp = self.residualCon2(x,crossAttentionOp)
        x = self.layerNorm2(residualOp)
        ffnOp = self.ffn(x)
        residualOp = self.residualCon3(x,ffnOp)
        output = self.layerNorm3(residualOp)
        return output

    
def casualMask(x):
    maskMatrix = torch.tril(torch.ones(size=(x.size(1),x.size(1)),device=x.device,dtype=torch.bool))
    return maskMatrix
    
# x = torch.arange(1,10).reshape(3,3)
# casualMask(x)