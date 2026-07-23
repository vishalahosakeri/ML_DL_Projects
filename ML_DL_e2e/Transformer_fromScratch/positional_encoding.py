import torch
import math
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self,seq_len,d_model):
        super().__init__()
        #create postinal encoding matrx and buffer it so that it cab moved to gpu with model 
        self.register_buffer("pe",get_positional_encoding(seq_len,d_model).unsqueeze(0))

    def forward(self, x) -> torch.Tensor:
        #add postional encoding vector to embedded token
            
        x = x + self.pe[:,:x.size(1),:]
        return x


'''
    positional encoding matrix :
    Get the postional encoding vector for each token in sequence using fixed postional encoding.
    Computes the fixed sinusoidal positional encoding introduced in the Attention Is All You Need paper. Even dimensions use sine and odd dimensions use cosine.
'''
def get_positional_encoding(seq_len,d_model) -> torch.Tensor:
    # print(f"embedded {seq_len}")
    pe = torch.zeros(seq_len,d_model,dtype=torch.float32)
    positions = torch.arange(0,seq_len,dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0,d_model,2).float() * - math.log(10000) / d_model)
    pe[:,0::2] = torch.sin(positions * div_term)
    pe[:,1::2] = torch.cos(positions * div_term)
    return pe

# pos = PositionalEncoding(3,10)
# x = torch.arange(1,31).reshape(3,10)
# print(pos(x).shape)
