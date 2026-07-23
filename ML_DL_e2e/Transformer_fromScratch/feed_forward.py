import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self,d_model,ffnDim):
        super().__init__()
        self.expandLayer = nn.Sequential(
            nn.Linear(in_features=d_model,out_features=ffnDim),
            nn.GELU(),
            nn.Linear(in_features=ffnDim,out_features=d_model)
        )

    def forward(self,x):
        return self.expandLayer(x)

# ffn = FeedForward(512,2048)
# x = torch.arange(1,1025,dtype=torch.float32).reshape(1,2,512)
# #self attention
# print(ffn(x))