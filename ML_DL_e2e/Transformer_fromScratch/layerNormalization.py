import torch
from torch import nn

class LayerNormalization(nn.Module):
    def __init__(self,d_model,epsilon=1e-4):
        super().__init__()
        self.d_model = d_model
        self.epsilon = epsilon #for numerical stability , if tensor is identical = [5,5,5], variance becomes 0
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, X):
        mean = torch.mean(X,dim=-1,keepdim=True)
        var = torch.var(X,dim=-1,keepdim=True,unbiased=False) #unbiased = false since we are not doing on samples
        X_normalized = (X - mean)/torch.sqrt(var+self.epsilon)
        out = self.gamma * X_normalized + self.beta
        return out






