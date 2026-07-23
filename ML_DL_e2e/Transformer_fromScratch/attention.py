'''
This attention mechanism receives the embedded with postional encoding tokens,X, as input.
Compute Q, K, V using  Q = WqX, K = WkX, V = WvV

'''

import torch
from torch import nn
import math

class MultiheadAttention(nn.Module):
    def __init__(self,d_model,heads):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("reminder of d_mode/heads should be zero.")
        self.head = heads
        self.d_model = d_model
        self.head_dim = d_model // heads
        self.scaledAttention = ScaledDotProductAttention()
        self.queryLayer = nn.Linear(in_features=d_model,out_features=d_model)
        self.keyLayer = nn.Linear(in_features=d_model,out_features=d_model)
        self.valueLayer = nn.Linear(in_features=d_model,out_features=d_model)
        self.finalLayer = nn.Linear(in_features=d_model,out_features=d_model)
    
    def forward(self, Q, K, V,mask):
        #get Q, K, V value
        query = self.queryLayer(Q)
        # print(query.shape)
        key = self.keyLayer(K)
        # print(key.shape)
        value = self.valueLayer(V)
        # print(value.shape)
        #reshape/split into heads
        query_head = (query.reshape(Q.size(0),Q.size(1),self.head,self.head_dim)).permute(0,2,1,3)
        key_head = (key.reshape(K.size(0),K.size(1),self.head,self.head_dim)).permute(0,2,1,3)
        val_head = (value.reshape(V.size(0),V.size(1),self.head,self.head_dim)).permute(0,2,1,3)
        context_vector = self.scaledAttention(query_head,key_head,val_head,mask)
        #combine the outputs
        # print(f"context_vector shape :{context_vector.shape}")
        output = (context_vector.permute(0,2,1,3)).reshape(Q.size(0),Q.size(1),Q.size(2))
        # print(output.shape)

        return self.finalLayer(output)

class ScaledDotProductAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,query,key,value,mask):
        attention_score = torch.matmul(query,torch.transpose(key,3,2))
        scale = math.sqrt(query.size(3)) #square root of d model 
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_score = attention_score.masked_fill(~mask,float('-inf'))
        # print(f"Shape of attention score:{attention_score.shape},{value.shape}")
        attention_weigts = torch.softmax(attention_score/scale,dim=3)
        context_vector = torch.matmul(attention_weigts, value)
        return context_vector

     
# attentionHead = MultiheadAttention(512,7)
# x = torch.arange(1,3073,dtype=torch.float32).reshape(2,3,512)
# #self attention
# print(attentionHead(x,x,x,mask=None))

