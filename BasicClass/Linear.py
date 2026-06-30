import numpy as np

# Z=XW+b
# X = input (batch_size × input_features)
# W = weights (input_features × output_features)
# b = bias (1 × output_features)
# Z = output

#back prop
# L→Z→W,b,X   i.e ex : dL/dW = dl/dZ . dZ/dW



class Linear:
    def __init__(self,in_features,out_features):
        self.W = np.random.randn(in_features,out_features) * 0.01
        self.b = np.zeros(1,out_features)
    
    #Linear Layer Forward Pass
    def forward(self,X):
        self.X = X
        self.Z = X @ self.W + self.b
        return self.Z
    
    #Linear Layer Backward Pass
    def backward(self,dZ):
        self.dZ = dZ
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ,axis=1,keepdims=True)
        self.dX = dZ @ self.W.T
        return self.dX
    
    

    