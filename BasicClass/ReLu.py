import numpy as np

class Relu:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0,self.Z)
    
    def backward(self, dA):
        dZ = dA.copy()
        dZ[self.Z<=0]=0    #practice
        return dZ

