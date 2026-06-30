import numpy as np

x = [2,3,1,4]
x_arr = np.array(x)
print(x_arr.shape)
print(x_arr.size)
print(x_arr.ndim)

print(np.random.rand(*x_arr.shape))

mask = ((np.random.rand(*x_arr.shape))<0.5).astype(x_arr.dtype)
print(mask)

class DropOut:

    def __init__(self,rate:float=0.5):
        assert 0.0 <= rate < 1.0
        self.training = True
        self.mask = None
        self.rate = rate

    def forward(self,x:np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0.0: # only during training or drop rate is more than 0
            return x   
        self.mask = ((np.random.rand(*x.shape)) < 1 -  self.rate).astype(x.dtype) #masking
        return x * self.mask/(1 -  self.rate) #inverted scaling
    
    



