import torch
import numpy as np
import torchvision

print(torch.__version__)
print(torchvision.__version__)

# x = torch.tensor([1, 2, 3],requires_grad=True,dtype=torch.float)
# # print(x)
# # print(torch.__version__)
# # print(torch.backends.mps.is_available())  # For Mac GPU (M1/M2/M3)

# #convert tensor to numpy
# # l1 = [2,3,4]
# t1 = torch.tensor(x)
# n1 = t1.numpy()

# n2 = np.array([2,6,8])
# t2 = torch.from_numpy(n2)

# print(n2,t2)

# print(t2.sort(descending=True).values)
# # torch.search

# def modify_str():
#     str1 = "I love python"
#     str1.replace("love","enjoy").split()
#     return str1

# print(modify_str())