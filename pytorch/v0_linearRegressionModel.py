
import matplotlib.pyplot as plt
import torch
import numpy as np
# create know parameters
weight = 0.7
bias = 0.3

#create
start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)
y = weight * X + bias

## splitting data into training and test data

split = int(0.8 * len(X))
x_train , y_train = X[:split],y[:split]
x_test, y_test = X[split:],y[split:]

def plot_predictions(train_data = x_train, train_labels= y_train,
                     test_data = x_test, test_labels= y_test, predictions=None):
  plt.figure(figsize=(10,7))

  #train data
  plt.scatter(train_data,train_labels,c="b",s=4,label ="Training dta")
  # test data
  plt.scatter(test_data,test_labels.cpu().numpy(),c="g",s=16,label="Test data")

  if predictions is not None:
    predictions = np.array(torch.tensor(predictions))
    plt.scatter(test_data,predictions,c="r",s=4,label= 'Predictions')

  plt.legend()
  plt.show()

print("Type of train_data:",type(x_train))
plot_predictions()

# create linear regression model
import torch
from torch import nn
class LinearRegressionModel(nn.Module):# nn module is base class for all modules. So, linear regression is extending nn.module
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float ))
    self.bias = nn.Parameter(torch.randn(1,requires_grad=True,dtype=torch.float))
 
  # forward propagation
  def forward(self, x:torch.tensor)-> torch.tensor:
    return self.weight * x + self.bias

# created model, but check it now
torch.manual_seed(42)

#create model instance of model
model_0 = LinearRegressionModel()

#check out parameteres
list(model_0.parameters())

# set up loss function
loss_fn = nn.L1Loss()
# output = loss(x_train,y_train)

#set up optimization
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.01)


epochs = 180
epoch_cnt = []
loss_val = []
test_loss_val = []

# 1. loop through data
for epoch in range(epochs) :
  #notifies specific layers (like Dropout and BatchNorm) that the model is now in training mode so they behave correctly. It does not actually start any training or optimization.
  model_0.train() # mode in pytorch that sets all parameteres that reuire gradients
  
  #forward pass
  y_pred = model_0(x_train) # or model_0(x_train)

  #loss
  loss = loss_fn(y_pred,y_train)

  #Optimizer zero grad
  optimizer.zero_grad()

  #backward prop
  loss.backward()

  #optimize step
  optimizer.step()

  model_0.eval() #  drop out etc whcih are not needed during evaluation
  with torch.inference_mode():#turn off gradient tracking
    #1 forward pass
    test_pred = model_0(x_test)
    test_loss = loss_fn(test_pred,y_test)
  
  if epoch % 10 == 0:
   epoch_cnt.append(epoch)
   loss_val.append(loss.item())
   test_loss_val.append(test_loss.item())

with torch.inference_mode():
  y_pred_new = model_0(x_test)

# print("y_pred",y_pred_new)
plot_predictions(predictions=y_pred_new)

plt.plot(epoch_cnt,loss_val,label = "trian loss value")
plt.plot(epoch_cnt,test_loss_val,label="test loss avalue")
plt.title("Loss map")
plt.xlabel("epoch cnt")
plt.ylabel("loss")
plt.legend()
plt.show()
