import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

#1 . prepare data
w = 0.7
b = 0.3

X = torch.arange(1,5,0.02).unsqueeze(dim=1)
y = w * X + b

split = int(len(X) * 0.8)
x_train = X[:split]
y_train = y[:split]

x_test = X[split:]
y_test = y[split:]

print(len(x_train),len(x_test))

# Visualization definition
def plot_predict(train_data = x_train, train_label = y_train, test_data = x_test, test_label = y_test,predicions = None):
    plt.scatter(x_train,y_train,c="blue",label="train data",s=4)
    plt.scatter(x_test, y_test,c="green",label="test data",s=6)
    
    if predicions != None:
        plt.scatter(test_data,predicions,c="red",label="predctions",s=8)

    plt.legend()
    plt.show()

# plot_predict()

#3 .create model - linear regression model
class LinearRegressionModelV1(nn.Module):

    def __init__(self):
        super().__init__()
        self.linearlayer = nn.Linear(in_features=1,out_features=1)

    def forward(self,x:torch.tensor)->torch.tensor:
        return self.linearlayer(x)
    
#4. create loss and optimize objects
model1 = LinearRegressionModelV1()

loss_fn = nn.L1Loss()
optimize = torch.optim.SGD(model1.parameters(),lr=0.01)

#5. train loop and test loop
epoches = 155
epoch_cnt = []
train_loss = []
test_loss = []

for epoch in range(epoches):

#train mode
    model1.train()
    y_pred = model1(x_train)
    loss = loss_fn(y_pred,y_train)
    optimize.zero_grad()
    loss.backward()
    optimize.step()

#evaluate mode
    model1.eval()
    with torch.inference_mode():
        y_pred_test = model1(x_test)
        test_loss_val = loss_fn(y_pred_test,y_test)

    if epoch % 10 == 0:
        epoch_cnt.append(epoch)
        train_loss.append(loss)
        test_loss.append(test_loss_val)    

with torch.inference_mode():
    y_pred_new = model1(x_test)

plot_predict(predicions=y_pred_new)

#visualize the losses in graph
train_loss = np.array(torch.tensor(train_loss).numpy())
test_loss = np.array(torch.tensor(test_loss).numpy())
plt.plot(epoch_cnt,train_loss,c="red",label="train loss")
plt.plot(epoch_cnt,test_loss,c="blue",label="test loss")
plt.title("Loss curves")
plt.xlabel("EPoch cnt")
plt.ylabel("Loss")
plt.legend()
plt.show()
    

