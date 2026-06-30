import sklearn
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from pathlib import Path
import requests


#1. get data and prepare it for model
X, y = make_circles(1000,noise=0.03,random_state=42)

circles = pd.DataFrame({"c1":X[:,0],"c2":X[:,1],"target":y})
# print(circles.head())

#2. scatter them and check how data looks like, split data
plt.scatter(X[:,0],X[:,1],c=y, cmap=plt.cm.RdBu)
plt.legend()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#3. scale them
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
# y_scaler = StandardScaler()
# y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
# y_test = y_scaler.transform(y_test.reshape(-1,1))

# print(X_train.shape,y_train.shape)
# print(X_test.shape,y_test.shape)

#4. Device agnostic code,move the data to device after converting numpy to tensor
device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = torch.from_numpy(X_train).to(device=device).type(torch.float32)
X_test = torch.from_numpy(X_test).to(device=device).type(torch.float32)
y_train = torch.from_numpy(y_train).to(device=device).type(torch.float32)
y_test = torch.from_numpy(y_test).to(device=device).type(torch.float32)

#5. Create model
class binaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linearStack = nn.Sequential(
            nn.Linear(in_features=2,out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50,out_features=5),
            nn.ReLU(),
            nn.Linear(in_features=5,out_features=1)
        )

    def forward(self,X):
        return self.linearStack(X)

model0_classify = binaryClassifier().to(device)
lstLayer = model0_classify.linearStack[-1]
# print(lstLayer.weight)

#6. create loss, optimizer and scheduler object
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model0_classify.parameters(),lr=0.1)
optimScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=5)

#7. evaluation method
def accuracy(actual:torch.tensor,predicted:torch.tensor):
    # print("Check::::",actual.shape,predicted.shape)
    correct = torch.eq(actual,predicted).sum().item()
    acc = (correct/len(actual))*100
    return acc

#8. training and evaluating loop
epochs = 1000
train_loss =[]
test_loss =[]
epoch_cnt =[]
#eralystopping variables
best_loss = float('inf')
delta = 1e-4
cnt_stop = 0
patience = 20
best_epoch = 0

for epoch in range(epochs):
    #train samples
    model0_classify.train()
    train_logits = model0_classify(X_train).squeeze() #forward prop, model gives logits
    train_pred = torch.round(torch.sigmoid(train_logits)).squeeze()#sigmod to convert the logits to probabilities, round - is for 1 or 0
    loss = loss_fn(train_logits,y_train)
    optimizer.zero_grad() #set grads to zero before back prop
    loss.backward() #back propagation
    optimizer.step() #update parameters
    train_acc = accuracy(y_train,train_pred)

    #evaluate model - validation dataset
    model0_classify.eval()
    with torch.no_grad():
        val_logits = model0_classify(X_test).squeeze()#forward prop
        val_pred = torch.round(torch.sigmoid(val_logits)).squeeze()#get 0 0r 1
        loss_test = loss_fn(val_logits,y_test).item()
        val_acc = accuracy(y_test,val_pred)

        #early stopping
        if loss_test<best_loss - delta:
            # still loss is decresing , improvements
            best_loss = loss_test
            cnt_stop = 0
            best_epoch = epoch
            torch.save(model0_classify.state_dict(),f="best_binaryModel.pth")
        else:
            #no improvements in loss decresing
            cnt_stop += 1

        if cnt_stop >= patience:
            print(f"There is no improvement in loss from last {patience}, so stopping at epoch:{epoch}.")
            break

    optimScheduler.step(loss_test)  

    epoch_cnt.append(epoch)
    train_loss.append(loss.item())
    test_loss.append(loss_test)  

    if epoch % 20 == 0:
        print(f"epoch :{epoch},train loss:{loss},test loss:{loss_test},train_acc:{train_acc},test_acc:{val_acc}")
        print(optimizer.param_groups[0]["lr"])


model0_classify.load_state_dict(torch.load("best_binaryModel.pth",weights_only=True))
#9. Test inference
with torch.inference_mode():
    y_pred = torch.round(torch.sigmoid(model0_classify(X_test))).squeeze()
    acc = accuracy(y_test,y_pred)

#10. get the helper function method

if Path("helper_functions.py").is_file():
    print("File exists")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py","wb") as f:
        f.write(request.content)


import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from helper_functions import plot_decision_boundary

# 11. plot model predictions
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title("train")
plot_decision_boundary(model0_classify,X_train,y_train)
plt.subplot(1,2,2)
plt.title("test")
plot_decision_boundary(model0_classify,X_test,y_test)
# plt.legend()
plt.show()

# 12. Loss curve
# train_loss = np.array(torch.tensor(train_loss).numpy())
plt.plot(epoch_cnt,train_loss,c="blue",label="train")
plt.plot(epoch_cnt,test_loss,c="green",label="test")
plt.title("loss curve")
plt.xlabel("Epoch cnt")
plt.ylabel("loss")
# plt.legend()
plt.show()




