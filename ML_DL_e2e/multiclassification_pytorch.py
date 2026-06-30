import sklearn
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from torch import nn
import requests
from pathlib import Path

torch.manual_seed(42)

#1. Get data
X_blob, y_blob = make_blobs(n_samples=1000,n_features=2,centers=4,random_state=42,cluster_std=1.5)
# print(X_blob.shape,y_blob.shape)
# print(X_blob[:5],y_blob[:5])

#2. scatter data and visualize
plt.scatter(X_blob[:,0],X_blob[:,1],c=y_blob)
plt.show()

#s3. split data 
X_train,X_test,y_train,y_test = train_test_split(X_blob,y_blob,test_size=0.3,random_state=42)

#3. Scale the inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#target not scaled since its classification 

#4. Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#5. convert numpy to tensor and move data to current device
X_train = torch.tensor(X_train).to(device).type(torch.float32)
X_test = torch.tensor(X_test).to(device).type(torch.float32)
y_train = torch.tensor(y_train).to(device).long()
y_test = torch.tensor(y_test).to(device).long()

#6. Create model
class multiClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linearStack = nn.Sequential(
            nn.Linear(in_features=2, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8,out_features=6),
            nn.ReLU(),
            nn.Linear(in_features=6,out_features=4)
        )
    
    def forward(self,X):
        return self.linearStack(X)

model0_multiClass = multiClassification().to(device)

#Testing 
y_test_logits = model0_multiClass(X_test)
print("logits:",y_test_logits[:5])
y_test_prob = torch.softmax(y_test_logits,dim=1)
print("prob:",y_test_prob[:5])
y_test_pred = torch.argmax(y_test_prob,dim=1)
print("pred:",y_test_pred[:5])
print("actual:",y_test[:5])

#7. Create loss, optimizer and scheduler 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model0_multiClass.parameters(),lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5)

#8. accuracy
def accuracy(actual, predict):
    correct = torch.eq(actual,predict).sum().item()
    acc = (correct/len(predict))*100
    return acc

#9. get visualization method
if Path("helper_functions.py").is_file():
    print("File aleady exists")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py","wb")as f:
        f.write(request.content)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from helper_functions import plot_decision_boundary

#10. training and evaluation loop
epoches = 300
train_loss = []
test_loss = []
epoch_cnt = []
#earky stopping
delta = 1e-4
cnt_stop =0
patience = 10
best_loss = float("inf")
best_epoch = 0

for epoch in range(epoches):
    #train
    model0_multiClass.train()
    train_logits = model0_multiClass(X_train)#forward prop
    train_prob = torch.softmax(train_logits,dim=1)#get prob from logits
    train_pred = torch.argmax(train_prob,dim=1) # get class index
    loss = loss_fn(train_logits,y_train) #cross entropy handles the softmax internally, so sending logits
    train_acc = accuracy(y_train,train_pred)

    optimizer.zero_grad()#set grads to zero before back prop
    loss.backward()#back prop
    optimizer.step()#update parameters

    #evaluate
    model0_multiClass.eval()
    with torch.no_grad():
        test_logits = model0_multiClass(X_test)
        test_pred = torch.argmax(torch.softmax(test_logits,dim=1),dim=1)
        loss_test = loss_fn(test_logits,y_test)
        test_acc = accuracy(y_test,test_pred)

        #early stopping
        if loss_test <= best_loss-delta:
            #still loss is improving
            best_loss = loss_test
            cnt_stop = 0
            best_epoch = epoch
            torch.save(model0_multiClass.state_dict(),"best_multiClassModel.pth")
        else:
            cnt_stop += 1

        if cnt_stop >= patience:
            print(f"Loss stopped decreasing at {best_epoch}")
            break

    scheduler.step(loss_test)

    epoch_cnt.append(epoch)
    train_loss.append(loss.item())
    test_loss.append(loss_test.item())

    if epoch % 20 ==0 :
        print(f"epoch :{epoch},train_loss:{loss.item():.4f},test_loss:{loss_test.item():.4f},train_acc:{train_acc},test_acc:{test_acc}")
        # print(optimizer.param_groups[0]["lr"])

model0_multiClass.load_state_dict(torch.load("best_multiClassModel.pth",weights_only=True))

#11. Infernce
with torch.inference_mode():
    y_test_logits = model0_multiClass(X_test)
    y_test_pred = torch.argmax(torch.softmax(y_test_logits,dim=1),dim=1)
    acc_inf = accuracy(y_test,y_test_pred)
    print(f"final accuracy on best parameters : {acc_inf}")
    
    # print(f"Final best weights:{model0_multiClass.state_dict()}")

#12. visualize
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model0_multiClass,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model0_multiClass,X_test,y_test)
plt.show()

#13. curve loss
plt.plot(epoch_cnt,train_loss,c="blue",label="train_loss")
plt.plot(epoch_cnt,test_loss,c="green",label="test loss")
plt.xlabel("Epoch cnt")
plt.ylabel("loss")
plt.legend()
plt.show()