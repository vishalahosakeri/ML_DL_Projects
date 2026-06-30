import torch
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import nn

torch.manual_seed(42)

#load data
X,y = fetch_california_housing(return_X_y=True)
# print(len(X),X.shape, y.shape) #20640 (20640, 8) (20640,)

#split data
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.2,random_state=42)

#scale them
scalerX = StandardScaler()
X_train = scalerX.fit_transform(X_train)
X_test = scalerX.transform(X_test)
scalerY = StandardScaler()
y_train = scalerY.fit_transform(y_train.reshape(-1,1))
y_test_notScaled = y_test.reshape(-1,1)
y_test = scalerY.transform(y_test.reshape(-1,1))

#write device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

#Convert them to pytorch tensors
X_train = torch.tensor(X_train,device=device).type(torch.float32)
X_test = torch.tensor(X_test,device=device).type(torch.float32)
y_train = torch.tensor(y_train,device=device).type(torch.float32)
y_test_notScaled = torch.tensor(y_test_notScaled,device=device).type(torch.float32)
y_test = torch.tensor(y_test,device=device).type(torch.float32)


#create model
class HousingModel(nn.Module):
    def __init__(self,inputFeaturesCnt):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=inputFeaturesCnt, out_features=32),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(in_features=16, out_features=1)
        )
    
    def forward(self,X):
        return self.linear_stack(X)
        

regModel = HousingModel(inputFeaturesCnt=X_train.shape[1]).to(device)
print(sum(p.numel() for p in regModel.parameters()))
# lastLayer = regModel.linear_stack[-1]

#Create loss , optimizer and LR scheduler objects
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(regModel.parameters(),lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=0.2)

#create train and test loop
epochs = 500
train_loss = []
test_loss = []
epoch_cnt = []
patience = 20
#early stopping variables
best_loss = float("inf")
cnt_stop = 0
min_delta = 1e-4
best_epoch = 0


for epoch in range(epochs):
  epoch_cnt.append(epoch)
  regModel.train()
  y_pred = regModel(X_train) # forward pass
  loss = loss_fn(y_pred,y_train) #calcualte loss
  optimizer.zero_grad() # set grad to zero before backward prop
  loss.backward() #backward prop
  optimizer.step() #update weight
  train_loss.append(loss.item())

  regModel.eval()
  with torch.no_grad():
    y_pred_test = regModel(X_test)
    loss_test = loss_fn(y_pred_test,y_test)
    current_test_loss = loss_test.item()
    test_loss.append(loss_test.item())
    #evaluate metric
    mae = torch.mean(torch.abs(y_pred_test - y_test))

    #check improvement
    if current_test_loss<best_loss - min_delta:
     best_loss = current_test_loss
     cnt_stop = 0
     best_epoch = epoch
     torch.save(regModel.state_dict(), "best_model.pth")
    else:
     cnt_stop += 1

    if cnt_stop >= patience:
        print(f"Current loss is not getting changed from last {patience} epoches, so early stopping at epoch {epoch},test loss: {loss_test},train loss:{loss},best model found at {best_epoch}")
        break
    
    scheduler.step(current_test_loss) 

  if epoch % 100 == 0 or epoch == (epochs-1):
    print(f"for {epoch}: train_loss :{loss}, test_loss :{loss_test},mae:{mae}") 
     
    
regModel.load_state_dict(torch.load("best_model.pth",weights_only=True))
with torch.inference_mode():
    y_pred_inf = regModel(X_test)
    #get the exact y_pred_inf without scaling
    y_pred_inf = torch.tensor(scalerY.inverse_transform(y_pred_inf.cpu().numpy())).to(device)
    mae = torch.mean(torch.abs(y_pred_inf - y_test_notScaled))
    print(f"Without scaled target, mae with out scaling:{mae}")

#visualize the loss curves
plt.plot(epoch_cnt,train_loss,c="blue",label="train")
plt.plot(epoch_cnt,test_loss,c="green",label="test")
plt.title("Loss curve")
plt.xlabel("Epoch cnt")
plt.ylabel("Loss")
plt.legend()
plt.show()






     


