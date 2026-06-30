import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torchmetrics
from torchmetrics import Accuracy
from timeit import default_timer as timer
from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

torch.manual_seed(42)
train_data = FashionMNIST(root="data",download=True,train=True,transform=transforms.ToTensor())
test_data = FashionMNIST(root="data",download=True,train=False,transform=transforms.ToTensor())

class_names = train_data.classes

print(f"Length of train and test data: {len(train_data)},{len(test_data)}")
print(f"type of train and test data:{type(train_data)},{type(test_data)}")

#get to know the shape of image, channels
image,label = train_data[0]
print(image.shape,type(image),label)
plt.imshow(image.squeeze(dim=0),cmap="gray")
plt.title(class_names[label])
plt.show()

#use dataloader for batch
batch_size = 32
train_dataLoader = DataLoader(batch_size=batch_size,dataset=train_data,shuffle=True)
test_dataLoader = DataLoader(batch_size=batch_size,dataset=test_data,shuffle=False)

batchImages,label = next(iter(train_dataLoader))
print(batchImages.shape,len(label))

#device agnostic code
device = ("cuda" if torch.cuda.is_available() 
          else "mps" if torch.backends.mps.is_available()
          else "cpu")

#start creating model
#model1
class FashionMLP_0(nn.Module):
    def __init__(self,ip_shape,op_shape,hidden_units):
        super().__init__()
        self.cnnLayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=ip_shape,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=op_shape)
        )

    def forward(self,X):
        return self.cnnLayers(X)
    
#get random shape and check the model
model_0 = FashionMLP_0((28*28),10,len(class_names)).to(device)
# print(model_0.state_dict())

#model2
class FashionMLP_1(nn.Module):
    def __init__(self,ip_shape,op_shape,hidden_units):
        super().__init__()
        self.cnnLayers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=ip_shape,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=op_shape)
        )

    def forward(self,X):
        return self.cnnLayers(X)
    
#get random shape and check the model
model_1 = FashionMLP_1((28*28),10,len(class_names)).to(device)

#model 3
class FashionMLP_2(nn.Module):
    def __init__(self,ip_channels,hidden_channels,op_channels):
        super().__init__()
        self.cnnLayers_1 = nn.Sequential(
            nn.Conv2d(in_channels=ip_channels,out_channels=hidden_channels,kernel_size=3,padding=1,stride=1), #ip dimension : [32,1,28,28], op dimesnion: [32,10,28,28]
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,padding=1,stride=1),#ip dimension : [32,10,28,28], op dimesnion: [32,10,28,28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))#ip dimension : [32,10,28,28], op dimesnion: [32,10,14,14]
        self.cnnLayers_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,padding=1,stride=1),#ip dimension : [32,10,14,14], op dimesnion: [32,10,14,14]
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels,out_channels=hidden_channels,kernel_size=3,padding=1,stride=1),#ip dimension : [32,10,14,14], op dimesnion: [32,10,14,14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)#ip dimension : [32,10,14,14], op dimesnion: [32,10,7,7]
        )
        self.cnnLayers_3 = nn.Sequential(
            nn.Flatten(),#ip dimension : [32,10,7,7], op dimesnion: [32,490]
            nn.Linear(in_features=hidden_channels*7*7,out_features=op_channels) #input features = multiplication of  hidden units * width and height of input
        )

    def forward(self,X):
        X = self.cnnLayers_1(X)
        X= self.cnnLayers_2(X)
        X = self.cnnLayers_3(X)
        return X

model_2 = FashionMLP_2(1,10,len(class_names)).to(device)

#create loss, optimizer and schedulers
loss_fn = nn.CrossEntropyLoss()
# optimizer_model0 = torch.optim.SGD(model_0.parameters(),lr=0.01)
# optimizer_model1 = torch.optim.SGD(model_1.parameters(),lr=0.01)
# optimizer_model2 = torch.optim.SGD(model_2.parameters(),lr=0.01)
# scheduler_model0 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_model0,patience=5)
# scheduler_model1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_model1,patience=5)
# scheduler_model2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_model2,patience=5)

#use torchmetrics instead of custom accuracy
acc_fn = Accuracy(task="multiclass",num_classes=len(class_names)).to(device)

#define training, testing and evaluating methods
def training_step(model:nn.Module,dataLoader:DataLoader,loss_fn:nn.Module,optimizer:torch.optim.Optimizer,epoch:int,device:torch.device=device):
    train_loss ,train_acc = 0, 0

    #train mode
    model.train()
    for batch_idx, (X_train,y_train) in enumerate(dataLoader):
        X_train = X_train.to(device)
        
        # print(f"Shape of X_train:{X_train.shape},shape of y_train:{y_train.shape}")
        y_train = y_train.to(device).long()

        y_pred_logits = model(X_train)
        y_pred = torch.argmax(torch.softmax(y_pred_logits,dim=1),dim=1).long()#forward prop
        loss = loss_fn(y_pred_logits,y_train) #calcualte loss
        optimizer.zero_grad() #set grad to zero before grad calcualtion
        loss.backward() #back prop
        optimizer.step() #update model parameters
        acc = acc_fn(y_pred,y_train) #calcualte accuaracy for each batch

        train_loss += loss # loss for all batches
        train_acc += acc # acc for all batches
     #calcaulte 
    train_loss /= len(dataLoader)
    train_acc /= len(dataLoader)

    
    print(f"epoch : {epoch},Train loss :{train_loss.item():.4f},Train acc :{train_acc.item()}",end=" ")

def testing_step(model:nn.Module,dataLoader:DataLoader,loss_fn:nn.Module,epoch:int,device:torch.device=device):
    test_loss , test_acc = 0, 0

    model.eval()
    with torch.no_grad():
        for (X_test,y_test) in dataLoader:
            X_test = X_test.to(device)
            y_test = y_test.to(device).long()

            y_pred_logits = model(X_test)
            y_pred_test = torch.argmax(torch.softmax(y_pred_logits,dim=1),dim=1)#forward prop
            loss_test = loss_fn(y_pred_logits,y_test)#loss for test
            acc_test = acc_fn(y_pred_test,y_test)# accuracy for each batch

            test_loss += loss_test
            test_acc += acc_test
    
    #calcualte loss and acc per batch
    test_loss /= len(dataLoader)
    test_acc /= len(dataLoader)

    
    print(f",Test loss: {test_loss.item():.4f},Test acc:{test_acc.item()}")
    return test_loss

def eval_model(model:nn.Module,dataLoader:DataLoader,loss_fn:nn.Module,device:torch.device=device):
    loss, acc = 0, 0
    model.to(device)

    model.eval()
    with torch.inference_mode():
        for (X_eval,y_eval) in dataLoader:
            X_eval = X_eval.to(device)
            y_eval = y_eval.to(device).long()
            y_pred_logits = model(X_eval)
            y_pred_eval = torch.argmax(torch.softmax(y_pred_logits,dim=1),dim=1) #forward pass for infernce mode
            loss += loss_fn(y_pred_logits,y_eval) # calcualte loss for all batch
            acc += acc_fn(y_pred_eval,y_eval) # calcualte acc for all batch
        
        #acc, loss per batch
        loss /= len(dataLoader)
        acc /= len(dataLoader)

        model_result = {"model_name": model.__class__.__name__,"model_loss":loss.item(),"model_acc":acc.item()}

    return model_result

#get hypothesis
model_list = [model_0,model_1,model_2]
#train and test loop

model_results = []
for model in model_list:
    optimizer_model = torch.optim.SGD(model.parameters(),lr=0.01)
    scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_model,patience=5)
    epochs = 3
    best_loss = float("inf")
    delta = 1e-4
    stopping_patience = 20
    cnt_stop = 0
    print(f"*********{model.__class__.__name__}********")
    for epoch in range(epochs) :
        training_step(model,train_dataLoader,loss_fn,optimizer_model,epoch,device)
        test_loss = testing_step(model,test_dataLoader,loss_fn,epoch,device)
        scheduler_model.step(test_loss)

        if test_loss <= best_loss - delta:
            best_loss = test_loss
            cnt_stop =0
            torch.save(model.state_dict(),f= f"{model.__class__.__name__}.pth" )
            #not saving model here since we are doing comparison of models at end
        else:
            cnt_stop += 1

        if cnt_stop >= stopping_patience:
            print(f"{model.__class__.__name__} has best loss at epoch:{epoch}")
            break
    
    model.load_state_dict(torch.load(f=f"{model.__class__.__name__}.pth", weights_only=True))
    model_results.append(eval_model(model, test_dataLoader,loss_fn,device))

print(f"model results:{len(model_results)},{model_results}")
modelPerformance = pd.DataFrame(model_results)
print(modelPerformance)

#save the best model with less loss
idx = modelPerformance['model_loss'].idxmin()
best_model_name = modelPerformance.at[idx,"model_name"]
model_registry = {m.__class__.__name__:m for m in model_list}
best_model = model_registry[best_model_name]

print(f"Type of model:{type(best_model)}")


#evaluate on best model
best_model.eval()
with torch.inference_mode():
    best_model_result=eval_model(best_model,test_dataLoader,loss_fn,device)
    print("best model result:",best_model_result)

#draw confusion matrix
y_preds = []
for (X,y) in test_dataLoader:
    X = X.to(device)
    y = y.to(device)

    y_pred = torch.argmax(best_model(X),dim=1)
    y_preds.append(y_pred)
y_pred_tensor = torch.cat(y_preds) # its for confusion matrix

#prepare confusion matrix
cnf = ConfusionMatrix(task="multiclass",num_classes=len(class_names))
cnf = cnf(y_pred_tensor.cpu(),test_data.targets)
fig,ax =plot_confusion_matrix(cnf.numpy(),class_names=class_names)
plt.show()



    






    










