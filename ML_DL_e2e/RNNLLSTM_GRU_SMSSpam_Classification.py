import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
import pandas as pd
from nltk import word_tokenize
from torchmetrics import F1Score
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

file_path = r"/Users/sanvijanvi/Library/Mobile Documents/com~apple~CloudDocs/ML_DL_Projects/NLP/SMSSpamCollection.txt"

spamMessages = pd.read_csv(file_path,sep="\t",names=["labels","message"])

X = spamMessages["message"]
y = spamMessages["labels"]

device = "cuda" if torch.cuda.is_available() else "cpu"

#split data
X_train, X_test , y_train , y_test = train_test_split(X,y,test_size=0.3,shuffle=True,random_state=42,stratify=y)

print("****************",pd.Series(y_train).value_counts())#imablance dat set, so dont use accuarcy

#encode the labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
# print("Encoder classes:",encoder.classes_)
y_test_encoded = encoder.transform(y_test)

#get the vocabulary
all_tokens = []

torch.manual_seed(42)
# get all tokens
for msg in X_train:
    tokens = word_tokenize(msg.lower())
    all_tokens.extend(tokens)
#get vocabulary
vocab = {"<PAD>":0,"<UNK>":1}
for token in set(all_tokens):
    if token not in vocab:
        vocab[token] = len(vocab)

msg_seq_list = []
for msg in X_train:
    msg_seq = []
    tokens = word_tokenize(msg.lower())
    for token in tokens:
        msg_seq.append(vocab.get(token,1))
    msg_seq_list.append(torch.tensor(msg_seq))

#get ids for test data, vocabulary should be ready with only train data
msg_seq_list_test = []
for msg in X_test:
    msg_seq = []
    tokens = word_tokenize(msg.lower())
    for token in tokens:
        msg_seq.append(vocab.get(token,1))
    msg_seq_list_test.append(torch.tensor(msg_seq))

#train_labels
train_labels = torch.tensor(y_train_encoded,dtype=torch.float32)
#test_labels
test_labels = torch.tensor(y_test_encoded,dtype=torch.float32)
# test_labels = test_labels.to(device)

#padding sequence
train_padded = pad_sequence(msg_seq_list,batch_first=True)  # torch.Size([3900, 216])
# train_padded = train_padded.to(device)
test_padded = pad_sequence(msg_seq_list_test,batch_first=True)  # torch.Size([3900, 216])
# test_padded = test_padded.to(device)

# tensor dataset
train_set = TensorDataset(train_padded,train_labels)
test_set = TensorDataset(test_padded,test_labels)
#dataloader
train_set = DataLoader(train_set,batch_size=32,shuffle = True)
test_set = DataLoader(test_set,batch_size=32,shuffle = False)


#create model
class rnn_classification(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=embed_dim,padding_idx=0)# [3900,216,128]
        self.drop = nn.Dropout(0.3)
        # self.rnn = nn.LSTM(batch_first=True,input_size=embed_dim,hidden_size=hidden_dim,num_layers=1)#h_final = [1,3900,3]
        self.rnn = nn.GRU(batch_first=True,input_size=embed_dim,hidden_size=hidden_dim,num_layers=1)
        # Linear input must be hidden_dim * 2
        #if bidirectional = True, self.linear = nn.Linear(hidden_dim * 2, 1)
        self.linear = nn.Linear(in_features=hidden_dim,out_features=1) #[1]

    def forward(self,X):
        # print("X shape:",X.shape)
        embed = self.drop(self.embed(X))
        # print("embed shape:",embed.shape)
          # compute actual lengths (non-padding tokens)
        lengths = (X != 0).sum(dim=1).cpu()
        packed = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        # out,(h_final,c_final) = self.rnn(packed)
        out, h_final = self.rnn(packed)
        # print("h_final std:", h_final[-1].std().item())
        # if bidirectional = true, h_final shape: (num_layers * 2, batch, hidden_dim)
        # For single layer:
        #   h_final[0] = forward direction final hidden state
        #   h_final[1] = backward direction final hidden state  
        # last = torch.cat([h_final[0], h_final[1]], dim=1)  # (batch, hidden_dim * 2)
        # return self.linear(last)
        linear_op= self.linear(h_final[-1])
        return linear_op
    
rnn_model = rnn_classification(vocab_size=len(vocab),embed_dim=32,hidden_dim=32).to(device)

#define pos_weight due to imbalnced data set for train
spam_cnt = (train_labels == 1).sum()
ham_cnt = (train_labels == 0).sum()
pos_weight = torch.tensor([ham_cnt/spam_cnt],dtype=torch.float32,device=device) #Missing a spam message is about 6.5 times more costly than missing a ham message.

#define loss, optimizer and lr scheduler
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  #sigmid with loss
optimizer = torch.optim.Adam(rnn_model.parameters(),lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5)
acc_fn = F1Score(task="binary").to(device)

def train_step(model,trainData, loss_fn, optimizer, acc_fn, epoch, device = device):
     train_loss, train_acc = 0, 0

     acc_fn.reset()
     model.train()
     for batchIdx, (msg,y_label) in enumerate(trainData):
        #  print("Message:",msg[:3])
         optimizer.zero_grad() # set grad to zero, before backward prop starts
         msg = msg.to(device)
         y_label = y_label.to(device)
         y_logit = model(msg).squeeze(1) # get logits, forward pass
         y_prob = torch.sigmoid(y_logit)
         y_pred = torch.round(y_prob) #get prediction, sigmoid converts logits to probabilities
        #  print(y_pred)
         loss = loss_fn(y_logit,y_label)
         
         loss.backward() # backward propagation
         torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
         optimizer.step() # update model parameters

         train_loss += loss # add loss for each loop
         acc_fn.update(y_pred,y_label)
     
    #  print("logits:", y_logit[:5])
    #  print("probs :", y_prob[:5])
     #get loss per batch
     train_loss /= len(trainData)
     train_acc = acc_fn.compute()

     print(f"At epoch {epoch}, train_loss = {train_loss.item():.4f}, train_acc = {train_acc.item():.4f}",end=" ")

def test_step(model,testdata, loss_fn, acc_fn, epoch, device = device):
    test_loss, test_acc = 0, 0
    acc_fn.reset()
    model.eval()
    with torch.no_grad():
        for (msg,y_label) in testdata:
            msg = msg.to(device)
            y_label = y_label.to(device)
            y_logit = model(msg).squeeze(1) # get logits, forward pass
            y_pred = torch.round(torch.sigmoid(y_logit)) #get prediction, sigmoid converts logits to probabilities
            
            loss = loss_fn(y_logit,y_label)
            test_loss += loss # add loss for each loop
            acc_fn.update(y_pred,y_label)
     
     #get loss per batch
    test_loss /= len(testdata)
    test_acc = acc_fn.compute()

    print(f"test_loss = {test_loss.item():.4f}, test_acc = {test_acc.item():.4f}")
    return test_loss

def evaluate_step(model,data, loss_fn, acc_fn, device = device):
    eval_loss, eval_acc = 0, 0
    y_preds_evaluate = []
    acc_fn.reset()
    model.eval()
    with torch.no_grad():
        for (msg,y_label) in data:
            msg = msg.to(device)
            y_label = y_label.to(device)
            y_logit = model(msg).squeeze(1) # get logits, forward pass
            y_pred = torch.round(torch.sigmoid(y_logit)) #get prediction, sigmoid converts logits to probabilities
            
            eval_loss += loss_fn(y_logit,y_label) # add loss for each loop
            acc_fn.update(y_pred,y_label)

            y_preds_evaluate.append(y_pred)
     
     #get loss per batch
    eval_loss /= len(data)
    eval_acc =  acc_fn.compute()

    model_results = {"model_name":model.__class__.__name__,"model_loss":eval_loss.item(),"model_acc":eval_acc.item()}
    return model_results, y_preds_evaluate


#train and test loop
epochs = 50

best_loss = float("inf")
delta = 1e-4
cnt = 0
patience_cnt = 20

for epoch in range(epochs):
    #train data
    train_step(rnn_model,train_set,loss_fn,optimizer,acc_fn,epoch, device)
    #test data
    test_loss = test_step(rnn_model,test_set,loss_fn,acc_fn,epoch,device)
    scheduler.step(test_loss)

    #early stopping 
    if test_loss <= best_loss - delta:
        best_loss = test_loss
        cnt = 0
        #still loss is decresing save it
        torch.save(rnn_model.state_dict(),f="rnn_model.pth")
    else:
        cnt += 1

    if cnt >= patience_cnt:
        print(f"Model stopped improving at epoch {epoch}")
        break


#evaluate the model
rnn_model.load_state_dict(torch.load("rnn_model.pth",weights_only=True))
rnn_model.eval()
with torch.inference_mode():
  model_result, y_preds_evaluate = evaluate_step(model=rnn_model,data=test_set,loss_fn=loss_fn,acc_fn=acc_fn,device=device)
  print(f"Model result after evaluation:{model_result}")

y_pred_tensor = torch.cat(y_preds_evaluate)
cnf = ConfusionMatrix(task="binary",num_classes=2)
cnf = cnf(y_pred_tensor.cpu(),test_labels)
print("cnf:",cnf)
fig, ax = plot_confusion_matrix(cnf.numpy(),class_names=["ham","spam"])
plt.show()























