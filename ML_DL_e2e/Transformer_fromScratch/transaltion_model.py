from datasets import load_dataset
from nltk import word_tokenize
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.nn.utils.rnn import pad_sequence
from embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlockPostLN
from decoder_block import DecoderBlockPostLNcrossAtt


dataset = load_dataset("bentrevett/multi30k")

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]


# Step1 : get tokens & tokenIds for them. Replace tokens in message with to tokenIds
#english 
#get tokenIds for tokens
eng_vocab = {"<PAD>":0,"<UNK>":1,"<SOS>":2,"<EOS>":3}
eng_msg_list = []

for engSent in train_dataset:
    sent = engSent["en"]
    eng_msg = []
    tokens = word_tokenize(sent.lower())
    for token in tokens:
        if token not in eng_vocab:
            eng_vocab[token] = len(eng_vocab)
    #prepare messages with tokenIds
    for token in tokens:
        eng_msg.append(eng_vocab.get(token,1))
    eng_msg_list.append(torch.tensor(eng_msg))
max_len_eng = 128 # better to hardcode, since test data might have bigger sentences


#german
#get tokenIds for tokens
german_vocab = {"<PAD>":0,"<UNK>":1,"<SOS>":2,"<EOS>":3}
german_msg_list = []
for gerSent in train_dataset:
    sent = gerSent["de"]
    german_msg = []
    tokens = word_tokenize(sent.lower(),language="german")
    for token in tokens:
        if token not in german_vocab:
            german_vocab[token] = len(german_vocab)
    #prepare messages with tokenIds
    for token in tokens:
        german_msg.append(german_vocab.get(token,1))
    german_msg_list.append(torch.tensor(german_msg))

encoder_input = pad_sequence(eng_msg_list,batch_first=True,padding_value=eng_vocab["<PAD>"])

#prepare decoder input & target - input start with <SOS> & trget end with <EOS>
decoder_input = []
target = []
sosTensor = torch.tensor([german_vocab.get("<SOS>")])
eosTensor = torch.tensor([german_vocab.get("<EOS>")])
for germanMsg in german_msg_list:
    decoder_input.append(torch.cat((sosTensor,germanMsg),dim=0))
    target.append(torch.cat((germanMsg,eosTensor),dim=0))
    

decoder_input = pad_sequence(decoder_input,batch_first=True,padding_value=german_vocab["<PAD>"])
target = pad_sequence(target,batch_first=True,padding_value=german_vocab["<PAD>"])

train_data_tensor = TensorDataset(encoder_input,decoder_input,target)
#final train dataset
train_data = DataLoader(train_data_tensor,batch_size=32,shuffle=True)

#prepare validation dataset
eng_msg_list = []
for engSent in validation_dataset:
    sent = engSent["en"]
    eng_msg = []
    tokens = word_tokenize(sent.lower())
    #prepare messages with tokenIds
    for token in tokens:
        eng_msg.append(eng_vocab.get(token,1))
    eng_msg_list.append(torch.tensor(eng_msg))

german_msg_list = []
for gerSent in validation_dataset:
    sent = gerSent["de"]
    ger_msg = []
    tokens = word_tokenize(sent.lower(),language="german")
    #prepare messages with tokenIds
    for token in tokens:
        ger_msg.append(german_vocab.get(token,1))
    german_msg_list.append(torch.tensor(ger_msg))

#prepare decoder input & target for german
decoder_input = []
target = []
sosTensor = torch.tensor([german_vocab.get("<SOS>")])
eosTensor = torch.tensor([german_vocab.get("<EOS>")])
for gerSent in german_msg_list:
    decoder_input.append(torch.cat((sosTensor,gerSent),dim=0))
    target.append(torch.cat((gerSent,eosTensor),dim=0))

#pad the messages
encoder_input = pad_sequence(batch_first=True, sequences=eng_msg_list,padding_value=eng_vocab["<PAD>"])
decoder_input = pad_sequence(batch_first=True, sequences=decoder_input,padding_value=german_vocab["<PAD>"])
target = pad_sequence(batch_first=True, sequences=target,padding_value=german_vocab["<PAD>"])

#get the tensor
validation_set_tensor = TensorDataset(encoder_input,decoder_input,target,)
validation_set = DataLoader(validation_set_tensor,batch_size=32,shuffle=False)
    
#prepare test dataset
eng_msg_list = []
for engSent in test_dataset:
    sent = engSent["en"]
    eng_msg = []
    tokens = word_tokenize(sent.lower())
    #prepare messages with tokenIds
    for token in tokens:
        eng_msg.append(eng_vocab.get(token,1))
    eng_msg_list.append(torch.tensor(eng_msg))

german_msg_list = []

for gerSent in test_dataset:
    sent = gerSent["de"]
    ger_msg = []
    tokens = word_tokenize(sent.lower(),language="german")
    #prepare messages with tokenIds
    for token in tokens:
        ger_msg.append(german_vocab.get(token,1))
    german_msg_list.append(torch.tensor(ger_msg))

#prepare decoder input & target for german
decoder_input = []
target = []
sosTensor = torch.tensor([german_vocab.get("<SOS>")])
eosTensor = torch.tensor([german_vocab.get("<EOS>")])
for gerSent in german_msg_list:
    decoder_input.append(torch.cat((sosTensor,gerSent),dim=0))
    target.append(torch.cat((gerSent,eosTensor),dim=0))

max_len_german = 128 # hardcoding, test data might have bigger sentences
print(f"german vocabSize : {len(german_vocab)}")

#pad the messages
encoder_input = pad_sequence(batch_first=True, sequences=eng_msg_list,padding_value=eng_vocab["<PAD>"])
decoder_input = pad_sequence(batch_first=True, sequences=decoder_input,padding_value=german_vocab["<PAD>"])
target = pad_sequence(batch_first=True, sequences=target,padding_value=german_vocab["<PAD>"])

#get the tensor
test_set_tensor = TensorDataset(encoder_input,decoder_input,target,)
test_set = DataLoader(test_set_tensor,batch_size=32,shuffle=False)



class TransaltionModel(nn.Module):
    def __init__(self,eng_vocab_size,german_vocab_size,max_len_eng,max_len_ger,d_model,no_encoder,no_decoder,heads,ffnDim):
        super().__init__()
        self.embedEng = TokenEmbedding(vocab_size=eng_vocab_size,embed_dim=d_model)
        self.embedGer = TokenEmbedding(vocab_size=german_vocab_size,embed_dim=d_model)
        self.posEng = PositionalEncoding(seq_len=max_len_eng,d_model=d_model)
        self.posGer = PositionalEncoding(seq_len=max_len_ger,d_model=d_model)
        self.encoders = nn.ModuleList([EncoderBlockPostLN(d_model=d_model,heads = heads, ffnDim = ffnDim) for i in range(no_encoder)])
        self.decoders = nn.ModuleList([DecoderBlockPostLNcrossAtt(d_model=d_model,heads=heads,ffnDim=ffnDim) for i in range(no_decoder)])
        self.linearLayer = nn.Linear(in_features=d_model,out_features=german_vocab_size)
        
    
    def forward(self,encoder_input,decoder_input,combined_mask,encoder_mask):
        # encoder_input = eng sentence, decoder_input = german sentence with <SOS> , target = german sentence with <EOS>
        encoder_embed_pos = self.posEng(self.embedEng(encoder_input))
        encoder_data = encoder_embed_pos
        for encoder in self.encoders:
            encoder_data = encoder(encoder_data,encoder_mask)
        decoder_embed_pos = self.posGer(self.embedGer(decoder_input))
        decoder_data = decoder_embed_pos
        # print(f"***decoder_data:{decoder_data.shape}")
        # print(f"**encoder_data:{encoder_data.shape}")
        for decoder in self.decoders:
            decoder_data = decoder(decoder_data,encoder_data,combined_mask,encoder_mask)
        output = self.linearLayer(decoder_data)
        return output, decoder_data
    
    def translate(self, encoder_input,german_vocab):
        decoder_input = torch.tensor([[german_vocab["<SOS>"]]])
        eosTokenID = german_vocab["<EOS>"]
        next_token = None
        generated_tokens = []

        while next_token != eosTokenID and decoder_input.size(1) < max_len_german :
            # print(f"Decoder input:{decoder_input},{decoder_input.shape}") #Decoder input:tensor([[2]]),torch.Size([1, 1])
            # print(f"encoder input:{encoder_input.shape}") #encoder input:torch.Size([32, 33])
            #padding mask 
            encoder_padding_mask = (encoder_input != eng_vocab["<PAD>"]) # [32,33]
            decoder_padding_mask = (decoder_input != german_vocab["<PAD>"]) # [32,33]
            casual_mask = torch.tril(torch.ones((decoder_input.size(1),decoder_input.size(1)),dtype= torch.bool)) #[33,33]
            #combine mask
            combined_mask = decoder_padding_mask.unsqueeze(1) & casual_mask #[32,1,33]
            # print(f"combined_mask : {combined_mask[0]}, combined_mask.shape : {combined_mask.shape}")
            
            logits, decoder_data = self.forward(encoder_input,decoder_input,combined_mask,encoder_padding_mask)
            # last_logits = logits[:,-1,:]
            # topk = torch.topk(last_logits,10)
            # print(f"top k indices : {topk.indices}, tok k values : {topk.values}")
            next_token = torch.argmax(logits[:,-1,:],dim=-1)
            # print("Last hidden state:", decoder_data[0, -1, :10])
            # print("Last logits:", logits[0, -1, :10])
            if next_token.item() == eosTokenID:
                break
            generated_tokens.append(next_token)
            decoder_input = torch.cat((decoder_input,next_token.unsqueeze(1)),dim=1)

        inv_german_tokens = {v:k for k, v in german_vocab.items()}
        print(f"Generated tokens:{generated_tokens}")
        predicted_tokens = [inv_german_tokens[t.item()] for t in generated_tokens]
        return " ".join(predicted_tokens)


    
translateModel = TransaltionModel(eng_vocab_size=len(eng_vocab),german_vocab_size= len(german_vocab),max_len_eng=max_len_eng,max_len_ger=max_len_german,
                                  d_model = 512, no_encoder=5, no_decoder=5,heads=4, ffnDim=1024)

#create loss, optimizer, lr_scheduler 
loss_fn = nn.CrossEntropyLoss(ignore_index=german_vocab["<PAD>"])
optimizer = torch.optim.Adam(translateModel.parameters(),lr=0.00001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,patience=5)

#train loop
epochs = 5

for epoch in range(epochs):
    translateModel.train()
    train_loss = 0
    
    for batch,(encoder_input, decoder_input, target) in enumerate(train_data):
        # print(f"{batch} id")
        #padding mask 
            encoder_padding_mask = (encoder_input != eng_vocab["<PAD>"])
            decoder_padding_mask = (decoder_input != german_vocab["<PAD>"])
            casual_mask = torch.tril(torch.ones((decoder_input.size(1),decoder_input.size(1)),dtype= torch.bool))
            #combine mask
            combined_mask = decoder_padding_mask.unsqueeze(1) & casual_mask 

            #set grad to zero
            optimizer.zero_grad()

            #forward method
            logits,_ = translateModel(encoder_input,decoder_input,combined_mask,encoder_padding_mask)

            #calcualte loss
            loss = loss_fn(logits.permute(0,2,1),target)
            #backward prop
            loss.backward()
            #optimize the model param
            optimizer.step()

            train_loss += loss.item()
    
    train_loss /= train_data
    print(f"Train loss per bacth:{train_loss}",end=" ")

    #validation set 
    translateModel.eval()
    torch.no_grad()
    validation_loss = 0
    for batch,(encoder_input, decoder_input, target) in enumerate(validation_set):
        #padding mask 
            encoder_padding_mask = (encoder_input != eng_vocab["<PAD>"])
            decoder_padding_mask = (decoder_input != german_vocab["<PAD>"])
            casual_mask = torch.tril(torch.ones((decoder_input.size(1),decoder_input.size(1)),dtype= torch.bool))
            #combine mask
            combined_mask = decoder_padding_mask.unsqueeze(1) & casual_mask 

            #forward method
            logits,_  = translateModel(encoder_input,decoder_input,combined_mask,encoder_padding_mask)
            #calcualte loss
            loss = loss_fn(logits.permute(0,2,1),target)
            validation_loss += loss.item()
    
    validation_loss /= validation_set
    print(f"Validation loss per bacth:{validation_loss}")

torch.save(translateModel.state_dict(), "transformer_checkpoint.pt")
print("Model saved.")

translateModel.eval()
with torch.inference_mode():
    inv_german_tokens = {v:k for k, v in german_vocab.items()}
    for batch,(encoder_input, decoder_input, target) in enumerate(test_set):
        if batch==0:
            for i in range(10):
                singl_sentence = encoder_input[i].unsqueeze(0)
                print(f"***********Testing*******")
                inv_eng_vocab = {v:k for k,v in eng_vocab.items()}
                actual_english = " ".join([inv_eng_vocab.get(t.item(),"<UNK>") for t in singl_sentence[0] if t.item() != eng_vocab["<PAD>"]])
                print(f"Actual english sentence:{actual_english}")

                actual_german = " ".join([inv_german_tokens.get(t.item(),"<UNK>") for t in target[i] if t.item != german_vocab["<PAD>"]])
                print(f"Actual german message:{actual_german}")
                predicted = translateModel.translate(singl_sentence,german_vocab)
                print(f"Predicte message:{predicted}")

    


