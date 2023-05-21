import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import os
import random
import gensim

def get_data(file, num=None):
    with open(file, "r", encoding="utf-8") as f:
        all_data = f.read().split("\n")

    if num is None:
        return all_data
    else:
        return all_data[:num]

def train_word2vec(file):
    with open(file, encoding="utf-8") as f:
        all_data = f.read().split("\n")
    model = gensim.models.Word2Vec(all_data, vector_size=128, window=10, min_count=1,sg=1,hs=0,workers=5)
    model.save("word2vec.vec")
    return model

class PDataset(Dataset):
    def __init__(self, all_text, word_2_vec):
        self.all_text = all_text
        self.word_2_vec =word_2_vec


    def __getitem__(self, index):
        poetry = self.all_text[index]
        emb = [self.word_2_vec.wv[i] for i in poetry]
        poetry_label = [self.word_2_vec.wv.key_to_index[i] for i in poetry]
        return torch.tensor(emb),torch.tensor(poetry_label)

    def __len__(self):
        return len(self.all_text)

def auto_generate_p():
    global model, word2vec, device,hidden_num
    result = []
    letter = random.choice(word2vec.wv.index_to_key)
    while letter=="，" or letter=="。":
        letter = random.choice(random.choice(word2vec.wv.index_to_key))
    result.append(letter)


    h_0 = torch.zeros(1, hidden_num).reshape(1, 1, hidden_num).to(device)
    for i in range(23):
        letter_emb = torch.tensor([[word2vec.wv[letter]]]).to(device)
        letter_idx, h_0 = model.forward(letter_emb, h_0=h_0)
        letter = word2vec.wv.index_to_key[letter_idx]
        result.append(letter)
    return "".join(result)

class Pmodel(nn.Module):
    def __init__(self, corpus_len, embedding_num, hidden_num):
        super().__init__()
        self.rnn = nn.RNN(embedding_num, hidden_num,batch_first=True,num_layers=1,bidirectional=False)
        self.V = nn.Linear(hidden_num, corpus_len)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,x,label=None, h_0 = None):

        rnn_out1, rnn_out2 = self.rnn.forward(x, h_0)
        pre = self.V.forward(rnn_out1)
        if label is not None:
            pre = pre.reshape(x.shape[0] * x.shape[1] , -1)
            loss = self.loss_fun(pre, label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1), torch.mean(rnn_out2, dim=0, keepdim=True)

if __name__=="__main__":

    all_data = get_data(os.path.join("..","..","data",'古诗生成',"poetry_5.txt"), 3000)
    word2vec = gensim.models.Word2Vec.load("word2vec.vec")
    batch_size = 10
    epoch = 100
    lr = 0.002
    embedding_num, hidden_num = word2vec.vector_size, 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataset = PDataset(all_data,word2vec)
    train_dataloader = DataLoader(train_dataset, batch_size)

    model = Pmodel(len(word2vec.wv.index_to_key), embedding_num, hidden_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)


    for e in range(epoch):
        print("*" * 100)
        for batch_x,batch_y in train_dataloader:
            batch_index_x = batch_x[:,:-1].to(device)
            batch_index_y = batch_y[:,1:].to(device)

            loss = model.forward(batch_index_x, batch_index_y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")
        p = auto_generate_p()
        print(p)