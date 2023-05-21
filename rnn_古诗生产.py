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

def build_data(train_data):
    word_2_index = {"UNK": 0}
    for lines in train_data:
        for word in lines:
            if word not in word_2_index:
                word_2_index[word] = len(word_2_index)
    return word_2_index

def train_word2vec(file):
    with open(file, encoding="utf-8") as f:
        all_data = f.read().split("\n")
    model = gensim.models.Word2Vec(all_data, vector_size=128, window=10, min_count=1,sg=1,hs=0,workers=5)
    model.save("word2vec.vec")


class PDataset(Dataset):
    def __init__(self, all_text, word_2_index):
        self.all_text = all_text
        self.word_2_index = word_2_index

    def __getitem__(self, index):
        poetry = self.all_text[index]
        poetry_index = [word_2_index.get(i, 0) for i in poetry]
        return torch.tensor(poetry_index)

    def __len__(self):
        return len(self.all_text)

def auto_generate_p():
    global model, word_2_index, index_2_word, device,hidden_num
    result = []
    letter = random.choice(index_2_word)
    while letter=="，" or letter=="。":
        letter = random.choice(index_2_word)
    result.append(letter)
    letter_idx = word_2_index[letter]
    letter_idx = torch.tensor([[letter_idx]]).to(device)
    h_0 = torch.zeros(1, hidden_num).reshape(1, 1, hidden_num).to(device)
    for i in range(23):
        letter_idx, h_0 = model.forward(letter_idx, h_0=h_0)
        letter = index_2_word[letter_idx]
        result.append(letter)
    return "".join(result)

class Pmodel(nn.Module):
    def __init__(self, corpus_len, embedding_num, hidden_num):
        super().__init__()
        self.corpus_len = corpus_len
        self.embedding = nn.Embedding(corpus_len, embedding_num)
        self.rnn = nn.RNN(embedding_num, hidden_num,batch_first=True,num_layers=1,bidirectional=False)
        self.V = nn.Linear(hidden_num, corpus_len)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,x,label=None, h_0 = None):
        batch_emb = self.embedding.forward(x)
        rnn_out1, rnn_out2 = self.rnn.forward(batch_emb, h_0)
        pre = self.V.forward(rnn_out1)
        if label is not None:
            pre = pre.reshape(-1 , self.corpus_len)
            loss = self.loss_fun(pre, label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1), torch.mean(rnn_out2, dim=0, keepdim=True)

if __name__=="__main__":
    all_data = get_data(os.path.join("..","..","data",'古诗生成',"poetry_5.txt"))
    word_2_index = build_data(all_data)
    index_2_word = list(word_2_index.keys())

    batch_size = 10
    epoch = 100
    lr = 0.002
    embedding_num, hidden_num = 200, 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataset = PDataset(all_data,word_2_index)
    train_dataloader = DataLoader(train_dataset, batch_size)

    model = Pmodel(len(word_2_index), embedding_num, hidden_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)


    for e in range(epoch):
        print("*" * 100)
        for batch_index in train_dataloader:
            batch_index_x = batch_index[:,:-1].to(device)
            batch_index_y = batch_index[:,1:].to(device)

            loss = model.forward(batch_index_x, batch_index_y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        print(f"loss:{loss:.3f}")
        p = auto_generate_p()
        print(p)