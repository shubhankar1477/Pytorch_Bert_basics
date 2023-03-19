import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self,embedding_matrix) -> None:
        super(LSTM,self).__init__()
        ## Define your embedding layer
        num_words=embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(num_embeddings=num_words,embedding_dim=embed_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        ## define LSTM layer
        self.lstm = nn.LSTM(embed_dim,128,bidirectional = True , batch_first = True)
        self.out = nn.Linear(512,1)

    def forward(self,x):
        ## Inpput will be tokens for embedding layer
        x = self.embedding(x)
        # pass it through LSTM
        x,_ = self.lstm(x)
        avg_pool = torch.mean(x,1)
        max_pool,_ = torch.max(x,1)
        ## concat mean and max pooling
        out = torch.cat((avg_pool,max_pool))
        out = self.out(out)
        return out

