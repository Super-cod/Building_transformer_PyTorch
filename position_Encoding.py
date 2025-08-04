import torch
import torch.nn as nn
import numpy as np
class PositionEncoding(nn.Module):
    def __init__(self,d_length=10,max_length_of_sentence=10):
        super().__init__()

        pe=np.zeros( d_length , max_length_of_sentence)

        position= torch.arange(start=0,end=max_length_of_sentence,step=1).float().unsqueeze(1)
        #[[0.],[1.],[2.],[3.]]
        embedding_index=torch.arange(start=0,end=d_length,step=2).float()

        div_term=1/torch.tensor(10000.0)**(embedding_index/d_length)

        pe[:,0::2]=torch.sin(position * div_term)
        pe[:,1::2]=torch.sin(position * div_term)

        self.register_buffer("pe",pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0),:]



