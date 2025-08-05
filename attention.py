import torch
import torch.nn as nn
import torch.nn.functional as F


class attention(nn.Module):
    def __init__(self, d_vector):
        super().__init__()

        self.W_q = nn.Linear(in_features=d_vector,out_features=d_vector,bias=False)
        self.W_k = nn.Linear(in_features=d_vector, out_features=d_vector, bias=False)
        self.W_v = nn.Linear(in_features=d_vector, out_features=d_vector, bias=False)

        self.row_dim =0
        self.col_dim =1

    def forward(self, encoding_for_q,encoding_for_k,encoding_for_v,mask=None):
        q=self.W_q(encoding_for_q)
        k=self.W_k(encoding_for_k)
        v = self.W_v(encoding_for_v)

        sims=torch.matmul(q, k.transpose(dim0=-2,dim1=-1))
        scaled_sim=sims/torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sim=scaled_sim.masked_fill(mask=mask,value=-1e4)

        attention_percents=F.softmax(scaled_sim,dim=self.col_dim)

        attention_percents=torch.matmul(attention_percents,v)

        return attention_percents

