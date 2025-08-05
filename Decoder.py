import torch
import torch.nn as nn
import lightning as L
from transformers.models.ctrl.modeling_ctrl import positional_encoding

from position_Encoding import PositionEncoding
from attention import attention

class DecoderonlyTransformer(L.LightningModule):
    def __init__(self, num_tokens,d_vector, max_len):
        super().__init__()

        self.we=nn.Embedding(num_embeddings=num_tokens,embedding_dim=d_vector)

        self.pe=PositionEncoding(d_length=d_vector,max_length_of_sentence=max_len)

        self.attention=attention(d_vector=d_vector)
        self.fc = nn.Linear(d_vector, num_tokens)  # vocab_size

        self.loss=nn.CrossEntropyLoss()


    def forward(self,toke_ids):
        word_embedings=self.we(toke_ids)
        positional_encoding=self.pe(word_embedings)

        seq_len = toke_ids.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(toke_ids.device)
        mask = mask == 0  # Convert to boolean mask (True = masked)

        attention=self.attention(positional_encoding,
                                 positional_encoding,
                                 positional_encoding,
                                 mask=mask)

        residual_connection_values=positional_encoding + attention

        fc_layer=self.fc(residual_connection_values)

        return fc_layer

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        logits = logits.view(-1, logits.size(-1))  # Flatten for cross-entropy
        y = y.view(-1)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss
