import torch
from torch.utils.data import DataLoader
from vocabulary import Vocabulary
from dataset import WordLevelDataset
from Decoder import DecoderonlyTransformer
import lightning as L

block_size = 128
batch_size = 32
frequency_threshold = 2
max_len = block_size
d_model = 128
vocab_file = "DailyDialog.txt"

# --- Step 1: Build Vocab ---
with open(vocab_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

vocab = Vocabulary(frequency_threshold=frequency_threshold)
vocab.build_vocabulary(lines)
print(f"✅ Vocabulary built. Vocab size: {len(vocab)}")

# --- Step 2: Dataset + Dataloader ---
dataset = WordLevelDataset(vocab_file, vocab, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


model = DecoderonlyTransformer(
    num_tokens=len(vocab),
    d_vector=d_model,
    max_len=max_len
)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    precision=16,
    log_every_n_steps=14
)

trainer.fit(model, dataloader)


# Save model weights
torch.save(model.state_dict(), "gpt_model_of_hello.pth")
print("✅ Model saved to gpt_model.pth")
