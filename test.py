import torch
import torch.nn.functional as F
from Decoder import DecoderonlyTransformer
from vocabulary import Vocabulary

# === Load vocab ===
vocab = Vocabulary(frequency_threshold=2)
with open("DailyDialog.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
vocab.build_vocabulary(lines)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DecoderonlyTransformer(num_tokens=len(vocab), d_vector=128, max_len=128)
model.load_state_dict(torch.load("gpt_model_of_hello.pth", map_location=device))
model.to(device)
model.eval()

# === Chat function ===
def talk(prompt, max_new_tokens=30):
    input_ids = vocab.numericalize(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        if input_tensor.size(1) > 128:
            input_tensor = input_tensor[:, -128:]

        with torch.no_grad():
            logits = model(input_tensor)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_tensor = torch.cat([input_tensor, next_token], dim=1)

    output_ids = input_tensor[0].tolist()
    output_words = [vocab.itos.get(idx, "<UNK>") for idx in output_ids]
    return " ".join(output_words)

# === Interactive loop ===
print("🤖 Welcome! Talk to your GPT. Type 'exit' to quit.")
while True:
    user_input = input("\n👤 You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("🤖 GPT: Goodbye!")
        break
    reply = talk(user_input, max_new_tokens=30)
    print("🤖 GPT:", reply)
