import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
import tkinter as tk
import random

# Setăm dispozitivul
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definim modelul Transformer
class ReasoningTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, ff_hidden_dim=1024, num_layers=3):
        super(ReasoningTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, embed_dim))
        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=ff_hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, target_ids=None, mask=None):
        B, T = input_ids.shape
        src = self.token_embedding(input_ids) + self.positional_encoding[:, :T, :]
        src = self.dropout(src)

        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        if target_ids is not None:
            tgt = self.token_embedding(target_ids) + self.positional_encoding[:, :T, :]
            tgt = self.dropout(tgt)
            output = self.transformer(src, tgt, src_mask=mask, tgt_mask=mask)
        else:
            output = self.transformer(src, src, src_mask=mask)

        logits = self.output_layer(output)
        return logits

    def generate(self, tokenizer, prompt, max_len=50, temperature=0.7):
        self.eval()
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_len):
                logits = self(generated)[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        return tokenizer.decode(generated[0], skip_special_tokens=True)

# Încărcăm tokenizer-ul
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Funcție pentru a masca tokeni aleatoriu (MLM)
def mask_tokens(input_ids, tokenizer, mask_prob=0.15):
    masked_ids = input_ids.clone()
    labels = input_ids.clone()
    mask = torch.rand(input_ids.shape) < mask_prob
    mask[input_ids == tokenizer.pad_token_id] = False
    mask[input_ids == tokenizer.cls_token_id] = False
    mask[input_ids == tokenizer.sep_token_id] = False
    masked_ids[mask] = tokenizer.mask_token_id
    labels[~mask] = -100  # Ignorăm tokenii nemascați în pierdere
    return masked_ids, labels

# Citim și tokenizăm textul
def load_and_tokenize(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    tokenized_text = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=5000)
    return tokenized_text

# Funcția de antrenare cu MLM și NTP
def train(model, tokenized_text, epochs=10, seq_length=64, batch_size=4):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    dataset = tokenized_text.squeeze(0)
    for ep in range(epochs):
        total_loss = 0
        num_batches = 0

        for i in range(0, dataset.size(0) - seq_length, seq_length * batch_size):
            batch_inputs = []
            batch_targets = []
            batch_masked = []
            batch_labels = []

            for b in range(batch_size):
                start = i + b * seq_length
                if start + seq_length + 1 >= dataset.size(0):
                    break
                input_seq = dataset[start:start + seq_length]
                target_seq = dataset[start + 1:start + seq_length + 1]
                masked_seq, labels = mask_tokens(input_seq, tokenizer)
                batch_inputs.append(input_seq)
                batch_targets.append(target_seq)
                batch_masked.append(masked_seq)
                batch_labels.append(labels)

            if not batch_inputs:
                break

            input_batch = torch.stack(batch_inputs).to(device)
            target_batch = torch.stack(batch_targets).to(device)
            masked_batch = torch.stack(batch_masked).to(device)
            label_batch = torch.stack(batch_labels).to(device)

            optimizer.zero_grad()

            # Pierdere pentru MLM
            logits_mlm = model(masked_batch)
            loss_mlm = criterion(logits_mlm.view(-1, model.vocab_size), label_batch.view(-1))

            # Pierdere pentru NTP
            logits_ntp = model(input_batch)
            loss_ntp = criterion(logits_ntp.view(-1, model.vocab_size), target_batch.view(-1))

            # Combinăm pierderile
            loss = 0.5 * loss_mlm + 0.5 * loss_ntp
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoca {ep+1}/{epochs}, Pierdere medie: {avg_loss:.4f}")
            sample = model.generate(tokenizer, "Plouă afară, deci", max_len=20)
            print(f"Exemplu generat: {sample}")

# Inițializăm modelul
model = ReasoningTransformer(vocab_size=tokenizer.vocab_size).to(device)
tokenized_text = load_and_tokenize("train.txt")

# Interfața Tkinter
st = model

def runButton():
    train(st, tokenized_text, epochs=10, seq_length=64)
    user_text = text_area.get("1.0", tk.END).strip()
    generated_text = st.generate(tokenizer, user_text, max_len=20)
    print("\nText generat:\n", generated_text)
    text_area.delete("1.0", tk.END)
    text_area.insert(tk.END, generated_text)

def testButton():
    user_text = text_area.get("1.0", tk.END).strip()
    generated_text = st.generate(tokenizer, user_text, max_len=20)
    print("\nText generat:\n", generated_text)
    text_area.delete("1.0", tk.END)
    text_area.insert(tk.END, generated_text)

root = tk.Tk()
root.title("Model Transformer cu Reasoning")
root.geometry("800x600")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

text_area = tk.Text(root, height=10, width=80)
text_area.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
text_area.insert(tk.END, "Plouă afară, deci")

button = tk.Button(root, text="Run (Train + Generate)", command=runButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=5)

button = tk.Button(root, text="Test (Generate)", command=testButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=5)

scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_x = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=tk.TOP, fill=tk.X)

canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=second_frame, anchor="nw")

root.mainloop()