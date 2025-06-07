import tkinter as tk
import numpy as np
import os
import random
from numba import jit, prange
from RLayer import RLayer
from transformers import AutoTokenizer

@jit(nopython=True)
def softmax_1d(x):
    """Calculează softmax pentru un vector 1D."""
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

@jit(nopython=True)
def compute_loss(output_probs, target_seq):
    T = output_probs.shape[0]
    loss = 0.0
    for t in range(T):
        target_idx = int(target_seq[t, 0])
        loss -= np.log(output_probs[t, target_idx] + 1e-8)
    return loss / T

def find_next_utf8_char_start(file, pos, block_size=1024):
    """Găsește următorul punct de start valid pentru un caracter UTF-8 citind un bloc de date."""
    file.seek(pos)
    data = file.read(block_size)
    for i, byte in enumerate(data):
        byte_val = byte if isinstance(byte, int) else ord(byte)
        if byte_val < 128 or (byte_val & 0xC0) != 0x80:
            return pos + i
    return None

class RNN:
    def __init__(self, model_name="distilbert-base-uncased", embed_dim=128, hidden_dim=256, lr=0.01, max_seq_length=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = lr
        self.max_seq_length = max_seq_length  # Lungimea maximă a secvenței pentru pre-alocare

        scale = np.sqrt(6.0 / (self.vocab_size + embed_dim))
        self.embedding = np.random.uniform(-scale, scale, (self.vocab_size, embed_dim)).astype(np.float32)

        self.rnn1 = RLayer(hidden_dim=self.hidden_dim, vocab_size=self.vocab_size, embed_dim=self.embed_dim, learning_rate=self.learning_rate)

        # Pre-alocăm array-urile pentru a evita alocările repetate
        self.embedded_seq = np.zeros((self.max_seq_length, self.embed_dim), dtype=np.float32)
        self.hidden_states = np.zeros((self.max_seq_length, self.hidden_dim), dtype=np.float32)

        self.load_weights("weights.npz")

    def _prepare_sequence(self, token_ids):
        """Pregătește secvența de tokeni pentru procesare."""
        return np.array(token_ids, dtype=np.float32).reshape(-1, 1)

    def train(self, file_path, epochs=10, seq_length=40, num_batches=100):
        """
        Antrenează modelul folosind secvențe random extrase din fișier.
        """
        if not os.path.exists(file_path):
            print(f"Eroare: Fișierul {file_path} nu există!")
            return

        file_size = os.path.getsize(file_path)

        best_loss = float('inf')
        patience = 5
        no_improvement = 0

        for ep in range(epochs):
            print(f"\n=== EPOCH {ep+1}/{epochs} === (lr={self.learning_rate:.6f})")
            total_loss = 0.0
            batches = 0

            with open(file_path, 'rb') as f:
                for _ in range(num_batches):
                    random_pos = random.randint(0, file_size - seq_length * 4)
                    start_pos = find_next_utf8_char_start(f, random_pos)
                    if start_pos is None:
                        continue

                    f.seek(start_pos)
                    segment_bytes = f.read(seq_length * 4)
                    segment = segment_bytes.decode('utf-8', errors='replace')

                    token_ids = self.tokenizer.encode(segment, add_special_tokens=False)
                    if len(token_ids) < 2:
                        continue

                    input_ids = token_ids[:-1]
                    target_ids = token_ids[1:]

                    input_seq = self._prepare_sequence(input_ids)
                    target_seq = self._prepare_sequence(target_ids)

                    output_probs, hidden_states, embedded_seq = self.forward(input_seq)
                    loss = compute_loss(output_probs, target_seq)
                    total_loss += loss

                    self.backward(embedded_seq, hidden_states, target_seq, output_probs, input_seq)
                    batches += 1

            if batches > 0:
                avg_loss = total_loss / batches
                print(f"Average loss: {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= patience:
                    print("Early stopping.")
                    break
            else:
                print("Nu s-au procesat batch-uri valide.")

        self.save_weights("weights.npz")

    def forward(self, input_seq):
        """Propagarea înainte prin rețea."""
        T = input_seq.shape[0]
        for t in range(T):
            token_id = int(input_seq[t, 0])
            self.embedded_seq[t] = self.embedding[token_id]
        hidden_states, _ = self.rnn1.forward(self.embedded_seq[:T])
        output_probs = self.rnn1.compute_output(hidden_states)
        return output_probs, hidden_states, self.embedded_seq[:T]

    def backward(self, embedded_seq, hidden_states, target_seq, output_probs, input_seq):
        """Propagarea înapoi pentru actualizarea greutăților."""
        dEmbed = self.rnn1.backward(embedded_seq, hidden_states, target_seq, output_probs)

        T = embedded_seq.shape[0]
        for t in range(T):
            token_id = int(input_seq[t, 0])
            self.embedding[token_id] -= self.learning_rate * dEmbed[t]
    def nucleus_sampling(self, logits, p=0.9):
            """Implementare top-p (nucleus) sampling."""
            sorted_indices = np.argsort(logits)[::-1]
            sorted_probs = softmax_1d(logits[sorted_indices])
            cumulative_probs = np.cumsum(sorted_probs)
            mask = cumulative_probs <= p
            if not np.any(mask):  # Dacă nu există valori sub p, ia cel mai probabil token
                return sorted_indices[0]
            top_p_indices = sorted_indices[mask]
            top_p_probs = sorted_probs[mask]
            top_p_probs /= np.sum(top_p_probs)  # Renormalizare
            return np.random.choice(top_p_indices, p=top_p_probs)
    def generate_text(self, prompt, max_len=50, temperature=0.8, top_p=0.9):
        """Generează text pornind de la un prompt folosind nucleus sampling."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_tokens = list(input_ids)

        for _ in range(max_len):
            input_seq = self._prepare_sequence(generated_tokens)
            output_probs, hidden_states, embedded_seq = self.forward(input_seq)
            logits = output_probs[-1] / temperature
            next_token = self.nucleus_sampling(logits, p=top_p)
            generated_tokens.append(next_token)

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def run(self, text_area):
        """Integrare cu tkinter: antrenează și generează text."""
        text = text_area.get("1.0", tk.END).strip()
        self.train("train2.txt", epochs=10, seq_length=40)
        self.save_weights('weights.npz')
        generated_text = self.generate_text(text, max_len=20)
        print("\nText generat:\n", generated_text)

    def test(self, text_area):
        """Generează text fără re-antrenare."""
        text = text_area.get("1.0", tk.END).strip()
        generated_text = self.generate_text(text, max_len=20)
        print("\nText generat:\n", generated_text)

    def save_weights(self, file_path):
        """Salvează greutățile modelului."""
        np.savez(file_path,
                 embedding=self.embedding,
                 W_in=self.rnn1.W_in,
                 W_rec=self.rnn1.W_rec,
                 W_out=self.rnn1.W_out,
                 b_rec=self.rnn1.b_rec,
                 b_out=self.rnn1.b_out)
        print(f"Greutăți salvate în {file_path}")

    def load_weights(self, file_path):
        """Încarcă greutățile modelului, dacă există."""
        if os.path.exists(file_path):
            data = np.load(file_path)
            self.embedding = data['embedding']
            self.rnn1.W_in = data['W_in']
            self.rnn1.W_rec = data['W_rec']
            self.rnn1.W_out = data['W_out']
            self.rnn1.b_rec = data['b_rec']
            self.rnn1.b_out = data['b_out']
            print(f"Greutăți încărcate din {file_path}")