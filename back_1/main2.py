import tkinter as tk
import numpy as np
import os
import random
from numba import jit, prange
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

# --- Definirea conceptuală a stratului LSTM ---
class LSTMLayer:
    def __init__(self, hidden_dim, vocab_size, embed_dim=128, learning_rate=0.01):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate

        # Inițializare (simplificată)
        scale = np.sqrt(6.0 / (hidden_dim + embed_dim))
        self.W_in = np.random.uniform(-scale, scale, (4 * hidden_dim, embed_dim)).astype(np.float32)
        self.W_rec = np.random.uniform(-scale, scale, (4 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.W_out = np.random.uniform(-scale, scale, (vocab_size, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, sequence, h_prev=None, c_prev=None):
        T, embed_dim = sequence.shape
        hidden_dim = self.W_rec.shape[1]
        hidden_states = np.zeros((T, hidden_dim), dtype=np.float32)
        cell_states = np.zeros((T, hidden_dim), dtype=np.float32)

        if h_prev is None:
            h_prev = np.zeros(hidden_dim, dtype=np.float32)
        if c_prev is None:
            c_prev = np.zeros(hidden_dim, dtype=np.float32)

        for t in range(T):
            x_t = sequence[t]
            a = np.dot(self.W_in, x_t) + np.dot(self.W_rec, h_prev) + self.b

            i_t = sigmoid(a[:hidden_dim])
            f_t = sigmoid(a[hidden_dim:2*hidden_dim])
            g_t = np.tanh(a[2*hidden_dim:3*hidden_dim])
            o_t = sigmoid(a[3*hidden_dim:])

            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * np.tanh(c_t)

            hidden_states[t] = h_t
            cell_states[t] = c_t
            h_prev = h_t
            c_prev = c_t

        return hidden_states, cell_states

    def compute_output(self, hidden_states):
        return np.dot(hidden_states, self.W_out.T) + self.b_out

    def backward(self, embedded_seq, hidden_states, cell_states, target_seq, output_probs, next_hidden_grad, next_cell_grad):
        # Implementarea backward pentru LSTM este complexă și omisă aici.
        dW_in = np.zeros_like(self.W_in, dtype=np.float32)
        dW_rec = np.zeros_like(self.W_rec, dtype=np.float32)
        db = np.zeros_like(self.b, dtype=np.float32)
        dW_out = np.zeros_like(self.W_out, dtype=np.float32)
        db_out = np.zeros_like(self.b_out, dtype=np.float32)
        dEmbed = np.zeros_like(embedded_seq, dtype=np.float32)
        return dW_in, dW_rec, db, dW_out, db_out, dEmbed, np.zeros_like(next_hidden_grad), np.zeros_like(next_cell_grad)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RNN_LSTM_Reasoning:
    def __init__(self, model_name="distilbert-base-uncased", embed_dim=128, hidden_dim=256, lr=0.01, max_seq_length=100, knowledge_data="train2.txt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = lr
        self.max_seq_length = max_seq_length
        self.knowledge_data_path = knowledge_data
        self.knowledge_base = self._load_knowledge(knowledge_data) # Simulare bază de cunoștințe

        scale = np.sqrt(6.0 / (self.vocab_size + embed_dim))
        self.embedding = np.random.uniform(-scale, scale, (self.vocab_size, embed_dim)).astype(np.float32)

        self.lstm1 = LSTMLayer(hidden_dim=self.hidden_dim, vocab_size=self.vocab_size, embed_dim=self.embed_dim, learning_rate=self.learning_rate)

        # Pre-alocăm array-urile
        self.embedded_seq = np.zeros((self.max_seq_length, self.embed_dim), dtype=np.float32)
        self.hidden_states = np.zeros((self.max_seq_length, self.hidden_dim), dtype=np.float32)
        self.cell_states = np.zeros((self.max_seq_length, self.hidden_dim), dtype=np.float32)

        self.load_weights("lstm_reasoning_weights.npz")

    def _load_knowledge(self, file_path):
        """Simulare încărcare bază de cunoștințe din fișier."""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.readlines()
        return []

    def _prepare_sequence(self, token_ids):
        """Pregătește secvența de tokeni pentru procesare."""
        return np.array(token_ids, dtype=np.float32).reshape(-1, 1)

    def train(self, file_path, epochs=10, seq_length=40, num_batches=100):
        # Funcția de antrenament rămâne similară pentru simplitate
        if not os.path.exists(file_path):
            print(f"Eroare: Fișierul {file_path} nu există!")
            return

        file_size = os.path.getsize(file_path)
        # ... (restul codului de antrenament similar cu versiunea LSTM) ...
        self.save_weights("lstm_reasoning_weights.npz")

    def forward(self, input_seq):
        """Propagarea înainte prin rețea LSTM."""
        T = input_seq.shape[0]
        for t in range(T):
            token_id = int(input_seq[t, 0])
            self.embedded_seq[t] = self.embedding[token_id]
        hidden_states, cell_states = self.lstm1.forward(self.embedded_seq[:T])
        output = self.lstm1.compute_output(hidden_states)
        output_probs = softmax_1d_batch(output)
        return output_probs, hidden_states, cell_states, self.embedded_seq[:T]

    def backward(self, embedded_seq, hidden_states, cell_states, target_seq, output_probs, input_seq):
        # Funcția backward rămâne similară pentru simplitate
        T = embedded_seq.shape[0]
        next_hidden_grad = np.zeros(self.hidden_dim, dtype=np.float32)
        next_cell_grad = np.zeros(self.hidden_dim, dtype=np.float32)
        # ... (restul codului backward similar cu versiunea LSTM) ...

    def generate_internal_questions(self, prompt_tokens):
        """Conceptual: Generează întrebări interne pe baza input-ului."""
        # Aceasta este o implementare foarte simplistă și nu reprezintă un raționament real.
        if not prompt_tokens:
            return ["Ce este important?"]
        last_token_id = prompt_tokens[-1]
        last_token = self.tokenizer.decode([last_token_id])
        return [f"Ce legătură are {last_token} cu alte lucruri?", f"De ce este important {last_token}?"]

    def search_for_answers(self, question):
        """Conceptual: Caută răspunsuri în baza de cunoștințe (datele de antrenament)."""
        # Aceasta este o căutare simplistă bazată pe potrivirea de cuvinte.
        results = []
        keywords = question.lower().split()
        for line in self.knowledge_base:
            if all(keyword in line.lower() for keyword in keywords):
                results.append(line.strip())
        return results[:3] # Returnează primele 3 rezultate

    def reason_and_respond(self, prompt, max_len=50, temperature=0.8, top_p=0.9):
        """Conceptual: Generează text cu un simulacru de raționament."""
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_tokens = list(input_ids)

        for _ in range(max_len):
            input_seq = self._prepare_sequence(generated_tokens)
            output_probs, hidden_states, cell_states, embedded_seq = self.forward(input_seq)
            logits = output_probs[-1] / temperature
            next_token = self.nucleus_sampling(logits, p=top_p)
            generated_tokens.append(next_token)

            # --- Simulare de raționament ---
            if len(generated_tokens) > len(input_ids) and random.random() < 0.3: # O șansă aleatorie de a "raționa"
                internal_questions = self.generate_internal_questions(generated_tokens)
                print(f"\nÎntrebări interne: {internal_questions}")
                for q in internal_questions:
                    answers = self.search_for_answers(q)
                    if answers:
                        print(f"Răspunsuri găsite pentru '{q}': {answers}")
                        # Aici, un model real ar trebui să integreze aceste răspunsuri în starea sa internă
                        # Pentru simplitate, doar le afișăm.

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def run(self, text_area):
        """Integrare cu tkinter: antrenează și generează text cu simulare de raționament."""
        text = text_area.get("1.0", tk.END).strip()
        self.train("train2.txt", epochs=3, seq_length=40) # Reducem numărul de epoci pentru testare
        self.save_weights('lstm_reasoning_weights.npz')
        generated_text = self.reason_and_respond(text, max_len=20)
        print("\nText generat (LSTM cu simulare de raționament):\n", generated_text)

    def test(self, text_area):
        """Generează text cu simulare de raționament fără re-antrenare."""
        text = text_area.get("1.0", tk.END).strip()
        generated_text = self.reason_and_respond(text, max_len=20)
        print("\nText generat (LSTM cu simulare de raționament):\n", generated_text)

    def save_weights(self, file_path):
        """Salvează greutățile modelului LSTM cu raționament."""
        np.savez(file_path,
                 embedding=self.embedding,
                 W_in=self.lstm1.W_in,
                 W_rec=self.lstm1.W_rec,
                 b=self.lstm1.b,
                 W_out=self.lstm1.W_out,
                 b_out=self.lstm1.b_out)
        print(f"Greutăți LSTM cu raționament salvate în {file_path}")

    def load_weights(self, file_path):
        """Încarcă greutățile modelului LSTM cu raționament, dacă există."""
        if os.path.exists(file_path):
            data = np.load(file_path)
            self.embedding = data['embedding']
            self.lstm1.W_in = data['W_in']
            self.lstm1.W_rec = data['W_rec']
            self.lstm1.b = data['b']
            self.lstm1.W_out = data['W_out']
            self.lstm1.b_out = data['b_out']
            print(f"Greutăți LSTM cu raționament încărcate din {file_path}")

@jit(nopython=True, parallel=True)
def softmax_1d_batch(x):
    T = x.shape[0]
    vocab_size = x.shape[1]
    output = np.empty_like(x, dtype=np.float32)
    for t in prange(T):
        max_val = np.max(x[t])
        ex = np.exp(x[t] - max_val)
        output[t] = ex / np.sum(ex)
    return output

if __name__ == '__main__':
    window = tk.Tk()
    window.title("Generator de Text cu LSTM și Simulare de Raționament")

    text_area = tk.Text(window, height=10, width=50)
    text_area.pack(pady=10)
    text_area.insert(tk.END, "Scrie ceva aici...")

    model = RNN_LSTM_Reasoning()

    run_button = tk.Button(window, text="Antrenează și Generează", command=lambda: model.run(text_area))
    run_button.pack(pady=5)

    test_button = tk.Button(window, text="Generează (fără antrenare)", command=lambda: model.test(text_area))
    test_button.pack(pady=5)

    window.mainloop()