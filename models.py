# coding: utf-8
"""Unified models for RNN text generation and LSTM reasoning."""
import os
import random
import numpy as np
from numba import jit, prange
from transformers import AutoTokenizer
from RLayer import RLayer


@jit(nopython=True)
def softmax_1d(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


def find_next_utf8_char_start(file, pos, block_size=1024):
    file.seek(pos)
    data = file.read(block_size)
    for i, byte in enumerate(data):
        byte_val = byte if isinstance(byte, int) else ord(byte)
        if byte_val < 128 or (byte_val & 0xC0) != 0x80:
            return pos + i
    return None


@jit(nopython=True)
def compute_loss(output_probs, target_seq):
    T = output_probs.shape[0]
    loss = 0.0
    for t in range(T):
        target_idx = int(target_seq[t, 0])
        loss -= np.log(output_probs[t, target_idx] + 1e-8)
    return loss / T


class BasicRNN:
    """Simple RNN language model."""

    def __init__(self, model_name="distilbert-base-uncased", embed_dim=128,
                 hidden_dim=256, lr=0.01, max_seq_length=100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = lr
        self.max_seq_length = max_seq_length

        scale = np.sqrt(6.0 / (self.vocab_size + embed_dim))
        self.embedding = np.random.uniform(-scale, scale,
                                           (self.vocab_size, embed_dim)).astype(np.float32)
        self.rnn1 = RLayer(hidden_dim=self.hidden_dim, vocab_size=self.vocab_size,
                           embedding_dim=self.embed_dim, learning_rate=self.learning_rate)

        self.embedded_seq = np.zeros((self.max_seq_length, self.embed_dim), dtype=np.float32)
        self.hidden_states = np.zeros((self.max_seq_length, self.hidden_dim), dtype=np.float32)

        self.load_weights("weights.npz")

    def _prepare_sequence(self, token_ids):
        return np.array(token_ids, dtype=np.float32).reshape(-1, 1)

    def train(self, file_path, epochs=10, seq_length=40, num_batches=100):
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
                    random_pos = random.randint(0, max(0, file_size - seq_length * 4))
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
        T = input_seq.shape[0]
        for t in range(T):
            token_id = int(input_seq[t, 0])
            self.embedded_seq[t] = self.embedding[token_id]
        hidden_states, _ = self.rnn1.forward(self.embedded_seq[:T])
        output_probs = self.rnn1.compute_output(hidden_states)
        return output_probs, hidden_states, self.embedded_seq[:T]

    def backward(self, embedded_seq, hidden_states, target_seq, output_probs, input_seq):
        dEmbed = self.rnn1.backward(embedded_seq, hidden_states, target_seq, output_probs)
        T = embedded_seq.shape[0]
        for t in range(T):
            token_id = int(input_seq[t, 0])
            self.embedding[token_id] -= self.learning_rate * dEmbed[t]

    def nucleus_sampling(self, logits, p=0.9):
        sorted_indices = np.argsort(logits)[::-1]
        sorted_probs = softmax_1d(logits[sorted_indices])
        cumulative_probs = np.cumsum(sorted_probs)
        mask = cumulative_probs <= p
        if not np.any(mask):
            return sorted_indices[0]
        top_p_indices = sorted_indices[mask]
        top_p_probs = sorted_probs[mask]
        top_p_probs /= np.sum(top_p_probs)
        return np.random.choice(top_p_indices, p=top_p_probs)

    def generate_text(self, prompt, max_len=50, temperature=0.8, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_tokens = list(input_ids)
        for _ in range(max_len):
            input_seq = self._prepare_sequence(generated_tokens)
            output_probs, _, _ = self.forward(input_seq)
            logits = output_probs[-1] / temperature
            next_token = self.nucleus_sampling(logits, p=top_p)
            generated_tokens.append(next_token)
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def save_weights(self, file_path):
        np.savez(file_path,
                 embedding=self.embedding,
                 W_in=self.rnn1.w_input,
                 W_rec=self.rnn1.w_recurrent,
                 W_out=self.rnn1.w_output,
                 b_rec=self.rnn1.bias_recurrent,
                 b_out=self.rnn1.bias_output)
        print(f"Greutăți salvate în {file_path}")

    def load_weights(self, file_path):
        if os.path.exists(file_path):
            data = np.load(file_path)
            self.embedding = data['embedding']
            self.rnn1.w_input = data['W_in']
            self.rnn1.w_recurrent = data['W_rec']
            self.rnn1.w_output = data['W_out']
            self.rnn1.bias_recurrent = data['b_rec']
            self.rnn1.bias_output = data['b_out']
            print(f"Greutăți încărcate din {file_path}")


# ---------------------------------------------------------------------------
# LSTM with simple reasoning
# ---------------------------------------------------------------------------

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LSTMLayer:
    def __init__(self, hidden_dim, vocab_size, embed_dim=128, learning_rate=0.01):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        scale = np.sqrt(6.0 / (hidden_dim + embed_dim))
        self.W_in = np.random.uniform(-scale, scale, (4 * hidden_dim, embed_dim)).astype(np.float32)
        self.W_rec = np.random.uniform(-scale, scale, (4 * hidden_dim, hidden_dim)).astype(np.float32)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        self.W_out = np.random.uniform(-scale, scale, (vocab_size, hidden_dim)).astype(np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, sequence, h_prev=None, c_prev=None):
        T, _ = sequence.shape
        hidden_states = np.zeros((T, self.hidden_dim), dtype=np.float32)
        cell_states = np.zeros((T, self.hidden_dim), dtype=np.float32)
        if h_prev is None:
            h_prev = np.zeros(self.hidden_dim, dtype=np.float32)
        if c_prev is None:
            c_prev = np.zeros(self.hidden_dim, dtype=np.float32)
        for t in range(T):
            x_t = sequence[t]
            a = np.dot(self.W_in, x_t) + np.dot(self.W_rec, h_prev) + self.b
            i_t = sigmoid(a[:self.hidden_dim])
            f_t = sigmoid(a[self.hidden_dim:2*self.hidden_dim])
            g_t = np.tanh(a[2*self.hidden_dim:3*self.hidden_dim])
            o_t = sigmoid(a[3*self.hidden_dim:])
            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * np.tanh(c_t)
            hidden_states[t] = h_t
            cell_states[t] = c_t
            h_prev = h_t
            c_prev = c_t
        return hidden_states, cell_states

    def compute_output(self, hidden_states):
        return np.dot(hidden_states, self.W_out.T) + self.b_out

    def backward(self, *args, **kwargs):
        # Placeholder - full LSTM backward not implemented
        dW_in = np.zeros_like(self.W_in, dtype=np.float32)
        dW_rec = np.zeros_like(self.W_rec, dtype=np.float32)
        db = np.zeros_like(self.b, dtype=np.float32)
        dW_out = np.zeros_like(self.W_out, dtype=np.float32)
        db_out = np.zeros_like(self.b_out, dtype=np.float32)
        dEmbed = np.zeros(args[0].shape, dtype=np.float32)
        return dW_in, dW_rec, db, dW_out, db_out, dEmbed, np.zeros(self.hidden_dim), np.zeros(self.hidden_dim)


class LSTMReasoningModel(BasicRNN):
    """LSTM model that can ask simple internal questions based on training data."""

    def __init__(self, model_name="distilbert-base-uncased", embed_dim=128,
                 hidden_dim=256, lr=0.01, max_seq_length=100, knowledge_data="train2.txt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = lr
        self.max_seq_length = max_seq_length
        self.knowledge_data_path = knowledge_data
        self.knowledge_base = self._load_knowledge(knowledge_data)
        scale = np.sqrt(6.0 / (self.vocab_size + embed_dim))
        self.embedding = np.random.uniform(-scale, scale,
                                           (self.vocab_size, embed_dim)).astype(np.float32)
        self.lstm1 = LSTMLayer(hidden_dim=self.hidden_dim, vocab_size=self.vocab_size,
                               embed_dim=self.embed_dim, learning_rate=self.learning_rate)
        self.embedded_seq = np.zeros((self.max_seq_length, self.embed_dim), dtype=np.float32)
        self.hidden_states = np.zeros((self.max_seq_length, self.hidden_dim), dtype=np.float32)
        self.cell_states = np.zeros((self.max_seq_length, self.hidden_dim), dtype=np.float32)
        self.load_weights("lstm_reasoning_weights.npz")

    def _load_knowledge(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.readlines()
        return []

    def train(self, file_path, epochs=3, seq_length=40, num_batches=100):
        # Simplified: reuse BasicRNN training loop
        super().train(file_path, epochs=epochs, seq_length=seq_length, num_batches=num_batches)
        self.save_weights("lstm_reasoning_weights.npz")

    def forward(self, input_seq):
        T = input_seq.shape[0]
        for t in range(T):
            token_id = int(input_seq[t, 0])
            self.embedded_seq[t] = self.embedding[token_id]
        hidden_states, cell_states = self.lstm1.forward(self.embedded_seq[:T])
        output = self.lstm1.compute_output(hidden_states)
        output_probs = softmax_1d_batch(output)
        return output_probs, hidden_states, cell_states, self.embedded_seq[:T]

    def backward(self, *args, **kwargs):
        # placeholder - not used for now
        pass

    def generate_internal_questions(self, prompt_tokens):
        if not prompt_tokens:
            return ["Ce este important?"]
        last_token_id = prompt_tokens[-1]
        last_token = self.tokenizer.decode([last_token_id])
        return [f"Ce legătură are {last_token} cu alte lucruri?",
                f"De ce este important {last_token}?"]

    def search_for_answers(self, question):
        results = []
        keywords = question.lower().split()
        for line in self.knowledge_base:
            if all(keyword in line.lower() for keyword in keywords):
                results.append(line.strip())
        return results[:3]

    def reason_and_respond(self, prompt, max_len=50, temperature=0.8, top_p=0.9):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        generated_tokens = list(input_ids)
        for _ in range(max_len):
            input_seq = self._prepare_sequence(generated_tokens)
            output_probs, _, _, _ = self.forward(input_seq)
            logits = output_probs[-1] / temperature
            next_token = self.nucleus_sampling(logits, p=top_p)
            generated_tokens.append(next_token)
            if len(generated_tokens) > len(input_ids) and random.random() < 0.3:
                internal_questions = self.generate_internal_questions(generated_tokens)
                for q in internal_questions:
                    answers = self.search_for_answers(q)
                    if answers:
                        print(f"Răspunsuri găsite pentru '{q}': {answers}")
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def save_weights(self, file_path):
        np.savez(file_path,
                 embedding=self.embedding,
                 W_in=self.lstm1.W_in,
                 W_rec=self.lstm1.W_rec,
                 b=self.lstm1.b,
                 W_out=self.lstm1.W_out,
                 b_out=self.lstm1.b_out)
        print(f"Greutăți LSTM cu raționament salvate în {file_path}")

    def load_weights(self, file_path):
        if os.path.exists(file_path):
            data = np.load(file_path)
            self.embedding = data['embedding']
            self.lstm1.W_in = data['W_in']
            self.lstm1.W_rec = data['W_rec']
            self.lstm1.b = data['b']
            self.lstm1.W_out = data['W_out']
            self.lstm1.b_out = data['b_out']
            print(f"Greutăți LSTM cu raționament încărcate din {file_path}")
