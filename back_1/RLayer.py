from numba import jit, njit, prange
import numpy as np
import math
import random

@jit(nopython=True)
def softmax_1d(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)

@njit(parallel=True)
def forward_kernel(sequence, W_in, W_rec, b_rec):
    """
    sequence: shape (T, embed_dim)
    W_in: shape (hidden_dim, embed_dim)
    W_rec: shape (hidden_dim, hidden_dim)
    b_rec: shape (hidden_dim,)
    Return:
       hidden_states: (T, hidden_dim)
       last_hidden: (hidden_dim,) ultima stare
    """
    T, embed_dim = sequence.shape
    hidden_dim = W_in.shape[0]

    hidden_states = np.zeros((T, hidden_dim), dtype=np.float32)
    h_prev = np.zeros(hidden_dim, dtype=np.float32)

    for t in range(T):
        x_t = sequence[t]  # (embed_dim,)
        h_t = np.zeros(hidden_dim, dtype=np.float32)

        # Paralelizare peste dimensiunea hidden_dim
        for i in prange(hidden_dim):
            sum_val = b_rec[i]
            # Contribuția embeddingului
            for e in range(embed_dim):
                sum_val += W_in[i, e] * x_t[e]
            # Contribuția recurenței
            for j in range(hidden_dim):
                sum_val += W_rec[i, j] * h_prev[j]
            h_t[i] = np.tanh(sum_val)
        hidden_states[t] = h_t
        h_prev = h_t
    return hidden_states, h_prev

@njit(parallel=True)
def output_kernel(hidden_states, W_out, b_out):
    """
    hidden_states: (T, hidden_dim)
    W_out: (vocab_size, hidden_dim)
    b_out: (vocab_size,)
    Return:
      output_probs: (T, vocab_size)
    """
    T, hidden_dim = hidden_states.shape
    vocab_size = W_out.shape[0]
    output = np.zeros((T, vocab_size), dtype=np.float32)

    # Paralelizare peste T
    for t in prange(T):
        for v in range(vocab_size):
            sum_val = b_out[v]
            for h in range(hidden_dim):
                sum_val += W_out[v, h] * hidden_states[t, h]
            output[t, v] = sum_val
        # Aplicăm softmax
        ex = np.exp(output[t] - np.max(output[t]))
        output[t] = ex / np.sum(ex)
    return output

@njit(parallel=True)
def backward_kernel(embedded_seq, hidden_states, target_seq, output_probs,
                    W_in, W_rec, W_out, b_rec, b_out):
    """
    embedded_seq: (T, embed_dim)
    hidden_states: (T, hidden_dim)
    target_seq: (T, 1) - conține ID-ul țintă la fiecare pas
    output_probs: (T, vocab_size)

    W_in: (hidden_dim, embed_dim)
    W_rec: (hidden_dim, hidden_dim)
    W_out: (vocab_size, hidden_dim)
    b_rec: (hidden_dim,)
    b_out: (vocab_size,)

    Return:
      dW_in, dW_rec, dW_out, db_rec, db_out, dEmbed
    """
    T, embed_dim = embedded_seq.shape
    hidden_dim = W_in.shape[0]
    vocab_size = W_out.shape[0]

    dW_in = np.zeros_like(W_in, dtype=np.float32)
    dW_rec = np.zeros_like(W_rec, dtype=np.float32)
    dW_out = np.zeros_like(W_out, dtype=np.float32)
    db_rec = np.zeros_like(b_rec, dtype=np.float32)
    db_out = np.zeros_like(b_out, dtype=np.float32)
    dEmbed = np.zeros_like(embedded_seq, dtype=np.float32)

    dh_next = np.zeros(hidden_dim, dtype=np.float32)

    for t in range(T - 1, -1, -1):
        target_idx = int(target_seq[t, 0])
        dy = output_probs[t].copy()
        dy[target_idx] -= 1.0

        # Gradient b_out, W_out - paralelizat
        for v in prange(vocab_size):
            db_out[v] += dy[v]
            for h in range(hidden_dim):
                dW_out[v, h] += dy[v] * hidden_states[t, h]

        # Gradient dh
        dh = np.zeros(hidden_dim, dtype=np.float32)
        for v in range(vocab_size):
            for h in range(hidden_dim):
                dh[h] += dy[v] * W_out[v, h]
        dh += dh_next

        # Derivata tanh
        h_t = hidden_states[t]
        dh_raw = dh * (1.0 - h_t * h_t)

        # Gradient b_rec, W_in, W_rec - paralelizat
        for i in prange(hidden_dim):
            db_rec[i] += dh_raw[i]
            for e in range(embed_dim):
                dW_in[i, e] += dh_raw[i] * embedded_seq[t, e]
            if t > 0:
                h_prev = hidden_states[t - 1]
            else:
                h_prev = np.zeros(hidden_dim, dtype=np.float32)
            for j in range(hidden_dim):
                dW_rec[i, j] += dh_raw[i] * h_prev[j]

        # Gradient dEmbed - paralelizat
        for e in prange(embed_dim):
            for i in range(hidden_dim):
                dEmbed[t, e] += dh_raw[i] * W_in[i, e]

        # dh_next
        dh_next = np.zeros(hidden_dim, dtype=np.float32)
        if t > 0:
            for j in prange(hidden_dim):
                for i in range(hidden_dim):
                    dh_next[j] += dh_raw[i] * W_rec[i, j]

    return dW_in, dW_rec, dW_out, db_rec, db_out, dEmbed

class RLayer:
    def __init__(self, hidden_dim, vocab_size, embed_dim=128, learning_rate=0.01):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate

        # Inițializare Xavier
        scale = np.sqrt(6.0 / (hidden_dim + embed_dim))
        self.W_in = np.random.uniform(-scale, scale, (hidden_dim, embed_dim)).astype(np.float32)
        self.W_rec = np.random.uniform(-scale, scale, (hidden_dim, hidden_dim)).astype(np.float32)

        scale_out = np.sqrt(6.0 / (hidden_dim + vocab_size))
        self.W_out = np.random.uniform(-scale_out, scale_out, (vocab_size, hidden_dim)).astype(np.float32)

        self.b_rec = np.zeros(hidden_dim, dtype=np.float32)
        self.b_out = np.zeros(vocab_size, dtype=np.float32)

    def forward(self, embedded_seq):
        """
        embedded_seq: (T, embed_dim)
        Return:
          hidden_states: (T, hidden_dim)
          last_hidden: (hidden_dim,)
        """
        hidden_states, last_hidden = forward_kernel(embedded_seq, self.W_in, self.W_rec, self.b_rec)
        return hidden_states, last_hidden

    def compute_output(self, hidden_states):
        """
        Return logits softmax: (T, vocab_size)
        """
        return output_kernel(hidden_states, self.W_out, self.b_out)

    def backward(self, embedded_seq, hidden_states, target_seq, output_probs, clip_value=1.0):
        dW_in, dW_rec, dW_out, db_rec, db_out, dEmbed = backward_kernel(
            embedded_seq, hidden_states, target_seq, output_probs,
            self.W_in, self.W_rec, self.W_out,
            self.b_rec, self.b_out
        )

        # Gradient Clipping
        # np.clip(dW_in, -clip_value, clip_value, out=dW_in)
        # np.clip(dW_rec, -clip_value, clip_value, out=dW_rec)
        # np.clip(dW_out, -clip_value, clip_value, out=dW_out)
        # np.clip(db_rec, -clip_value, clip_value, out=db_rec)
        # np.clip(db_out, -clip_value, clip_value, out=db_out)
        # np.clip(dEmbed, -clip_value, clip_value, out=dEmbed)

        # Update parametri
        self.W_in -= self.learning_rate * dW_in
        self.W_rec -= self.learning_rate * dW_rec
        self.W_out -= self.learning_rate * dW_out
        self.b_rec -= self.learning_rate * db_rec
        self.b_out -= self.learning_rate * db_out

        return dEmbed