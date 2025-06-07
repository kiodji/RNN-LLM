from numba import jit, njit, prange
import numpy as np

# -------------------------------------------------
#  Funcții ajutătoare
# -------------------------------------------------
@jit(nopython=True)
def softmax_1d(x):
    """Softmax pentru un vector 1-D."""
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)


# -------------------------------------------------
#  Kernel de propagare înainte (forward)
# -------------------------------------------------
@njit(parallel=True)
def forward_kernel(
    embedded_sequence,               # (T, embedding_dim)
    w_input,                         # (hidden_dim, embedding_dim)
    w_recurrent,                     # (hidden_dim, hidden_dim)
    bias_recurrent                   # (hidden_dim,)
):
    """
    Returnează:
        hidden_states : (T, hidden_dim)
        last_hidden   : (hidden_dim,)  – ultima stare ascunsă
    """
    sequence_length, embedding_dim = embedded_sequence.shape
    hidden_dim = w_input.shape[0]

    hidden_states = np.zeros((sequence_length, hidden_dim), dtype=np.float32)
    prev_hidden   = np.zeros(hidden_dim, dtype=np.float32)     # h_{t-1}

    for t in range(sequence_length):
        x_t = embedded_sequence[t]                              # (embedding_dim,)
        curr_hidden = np.zeros(hidden_dim, dtype=np.float32)    # h_t (temporar)

        # → paralelizăm pe dimensiunea ascunsă
        for i in prange(hidden_dim):
            total = bias_recurrent[i]

            # contribuția de la input embedding
            for e in range(embedding_dim):
                total += w_input[i, e] * x_t[e]

            # contribuția recurentă
            for j in range(hidden_dim):
                total += w_recurrent[i, j] * prev_hidden[j]

            curr_hidden[i] = np.tanh(total)

        hidden_states[t] = curr_hidden
        prev_hidden = curr_hidden

    return hidden_states, prev_hidden


# -------------------------------------------------
#  Kernel de ieșire (output)
# -------------------------------------------------
@njit(parallel=True)
def output_kernel(
    hidden_states,                   # (T, hidden_dim)
    w_output,                        # (vocab_size, hidden_dim)
    bias_output                      # (vocab_size,)
):
    """
    Returnează probabilități softmax:
        output_probs : (T, vocab_size)
    """
    sequence_length, hidden_dim = hidden_states.shape
    vocab_size = w_output.shape[0]
    output_probs = np.zeros((sequence_length, vocab_size), dtype=np.float32)

    # paralelizăm pe timp (t)
    for t in prange(sequence_length):
        for v in range(vocab_size):
            total = bias_output[v]
            for h in range(hidden_dim):
                total += w_output[v, h] * hidden_states[t, h]
            output_probs[t, v] = total

        # aplicăm softmax pe linia curentă
        ex = np.exp(output_probs[t] - np.max(output_probs[t]))
        output_probs[t] = ex / np.sum(ex)

    return output_probs


# -------------------------------------------------
#  Kernel de back-propagation
# -------------------------------------------------
@njit(parallel=True)
def backward_kernel(
    embedded_sequence,               # (T, embedding_dim)
    hidden_states,                   # (T, hidden_dim)
    target_sequence,                 # (T, 1)
    output_probs,                    # (T, vocab_size)
    w_input, w_recurrent, w_output,
    bias_recurrent, bias_output
):
    """
    Returnează gradientele:
        dw_input, dw_recurrent, dw_output,
        dbias_recurrent, dbias_output, d_embedded
    """
    sequence_length, embedding_dim = embedded_sequence.shape
    hidden_dim = w_input.shape[0]
    vocab_size = w_output.shape[0]

    dw_input        = np.zeros_like(w_input,        dtype=np.float32)
    dw_recurrent    = np.zeros_like(w_recurrent,    dtype=np.float32)
    dw_output       = np.zeros_like(w_output,       dtype=np.float32)
    dbias_recurrent = np.zeros_like(bias_recurrent, dtype=np.float32)
    dbias_output    = np.zeros_like(bias_output,    dtype=np.float32)
    d_embedded      = np.zeros_like(embedded_sequence, dtype=np.float32)

    dh_next = np.zeros(hidden_dim, dtype=np.float32)  # pentru legătura pe timp

    for t in range(sequence_length - 1, -1, -1):
        target_idx = int(target_sequence[t, 0])
        dy = output_probs[t].copy()
        dy[target_idx] -= 1.0  # ∇L/∇logits

        # --- gradient w_output & bias_output
        for v in prange(vocab_size):
            dbias_output[v] += dy[v]
            for h in range(hidden_dim):
                dw_output[v, h] += dy[v] * hidden_states[t, h]

        # --- back-prop spre stare ascunsă
        dh = np.zeros(hidden_dim, dtype=np.float32)
        for v in range(vocab_size):
            for h in range(hidden_dim):
                dh[h] += dy[v] * w_output[v, h]
        dh += dh_next

        # derivata tanh
        h_t = hidden_states[t]
        dh_raw = dh * (1.0 - h_t * h_t)

        # --- gradient w_input, w_recurrent & bias_recurrent
        for i in prange(hidden_dim):
            dbias_recurrent[i] += dh_raw[i]
            # input
            for e in range(embedding_dim):
                dw_input[i, e] += dh_raw[i] * embedded_sequence[t, e]
            # recurrent
            prev_hidden = hidden_states[t - 1] if t > 0 else np.zeros(hidden_dim, dtype=np.float32)
            for j in range(hidden_dim):
                dw_recurrent[i, j] += dh_raw[i] * prev_hidden[j]

        # --- gradient față de embedded_sequence
        for e in prange(embedding_dim):
            for i in range(hidden_dim):
                d_embedded[t, e] += dh_raw[i] * w_input[i, e]

        # pregătim dh_next pentru pasul anterior
        dh_next = np.zeros(hidden_dim, dtype=np.float32)
        if t > 0:
            for j in prange(hidden_dim):
                for i in range(hidden_dim):
                    dh_next[j] += dh_raw[i] * w_recurrent[i, j]

    return (
        dw_input, dw_recurrent, dw_output,
        dbias_recurrent, dbias_output, d_embedded
    )


# -------------------------------------------------
#  Clasa RLayer
# -------------------------------------------------
class RLayer:
    def __init__(
        self,
        hidden_dim,
        vocab_size,
        embedding_dim=128,
        learning_rate=0.01
    ):
        self.hidden_dim     = hidden_dim
        self.vocab_size     = vocab_size
        self.embedding_dim  = embedding_dim
        self.learning_rate  = learning_rate

        # Inițializare Xavier/Glorot
        scale_in  = np.sqrt(6.0 / (hidden_dim + embedding_dim))
        scale_out = np.sqrt(6.0 / (hidden_dim + vocab_size))

        self.w_input      = np.random.uniform(-scale_in,  scale_in, (hidden_dim, embedding_dim)).astype(np.float32)
        self.w_recurrent  = np.random.uniform(-scale_in,  scale_in, (hidden_dim, hidden_dim)).astype(np.float32)
        self.w_output     = np.random.uniform(-scale_out, scale_out, (vocab_size, hidden_dim)).astype(np.float32)

        self.bias_recurrent = np.zeros(hidden_dim, dtype=np.float32)
        self.bias_output    = np.zeros(vocab_size, dtype=np.float32)

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, embedded_sequence):
        """
        Returnează:
            hidden_states : (T, hidden_dim)
            last_hidden   : (hidden_dim,)
        """
        return forward_kernel(
            embedded_sequence,
            self.w_input,
            self.w_recurrent,
            self.bias_recurrent
        )

    # -----------------------------
    # Output
    # -----------------------------
    def compute_output(self, hidden_states):
        """Logits softmax (T, vocab_size)."""
        return output_kernel(
            hidden_states,
            self.w_output,
            self.bias_output
        )

    # -----------------------------
    # Backward + update
    # -----------------------------
    def backward(
        self,
        embedded_sequence,
        hidden_states,
        target_sequence,
        output_probs,
        clip_value=1.0
    ):
        (
            dw_input, dw_recurrent, dw_output,
            dbias_recurrent, dbias_output, d_embedded
        ) = backward_kernel(
            embedded_sequence,
            hidden_states,
            target_sequence,
            output_probs,
            self.w_input, self.w_recurrent, self.w_output,
            self.bias_recurrent, self.bias_output
        )

        # Clipping (stabilitate numerică)
        for grad in (dw_input, dw_recurrent, dw_output,
                     dbias_recurrent, dbias_output, d_embedded):
            np.clip(grad, -clip_value, clip_value, out=grad)

        # Actualizare parametri
        self.w_input      -= self.learning_rate * dw_input
        self.w_recurrent  -= self.learning_rate * dw_recurrent
        self.w_output     -= self.learning_rate * dw_output
        self.bias_recurrent -= self.learning_rate * dbias_recurrent
        self.bias_output    -= self.learning_rate * dbias_output

        return d_embedded
