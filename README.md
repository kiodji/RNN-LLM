# RNN-LLM

This project contains simple implementations of recurrent networks for text generation.
It now exposes a single module `models.py` that provides several classes:

- `BasicRNN` – a small RNN language model.
- `LSTMReasoningModel` – a toy LSTM variant which can load a knowledge file and
  generate internal questions while producing text.
- `DynamicReasoningModel` – extends the LSTM model with a staged learning
  pipeline for grammar, semantics and cause–effect rules.

## Command line interface

Use `run.py` to train a model or generate text. The `--mode` argument selects
between the standard RNN, the reasoning LSTM model or the new dynamic mode.

```bash
# Train the basic RNN and generate text
python run.py "Hello" --mode rnn --train train.txt --epochs 5

# Generate with reasoning from an existing knowledge file
python run.py "Why" --mode reasoning

# Train in dynamic multi‑stage mode
python run.py "Salut" --mode dynamic --train train.txt --epochs 3
```

Training parameters such as `--epochs`, `--seq-length` and `--num-batches`
can be adjusted as needed.
