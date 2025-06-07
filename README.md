# RNN-LLM

This project contains simple implementations of recurrent networks for text generation.
It now exposes a single module `models.py` that provides two classes:

- `BasicRNN` – a small RNN language model.
- `LSTMReasoningModel` – a toy LSTM variant which can load a knowledge file and
  generate internal questions while producing text.

## Command line interface

Use `run.py` to train a model or generate text. The `--mode` argument selects
between the standard RNN and the reasoning LSTM model.

```bash
# Train the basic RNN and generate text
python run.py "Hello" --mode rnn --train train.txt --epochs 5

# Generate with reasoning from an existing knowledge file
python run.py "Why" --mode reasoning
```

Training parameters such as `--epochs`, `--seq-length` and `--num-batches`
can be adjusted as needed.
