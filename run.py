import argparse
from models import BasicRNN, LSTMReasoningModel, DynamicReasoningModel


def main():
    parser = argparse.ArgumentParser(description="RNN/LSTM text generation")
    parser.add_argument("prompt", help="Prompt for generation")
    parser.add_argument("--train", dest="train", help="Path to training file", default=None)
    parser.add_argument("--mode", choices=["rnn", "reasoning", "dynamic"], default="rnn", help="Model type")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq-length", type=int, default=40)
    parser.add_argument("--num-batches", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "rnn":
        model = BasicRNN()
    elif args.mode == "reasoning":
        model = LSTMReasoningModel()
    else:
        model = DynamicReasoningModel()

    if args.train:
        if args.mode == "dynamic":
            model.train_dynamic(args.train, epochs=args.epochs, seq_length=args.seq_length, num_batches=args.num_batches)
        else:
            model.train(args.train, epochs=args.epochs, seq_length=args.seq_length, num_batches=args.num_batches)

    if args.mode == "rnn":
        result = model.generate_text(args.prompt, max_len=20)
    else:
        result = model.reason_and_respond(args.prompt, max_len=20)
    print(result)


if __name__ == "__main__":
    main()
