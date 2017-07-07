import argparse

from mycroft import __version__, train, predict


def main():
    parser = argparse.ArgumentParser(description="Text Classifier")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", description="Train a model")

    data_group = train_parser.add_argument_group("data")
    data_group.add_argument("training", help="training data")
    data_group.add_argument("--limit", metavar="N", type=int,
                            help="only train on this many sample (default use all the data)")
    data_group.add_argument("--validation", metavar="PORTION", type=float,
                            help="portion of data to use for validation (default none)")
    data_group.add_argument("--text-name", metavar="NAME", default="text",
                            help="name of the text column (default 'text')")
    data_group.add_argument("--label-name", metavar="NAME", default="label",
                            help="name of the label column (default 'label')")

    model_group = train_parser.add_argument_group("model")
    model_group.add_argument("--rnn-units", metavar="N", type=int, default=128, help="RNN units (default 128)")
    model_group.add_argument("--dropout", metavar="RATE", type=float, default=0.2, help="Dropout rate (default 0.2)")

    train_group = train_parser.add_argument_group("model")
    train_group.add_argument("--epochs", metavar="N", type=int, default=10, help="training epochs (default 10)")
    train_group.add_argument("--batch-size", metavar="N", type=int, default=56, help="batch size (default 256)")
    train_group.add_argument("--model-filename", metavar="FILENAME", help="file in which to to store the model")

    train_parser.set_defaults(
        func=lambda args: train(args.training, args.text_name, args.label_name, args.limit, args.batch_size,
                                args.epochs, args.rnn_units, args.dropout, args.validation, args.model_filename))

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", description="Use a model to predict labels")
    predict_parser.add_argument("test", help="test data")
    predict_parser.add_argument("model_filename", metavar="filename", help="file containing the trained model")
    predict_parser.add_argument("--text-name", metavar="NAME", default="text",
                                help="name of the text column (default 'text')")
    predict_parser.add_argument("--limit", metavar="N", type=int,
                                help="only predict this many samples (default use all the data)")
    predict_parser.set_defaults(func=lambda args: predict(args.test, args.model_filename, args.text_name, args.limit))

    args = parser.parse_args()
    args.func(args)
