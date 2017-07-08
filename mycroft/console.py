import argparse
import textwrap

from mycroft import __version__, train, predict, evaluate


def main():
    parser = argparse.ArgumentParser(description="Text Classifier")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", description=textwrap.dedent("""
        Train a model to predict categorical labels from English text. The training data is a comma- or tab-delimited 
        file with column of text and a column of labels. The training and optional validation loss and accuracy are
        printed for each epoch."""))

    data_group = train_parser.add_argument_group("data", description="Arguments for specifying the data to train on")
    data_group.add_argument("training", help="training data")
    data_group.add_argument("--limit", metavar="N", type=int,
                            help="only train on this many samples (default use all the data)")
    data_group.add_argument("--validation", metavar="PORTION", type=float,
                            help="portion of data to use for validation (default none)")
    data_group.add_argument("--text-name", metavar="NAME", default="text",
                            help="name of the text column (default 'text')")
    data_group.add_argument("--label-name", metavar="NAME", default="label",
                            help="name of the label column (default 'label')")

    model_group = train_parser.add_argument_group("model",
                                                  description="Arguments for specifying the model configuration")
    model_group.add_argument("--rnn-units", metavar="N", type=int, default=128, help="RNN units (default 128)")
    model_group.add_argument("--dropout", metavar="RATE", type=float, default=0.5, help="Dropout rate (default 0.5)")
    model_group.add_argument("--max-tokens", metavar="M", type=int,
                             help="Maximum number of tokens to embed (default longest text in the training data)")

    train_group = train_parser.add_argument_group("training",
                                                  description="Arguments for controlling the training procedure")
    train_group.add_argument("--epochs", metavar="N", type=int, default=10, help="training epochs (default 10)")
    train_group.add_argument("--batch-size", metavar="M", type=int, default=256, help="batch size (default 256)")
    train_group.add_argument("--model-filename", metavar="FILENAME",
                             help="file in which to to store the model (default do not store a model)")

    train_parser.set_defaults(
        func=lambda args: train(args.training, args.limit, args.validation, args.text_name, args.label_name,
                                args.rnn_units, args.dropout, args.max_tokens, args.epochs, args.batch_size,
                                args.model_filename))

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", description=textwrap.dedent("""
        Use a model to predict labels. The test data is a comma- or tab-delimited file with a column of texts. This
        prints that file, adding columns containing predicted probabilities for each category."""))
    predict_parser.add_argument("test", help="test data")
    predict_parser.add_argument("model_filename", metavar="filename", help="file containing the trained model")
    predict_parser.add_argument("--text-name", metavar="NAME", default="text",
                                help="name of the text column (default 'text')")
    predict_parser.add_argument("--limit", metavar="N", type=int,
                                help="only use this many samples (default use all the data)")
    predict_parser.set_defaults(func=lambda args: predict(args.test, args.model_filename, args.text_name, args.limit))

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser("evaluate", description=textwrap.dedent("""
        Score the model's performance on a labeled data set. 
        The test data is a comma- or tab-delimited file with columns of texts and labels."""))
    evaluate_parser.add_argument("test", help="test data")
    evaluate_parser.add_argument("model_filename", metavar="filename", help="file containing the trained model")
    evaluate_parser.add_argument("--text-name", metavar="NAME", default="text",
                                 help="name of the text column (default 'text')")
    evaluate_parser.add_argument("--label-name", metavar="NAME", default="label",
                                 help="name of the label column (default 'label')")
    evaluate_parser.add_argument("--limit", metavar="N", type=int,
                                 help="only use this many samples (default use all the data)")
    evaluate_parser.set_defaults(
        func=lambda args: evaluate(args.test, args.model_filename, args.text_name, args.label_name, args.limit))

    args = parser.parse_args()
    args.func(args)
