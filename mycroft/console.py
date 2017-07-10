import argparse
import os
import textwrap

import pandas
from sklearn.datasets import fetch_20newsgroups

from mycroft import __version__, train, predict, evaluate, details


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    Mycroft classifies text to categorical labels.
                                         
    The training data is a comma- or tab-delimited file with column of text and a column of labels.
    The test data is in the same format without the labels."""))
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Commands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", description=textwrap.dedent("""
        Train a model to predict labels for text. The training and optional validation loss and accuracy are
        printed for each epoch."""))

    data_group = train_parser.add_argument_group("data", description="Arguments for specifying the data to train on")
    data_group.add_argument("training", help="training data file")
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

    language_group = train_parser.add_argument_group("language",
                                                     description="Arguments for controlling language processing")
    language_group.add_argument("--language-model", default="en", help="the spaCy language model to use (default 'en')")

    train_group = train_parser.add_argument_group("training",
                                                  description="Arguments for controlling the training procedure")
    train_group.add_argument("--epochs", metavar="N", type=int, default=10, help="training epochs (default 10)")
    train_group.add_argument("--batch-size", metavar="M", type=int, default=256, help="batch size (default 256)")
    train_group.add_argument("--model-filename", metavar="FILENAME",
                             help="file in which to to store the model (default do not store a model)")

    train_parser.set_defaults(
        func=lambda args: train(args.training, args.limit, args.validation, args.text_name, args.label_name,
                                args.rnn_units, args.dropout, args.max_tokens,
                                args.language_model,
                                args.epochs, args.batch_size, args.model_filename))

    shared_arguments = argparse.ArgumentParser(add_help=False)
    shared_arguments.add_argument("test", help="test data file")
    shared_arguments.add_argument("model_filename", metavar="model", help="file containing the trained model")
    shared_arguments.add_argument("--batch-size", metavar="M", type=int, default=256, help="batch size (default 256)")
    shared_arguments.add_argument("--limit", metavar="N", type=int,
                                  help="only use this many samples (default use all the data)")
    shared_arguments.add_argument("--language-model", default="en",
                                  help="the spaCy language model to use (default 'en')")
    shared_arguments.add_argument("--text-name", metavar="NAME", default="text",
                                  help="name of the text column (default 'text')")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", parents=[shared_arguments], description=textwrap.dedent("""
        Use a model to predict labels. This prints the test data, adding columns containing predicted probabilities for 
        each category and the most probable category."""))
    predict_parser.set_defaults(
        func=lambda args: predict(args.test, args.model_filename, args.batch_size, args.text_name, args.limit,
                                  args.language_model))

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser("evaluate", parents=[shared_arguments], description=textwrap.dedent("""
        Score the model's performance on a labeled data set. 
        The test data is a comma- or tab-delimited file with columns of texts and labels."""))
    evaluate_parser.add_argument("--label-name", metavar="NAME", default="label",
                                 help="name of the label column (default 'label')")
    evaluate_parser.set_defaults(
        func=lambda args: evaluate(args.test, args.model_filename, args.batch_size, args.text_name, args.label_name,
                                   args.limit, args.language_model))

    # Details subcommand
    details_parser = subparsers.add_parser("details", description="Show details of a trained model.")
    details_parser.add_argument("model_filename", metavar="model", help="file containing the trained model")
    details_parser.set_defaults(func=lambda args: details(args.model_filename))

    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", description="Run a demo on 20 newsgroups data.")
    demo_parser.add_argument("--limit", metavar="N", type=int,
                             help="only use this many samples (default use all the data)")
    demo_parser.add_argument("--output-directory", metavar="DIRECTORY", default=".",
                             help="where to write files (default working directory)")
    demo_parser.set_defaults(func=lambda args: demo(args.output_directory, args.limit))

    args = parser.parse_args()
    args.func(args)


def demo(output_directory, limit):
    def create_data_file(partition, filename):
        # TODO Drop the really long texts because they take up too much memory.
        data = pandas.DataFrame(
            {"text": partition.data,
             "label": [partition.target_names[target] for target in partition.target]}).dropna()[:limit]
        filename = os.path.join(output_directory, filename)
        data.to_csv(filename, index=False)
        return filename

    print("Download 20 Newsgroups data and create train.csv and test.csv.")
    newsgroups_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    newsgroups_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    train_filename = create_data_file(newsgroups_train, "train.csv")
    test_filename = create_data_file(newsgroups_test, "test.csv")
    model_filename = os.path.join(output_directory, "model.hd5")
    print("Train a model.")
    print("mycroft train %s --model-filename %s" % (train_filename, model_filename))
    train(train_filename, None, 0.2, "text", "label", 128, 0.5, None, "en", 10, 256, model_filename)
    print("Evaluate it on the test data.")
    print("mycroft evaluate %s --model-filename %s" % (test_filename, model_filename))
    evaluate(test_filename, model_filename, 256, "text", "label", None, "en")
