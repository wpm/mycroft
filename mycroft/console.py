"""
Command line interface to the text classifier.
"""
import argparse
import os
import textwrap
from functools import partial

import numpy
import pandas
from sklearn.datasets import fetch_20newsgroups

from mycroft import __version__
from .model import BagOfWordsEmbeddingClassifier, TextSequenceEmbeddingClassifier


def main(model_specifications, description=None, args=None):
    """
    Create and run a command line application that allows you to train and use the specified models. Each model is
    specified by a 3-tuple of model class, the name of its training subcommand, and text for the help description.

    This can be the main entry point of a program.

    :param model_specifications: descriptions of models to make available via the command line
    :type model_specifications: (mycroft.mode.TextEmbeddingClassifier, str, str)
    :param description: top level program description in the help message
    :type description: str
    :param args: command line arguments, if None, get them from sys.argv
    :type args: list of str or None
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=description)
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Commands")

    # Training subcommands
    train_parser = subparsers.add_parser("train", description="Train a model")
    model_parsers = train_parser.add_subparsers(title="Models")
    for model_class, model_command_name, description in model_specifications:
        model_parser = model_parsers.add_parser(model_command_name, parents=[training_argument_groups()],
                                                description=description)
        model_argument_group = \
            model_parser.add_argument_group("model", description="Arguments for specifying the model configuration:")
        model_class.command_line_arguments(model_argument_group)
        model_parser.set_defaults(func=partial(train_command, parser, model_class))

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", parents=[test_argument_groups("predict")],
                                           description=textwrap.dedent("""
        Use a model to predict labels. This prints the test data, adding columns containing predicted probabilities for 
        each category and the most probable category."""))
    predict_parser.set_defaults(func=predict_command)

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser("evaluate", parents=[test_argument_groups("evaluate")],
                                            description=textwrap.dedent("""
        Evaluate the model's performance on a labeled data set. 
        The test data is a comma- or tab-delimited file with columns of texts and labels."""))
    evaluate_parser.set_defaults(func=evaluate_command)

    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", description="Run a demo_command on 20 newsgroups data.")
    demo_parser.add_argument("--directory", default=".", help="directory in which to run demo (default current)")
    demo_parser.set_defaults(func=demo_command)

    parsed_args = parser.parse_args(args=args)
    parsed_args.func(parsed_args)


def default_main(args=None):
    """
    Entry point for the "mycroft" command line application.

    :param args: command line arguments, if None, get them from sys.argv
    :type args: list of str or None
    """
    model_specifications = [
        (TextSequenceEmbeddingClassifier, "nseq", textwrap.dedent("""
        Train a neural text sequence model.
        This applies a recursive neural network over a sequence of word embeddings to make a softmax prediction.""")),
        (BagOfWordsEmbeddingClassifier, "nbow", textwrap.dedent("""
        Train a neural bag of words model.
        This uses the mean of the word embeddings in a document to make a softmax prediction."""))
    ]
    # noinspection PyTypeChecker
    main(model_specifications, description=textwrap.dedent("""
    Mycroft classifies text to categorical labels.

    The training data is a comma- or tab-delimited file with column of text and a column of labels.
    The test data is in the same format without the labels."""), args=args)


def training_argument_groups():
    arguments = argparse.ArgumentParser(add_help=False)
    data_group = arguments.add_argument_group("data",
                                              description="Arguments for specifying the training data:")
    data_group.add_argument("training_data", metavar="TRAINING-FILE", help="training data file")
    data_group.add_argument("--limit", type=int, help="only train on this many samples (default use all the data)")
    data_group.add_argument("--validation-fraction", metavar="FRACTION", type=float,
                            help="fraction of data to use for validation (default none, use all data for training)")
    data_group.add_argument("--validation-data", metavar="FILE",
                            help="validation data file (default no validation data)")
    data_group.add_argument("--text-name", metavar="NAME", default="text",
                            help="name of the text column (default 'text')")
    data_group.add_argument("--label-name", metavar="NAME", default="label",
                            help="name of the label column (default 'label')")
    data_group.add_argument("--omit-labels", metavar="LABEL", nargs="*", help="omit samples with these label values")

    training_group = arguments.add_argument_group("training",
                                                  description="Arguments for controlling the training procedure:")
    training_group.add_argument("--epochs", type=int, default=10, help="number of training epochs (default 10)")
    training_group.add_argument("--batch-size", metavar="SIZE", type=int, default=32, help="batch size (default 32)")
    training_group.add_argument("--model-directory", metavar="DIRECTORY",
                                help="directory in which to store the model (default do not store a model)")
    training_group.add_argument("--logging", choices=["none", "progress", "epoch"], default="epoch",
                                help="no logging, a progress bar, one line per epoch (default per epoch)")
    training_group.add_argument("--tensor-board", metavar="DIRECTORY",
                                help="directory in which to create TensorBoard logs (default do not create them)")
    return arguments


def test_argument_groups(test_command):
    assert test_command in ["predict", "evaluate"]
    arguments = argparse.ArgumentParser(add_help=False)
    arguments.add_argument("model", help="directory or file containing the trained model")
    data_group = arguments.add_argument_group("data", description="Arguments for specifying the data to use:")

    data_group.add_argument("test", help="test data file")
    data_group.add_argument("--batch-size", metavar="SIZE", type=int, default=32, help="batch size (default 32)")
    data_group.add_argument("--limit", type=int, help="only use this many samples (default use all the data)")
    data_group.add_argument("--text-name", metavar="NAME", default="text",
                            help="name of the text column (default 'text')")
    if test_command == "evaluate":
        data_group.add_argument("--label-name", metavar="NAME", default="label",
                                help="name of the label column (default 'label')")
        data_group.add_argument("--omit-labels", metavar="LABEL", nargs="*",
                                help="omit samples with these label values")
    return arguments


def train_command(parser, model_class, args):
    if args.validation_fraction and args.validation_data:
        parser.error("Cannot specify both a validation fraction and a validation set.")
    # Preprocess training data.
    texts, labels, label_names = preprocess_labeled_data(args.training_data, args.limit, args.omit_labels,
                                                         args.text_name, args.label_name)
    if args.validation_data:
        validation_texts, validation_labels, _ = preprocess_labeled_data(args.validation_data, args.limit,
                                                                         args.omit_labels,
                                                                         args.text_name,
                                                                         args.label_name)
        validation_data = (validation_texts, validation_labels)
    else:
        validation_data = None
    # Train the model.
    model = model_class.create_from_command_line_arguments((texts, labels, label_names), args)
    verbose = {"none": 0, "progress": 1, "epoch": 2}[args.logging]
    history = model.train(texts, labels, epochs=args.epochs, batch_size=args.batch_size,
                          validation_fraction=args.validation_fraction, validation_data=validation_data,
                          model_directory=args.model_directory, tensor_board_directory=args.tensor_board,
                          verbose=verbose)
    losses = history.history[history.monitor]
    best_loss = min(losses)
    best_epoch = losses.index(best_loss)
    s = " - ".join("%s: %0.5f" % (score, values[best_epoch]) for score, values in sorted(history.history.items()))
    print("Best epoch %d of %d: %s" % (best_epoch + 1, args.epochs, s))


def predict_command(args):
    from .model import load_embedding_model

    model = load_embedding_model(args.model)
    data = read_data_file(args.test, args.limit)
    label_probabilities, predicted_labels = model.predict(data[args.text_name], args.batch_size)
    predictions = pandas.DataFrame(label_probabilities.reshape((len(data), model.num_labels)),
                                   columns=model.label_names)
    predictions["predicted label"] = predicted_labels
    data = data.join(predictions)
    print(data.to_csv(index=False))


def evaluate_command(args):
    from .model import load_embedding_model

    model = load_embedding_model(args.model)
    texts, labels, _ = preprocess_labeled_data(args.test, args.limit, args.omit_labels, args.text_name, args.label_name,
                                               model.label_names)
    results = model.evaluate(texts, labels, args.batch_size)
    print("\n" + " - ".join("%s: %0.5f" % (name, score) for name, score in results))


def preprocess_labeled_data(data_filename, limit, omit_labels, text_name, label_name, label_names=None):
    data = read_data_file(data_filename, limit)
    if omit_labels:
        data = data[~data[label_name].isin(omit_labels)]
    data[label_name] = pandas.Categorical(data[label_name].astype(str), categories=label_names)
    labels = numpy.array(data[label_name])
    label_names = list(data[label_name].cat.categories)
    return data[text_name], labels, label_names


def read_data_file(data_filename, limit):
    return pandas.read_csv(data_filename, sep=None, engine="python").dropna()[:limit]


def demo_command(args):
    def create_data_file(partition, filename, samples):
        data = pandas.DataFrame(
            {"text": partition.data,
             "label": [partition.target_names[target] for target in partition.target]}).dropna()[:samples]
        data.to_csv(filename, index=False)
        return filename

    os.makedirs(args.directory, exist_ok=True)
    print("Download a portion of the 20 Newsgroups data and create train.csv and test.csv.")
    newsgroups_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    newsgroups_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    train_filename = create_data_file(newsgroups_train, os.path.join(args.directory, "train.csv"), 1000)
    test_filename = create_data_file(newsgroups_test, os.path.join(args.directory, "test.csv"), 100)
    model_directory = os.path.join(args.directory, "model")
    print("Train a model.\n")
    cmd = "train nbow %s --model-directory %s --epochs 2\n" % (train_filename, model_directory)
    print("mycroft " + cmd)
    default_main(cmd.split())
    print("\nEvaluate it on the test data.\n")
    cmd = "evaluate %s %s\n" % (model_directory, test_filename)
    print("mycroft " + cmd)
    default_main(cmd.split())
