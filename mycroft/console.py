"""
Command line interface to the text classifier.
"""
from __future__ import print_function

import argparse
import textwrap

import numpy
import pandas
from sklearn.datasets import fetch_20newsgroups

from mycroft import __version__


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    Mycroft classifies text to categorical labels.
                                         
    The training data is a comma- or tab-delimited file with column of text and a column of labels.
    The test data is in the same format without the labels."""))
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Commands")

    # Training

    # train-nseq subcommand
    neural_sequence_parser = subparsers.add_parser("train-nseq", parents=[create_training_argument_groups("nseq")],
                                                   description=textwrap.dedent(
                                                       """Train a neural text sequence model.
                                                       This applies a recursive neural network over a sequence of word 
                                                       embeddings to make a softmax prediction."""))
    neural_sequence_parser.set_defaults(func=neural_sequence_command)

    # train-nbow subcommand
    neural_bow_parser = subparsers.add_parser("train-nbow", parents=[create_training_argument_groups("nbow")],
                                              description=textwrap.dedent(
                                                  """Train a neural bag of words model.
                                                  This uses the mean of the word embeddings in a document to make a
                                                  softmax prediction."""))
    neural_bow_parser.set_defaults(func=neural_bow_command)

    # train-svm subcommand
    svm_parser = subparsers.add_parser("train-svm", parents=[(create_training_argument_groups("svm"))],
                                       description="Train a support vector machine model over tf-idf counts.")
    svm_parser.set_defaults(func=svm_command)

    # Predict, evaluate, and demo
    shared_test_arguments = argparse.ArgumentParser(add_help=False)
    shared_test_arguments.add_argument("test", help="test data file")
    shared_test_arguments.add_argument("model_directory", metavar="model",
                                       help="directory containing the trained model")
    shared_test_arguments.add_argument("--batch-size", metavar="SIZE", type=int, default=256,
                                       help="batch size (default 256)")
    shared_test_arguments.add_argument("--limit", type=int,
                                       help="only use this many samples (default use all the data)")
    shared_test_arguments.add_argument("--text-name", metavar="NAME", default="text",
                                       help="name of the text column (default 'text')")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", parents=[shared_test_arguments], description=textwrap.dedent("""
        Use a model to predict labels. This prints the test data, adding columns containing predicted probabilities for 
        each category and the most probable category."""))
    predict_parser.set_defaults(func=predict_command)

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser("evaluate", parents=[shared_test_arguments], description=textwrap.dedent("""
        Score the model's performance on a labeled data set. 
        The test data is a comma- or tab-delimited file with columns of texts and labels."""))
    evaluate_parser.add_argument("--label-name", metavar="NAME", default="label",
                                 help="name of the label column (default 'label')")
    evaluate_parser.set_defaults(func=evaluate_command)

    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", description="Run a demo_command on 20 newsgroups data.")
    demo_parser.set_defaults(func=demo_command)

    args = parser.parse_args()
    args.func(args)


def create_training_argument_groups(training_command):
    assert training_command in ["nseq", "nbow", "svm"]
    training_arguments = argparse.ArgumentParser(add_help=False)

    model_group = training_arguments.add_argument_group("model",
                                                        description="Arguments for specifying the model configuration:")
    model_group.add_argument("--dropout", metavar="RATE", type=float, default=0.5, help="Dropout rate (default 0.5)")
    if training_command is "nseq":
        model_group.add_argument("--rnn-units", metavar="UNITS", type=int, default=128, help="RNN units (default 128)")
        model_group.add_argument("--max-tokens", metavar="TOKENS", type=int,
                                 help="Maximum number of tokens to embed (default longest text in the training data)")

    data_group = training_arguments.add_argument_group("data",
                                                       description="Arguments for specifying the data to train on:")
    data_group.add_argument("training", help="training data file")
    data_group.add_argument("--limit", type=int, help="only train on this many samples (default use all the data)")
    data_group.add_argument("--validation", metavar="PORTION", type=float,
                            help="portion of data to use for validation (default none)")
    data_group.add_argument("--text-name", metavar="NAME", default="text",
                            help="name of the text column (default 'text')")
    data_group.add_argument("--label-name", metavar="NAME", default="label",
                            help="name of the label column (default 'label')")
    data_group.add_argument("--omit-labels", metavar="LABEL", nargs="*", help="omit samples with these label values")

    train_group = training_arguments.add_argument_group("training",
                                                        description="Arguments for controlling the training procedure:")
    if training_command in ["nbow", "nseq"]:
        train_group.add_argument("--epochs", type=int, default=10, help="training epochs (default 10)")
        train_group.add_argument("--batch-size", metavar="SIZE", type=int, default=256, help="batch size (default 256)")
        train_group.add_argument("--model-directory", metavar="DIRECTORY",
                                 help="directory in which to to store the model (default do not store a model)")
        train_group.add_argument("--logging", choices=["none", "progress", "epoch"], default="epoch",
                                 help="no logging, a progress bar, one line per epoch (default epoch)")
    else:
        train_group.add_argument("--model-filename", metavar="FILENAME",
                                 help="file in which to to store the model (default do not store a model)")
        train_group.add_argument("--verbose", action="store_true", help="verbose training output")

    if training_command in ["nbow", "nseq"]:
        language_group = \
            training_arguments.add_argument_group("language",
                                                  description="Arguments for controlling language processing:")
        if training_command is "nseq":
            language_group.add_argument("--vocabulary-size", metavar="SIZE", default=20000,
                                        help="number of words in the vocabulary (default 20000)")
        language_group.add_argument("--language-model", metavar="MODEL", default="en",
                                    help="the spaCy language model to use (default 'en')")

    return training_arguments


def neural_sequence_command(args):
    from .model import TextSequenceEmbeddingClassifier
    from .text import longest_text

    texts, labels, label_names = preprocess_labeled_data(args.training, args.limit, args.omit_labels, args.text_name,
                                                         args.label_name)
    if args.max_tokens is None:
        args.max_tokens = longest_text(texts, args.language_model)
    model = TextSequenceEmbeddingClassifier.create(args.vocabulary_size, args.max_tokens, args.rnn_units, args.dropout,
                                                   label_names, args.language_model)
    train(args, texts, labels, model)


def neural_bow_command(args):
    from .model import BagOfWordsEmbeddingClassifier

    texts, labels, label_names = preprocess_labeled_data(args.training, args.limit, args.omit_labels, args.text_name,
                                                         args.label_name)
    model = BagOfWordsEmbeddingClassifier.create(args.dropout, label_names, args.language_model)
    train(args, texts, labels, model)


def svm_command(args):
    from .model import WordCountClassifier

    texts, labels, label_names = preprocess_labeled_data(args.training, args.limit, args.omit_labels, args.text_name,
                                                         args.label_name)
    model = WordCountClassifier(label_names, args.verbose)
    results = model.train(texts, labels, args.validation, args.model_filename)
    if results:
        print("Validation scores: %s" % " - ".join("%s: %0.5f" % (score, value) for score, value in sorted(results)))


def train(args, texts, labels, model):
    verbose = {"none": 0, "progress": 1, "epoch": 2}[args.logging]
    history = model.train(texts, labels, args.epochs, args.batch_size, args.validation, args.model_directory,
                          verbose=verbose)
    losses = history.history[history.monitor]
    best_loss = min(losses)
    best_epoch = losses.index(best_loss)
    s = " - ".join("%s: %0.5f" % (score, values[best_epoch]) for score, values in sorted(history.history.items()))
    print("Best epoch %d of %d: %s" % (best_epoch + 1, args.epochs, s))


def predict_command(args):
    from .model import TextEmbeddingClassifier

    model = TextEmbeddingClassifier.load_model(args.model_directory)
    data = read_data_file(args.test, args.limit)
    label_probabilities, predicted_labels = model.predict(data[args.text_name], args.batch_size)
    predictions = pandas.DataFrame(label_probabilities.reshape((len(data), model.num_labels)),
                                   columns=model.label_names)
    predictions["predicted label"] = [model.label_names[i] for i in predicted_labels]
    data = data.join(predictions)
    print(data.to_csv(index=False))


def evaluate_command(args):
    from .model import TextEmbeddingClassifier

    model = TextEmbeddingClassifier.load_model(args.model_directory)
    texts, labels, _ = preprocess_labeled_data(args.test, args.limit, args.omit_labels, args.text_name, args.label_name,
                                               model.label_names)
    results = model.evaluate(texts, labels, args.batch_size)
    print("\n" + " - ".join("%s: %0.5f" % (name, score) for name, score in results))


def demo_command(_):
    def create_data_file(partition, filename, samples):
        data = pandas.DataFrame(
            {"text": partition.data,
             "label": [partition.target_names[target] for target in partition.target]}).dropna()[:samples]
        data.to_csv(filename, index=False)
        return filename

    print("Download a portion of the 20 Newsgroups data and create train.csv and test.csv.")
    newsgroups_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    newsgroups_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    train_filename = create_data_file(newsgroups_train, "train.csv", 1000)
    test_filename = create_data_file(newsgroups_test, "test.csv", 100)
    model_directory = "model"
    print("Train a model.\n")
    print("mycroft train-nbow %s --model-directory %s --epochs 2\n" % (
        train_filename, model_directory))
    training_args = argparse.Namespace(training=train_filename, limit=None, text_name="text", label_name="label",
                                       validation=0.2, dropout=0.5,
                                       language_model="en",
                                       epochs=2, batch_size=256,
                                       model_directory=model_directory, logging="epoch")
    # noinspection PyTypeChecker
    neural_bow_command(training_args)
    print("\nEvaluate it on the test data.\n")
    print("mycroft evaluate %s model-directory %s\n" % (test_filename, model_directory))
    evaluate_args = argparse.Namespace(model_directory=model_directory, test=test_filename, limit=None,
                                       text_name="text", label_name="label", batch_size=256, language_model="en")
    # noinspection PyTypeChecker
    evaluate_command(evaluate_args)


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
