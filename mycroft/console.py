"""
Command line interface to the text classifier.
"""
from __future__ import print_function

import argparse
import os
import textwrap
from functools import partial

import numpy
import pandas
import six
from sklearn.datasets import fetch_20newsgroups

from mycroft import __version__


def main(args=None):
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
    from .model import TextSequenceEmbeddingClassifier
    neural_sequence_parser = subparsers.add_parser("train-nseq",
                                                   parents=[TextSequenceEmbeddingClassifier.training_argument_parser()],
                                                   description=textwrap.dedent(
                                                       """Train a neural text sequence model.
                                                       This applies a recursive neural network over a sequence of word 
                                                       embeddings to make a softmax prediction."""))
    neural_sequence_parser.set_defaults(func=partial(train_command, parser, create_neural_sequence_model))

    # train-nbow subcommand
    from .model import BagOfWordsEmbeddingClassifier
    neural_bow_parser = subparsers.add_parser("train-nbow",
                                              parents=[BagOfWordsEmbeddingClassifier.training_argument_parser()],
                                              description=textwrap.dedent(
                                                  """Train a neural bag of words model.
                                                  This uses the mean of the word embeddings in a document to make a
                                                  softmax prediction."""))
    neural_bow_parser.set_defaults(func=partial(train_command, parser, create_neural_bow_model))

    # Predict, evaluate, and demo

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", parents=[create_test_argument_groups("predict")],
                                           description=textwrap.dedent("""
        Use a model to predict labels. This prints the test data, adding columns containing predicted probabilities for 
        each category and the most probable category."""))
    predict_parser.set_defaults(func=predict_command)

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser("evaluate", parents=[create_test_argument_groups("evaluate")],
                                            description=textwrap.dedent("""
        Evaluate the model's performance on a labeled data set. 
        The test data is a comma- or tab-delimited file with columns of texts and labels."""))
    evaluate_parser.set_defaults(func=evaluate_command)

    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", description="Run a demo_command on 20 newsgroups data.")
    demo_parser.set_defaults(func=demo_command)

    parsed_args = parser.parse_args(args=args)
    parsed_args.func(parsed_args)


def create_test_argument_groups(test_command):
    assert test_command in ["predict", "evaluate"]
    test_arguments = argparse.ArgumentParser(add_help=False)
    test_arguments.add_argument("model", help="directory or file containing the trained model")
    data_group = test_arguments.add_argument_group("data", description="Arguments for specifying the data to use:")

    data_group.add_argument("test", help="test data file")
    data_group.add_argument("--batch-size", metavar="SIZE", type=int, default=32,
                            help="batch size, ignored for SVM models (default 32)")
    data_group.add_argument("--limit", type=int, help="only use this many samples (default use all the data)")
    data_group.add_argument("--text-name", metavar="NAME", default="text",
                            help="name of the text column (default 'text')")
    if test_command == "evaluate":
        data_group.add_argument("--label-name", metavar="NAME", default="label",
                                help="name of the label column (default 'label')")
        data_group.add_argument("--omit-labels", metavar="LABEL", nargs="*",
                                help="omit samples with these label values")
    return test_arguments


def train_command(parser, model_factory, args):
    if args.validation_fraction and args.validation_data:
        parser.error("Cannot specify both a validation fraction and a validation set.")
    label_names, labels, texts, validation_data = preprocess_training_data(args)
    model = model_factory(args, label_names=label_names, labels=labels, texts=texts)
    train(args, texts, labels, model, validation_data)


def create_neural_sequence_model(args, **kwargs):
    from .model import TextSequenceEmbeddingClassifier
    from .text import longest_text

    if args.max_tokens is None:
        args.max_tokens = longest_text(kwargs["texts"], args.language_model)
    return TextSequenceEmbeddingClassifier(args.vocabulary_size, args.max_tokens, args.rnn_type, args.rnn_units,
                                           args.dropout, kwargs["label_names"], args.language_model)


def create_neural_bow_model(args, **kwargs):
    from .model import BagOfWordsEmbeddingClassifier

    return BagOfWordsEmbeddingClassifier(args.dropout, kwargs["label_names"], args.language_model)


def preprocess_training_data(args):
    texts, labels, label_names = preprocess_labeled_data(args.training, args.limit, args.omit_labels, args.text_name,
                                                         args.label_name)
    if args.validation_data:
        validation_texts, validation_labels, _ = preprocess_labeled_data(args.validation_data, args.limit,
                                                                         args.omit_labels,
                                                                         args.text_name,
                                                                         args.label_name)
        validation_data = (validation_texts, validation_labels)
    else:
        validation_data = None
    return label_names, labels, texts, validation_data


def train(args, texts, labels, model, validation_data=None):
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
    from .model import TextEmbeddingClassifier

    model = load_model(args.model)
    data = read_data_file(args.test, args.limit)
    if isinstance(model, TextEmbeddingClassifier):
        label_probabilities, predicted_labels = model.predict(data[args.text_name], args.batch_size)
    else:
        label_probabilities, predicted_labels = model.predict(data[args.text_name])
    predictions = pandas.DataFrame(label_probabilities.reshape((len(data), model.num_labels)),
                                   columns=model.label_names)
    predictions["predicted label"] = predicted_labels
    data = data.join(predictions)
    if six.PY3:
        print(data.to_csv(index=False))
    else:
        print(data.to_csv(index=False, encoding="UTF8"))


def evaluate_command(args):
    from .model import TextEmbeddingClassifier

    model = load_model(args.model)
    texts, labels, _ = preprocess_labeled_data(args.test, args.limit, args.omit_labels, args.text_name, args.label_name,
                                               model.label_names)
    if isinstance(model, TextEmbeddingClassifier):
        results = model.evaluate(texts, labels, args.batch_size)
    else:
        results = model.evaluate(texts, labels)
    print("\n" + " - ".join("%s: %0.5f" % (name, score) for name, score in results))


def load_model(name):
    from .model import load_embedding_model

    if os.path.isdir(name):
        return load_embedding_model(name)
    else:
        raise ValueError("Invalid model name %s" % name)


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
    training_args = argparse.Namespace(training=train_filename,
                                       limit=None, text_name="text", label_name="label", omit_labels=None,
                                       validation_fraction=0.2, validation_data=None, dropout=0.5,
                                       language_model="en",
                                       epochs=2, batch_size=32, tensor_board=None,
                                       model_directory=model_directory, logging="epoch")
    # noinspection PyTypeChecker
    partial(train_command, None, create_neural_bow_model)(training_args)
    print("\nEvaluate it on the test data.\n")
    print("mycroft evaluate %s %s\n" % (model_directory, test_filename))
    evaluate_args = argparse.Namespace(model=model_directory, test=test_filename, limit=None,
                                       text_name="text", label_name="label", omit_labels=None, batch_size=32,
                                       language_model="en")
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
    if six.PY3:
        return pandas.read_csv(data_filename, sep=None, engine="python").dropna()[:limit]
    else:
        return pandas.read_csv(data_filename, sep=None, engine="python", encoding="UTF-8").dropna()[:limit]
