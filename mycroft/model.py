"""Machine learning components"""
import argparse
import errno
import json
import os
import pickle
import sys
from io import StringIO

import numpy
import six
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import mycroft.arguments as arguments


def load_embedding_model(model_directory):
    from keras.models import load_model
    with open(os.path.join(model_directory, TextEmbeddingClassifier.classifier_name), mode="rb") as f:
        model = pickle.load(f)
    model.model = load_model(os.path.join(model_directory, TextEmbeddingClassifier.model_name))
    return model


class TextEmbeddingClassifier(object):
    model_name = "model.hd5"
    classifier_name = "classifier.pk"
    description_name = "description.txt"
    history_name = "history.json"

    labels_attribute = "labels"

    def __init__(self, model, embedder, label_names):
        assert len(label_names) == len(set(label_names)), "Non-unique label names %s" % label_names
        self.model = model
        self.embedder = embedder
        self.label_names = label_names

    def __str__(self):
        def model_topology():
            if six.PY3:
                # Keras' model summary prints to standard out. This trick of capturing the output causes an error when
                # running under Python 2.7.
                old_stdout = sys.stdout
                sys.stdout = s = StringIO()
                self.model.summary()
                sys.stdout = old_stdout
                return s.getvalue()
            else:
                return ""

        return "%s\n\n%s" % (repr(self), model_topology())

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["model"]
        return d

    def train(self, texts, labels, epochs=10, batch_size=32, validation_fraction=None, validation_data=None,
              model_directory=None, verbose=1):
        def model_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.model_name)

        def classifier_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.classifier_name)

        def description_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.description_name)

        def history_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.history_name)

        def create_directory(directory):
            if six.PY3:
                os.makedirs(directory, exist_ok=True)
            else:
                try:
                    os.makedirs(directory)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

        assert not (
            validation_fraction and validation_data), "Both validation fraction and validation data are specified"
        doing_validation = validation_fraction or validation_data
        if doing_validation:
            monitor = "val_loss"
            if validation_data:
                validation_data = (self.embedder.encode(validation_data[0]), self.label_indexes(validation_data[1]))
        else:
            monitor = "loss"
        callbacks = None
        if model_directory is not None:
            create_directory(model_directory)
            if doing_validation:
                from keras.callbacks import ModelCheckpoint
                callbacks = [ModelCheckpoint(filepath=os.path.join(model_directory, TextEmbeddingClassifier.model_name),
                                             monitor=monitor, save_best_only=True, verbose=verbose)]
            with open(description_filename(), mode="w") as f:
                f.write("%s" % self)

        training_vectors = self.embedder.encode(texts)
        labels = self.label_indexes(labels)
        history = self.model.fit(training_vectors, labels, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_fraction, validation_data=validation_data,
                                 verbose=verbose, callbacks=callbacks)
        history.monitor = monitor

        if model_directory is not None:
            if not os.path.isfile(model_filename()):
                self.model.save(model_filename())
            with open(classifier_filename(), mode="wb") as f:
                pickle.dump(self, f)
            with open(history_filename(), mode="w") as f:
                h = {"epoch": history.epoch, "history": history.history, "monitor": history.monitor,
                     "params": history.params}
                json.dump(h, f, sort_keys=True, indent=4, separators=(",", ": "))

        return history

    def predict(self, texts, batch_size=32):
        embeddings = self.embedder.encode(texts)
        label_probabilities = self.model.predict(embeddings, batch_size=batch_size, verbose=0)
        predicted_labels = label_probabilities.argmax(axis=1)
        return label_probabilities, [self.label_names[label_index] for label_index in predicted_labels]

    def evaluate(self, texts, labels, batch_size=32):
        embeddings = self.embedder.encode(texts)
        labels = self.label_indexes(labels)
        metrics = self.model.evaluate(embeddings, labels, batch_size=batch_size, verbose=0)
        return list(zip(self.model.metrics_names, metrics))

    def label_indexes(self, labels):
        return [self.label_names.index(label) for label in labels]

    @property
    def dropout(self):
        return self.model.get_layer("dropout").rate

    @property
    def num_labels(self):
        return len(self.label_names)


class TextSequenceEmbeddingClassifier(TextEmbeddingClassifier):
    def __init__(self, vocabulary_size, sequence_length, rnn_type, rnn_units, dropout, label_names,
                 language_model="en"):
        from keras.models import Sequential
        from keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU, LSTM
        from .text import TextSequenceEmbedder

        embedder = TextSequenceEmbedder(vocabulary_size, sequence_length, language_model)
        model = Sequential()
        sequence_length = embedder.encoding_shape[0]
        embedding_size = embedder.embedding_size
        model.add(Embedding(embedder.vocabulary_size, embedding_size, weights=[embedder.embedding_matrix],
                            input_length=sequence_length, mask_zero=True, trainable=False))
        rnn = {"lstm": LSTM, "gru": GRU}[rnn_type]
        model.add(Bidirectional(rnn(rnn_units), name="rnn"))
        model.add(Dense(len(label_names), activation="softmax", name="softmax"))
        model.add(Dropout(dropout, name="dropout"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        super(self.__class__, self).__init__(model, embedder, label_names)

    @staticmethod
    def training_argument_parser():
        training_arguments = argparse.ArgumentParser(add_help=False)

        model_group = arguments.model_group_name(training_arguments)
        arguments.dropout_argument(model_group)
        model_group.add_argument("--rnn-type", metavar="TYPE", choices=["gru", "lstm"], default="gru",
                                 help="GRU or LSTM (default GRU)")
        model_group.add_argument("--rnn-units", metavar="UNITS", type=int, default=64,
                                 help="RNN units (default 64)")
        model_group.add_argument("--max-tokens", metavar="TOKENS", type=int,
                                 help="Maximum number of tokens to embed (default longest text in the training data)")

        arguments.data_group(training_arguments)

        arguments.training_group(training_arguments)

        language_group = arguments.langauge_group_name(training_arguments)
        arguments.vocabulary_size_argument(language_group)
        arguments.language_model_argument(language_group)

        return training_arguments

    def __repr__(self):
        return "Neural text sequence classifier: %d labels, %d RNN units, dropout rate %0.2f\n%s" % (
            self.num_labels, self.rnn_units, self.dropout, self.embedder)

    @property
    def rnn_units(self):
        return self.model.get_layer("rnn").layer.units

    @property
    def embeddings_per_text(self):
        return self.model.get_layer("rnn").input_shape[1]

    @property
    def embedding_size(self):
        return self.model.get_layer("rnn").input_shape[2]


class BagOfWordsEmbeddingClassifier(TextEmbeddingClassifier):
    def __init__(self, dropout, label_names, language_model="en"):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from .text import BagOfWordsEmbedder

        embedder = BagOfWordsEmbedder(language_model)
        model = Sequential()
        model.add(Dense(len(label_names), input_shape=embedder.encoding_shape, activation="softmax", name="softmax"))
        model.add(Dropout(dropout, name="dropout"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        super(self.__class__, self).__init__(model, embedder, label_names)

    @staticmethod
    def training_argument_parser():
        training_arguments = argparse.ArgumentParser(add_help=False)

        model_group = arguments.model_group_name(training_arguments)
        arguments.dropout_argument(model_group)

        arguments.data_group(training_arguments)

        arguments.training_group(training_arguments)

        language_group = arguments.langauge_group_name(training_arguments)
        arguments.language_model_argument(language_group)

        return training_arguments

    def __repr__(self):
        return "Neural bag of words classifier: %d labels, dropout rate %0.2f\n%s" % (
            self.num_labels, self.dropout, self.embedder)
