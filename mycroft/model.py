"""Machine learning components"""
import os
import pickle
import sys
from io import StringIO

import h5py
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class TextEmbeddingClassifier:
    model_name = "model.hd5"
    embedder_name = "embedder.pk"
    description_name = "description.txt"

    @classmethod
    def load_model(cls, model_directory):
        from keras.models import load_model
        model = load_model(os.path.join(model_directory, TextEmbeddingClassifier.model_name))
        with h5py.File(os.path.join(model_directory, TextEmbeddingClassifier.model_name), "r") as m:
            label_names = [name.decode("UTF-8") for name in list(m.attrs["categories"])]
        with open(os.path.join(model_directory, TextEmbeddingClassifier.embedder_name), mode="rb") as f:
            embedder = pickle.load(f)
        return cls(model, embedder, label_names)

    def __init__(self, model, embedder, label_names):
        self.model = model
        self.embedder = embedder
        self.label_names = label_names

    def __str__(self):
        def model_topology():
            # Keras' model summary prints to standard out.
            old_stdout = sys.stdout
            sys.stdout = s = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            return s.getvalue()

        return "%s\n\n%s" % (repr(self), model_topology())

    def train(self, texts, labels, epochs=10, batch_size=32, validation_fraction=None, model_directory=None, verbose=1):
        def model_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.model_name)

        def embedder_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.embedder_name)

        def description_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.description_name)

        if validation_fraction:
            monitor = "val_loss"
        else:
            monitor = "loss"
        callbacks = None
        if model_directory is not None:
            os.makedirs(model_directory, exist_ok=True)
            if validation_fraction:
                from keras.callbacks import ModelCheckpoint
                callbacks = [ModelCheckpoint(filepath=os.path.join(model_directory, TextEmbeddingClassifier.model_name),
                                             monitor=monitor, save_best_only=True, verbose=verbose)]
            with open(embedder_filename(), mode="wb") as f:
                pickle.dump(self.embedder, f)
            with open(description_filename(), mode="w") as f:
                f.write("%s" % self)

        training_vectors = self.embedder.encode(texts)
        history = self.model.fit(training_vectors, labels, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_fraction, verbose=verbose, callbacks=callbacks)

        if model_directory is not None:
            if not validation_fraction:
                self.model.save(model_filename())
            with h5py.File(model_filename()) as m:
                m.attrs["categories"] = numpy.array(
                    [numpy.string_(numpy.str_(label_name)) for label_name in self.label_names])
                m.attrs["language_model"] = numpy.string_(numpy.str_(self.embedder.language_model))

        history.monitor = monitor
        return history

    def predict(self, texts, batch_size=32):
        embeddings = self.embedder.encode(texts)
        label_probabilities = self.model.predict(embeddings, batch_size=batch_size, verbose=0)
        predicted_labels = label_probabilities.argmax(axis=1)
        return label_probabilities, predicted_labels

    def evaluate(self, texts, labels, batch_size=32):
        embeddings = self.embedder.encode(texts)
        metrics = self.model.evaluate(embeddings, labels, batch_size=batch_size, verbose=0)
        return list(zip(self.model.metrics_names, metrics))

    @property
    def dropout(self):
        return self.model.get_layer("dropout").rate

    @property
    def num_labels(self):
        return len(self.label_names)


class BagOfWordsEmbeddingClassifier(TextEmbeddingClassifier):
    @classmethod
    def create(cls, dropout, label_names, language_model="en"):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from mycroft.text import BagOfWordsEmbedder

        embedder = BagOfWordsEmbedder(language_model)
        model = Sequential()
        model.add(Dense(len(label_names), input_shape=embedder.encoding_shape, activation="softmax", name="softmax"))
        model.add(Dropout(dropout, name="dropout"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return cls(model, embedder, label_names)

    def __repr__(self):
        return "Neural bag of words classifier: %d labels, dropout rate %0.2f\n%s" % (
            self.num_labels, self.dropout, self.embedder)


class TextSequenceEmbeddingClassifier(TextEmbeddingClassifier):
    @classmethod
    def create(cls, vocabulary_size, sequence_length, rnn_units, dropout, label_names, language_model="en"):
        from keras.models import Sequential
        from keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
        from mycroft.text import TextSequenceEmbedder

        embedder = TextSequenceEmbedder(vocabulary_size, sequence_length, language_model)
        model = Sequential()
        sequence_length = embedder.encoding_shape[0]
        embedding_size = embedder.embedding_size
        model.add(Embedding(embedder.vocabulary_size, embedding_size, weights=[embedder.embedding_matrix],
                            input_length=sequence_length, mask_zero=True, trainable=False))
        model.add(Bidirectional(LSTM(rnn_units), name="rnn"))
        model.add(Dense(len(label_names), activation="softmax", name="softmax"))
        model.add(Dropout(dropout, name="dropout"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return cls(model, embedder, label_names)

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


class WordCountClassifier:
    def __init__(self, label_names, verbose=False, stop_words="english"):
        self.label_names = label_names
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(sublinear_tf=True, stop_words=stop_words)),
            ("svm", SVC(probability=True, verbose=verbose))
        ])

    def __repr__(self):
        return "SVM TF-IDF classifier: %d labels" % self.num_labels

    def train(self, texts, labels, validation_fraction=None, model_filename=None):
        if validation_fraction:
            train_texts, validation_texts, train_labels, validation_labels = \
                train_test_split(texts, labels, test_size=validation_fraction)
        else:
            train_texts, train_labels = texts, labels
        self.model.fit(train_texts, train_labels)
        if model_filename:
            self.save(model_filename)
        if validation_fraction:
            # noinspection PyUnboundLocalVariable,PyNoneFunctionAssignment
            validation_results = self.evaluate(validation_texts, validation_labels)
        else:
            validation_results = None
        return validation_results

    def predict(self, texts):
        label_probabilities = numpy.array(self.model.predict_proba(texts))
        predicted_labels = label_probabilities.argmax(axis=1)
        return label_probabilities, predicted_labels

    def evaluate(self, texts, labels):
        label_probabilities, predicted_labels = self.predict(texts)
        return [
            ("acc", accuracy_score(labels, predicted_labels)),
            ("loss", log_loss(labels, label_probabilities, labels=range(self.num_labels)))
        ]

    @property
    def num_labels(self):
        return len(self.label_names)

    def save(self, model_filename):
        with open(model_filename, "wb") as f:
            return pickle.dump(self, f)

    @staticmethod
    def load_model(model_filename):
        with open(model_filename, "rb") as f:
            return pickle.load(f)
