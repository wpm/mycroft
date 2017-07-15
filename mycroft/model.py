"""Machine learning components"""
import os
import pickle
import sys
from io import StringIO

import h5py
import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from keras.models import load_model, Sequential

from mycroft.text import BagOfWordsEmbedder, TextSequenceEmbedder


class TextEmbeddingClassifier:
    model_name = "model.hd5"
    embedder_name = "embedder.pk"
    description_name = "description.txt"

    @classmethod
    def load_model(cls, model_directory):
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
        if model_directory is not None:
            os.makedirs(model_directory, exist_ok=True)
            monitor = "val_loss"
            callbacks = [ModelCheckpoint(filepath=os.path.join(model_directory, TextEmbeddingClassifier.model_name),
                                         monitor=monitor, save_best_only=True, verbose=verbose)]
            with open(os.path.join(model_directory, TextEmbeddingClassifier.embedder_name), mode="wb") as f:
                pickle.dump(self.embedder, f)
            with open(os.path.join(model_directory, TextEmbeddingClassifier.description_name), mode="w") as f:
                f.write("%s" % self)
        else:
            monitor = "loss"
            callbacks = None

        training_vectors = self.embedder.encode(texts)
        history = self.model.fit(training_vectors, labels, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_fraction, verbose=verbose, callbacks=callbacks)

        if model_directory is not None:
            with h5py.File(os.path.join(model_directory, TextEmbeddingClassifier.model_name)) as m:
                m.attrs["categories"] = numpy.array(
                    [numpy.string_(numpy.str_(label_name)) for label_name in self.label_names])
                m.attrs["language_model"] = numpy.string_(numpy.str_(self.embedder.language_model))

        history.monitor = monitor
        return history

    def predict(self, texts, batch_size=32):
        embeddings = self.embedder.encode(texts)
        label_probabilities = self.model.predict(embeddings, batch_size=batch_size)
        predicted_labels = label_probabilities.argmax(axis=1)
        return label_probabilities, predicted_labels

    def evaluate(self, texts, labels, batch_size=32):
        embeddings = self.embedder.encode(texts)
        metrics = self.model.evaluate(embeddings, labels, batch_size=batch_size)
        return list(zip(self.model.metrics_names, metrics))

    @property
    def rnn_units(self):
        return self.model.get_layer("rnn").layer.units

    @property
    def dropout(self):
        return self.model.get_layer("dropout").rate

    @property
    def embeddings_per_text(self):
        return self.model.get_layer("rnn").input_shape[1]

    @property
    def embedding_size(self):
        return self.model.get_layer("rnn").input_shape[2]

    @property
    def num_labels(self):
        return len(self.label_names)


class BagOfWordsEmbeddingClassifier(TextEmbeddingClassifier):
    @classmethod
    def create(cls, dropout, label_names, language_model="en"):
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
