"""Machine learning components"""
import sys
from io import StringIO

import h5py
import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import load_model, Sequential


class TextEmbeddingClassifier:
    @classmethod
    def create(cls, tokens_per_text, embedding_size, rnn_units, dropout, label_names):
        model = Sequential()
        model.add(Bidirectional(LSTM(rnn_units), input_shape=(tokens_per_text, embedding_size), name="rnn"))
        model.add(Dense(len(label_names), activation="softmax", name="softmax"))
        model.add(Dropout(dropout, name="dropout"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return cls(model, label_names)

    @classmethod
    def load_model(cls, model_filename):
        model = load_model(model_filename)
        with h5py.File(model_filename, "r") as m:
            label_names = [name.decode("UTF-8") for name in list(m.attrs["categories"])]
        return cls(model, label_names)

    def __init__(self, model, label_names):
        self.model = model
        self.label_names = label_names

    def __repr__(self):
        return "Text embedding classifier: embedding %d x %d, %d labels, %d RNN units, dropout rate %0.2f" % (
            self.tokens_per_text, self.embedding_size, self.num_labels, self.rnn_units, self.dropout)

    def __str__(self):
        def model_topology():
            # Keras' model summary prints to standard out.
            old_stdout = sys.stdout
            sys.stdout = s = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            return s.getvalue()

        return "%s\n\n%s" % (repr(self), model_topology())

    def train(self, training, validation, epochs, model_filename=None):
        if validation is not None:
            monitor = "val_loss"
        else:
            monitor = "loss"
        if model_filename is not None:
            callbacks = [ModelCheckpoint(filepath=model_filename, monitor=monitor, save_best_only=True, verbose=1)]
        else:
            callbacks = None

        history = self.model.fit_generator(training(), steps_per_epoch=training.batches_per_epoch,
                                           validation_data=validation(), validation_steps=validation.batches_per_epoch,
                                           epochs=epochs, callbacks=callbacks)

        if model_filename is not None:
            with h5py.File(model_filename) as m:
                m.attrs["categories"] = numpy.array(
                    [numpy.string_(numpy.str_(label_name)) for label_name in self.label_names])

        history.monitor = monitor
        return history

    def predict(self, embeddings):
        return self.model.predict_generator(embeddings(), embeddings.batches_per_epoch)

    def evaluate(self, embeddings):
        metrics = self.model.evaluate_generator(embeddings(), embeddings.batches_per_epoch)
        return list(zip(self.model.metrics_names, metrics))

    @property
    def rnn_units(self):
        return self.model.get_layer("rnn").layer.units

    @property
    def dropout(self):
        return self.model.get_layer("dropout").rate

    @property
    def tokens_per_text(self):
        return self.model.get_layer("rnn").input_shape[1]

    @property
    def embedding_size(self):
        return self.model.get_layer("rnn").input_shape[2]

    @property
    def num_labels(self):
        return len(self.label_names)
