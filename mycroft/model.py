from itertools import tee

import h5py
import numpy
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Bidirectional, Dense, Dropout
from keras.models import load_model, Sequential


class TextEmbeddingClassifier:
    @classmethod
    def load_model(cls, model_filename):
        model = load_model(model_filename)
        with h5py.File(model_filename, "r") as m:
            label_names = [name.decode("UTF-8") for name in list(m.attrs["categories"])]
        return cls(model, label_names)

    @classmethod
    def create(cls, max_tokens_per_text, embedding_size, rnn_units, dropout, class_names):
        model = Sequential()
        model.add(
            Bidirectional(LSTM(rnn_units),
                          input_shape=(max_tokens_per_text, embedding_size), name="rnn"))
        model.add(Dense(len(class_names), activation="softmax"))
        model.add(Dropout(dropout))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return cls(model, class_names)

    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names

    def __repr__(self):
        return "Text classifier: %d classes" % self.classes

    @property
    def max_tokens(self):
        return self.model.get_layer("rnn").input_shape[1]

    @property
    def classes(self):
        return len(self.class_names)

    def train(self, embeddings, class_labels, validation, epochs, batch_size, model_filename):
        if validation is not None:
            monitor = "val_loss"
        else:
            monitor = "loss"
        if model_filename is not None:
            callbacks = [ModelCheckpoint(filepath=model_filename, monitor=monitor, save_best_only=True, verbose=1)]
        else:
            callbacks = None

        x = numpy.stack([embeddings])
        y = numpy.array(class_labels)
        history = self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=validation,
                                 callbacks=callbacks)

        if model_filename is not None:
            with h5py.File(model_filename) as m:
                m.attrs["categories"] = numpy.array(
                    [numpy.string_(numpy.str_(label_name)) for label_name in self.class_names])

        return history.history[monitor]

    def predict(self, embeddings):
        return self.model.predict(embeddings)


class TextSetEmbedder:
    def __init__(self, text_parser):
        self.text_parser = text_parser

    def __call__(self, texts, max_tokens_per_text=None):
        def padded_segment_embedding(parsed_text):
            embedding = numpy.array([token.vector for token in parsed_text])
            m = max(max_tokens_per_text - embedding.shape[0], 0)
            padded_embedding = numpy.pad(embedding[:max_tokens_per_text], ((0, m), (0, 0)), "constant")
            return padded_embedding

        parsed_texts = self.text_parser.pipe(texts)
        if max_tokens_per_text is not None:
            a = parsed_texts
        else:
            a, b = tee(parsed_texts)
            max_tokens_per_text = max(len(parse) for parse in b)
        embeddings = (padded_segment_embedding(parse) for parse in a)
        return embeddings, max_tokens_per_text

    @property
    def embedding_size(self):
        return self.text_parser.vocab.vectors_length
