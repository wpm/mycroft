from __future__ import print_function

from itertools import tee

import h5py
import numpy
import pandas

__version__ = "1.0.0"


def train(training_filename, text_name, label_name, limit, batch_size, epochs, rnn_units, dropout, validation,
          model_filename):
    # Read in the training data.
    data = read_data_file(training_filename, limit)
    texts, labels, label_names = preprocess_training_data(data, text_name, label_name)
    # Get vector embeddings for the training text.
    embeddings, max_tokens, embedding_size = embed_text(texts)
    # Train and save the model.
    losses = train_model(batch_size, dropout, embedding_size, embeddings, epochs, label_names, labels, max_tokens,
                         model_filename, rnn_units, validation)
    # Pint history results.
    best_loss = min(losses)
    print("Best loss %0.5f in epoch %d of %d" % (best_loss, losses.index(best_loss) + 1, epochs))


def predict(test_filename, model_filename, text_name, limit):
    # Read in the test data.
    data = read_data_file(test_filename, limit)
    texts = data[text_name]
    # Load the trained model.
    model, categories, max_tokens = load_text_classifier_model(model_filename)
    # Get vector embeddings for the test text.
    embeddings = embed_text(texts, max_tokens)[0]
    # Use the model to make predictions.
    label_probabilities = model.predict(embeddings)
    # Add the predictions to the input CSV and print it.
    predictions = pandas.DataFrame(label_probabilities.reshape((len(texts), len(categories))), columns=categories)
    data = data.join(predictions)
    print(data.to_csv(index=False))


def read_data_file(data_filename, limit):
    return pandas.read_csv(data_filename, sep=None, engine="python")[:limit]


def preprocess_training_data(data, text_name="text", label_name="label"):
    data[label_name] = data[label_name].astype("category")
    labels = numpy.array(data[label_name].cat.codes)
    label_names = data[label_name].cat.categories
    return data[text_name], labels, label_names


def embed_text(text, max_tokens=None):
    def padded_segment_embedding(parsed_segment):
        embedding = numpy.array([token.vector for token in parsed_segment])
        m = max(max_tokens - embedding.shape[0], 0)
        padded_embedding = numpy.pad(embedding[:max_tokens], ((0, m), (0, 0)), "constant")
        return padded_embedding

    import spacy
    nlp = spacy.load("en", tagger=None, parser=None, entity=None)
    parsed_1, parsed_2 = tee(nlp.pipe(text))
    if max_tokens is None:
        max_tokens = max(len(parse) for parse in parsed_1)
    embeddings = numpy.stack([padded_segment_embedding(parse) for parse in parsed_2])
    return embeddings, max_tokens, nlp.vocab.vectors_length


def train_model(batch_size, dropout, embedding_size, embeddings, epochs, label_names, labels, max_tokens,
                model_filename, rnn_units, validation):
    from keras.callbacks import ModelCheckpoint
    from keras.layers import LSTM, Bidirectional, Dense, Dropout
    from keras.models import Sequential

    # Construct the model topology.
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_units), input_shape=(max_tokens, embedding_size), name="rnn"))
    model.add(Dense(len(label_names), activation="softmax"))
    model.add(Dropout(dropout))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model.
    if validation is not None:
        monitor = "val_loss"
    else:
        monitor = "loss"
    if model_filename is not None:
        callbacks = [ModelCheckpoint(filepath=model_filename, monitor=monitor, save_best_only=True, verbose=1)]
    else:
        callbacks = None
    history = model.fit(embeddings, labels, epochs=epochs, batch_size=batch_size, validation_split=validation,
                        callbacks=callbacks)
    # Write category labels to the saved model.
    if model_filename is not None:
        with h5py.File(model_filename) as m:
            m.attrs["categories"] = numpy.array([numpy.string_(numpy.str_(label_name)) for label_name in label_names])
    # Return losses for each epoch.
    return history.history[monitor]


def load_text_classifier_model(model_filename):
    from keras.models import load_model

    model = load_model(model_filename)
    with h5py.File(model_filename, "r") as m:
        categories = [name.decode("UTF-8") for name in list(m.attrs["categories"])]
    max_tokens = model.get_layer("rnn").input_shape[1]
    return model, categories, max_tokens
