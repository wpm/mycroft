from __future__ import print_function

import argparse
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
    print(data.to_csv())


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


def main():
    parser = argparse.ArgumentParser(description="Text Classifier")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Commands")

    train_parser = subparsers.add_parser("train", description="Train a model")
    train_parser.add_argument("training", help="training data")
    train_parser.add_argument("--text-name", default="text", help="name of the text column (default 'text')")
    train_parser.add_argument("--label-name", default="label", help="name of the label column (default 'label')")
    train_parser.add_argument("--limit", type=int, help="limit the data to the specified length (default use all data)")
    train_parser.add_argument("--batch-size", type=int, default=56, help="batch size (default 256)")
    train_parser.add_argument("--epochs", type=int, default=10, help="training epochs (default 10)")
    train_parser.add_argument("--rnn-units", type=int, default=128, help="RNN units (default 128)")
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default 0.2)")
    train_parser.add_argument("--validation", type=float, help="portion of data to use for validation (default none)")
    train_parser.add_argument("--model-filename", help="file in which to to store the model")
    train_parser.set_defaults(
        func=lambda args: train(args.training, args.text_name, args.label_name, args.limit, args.batch_size,
                                args.epochs, args.rnn_units, args.dropout, args.validation, args.model_filename))

    predict_parser = subparsers.add_parser("predict", description="Use a model to predict labels")
    predict_parser.add_argument("test", help="test data")
    predict_parser.add_argument("model_filename", metavar="model-filename", help="file containing the trained model")
    predict_parser.add_argument("--text-name", default="text", help="name of the text column (default 'text')")
    predict_parser.add_argument("--limit", type=int,
                                help="limit the data to the specified length (default use all data)")
    predict_parser.set_defaults(func=lambda args: predict(args.test, args.model_filename, args.text_name, args.limit))

    args = parser.parse_args()
    args.func(args)
