from __future__ import print_function

import argparse

import h5py
import numpy
import pandas

__version__ = "1.0.0"


def train(training_filename, limit, batch_size, epochs, rnn_units, dropout, validation, model_filename):
    from keras.callbacks import ModelCheckpoint
    from keras.layers import LSTM, Bidirectional, Dense, Dropout
    from keras.models import Sequential

    # Read in the training data.
    data = pandas.read_csv(training_filename)[:limit]
    data.label = data.label.astype("category")
    classes = len(data.label.unique())

    # Get vector embeddings for the training text.
    max_tokens, embedding_size, data = get_embeddings(data)

    # Construct the model topology.
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_units), input_shape=(max_tokens, embedding_size), name="rnn"))
    model.add(Dense(classes, activation="softmax"))
    model.add(Dropout(dropout))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model.
    embeddings = numpy.array(numpy.array(list(data.embedding)))
    labels = numpy.array(data.label.cat.codes)
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
            m.attrs["categories"] = numpy.array([numpy.string_(numpy.str_(s)) for s in data.label.cat.categories])

    # Print training history information.
    losses = history.history[monitor]
    best_loss = min(losses)
    print("Best loss %0.5f in epoch %d of %d" % (best_loss, losses.index(best_loss) + 1, epochs))


def predict(test_filename, model_filename, limit):
    from keras.models import load_model

    # Read in the test data.
    data = pandas.read_csv(test_filename)[:limit]

    # Load the model.
    model = load_model(model_filename)
    with h5py.File(model_filename, "r") as m:
        categories = [c.decode("UTF-8") for c in list(m.attrs["categories"])]

    # Get vector embeddings for the test text.
    _, __, data = get_embeddings(data, model.get_layer("rnn").input_shape[1])

    # Use the model to make predictions.
    embeddings = numpy.stack(data.embedding)
    label_probabilities = model.predict(embeddings)
    data = data.drop(["embedding"], axis=1)

    # Add the predictions to the input CSV and print it.
    predictions = pandas.DataFrame(label_probabilities.reshape((len(data), len(categories))), columns=categories)
    data = data.join(predictions)
    print(data.to_csv())


def get_embeddings(data, max_tokens=None):
    def padded_segment_embedding(parsed_segment):
        embedding = numpy.array([token.vector for token in parsed_segment])
        m = max(max_tokens - embedding.shape[0], 0)
        padded_embedding = numpy.pad(embedding[:max_tokens], ((0, m), (0, 0)), "constant")
        return padded_embedding

    import spacy
    nlp = spacy.load("en", tagger=None, parser=None, entity=None)
    data["parsed"] = list(nlp.pipe(data.text, n_threads=-1))
    max_tokens = max_tokens or max(data.parsed.apply(lambda p: len(p)))
    data["embedding"] = data.parsed.apply(padded_segment_embedding)
    return max_tokens, nlp.vocab.vectors_length, data.drop(["parsed"], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Text Classifier")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    parser.set_defaults(func=lambda _: parser.print_usage())

    subparsers = parser.add_subparsers(title="Commands")

    train_parser = subparsers.add_parser("train", description="Train a model")
    train_parser.add_argument("training", help="training data")
    train_parser.add_argument("--limit", type=int, help="limit the data to the specified length (default use all data)")
    train_parser.add_argument("--batch-size", type=int, default=56, help="batch size (default 256)")
    train_parser.add_argument("--epochs", type=int, default=10, help="training epochs (default 10)")
    train_parser.add_argument("--rnn-units", type=int, default=128, help="RNN units (default 128)")
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default 0.2)")
    train_parser.add_argument("--validation", type=float, help="portion of data to use for validation (default none)")
    train_parser.add_argument("--model-filename", help="file in which to to store the model")
    train_parser.set_defaults(
        func=lambda args: train(args.training, args.limit, args.batch_size, args.epochs, args.rnn_units, args.dropout,
                                args.validation, args.model_filename))

    predict_parser = subparsers.add_parser("predict", description="Use a model to predict labels")
    predict_parser.add_argument("test", help="test data")
    predict_parser.add_argument("model_filename", metavar="model-filename", help="file containing the trained model")
    predict_parser.add_argument("--limit", type=int,
                                help="limit the data to the specified length (default use all data)")
    predict_parser.set_defaults(func=lambda args: predict(args.test, args.model_filename, args.limit))

    args = parser.parse_args()
    args.func(args)
