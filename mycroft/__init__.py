from __future__ import print_function

import numpy
import pandas

__version__ = "1.0.0"

text_parser_singleton = None


def text_parser(name="en"):
    global text_parser_singleton
    if text_parser_singleton is None:
        import spacy
        text_parser_singleton = spacy.load(name, tagger=None, parser=None, entity=None)
    return text_parser_singleton


def train(training_filename, limit, validation, text_name, label_name,
          rnn_units, dropout, max_tokens,
          language_model,
          epochs, batch_size, model_filename):
    from mycroft.model import TextSetEmbedder, TextEmbeddingClassifier

    def preprocess_training_data():
        data[label_name] = data[label_name].astype("category")
        labels = numpy.array(data[label_name].cat.codes)
        label_names = data[label_name].cat.categories
        return data[text_name], labels, label_names

    data = read_data_file(training_filename, limit)
    texts, classes, class_names = preprocess_training_data()
    embedder = TextSetEmbedder(text_parser(language_model))
    embeddings, max_tokens_per_text = embedder(texts, max_tokens_per_text=max_tokens)
    model = TextEmbeddingClassifier.create(max_tokens_per_text, embedder.embedding_size, rnn_units, dropout,
                                           class_names)
    print(repr(model))
    history = model.train(embeddings, classes, validation, epochs, batch_size, model_filename)
    losses = history.history[history.monitor]
    best_loss = min(losses)
    best_epoch = losses.index(best_loss)
    s = " - ".join(["%s: %0.5f" % (score, values[best_epoch]) for score, values in sorted(history.history.items())])
    print("Best epoch %d of %d: %s" % (best_epoch + 1, epochs, s))


def predict(test_filename, model_filename, text_name, limit, language_model):
    data, embeddings, model = model_and_test_embeddings(limit, model_filename, test_filename, text_name, language_model)
    label_probabilities = model.predict(embeddings)
    predicted_label = label_probabilities.argmax(axis=1)
    predictions = pandas.DataFrame(label_probabilities.reshape((len(data[text_name]), model.classes)),
                                   columns=model.class_names)
    predictions["predicted label"] = [model.class_names[i] for i in predicted_label]
    data = data.join(predictions)
    print(data.to_csv(index=False))


def evaluate(test_filename, model_filename, text_name, label_name, limit, language_model):
    data, embeddings, model = model_and_test_embeddings(limit, model_filename, test_filename, text_name, language_model)
    print("\n" +
          " - ".join("%s: %0.5f" % (name, score) for name, score in model.evaluate(embeddings, data[label_name])))


def model_and_test_embeddings(limit, model_filename, test_filename, text_name, language_model):
    from mycroft.model import TextSetEmbedder, TextEmbeddingClassifier
    data = read_data_file(test_filename, limit)
    embedder = TextSetEmbedder(text_parser(language_model))
    model = TextEmbeddingClassifier.load_model(model_filename)
    embeddings, _ = embedder(data[text_name], max_tokens_per_text=model.max_tokens_per_text)
    return data, embeddings, model


def details(model_filename):
    from mycroft.model import TextEmbeddingClassifier
    print(TextEmbeddingClassifier.load_model(model_filename))


def read_data_file(data_filename, limit):
    return pandas.read_csv(data_filename, sep=None, engine="python")[:limit]
