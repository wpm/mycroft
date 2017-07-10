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


def train(training_filename, limit, validation_fraction, text_name, label_name,
          rnn_units, dropout, tokens_per_text,
          language_model,
          epochs, batch_size, model_filename):
    from mycroft.model import TextEmbeddingClassifier
    from data import EmbeddingsGenerator

    def preprocess_data():
        data = read_data_file(training_filename, limit)
        data[label_name] = data[label_name].astype("category")
        labels = numpy.array(data[label_name].cat.codes)
        label_names = data[label_name].cat.categories
        return data[text_name], labels, label_names

    def partition_data():
        if validation_fraction:
            m = int((1 - validation_fraction) * len(texts))
            training_texts, training_labels = texts[:m], labels[:m]
            validation_texts, validation_labels = texts[m:], labels[m:]
            validation = EmbeddingsGenerator(validation_texts, tokens_per_text, batch_size, validation_labels, parser)
        else:
            training_texts, training_labels = texts, labels
            validation = None
        training = EmbeddingsGenerator(training_texts, tokens_per_text, batch_size, training_labels, parser)
        return training, validation

    parser = text_parser(language_model)
    texts, labels, label_names = preprocess_data()
    if tokens_per_text is None:
        tokens_per_text = EmbeddingsGenerator.maximum_tokens_per_text(texts, parser)
    training, validation = partition_data()

    model = TextEmbeddingClassifier.create(tokens_per_text, parser.vocab.vectors_length, rnn_units, dropout,
                                           label_names)
    print(repr(model))
    print(training)
    if validation is not None:
        print(validation)
    history = model.train(training, validation, epochs, model_filename)
    losses = history.history[history.monitor]
    best_loss = min(losses)
    best_epoch = losses.index(best_loss)
    s = " - ".join(["%s: %0.5f" % (score, values[best_epoch]) for score, values in sorted(history.history.items())])
    print("Best epoch %d of %d: %s" % (best_epoch + 1, epochs, s))


def predict(test_filename, model_filename, batch_size, text_name, limit, language_model):
    data, embeddings, model = model_and_test_embeddings(limit, model_filename, test_filename, batch_size, text_name,
                                                        None, language_model)
    label_probabilities = model.predict(embeddings)
    predicted_label = label_probabilities.argmax(axis=1)
    predictions = pandas.DataFrame(label_probabilities.reshape((len(data[text_name]), model.num_labels)),
                                   columns=model.class_names)
    predictions["predicted label"] = [model.class_names[i] for i in predicted_label]
    data = data.join(predictions)
    print(data.to_csv(index=False))


def evaluate(test_filename, model_filename, batch_size, text_name, label_name, limit, language_model):
    data, embeddings, model = model_and_test_embeddings(limit, model_filename, test_filename, batch_size, text_name,
                                                        label_name, language_model)
    print("\n" +
          " - ".join("%s: %0.5f" % (name, score) for name, score in model.evaluate(embeddings, data[label_name])))


def model_and_test_embeddings(limit, model_filename, test_filename, batch_size, text_name, label_name, language_model):
    from mycroft.model import TextEmbeddingClassifier
    from data import EmbeddingsGenerator
    model = TextEmbeddingClassifier.load_model(model_filename)
    data = read_data_file(test_filename, limit)
    if label_name is None:
        labels = None
    else:
        labels = data[label_name]
    embeddings = EmbeddingsGenerator(data[text_name], model.tokens_per_text, batch_size, labels=labels,
                                     text_parser=text_parser(language_model))
    return data, embeddings, model


def details(model_filename):
    from mycroft.model import TextEmbeddingClassifier
    print(TextEmbeddingClassifier.load_model(model_filename))


def read_data_file(data_filename, limit):
    return pandas.read_csv(data_filename, sep=None, engine="python").dropna()[:limit]
