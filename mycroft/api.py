"""
Programmatic interface to the text classifier.
"""
from mycroft.text import EmbeddingsGenerator, text_parser


def train(texts, labels, label_names, validation_fraction,
          rnn_units, dropout, tokens_per_text,
          language_model,
          epochs, batch_size, model_filename):
    from mycroft.model import TextEmbeddingClassifier

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
    if tokens_per_text is None:
        tokens_per_text = EmbeddingsGenerator.maximum_tokens_per_text(texts, parser)
    training, validation = partition_data()

    model = TextEmbeddingClassifier.create(tokens_per_text, parser.vocab.vectors_length, rnn_units, dropout,
                                           label_names)
    return model.train(training, validation, epochs, model_filename)


def predict(model, texts, batch_size, language_model):
    embeddings = EmbeddingsGenerator(texts, model.tokens_per_text, batch_size, text_parser=text_parser(language_model))
    label_probabilities = model.predict(embeddings)
    predicted_labels = label_probabilities.argmax(axis=1)
    return label_probabilities, predicted_labels


def evaluate(model, texts, labels, batch_size, language_model):
    embeddings = EmbeddingsGenerator(texts, model.tokens_per_text, batch_size, labels=labels,
                                     text_parser=text_parser(language_model))
    return model.evaluate(embeddings)
