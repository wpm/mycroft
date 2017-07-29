"""
Natural language processing components.
"""
import operator
from functools import partial

import numpy


def longest_text(texts, language_model="en"):
    """
    :param texts: texts in training data
    :type texts: sequence of str
    :param language_model: spaCy language model name
    :type language_model: str
    :return: the number of tokens in the longest text in the data
    :rtype: int
    """
    return max(len(document) for document in text_parser(language_model).pipe(texts))


class Embedder:
    """
    Base class of classes that convert text to continuous vector embeddings. Derived classes must implement the encode
    function.

    Embedders use the spaCy package to process the text and map it to embedding vectors.
    """

    def __init__(self, language_model="en"):
        """
        :param language_model: the name of the spaCy language model to use
        :type language_model: str
        """
        self.text_parser = text_parser(language_model)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["text_parser"] = self.language_model
        return d

    def __setstate__(self, d):
        d["text_parser"] = text_parser(d["text_parser"])
        self.__dict__.update(d)

    @property
    def language_model(self):
        return self.text_parser.vocab.lang

    @property
    def embedding_size(self):
        return self.text_parser.vocab.vectors_length

    def encode(self, texts):
        """
        Encode a sequence of texts as distributed vectors

        :param texts: texts to encode
        :type texts: sequence of str
        :return: text encodings
        :rtype: numpy.array
        """
        raise NotImplementedError()


class BagOfWordsEmbedder(Embedder):
    """
    Encode a sequence of words as a single vector that is mean of their embeddings.
    """

    def encode(self, texts):
        return numpy.array([document.vector for document in self.text_parser.pipe(texts)])

    def __repr__(self):
        return "Bag of words embedder: %s" % (self.text_parser.meta["name"])


class TextSequenceEmbedder(Embedder):
    """
    Encode a sequence of words as a matrix of their embeddings.
    """

    def __init__(self, vocabulary_size, sequence_length, language_model="en"):
        def lexeme_embeddings(parser, vocabulary_size):
            lexemes = sorted((lexeme for lexeme in parser.vocab if lexeme.has_vector),
                             key=operator.attrgetter("rank"))[:vocabulary_size - 1]
            for i, lexeme in enumerate(lexemes, 1):
                yield i, lexeme.orth_, lexeme.vector

        super(self.__class__, self).__init__(language_model)
        self.sequence_length = sequence_length
        self.embedding_matrix = numpy.zeros((vocabulary_size, self.text_parser.vocab.vectors_length))
        self.vocabulary = {}
        for index, token, vector in lexeme_embeddings(self.text_parser, vocabulary_size):
            self.embedding_matrix[index] = vector
            self.vocabulary[token] = index

    def encode(self, texts):
        from keras.preprocessing.sequence import pad_sequences

        token_index_sequences = pad_sequences(
            list([self.vocabulary.get(token.orth_, 0) for token in document]
                 for document in self.text_parser.pipe(texts)),
            maxlen=self.sequence_length)
        return numpy.array(token_index_sequences)

    def embedding_layer_factory(self):
        from keras.layers import Embedding
        return partial(Embedding, self.vocabulary_size, self.embedding_size, weights=[self.embedding_matrix])

    def __repr__(self):
        return "Text sequence embedder: %s, embedding matrix %s" % (
            self.text_parser.meta["name"], self.embedding_matrix.shape)

    @property
    def vocabulary_size(self):
        return self.embedding_matrix.shape[0]


text_parser_singletons = {}


def text_parser(language_model, tagger=None, parser=None, entity=None):
    global text_parser_singletons
    if language_model not in text_parser_singletons:
        import spacy
        text_parser_singletons[language_model] = spacy.load(language_model, tagger=tagger, parser=parser, entity=entity)
    return text_parser_singletons[language_model]
