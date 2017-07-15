"""
Natural language processing components.
"""
import operator

import numpy
from keras.preprocessing.sequence import pad_sequences


def longest_text(texts, language_model="en"):
    return max(len(document) for document in text_parser(language_model).pipe(texts))


class Embedder:
    def __init__(self, language_model="en"):
        self.text_parser = text_parser(language_model)

    def __repr__(self):
        return "%s: %s, embedding shape %s" % (
            self.__class__.__name__, self.text_parser.meta["name"], self.encoding_shape)

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
        raise NotImplementedError()

    @property
    def encoding_shape(self):
        raise NotImplementedError()


class BagOfWordsEmbedder(Embedder):
    def encode(self, texts):
        return numpy.array([document.vector for document in self.text_parser.pipe(texts)])

    @property
    def encoding_shape(self):
        return tuple((self.text_parser.vocab.vectors_length,))


class TextSequenceEmbedder(Embedder):
    def __init__(self, vocabulary_size, sequence_length, language_model="en"):
        def lexeme_embeddings(parser, vocabulary_size):
            lexemes = sorted((lexeme for lexeme in parser.vocab if lexeme.has_vector),
                             key=operator.attrgetter("rank"))[:vocabulary_size - 1]
            for i, lexeme in enumerate(lexemes, 1):
                yield i, lexeme.orth_, lexeme.vector

        super().__init__(text_parser(language_model))
        self.sequence_length = sequence_length
        self.embedding_matrix = numpy.zeros((vocabulary_size, self.text_parser.vocab.vectors_length))
        self.vocabulary = {}
        for index, token, vector in lexeme_embeddings(self.text_parser, vocabulary_size):
            self.embedding_matrix[index] = vector
            self.vocabulary[token] = index

    def __repr__(self):
        return super().__repr__() + ", embedding matrix %s" % (self.embedding_matrix.shape,)

    def encode(self, texts):
        token_index_sequences = pad_sequences(
            list([self.vocabulary.get(token.orth_, 0) for token in document]
                 for document in self.text_parser.pipe(texts)),
            maxlen=self.sequence_length)
        return numpy.array(token_index_sequences)

    @property
    def vocabulary_size(self):
        return self.embedding_matrix.shape[0]

    @property
    def encoding_shape(self):
        return tuple((self.sequence_length,))


text_parser_singleton = None


def text_parser(language_model):
    global text_parser_singleton
    if text_parser_singleton is None:
        import spacy
        text_parser_singleton = spacy.load(language_model, tagger=None, parser=None, entity=None)
    return text_parser_singleton
