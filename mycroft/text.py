"""
Natural language processing components.
"""
from math import ceil

import numpy
from cytoolz import partition_all

text_parser_singleton = None


def text_parser(name="en"):
    global text_parser_singleton
    if text_parser_singleton is None:
        import spacy
        text_parser_singleton = spacy.load(name, tagger=None, parser=None, entity=None)
    return text_parser_singleton


class EmbeddingsGenerator:
    def __init__(self, texts, tokens_per_text, batch_size, labels=None, text_parser=text_parser()):
        self.texts = texts
        self.tokens_per_text = tokens_per_text
        self.batch_size = batch_size
        self.text_parser = text_parser
        self.labels = labels

    @staticmethod
    def maximum_tokens_per_text(texts, text_parser=text_parser()):
        return max(len(tokens) for tokens in text_parser.pipe(texts))

    def __len__(self):
        return len(self.texts)

    def __repr__(self):
        return "Embeddings iterator: embedding %d x %d, %d samples, batch size %d, %d batches per epoch, model %s" % (
            self.embedding_size[0], self.embedding_size[1], len(self), self.batch_size, self.batches_per_epoch,
            self.text_parser.meta["name"])

    def __call__(self):
        while True:
            if self.labels is None:
                batches = self.epoch()
            else:
                # noinspection PyArgumentList
                batches = zip(self.epoch(),
                              (numpy.array(label_batch) for label_batch in partition_all(self.batch_size, self.labels)))
            for batch in batches:
                yield batch

    def epoch(self):
        # noinspection PyArgumentList
        for batch in partition_all(self.batch_size, self.embeddings()):
            yield numpy.stack(batch)

    def embeddings(self):
        def uniform_length_embedding(tokens):
            embedding = numpy.array([token.vector for token in tokens])
            m = max(self.tokens_per_text - embedding.shape[0], 0)
            padded_embedding = numpy.pad(embedding[:self.tokens_per_text], ((0, m), (0, 0)), "constant")
            return padded_embedding

        for tokens in self.text_parser.pipe(self.texts):
            yield uniform_length_embedding(tokens)

    @property
    def embedding_size(self):
        return self.tokens_per_text, self.text_parser.vocab.vectors_length

    @property
    def batches_per_epoch(self):
        return ceil(len(self) / self.batch_size)
