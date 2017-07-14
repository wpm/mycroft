import os
import pickle
import shutil
import tempfile
from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal

from text import text_parser, BagOfWordsEmbedder, TextSequenceEmbedder


class TestText(TestCase):
    def test_text_parser(self):
        english_1 = text_parser("en")
        english_2 = text_parser("en")
        self.assertEqual(english_1, english_2)

    def test_bag_of_words_embedder(self):
        embedder = BagOfWordsEmbedder()
        self.assertEqual("BagOfWordsEmbedder: core_web_sm, embedding shape (300,)", str(embedder))
        self.assertEqual("en", embedder.language_model)
        self.assertEqual((300,), embedder.encoding_shape)
        self.assertEqual(300, embedder.embedding_size)
        self.assertFalse(hasattr(embedder, "embedding_matrix"))
        embedding = embedder.encode(["The quick brown fox", "jumped over the lazy dog."])
        self.assertEqual((2, 300), embedding.shape)
        self.assertEqual(numpy.dtype("float32"), embedding.dtype)

    def test_text_sequence_embedder(self):
        embedder = TextSequenceEmbedder(10000, 50)
        self.assertEqual("TextSequenceEmbedder: core_web_sm, embedding shape (50,), embedding matrix (10000, 300)",
                         str(embedder))
        self.assertEqual("en", embedder.language_model)
        self.assertEqual((50,), embedder.encoding_shape)
        self.assertEqual(300, embedder.embedding_size)
        self.assertEqual(10000, embedder.vocabulary_size)
        self.assertTrue(hasattr(embedder, "embedding_matrix"))
        self.assertEqual((10000, 300), embedder.embedding_matrix.shape)
        embedding = embedder.encode(["The quick brown fox", "jumped over the lazy dog."])
        self.assertEqual((2, 50), embedding.shape)
        self.assertEqual(numpy.dtype("int32"), embedding.dtype)


class TestTextSerialization(TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)

    def test_serialize_bag_of_words_embedder(self):
        embedder_1 = BagOfWordsEmbedder()
        embedder_2 = self.serialization_round_trip(embedder_1, "bow.pk")
        texts = ["The quick brown fox", "jumped over the lazy dog."]
        embedding_1 = embedder_1.encode(texts)
        embedding_2 = embedder_2.encode(texts)
        assert_array_equal(embedding_1, embedding_2)

    def test_serialize_text_sequence_embedder(self):
        embedder_1 = TextSequenceEmbedder(10000, 50)
        embedder_2 = self.serialization_round_trip(embedder_1, "seq.pk")
        texts = ["The quick brown fox", "jumped over the lazy dog."]
        embedding_1 = embedder_1.encode(texts)
        embedding_2 = embedder_2.encode(texts)
        assert_array_equal(embedding_1, embedding_2)

    def serialization_round_trip(self, obj, name):
        with open(os.path.join(self.temporary_directory, name), mode="wb") as f:
            pickle.dump(obj, f)
        with open(os.path.join(self.temporary_directory, name), mode="rb") as f:
            return pickle.load(f)
