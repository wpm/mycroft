import os
import pickle
import shutil
import tempfile
from unittest import TestCase

import numpy
from mycroft.text import text_parser, BagOfWordsEmbedder, TextSequenceEmbedder, Embedder, text_statistics
from numpy.testing import assert_array_equal


class TestText(TestCase):
    def setUp(self):
        self.texts = ["The quick brown fox", "jumped over the lazy dog."]

    def test_text_parser(self):
        english_1 = text_parser("en")
        english_2 = text_parser("en")
        self.assertEqual(english_1, english_2)

    def test_text_statistics(self):
        longest_text, vocabulary_size = text_statistics(["Jabberwocky", "the cat is in the hat", "the dog is tired"])
        self.assertEqual(6, longest_text)
        # "Jabberwocky" does not have an embedding vector.
        self.assertEqual(7, vocabulary_size)

    def test_base_class(self):
        embedder = Embedder()
        with self.assertRaises(NotImplementedError):
            embedder.encode(["one two three"])

    def test_bag_of_words_embedder(self):
        embedder = BagOfWordsEmbedder()
        self.assertEqual("Bag of words embedder: core_web_sm", str(embedder))
        self.assertEqual("en", embedder.language_model)
        self.assertEqual(300, embedder.embedding_size)
        self.assertFalse(hasattr(embedder, "embedding_matrix"))
        embedding = embedder.encode(self.texts)
        self.assertEqual((2, 300), embedding.shape)
        self.assertEqual(numpy.dtype("float32"), embedding.dtype)

    def test_text_sequence_embedder(self):
        embedder = TextSequenceEmbedder(10000, 50)
        self.assertEqual("Text sequence embedder: core_web_sm, embedding matrix (10000, 300)",
                         str(embedder))
        self.assertEqual("en", embedder.language_model)
        self.assertEqual(300, embedder.embedding_size)
        self.assertEqual(10000, embedder.vocabulary_size)
        self.assertTrue(hasattr(embedder, "embedding_matrix"))
        self.assertEqual((10000, 300), embedder.embedding_matrix.shape)
        embedding = embedder.encode(self.texts)
        self.assertEqual((2, 50), embedding.shape)
        self.assertEqual(numpy.dtype("int32"), embedding.dtype)


class TestTextSerialization(TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.mkdtemp()
        self.texts = ["The quick brown fox", "jumped over the lazy dog."]

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)

    def test_serialize_bag_of_words_embedder(self):
        embedder_1 = BagOfWordsEmbedder()
        embedder_2 = self.serialization_round_trip(embedder_1, "bow.pk")
        self.assertEqual(embedder_1, embedder_2)
        embedding_1 = embedder_1.encode(self.texts)
        embedding_2 = embedder_2.encode(self.texts)
        assert_array_equal(embedding_1, embedding_2)

    def test_serialize_text_sequence_embedder(self):
        embedder_1 = TextSequenceEmbedder(10000, 50)
        embedder_2 = self.serialization_round_trip(embedder_1, "seq.pk")
        self.assertEqual(embedder_1, embedder_2)
        embedding_1 = embedder_1.encode(self.texts)
        embedding_2 = embedder_2.encode(self.texts)
        assert_array_equal(embedding_1, embedding_2)

    def serialization_round_trip(self, obj, name):
        with open(os.path.join(self.temporary_directory, name), mode="wb") as f:
            pickle.dump(obj, f)
        with open(os.path.join(self.temporary_directory, name), mode="rb") as f:
            return pickle.load(f)
