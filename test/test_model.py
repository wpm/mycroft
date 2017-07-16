from __future__ import unicode_literals

import os
import shutil
import tempfile
from itertools import tee
from random import shuffle
from unittest import TestCase

import numpy
from keras.callbacks import History

from mycroft.model import TextEmbeddingClassifier, BagOfWordsEmbeddingClassifier, TextSequenceEmbeddingClassifier, \
    WordCountClassifier
from test import to_lines


class TestModel(TestCase):
    @staticmethod
    def create_data_set():
        joyce_samples = to_lines("joyce.txt")
        kafka_samples = to_lines("kafka.txt")
        samples = [(s, "Joyce") for s in joyce_samples] + [(s, "Kafka") for s in kafka_samples]
        shuffle(samples)
        s1, s2 = tee(samples)
        texts = [s[0] for s in s1]
        labels = [s[1] for s in s2]
        return texts, labels, sorted(set(labels))

    def setUp(self):
        self.model_directory = tempfile.mkdtemp()
        self.texts, self.labels, self.label_names = self.create_data_set()

    def tearDown(self):
        shutil.rmtree(self.model_directory)

    def test_bag_of_words(self):
        model = BagOfWordsEmbeddingClassifier.create(0.5, self.label_names)
        self.assertEqual(2, model.num_labels)
        self.assertEqual(0.5, model.dropout)
        self.embedding_model_train_predict_evaluate(model)
        self.embedding_model_train_without_validation(model)

    def test_text_sequence(self):
        model = TextSequenceEmbeddingClassifier.create(20000, 10, "lstm", 32, 0.5, self.label_names)
        self.assertEqual(2, model.num_labels)
        self.assertEqual(0.5, model.dropout)
        self.assertEqual(10, model.embeddings_per_text)
        self.assertEqual(300, model.embedding_size)
        self.assertEqual(32, model.rnn_units)
        self.embedding_model_train_predict_evaluate(model)

    def embedding_model_train_predict_evaluate(self, model):
        # Train
        history = model.train(self.texts, self.labels, epochs=2, batch_size=10, validation_fraction=0.1,
                              model_directory=self.model_directory, verbose=0)
        self.assertIsInstance(history, History)
        self.assertTrue(os.path.exists(os.path.join(self.model_directory, "model.hd5")))
        self.assertTrue(os.path.exists(os.path.join(self.model_directory, "embedder.pk")))
        self.assertTrue(os.path.exists(os.path.join(self.model_directory, "description.txt")))
        # Predict
        loaded_model = TextEmbeddingClassifier.load_model(self.model_directory)
        n = len(self.texts)
        label_probabilities, predicted_labels = loaded_model.predict(self.texts)
        self.assertEqual((n, 2), label_probabilities.shape)
        self.assertEqual(numpy.dtype("float32"), label_probabilities.dtype)
        self.assertEqual(n, len(predicted_labels))
        self.assertTrue(set(predicted_labels).issubset({"Joyce", "Kafka"}))
        # Evaluate
        scores = loaded_model.evaluate(self.texts, self.labels)
        self.is_loss_and_accuracy(scores)

    def embedding_model_train_without_validation(self, model):
        history = model.train(self.texts, self.labels, epochs=2, batch_size=10, model_directory=self.model_directory,
                              verbose=0)
        self.assertIsInstance(history, History)
        self.assertTrue(os.path.exists(os.path.join(self.model_directory, "model.hd5")))
        self.assertTrue(os.path.exists(os.path.join(self.model_directory, "embedder.pk")))
        self.assertTrue(os.path.exists(os.path.join(self.model_directory, "description.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.model_directory, "history.json")))

    def test_word_count(self):
        model_filename = os.path.join(self.model_directory, "word-count.pk")
        model = WordCountClassifier(self.label_names)
        self.assertEqual("SVM TF-IDF classifier: 2 labels", str(model))
        # Train
        model.train(self.texts, self.labels, validation_fraction=0.1, model_filename=model_filename)
        self.assertTrue(os.path.exists(model_filename))
        # Predict
        loaded_model = WordCountClassifier.load_model(model_filename)
        n = len(self.texts)
        label_probabilities, predicted_labels = loaded_model.predict(self.texts)
        self.assertEqual((n, 2), label_probabilities.shape)
        self.assertEqual(numpy.dtype("float64"), label_probabilities.dtype)
        self.assertEqual(n, len(predicted_labels))
        self.assertTrue(set(predicted_labels).issubset({"Joyce", "Kafka"}))
        # Evaluate
        scores = loaded_model.evaluate(self.texts, self.labels)
        self.is_loss_and_accuracy(scores)

    def test_word_count_no_validation(self):
        model = WordCountClassifier(self.label_names)
        validation_results = model.train(self.texts, self.labels, validation_fraction=0.1)
        self.is_loss_and_accuracy(validation_results)
        validation_results = model.train(self.texts, self.labels)
        self.assertEqual(None, validation_results)

    def is_loss_and_accuracy(self, scores):
        self.assertIsInstance(scores, list)
        self.assertEqual(2, len(scores))
        loss = [s[1] for s in scores if s[0] == "loss"][0]
        self.assertIsInstance(loss, float)
        acc = [s[1] for s in scores if s[0] == "acc"][0]
        self.assertIsInstance(acc, float)
