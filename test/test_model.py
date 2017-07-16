# -*- coding: UTF-8 -*-
from __future__ import unicode_literals

import os
import shutil
import tempfile
import textwrap
from itertools import tee
from random import shuffle
from unittest import TestCase

import numpy
from keras.callbacks import History

from mycroft.model import TextEmbeddingClassifier, BagOfWordsEmbeddingClassifier, TextSequenceEmbeddingClassifier, \
    WordCountClassifier


class TestModel(TestCase):
    joyce = textwrap.dedent("""
    Stately, plump Buck Mulligan came from the stairhead, bearing a bowl of
    lather on which a mirror and a razor lay crossed. A yellow dressinggown,
    ungirdled, was sustained gently behind him on the mild morning air. He
    held the bowl aloft and intoned:
    
    —Introibo ad altare Dei.
    
    Halted, he peered down the dark winding stairs and called out coarsely:
    
    —Come up, Kinch! Come up, you fearful jesuit!
    
    Solemnly he came forward and mounted the round gunrest. He faced about
    and blessed gravely thrice the tower, the surrounding land and the
    awaking mountains. Then, catching sight of Stephen Dedalus, he bent
    towards him and made rapid crosses in the air, gurgling in his throat
    and shaking his head. Stephen Dedalus, displeased and sleepy, leaned
    his arms on the top of the staircase and looked coldly at the shaking
    gurgling face that blessed him, equine in its length, and at the light
    untonsured hair, grained and hued like pale oak.
    
    Buck Mulligan peeped an instant under the mirror and then covered the
    bowl smartly.
    
    —Back to barracks! he said sternly.
    """)

    kafka = textwrap.dedent("""
    One morning, when Gregor Samsa woke from troubled dreams, he found
    himself transformed in his bed into a horrible vermin.  He lay on
    his armour-like back, and if he lifted his head a little he could
    see his brown belly, slightly domed and divided by arches into stiff
    sections.  The bedding was hardly able to cover it and seemed ready
    to slide off any moment.  His many legs, pitifully thin compared
    with the size of the rest of him, waved about helplessly as he
    looked.
    
    "What's happened to me?" he thought.  It wasn't a dream.  His room,
    a proper human room although a little too small, lay peacefully
    between its four familiar walls.  A collection of textile samples
    lay spread out on the table - Samsa was a travelling salesman - and
    above it there hung a picture that he had recently cut out of an
    illustrated magazine and housed in a nice, gilded frame.  It showed
    a lady fitted out with a fur hat and fur boa who sat upright,
    raising a heavy fur muff that covered the whole of her lower arm
    towards the viewer.
    
    Gregor then turned to look out the window at the dull weather.
    Drops of rain could be heard hitting the pane, which made him feel
    quite sad.  "How about if I sleep a little bit longer and forget all
    this nonsense", he thought, but that was something he was unable to
    do because he was used to sleeping on his right, and in his present
    state couldn't get into that position.  However hard he threw
    himself onto his right, he always rolled back to where he was.  He
    must have tried it a hundred times, shut his eyes so that he
    wouldn't have to look at the floundering legs, and only stopped when
    he began to feel a mild, dull pain there that he had never felt
    before.""")

    @staticmethod
    def create_data_set():
        joyce_samples = TestModel.to_lines(TestModel.joyce)
        kafka_samples = TestModel.to_lines(TestModel.kafka)
        samples = [(s, "Joyce") for s in joyce_samples] + [(s, "Kafka") for s in kafka_samples]
        shuffle(samples)
        s1, s2 = tee(samples)
        texts = [s[0] for s in s1]
        labels = [s[1] for s in s2]
        return texts, labels, sorted(set(labels))

    @staticmethod
    def to_lines(text):
        return list(filter(None, text.split("\n")))

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
        model = TextSequenceEmbeddingClassifier.create(20000, 10, 32, 0.5, self.label_names)
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
