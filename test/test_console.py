import os
import shutil
import tempfile
from unittest import TestCase

import pandas
import six

from mycroft.console import main
from test import to_lines


class TestConsole(TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        joyce_samples = to_lines("joyce.txt")
        joyce = pandas.DataFrame({"text": joyce_samples, "label": ["Joyce"] * len(joyce_samples)})
        kafka_samples = to_lines("kafka.txt")
        kafka = pandas.DataFrame({"text": kafka_samples, "label": ["Kafka"] * len(kafka_samples)})
        # noinspection PyUnresolvedReferences
        data = pandas.concat([joyce, kafka]).sample(frac=1)
        self.data_filename = os.path.join(self.directory, "data.csv")
        if six.PY3:
            data.to_csv(self.data_filename, index=False)
        else:
            data.to_csv(self.data_filename, index=False, encoding="UTF-8")
        self.model_directory = os.path.join(self.directory, "model")

    def tearDown(self):
        shutil.rmtree(self.directory)

    def test_nbow(self):
        self.run_command(
            "train-nbow %s --model-directory %s --logging none" % (self.data_filename, self.model_directory))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.hd5")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "classifier.pk")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "description.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "history.json")))
        self.run_command("predict %s %s" % (self.model_directory, self.data_filename))
        self.run_command("evaluate %s %s" % (self.model_directory, self.data_filename))

    def test_nseq(self):
        self.run_command(
            "train-nseq %s --model-directory %s --logging none" % (self.data_filename, self.model_directory))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "model.hd5")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "classifier.pk")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "description.txt")))
        self.assertTrue(os.path.isfile(os.path.join(self.model_directory, "history.json")))
        self.run_command("predict %s %s" % (self.model_directory, self.data_filename))
        self.run_command("evaluate %s %s" % (self.model_directory, self.data_filename))

    @staticmethod
    def run_command(s):
        return main(s.split())
