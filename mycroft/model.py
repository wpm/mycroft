"""Machine learning components"""
import argparse
import inspect
import json
import os
import pickle
import sys
from io import StringIO

from .text import longest_text


def load_embedding_model(model_directory):
    """
    Load a trained model.

    :param model_directory: directory in which the model was saved
    :type model_directory: str
    :return: the model
    :rtype: TextEmbeddingClassifier
    """
    from keras.models import load_model

    with open(os.path.join(model_directory, TextEmbeddingClassifier.classifier_name), mode="rb") as f:
        model = pickle.load(f)
    model.model = load_model(os.path.join(model_directory, TextEmbeddingClassifier.model_name))
    return model


class TextEmbeddingClassifier:
    """
    Base class for models that can do text classification using text vector embeddings.

    Derived classes must define a constructor that takes a training data argument followed by model hyper-parameters.
    The training data argument is a 3-ple (texts, labels, label names). The label names must be passed up to this
    constructor. The texts and labels are present in case they are needed to determine hyper-parameters. (See
    TextSequenceEmbeddingClassifier for an example of this.

    When one of these classes is passed a part of the model_specification argument to mycroft.console.main, Mycroft will
    create command line arguments for all the hyper-parameter arguments in the construction. Positional constructor
    arguments will be positional command line arguments, and keyword constructor arguments will be command line options.
    Derived classes may optionally supply a CUSTOM_COMMAND_LINE_OPTIONS dictionary, which specifies additional keyword
    arguments to provide to the argparse.addArgument command.
    """
    # Names of files created in the model directory.
    model_name = "model.hd5"
    classifier_name = "classifier.pk"
    description_name = "description.txt"
    history_name = "history.json"

    CUSTOM_COMMAND_LINE_OPTIONS = {}

    def __init__(self, model, embedder, label_names):
        """
        Derived classes instantiate an embedder and compile a model, which are passed up to here.

        :param model: the model
        :type model: keras.models.Sequential
        :param embedder: embedder that converts text to vector embeddings
        :type embedder:  mycroft.text.Embedder
        :param label_names: all the label names in the data
        :type label_names: list of str
        """
        assert len(label_names) == len(set(label_names)), "Non-unique label names %s" % label_names
        self.model = model
        self.embedder = embedder
        self.label_names = label_names

    def __str__(self):
        def model_topology():
            old_stdout = sys.stdout
            sys.stdout = s = StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            return s.getvalue()

        return "%s\n\n%s" % (repr(self), model_topology())

    def __getstate__(self):
        d = self.__dict__.copy()
        del d["model"]
        return d

    def train(self, texts, labels, epochs=10, batch_size=32, validation_fraction=None, validation_data=None,
              model_directory=None, tensor_board_directory=None, verbose=1):
        def model_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.model_name)

        def classifier_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.classifier_name)

        def description_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.description_name)

        def history_filename():
            return os.path.join(model_directory, TextEmbeddingClassifier.history_name)

        def create_directory(directory):
            os.makedirs(directory, exist_ok=True)

        assert not (
            validation_fraction and validation_data), "Both validation fraction and validation data are specified"
        doing_validation = validation_fraction or validation_data
        if doing_validation:
            monitor = "val_loss"
            if validation_data:
                validation_data = (self.embedder.encode(validation_data[0]), self.label_indexes(validation_data[1]))
        else:
            monitor = "loss"
        if tensor_board_directory:
            from keras.callbacks import TensorBoard

            callbacks = [TensorBoard(log_dir=tensor_board_directory)]
        else:
            callbacks = []
        if model_directory is not None:
            create_directory(model_directory)
            if doing_validation:
                from keras.callbacks import ModelCheckpoint
                # noinspection PyTypeChecker
                callbacks.append(
                    ModelCheckpoint(filepath=os.path.join(model_directory, TextEmbeddingClassifier.model_name),
                                    monitor=monitor, save_best_only=True, verbose=verbose))
            with open(description_filename(), mode="w") as f:
                f.write("%s" % self)

        training_vectors = self.embedder.encode(texts)
        labels = self.label_indexes(labels)
        history = self.model.fit(training_vectors, labels, epochs=epochs, batch_size=batch_size,
                                 validation_split=validation_fraction, validation_data=validation_data,
                                 verbose=verbose, callbacks=callbacks)
        history.monitor = monitor

        if model_directory is not None:
            if not os.path.isfile(model_filename()):
                self.model.save(model_filename())
            with open(classifier_filename(), mode="wb") as f:
                pickle.dump(self, f)
            with open(history_filename(), mode="w") as f:
                h = {"epoch": history.epoch, "history": history.history, "monitor": history.monitor,
                     "params": history.params}
                json.dump(h, f, sort_keys=True, indent=4, separators=(",", ": "))

        return history

    def predict(self, texts, batch_size=32):
        embeddings = self.embedder.encode(texts)
        label_probabilities = self.model.predict(embeddings, batch_size=batch_size, verbose=0)
        predicted_labels = label_probabilities.argmax(axis=1)
        return label_probabilities, [self.label_names[label_index] for label_index in predicted_labels]

    def evaluate(self, texts, labels, batch_size=32):
        embeddings = self.embedder.encode(texts)
        labels = self.label_indexes(labels)
        metrics = self.model.evaluate(embeddings, labels, batch_size=batch_size, verbose=0)
        return list(zip(self.model.metrics_names, metrics))

    def label_indexes(self, labels):
        return [self.label_names.index(label) for label in labels]

    @property
    def num_labels(self):
        return len(self.label_names)

    @classmethod
    def create_from_command_line_arguments(cls, training, command_line_arguments):
        args = command_line_arguments.__dict__
        spec = inspect.getfullargspec(cls.__init__)
        filtered_args = dict((k, v) for k, v in args.items() if k in spec.args)
        return cls(tuple(training), **filtered_args)

    @classmethod
    def command_line_arguments(cls, parser=argparse.ArgumentParser(add_help=False)):
        def additional_options(name):
            options = cls.CUSTOM_COMMAND_LINE_OPTIONS.get(name, {})
            # The type= argument is handled by custom_type.
            options.pop("type", None)
            return options

        def custom_type(name, default):
            if name in cls.CUSTOM_COMMAND_LINE_OPTIONS:
                return cls.CUSTOM_COMMAND_LINE_OPTIONS.get("type", type(default))
            return type(default)

        spec = inspect.getfullargspec(cls.__init__)
        # Build command line arguments out of keyword arguments to the constructor and their defaults.
        for name, default in zip(spec.args[-len(spec.defaults):], spec.defaults):
            argument_type = custom_type(name, default)
            if not argument_type == bool:
                parser.add_argument("--" + name.replace("_", "-"), type=argument_type, default=default,
                                    **additional_options(name))
            else:
                action = ["store_true", "store_false"][int(default)]
                parser.add_argument("--" + name.replace("_", "-"), action=action, **additional_options(name))
        return parser


class TextSequenceEmbeddingClassifier(TextEmbeddingClassifier):
    VOCABULARY_SIZE = 20000
    TRAIN_EMBEDDINGS = False
    RNN_UNITS = 64
    RNN_TYPE = "gru"
    BIDIRECTIONAL = False
    DROPOUT = 0.5
    LANGUAGE_MODEL = "en"

    CUSTOM_COMMAND_LINE_OPTIONS = {
        "sequence_length": {"help": "Maximum number of tokens per text (default use longest in the data)", "type": int,
                            "metavar": "LENGTH"},
        "vocabulary_size": {"help": "number of words in the vocabulary (default %d)" % VOCABULARY_SIZE,
                            "metavar": "SIZE"},
        "train_embeddings": {"help": "train word embeddings? (default %s)" % TRAIN_EMBEDDINGS},
        "rnn_type": {"choices": ["gru", "lstm"], "help": "RNN type (default %s)" % RNN_TYPE},
        "rnn_units": {"help": "RNN units (default %d)" % RNN_UNITS, "metavar": "UNITS"},
        "bidirectional": {"help": "bidirectional RNN? (default %s)" % BIDIRECTIONAL},
        "dropout": {"help": "dropout rate (default %0.2f)" % DROPOUT},
        "language_model": {"help": "The spaCy language model to use (default '%s')" % LANGUAGE_MODEL, "metavar": "NAME"}
    }

    def __init__(self, training,
                 sequence_length=None, vocabulary_size=VOCABULARY_SIZE, train_embeddings=TRAIN_EMBEDDINGS,
                 language_model=LANGUAGE_MODEL, rnn_type=RNN_TYPE, rnn_units=RNN_UNITS, bidirectional=BIDIRECTIONAL,
                 dropout=DROPOUT):
        from keras.models import Sequential
        from keras.layers import Bidirectional, Dense, Dropout, GRU, LSTM
        from .text import TextSequenceEmbedder

        label_names = training[2]
        if sequence_length is None:
            sequence_length = longest_text(training[0], language_model)

        embedder = TextSequenceEmbedder(vocabulary_size, sequence_length, language_model)
        model = Sequential()
        model.add(embedder.embedding_layer_factory()(input_length=sequence_length, mask_zero=True,
                                                     trainable=train_embeddings, name="embedding"))
        rnn_class = {"lstm": LSTM, "gru": GRU}[rnn_type]
        if bidirectional:
            rnn = Bidirectional(rnn_class(rnn_units), name="rnn")
        else:
            rnn = rnn_class(rnn_units, name="rnn")
        model.add(rnn)
        model.add(Dense(len(label_names), activation="softmax", name="softmax"))
        model.add(Dropout(dropout, name="dropout"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        super().__init__(model, embedder, label_names)

    def __repr__(self):
        if self.bidirectional:
            bidi = " bidirectional,"
        else:
            bidi = ""
        return "Neural text sequence classifier: %d labels, %d RNN units,%s dropout rate %0.2f\n%s" % (
            self.num_labels, self.rnn_units, bidi, self.dropout, self.embedder)

    @property
    def rnn_units(self):
        from keras.layers import Bidirectional

        rnn = self.model.get_layer("rnn")
        if isinstance(rnn, Bidirectional):
            rnn = rnn.layer
        return rnn.units

    @property
    def bidirectional(self):
        from keras.layers import Bidirectional

        return isinstance(self.model.get_layer("rnn"), Bidirectional)

    @property
    def embeddings_per_text(self):
        return self.model.get_layer("rnn").input_shape[1]

    @property
    def embedding_size(self):
        return self.model.get_layer("rnn").input_shape[2]

    @property
    def dropout(self):
        return self.model.get_layer("dropout").rate


class BagOfWordsEmbeddingClassifier(TextEmbeddingClassifier):
    DROPOUT = 0.5
    LANGUAGE_MODEL = "en"

    CUSTOM_COMMAND_LINE_OPTIONS = {
        "dropout": {"help": "dropout rate (default %0.2f)" % DROPOUT},
        "language_model": {"help": "the spaCy language model to use (default '%s')" % LANGUAGE_MODEL, "metavar": "NAME"}
    }

    def __init__(self, training, dropout=DROPOUT, language_model=LANGUAGE_MODEL):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        from .text import BagOfWordsEmbedder

        label_names = training[2]
        embedder = BagOfWordsEmbedder(language_model)
        model = Sequential()
        model.add(Dense(len(label_names), input_shape=(embedder.embedding_size,), activation="softmax", name="softmax"))
        model.add(Dropout(dropout, name="dropout"))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        super().__init__(model, embedder, label_names)

    def __repr__(self):
        return "Neural bag of words classifier: %d labels, dropout rate %0.2f\n%s" % (
            self.num_labels, self.dropout, self.embedder)

    @property
    def dropout(self):
        return self.model.get_layer("dropout").rate
