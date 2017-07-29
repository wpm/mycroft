"""Machine learning components"""
import inspect
import json
import os
import pickle
import sys
import textwrap
from io import StringIO

from .text import TextSequenceEmbedder, longest_text


def load_embedding_model(model_directory):
    """
    Load a trained model.

    :param model_directory: directory in which the model was saved
    :type model_directory: str
    :return: the model
    :rtype: TextEmbeddingClassifier
    """
    with open(os.path.join(model_directory, TextEmbeddingClassifier.classifier_name), mode="rb") as f:
        model = pickle.load(f)
    model.load_model(model_directory)
    return model


class TextEmbeddingClassifier:
    """
    Base class for models that can do text classification using text vector embeddings.

    Derived classes must define a constructor that takes a training data argument followed by model hyper-parameters.
    The training data argument is a 3-ple (texts, labels, label names). The label names must be passed up to this
    constructor. The texts and labels are present in case they are needed to determine hyper-parameters. (See
    the constructors of ConvolutionNetClassifier and RNNClassifier for examples of this.

    When one of these classes is passed a part of the model_specification argument to mycroft.console.main, Mycroft will
    create command line arguments for all the hyper-parameter arguments in its constructor. Positional constructor
    arguments will be positional command line arguments, and keyword constructor arguments will be command line options.
    Derived classes may optionally supply a CUSTOM_COMMAND_LINE_OPTIONS dictionary, which specifies additional keyword
    arguments to provide to the argparse.addArgument command.
    """
    EPOCHS = 10
    BATCH_SIZE = 32

    # Names of files created in the model directory.
    model_name = "model.hd5"
    classifier_name = "classifier.pk"
    description_name = "description.txt"
    history_name = "history.json"

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

    def train(self, texts, labels, epochs=EPOCHS, early_stop=8, reduce=4, batch_size=32, validation_fraction=None,
              validation_data=None, model_directory=None, tensor_board_directory=None, verbose=1):
        from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

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
        callbacks = []
        if verbose == 0:
            callback_verbosity = 0
        else:
            callback_verbosity = 1
        if tensor_board_directory:
            callbacks.append(TensorBoard(log_dir=tensor_board_directory))
        if early_stop:
            callbacks.append(EarlyStopping(monitor=monitor, patience=early_stop, verbose=callback_verbosity))
        if reduce:
            callbacks.append(ReduceLROnPlateau(monitor=monitor, patience=reduce, verbose=callback_verbosity))
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
                # JSON requires float, not numpy.float32.
                if "lr" in h["history"]:
                    # noinspection PyTypeChecker
                    h["history"]["lr"] = [float(x) for x in h["history"]["lr"]]
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

    def load_model(self, model_directory):
        from keras.models import load_model
        self.model = load_model(os.path.join(model_directory, TextEmbeddingClassifier.model_name))

    @property
    def num_labels(self):
        return len(self.label_names)

    @classmethod
    def custom_command_line_options(cls):
        return {}

    @classmethod
    def create_from_command_line_arguments(cls, training, command_line_arguments):
        args = command_line_arguments.__dict__
        spec = inspect.getfullargspec(cls.__init__)
        filtered_args = dict((k, v) for k, v in args.items() if k in spec.args)
        return cls(tuple(training), **filtered_args)

    @classmethod
    def command_line_arguments(cls, parser):
        """
        Build command line options for this model out of the constructor arguments and their defaults and add them to
        an argument parser.

        This creates a command line option for each keyword argument in the constructor. If the constructor default
        value is None, the command line option will not have a default and the argument type must be specified in the
        dictionary returned by custom_command_line_options.

        :param parser: command line parser to add this model's options to
        :type parser: argparse._ArgumentGroup
        """

        def get_argument_type(name, default):
            try:
                argument_type = cls.custom_command_line_options()[name]["type"]
            except KeyError:
                if default is None:
                    raise ValueError(textwrap.dedent("""
                    constructor argument %s = None must have its type specified in the custom command line options""")
                                     % name)
                argument_type = type(default)
            return argument_type

        def additional_options(name):
            options = cls.custom_command_line_options().get(name, {})
            options.pop("type", None)  # type is handled by argument_type()
            return options

        def option_string(name):
            return "--" + name.replace("_", "-")

        spec = inspect.getfullargspec(cls.__init__)
        for name, default in zip(spec.args[-len(spec.defaults):], spec.defaults):
            argument_type = get_argument_type(name, default)
            if default is None:
                parser.add_argument(option_string(name), type=argument_type, **additional_options(name))
            elif argument_type == bool:
                action = ["store_true", "store_false"][int(default)]
                parser.add_argument(option_string(name), action=action, **additional_options(name))
            elif argument_type == tuple:
                parser.add_argument(option_string(name), nargs="+", type=type(default[0]), default=default,
                                    **additional_options(name))
            else:
                parser.add_argument(option_string(name), type=argument_type, default=default,
                                    **additional_options(name))


class RNNClassifier(TextEmbeddingClassifier):
    """
    This model uses GloVe vectors to embed the text into matrices of size sequence length × 300, clipping or padding
    the first dimension for each individual text as needed. A recurrent neural network (either a GRU or an LSTM)
    converts these embeddings to a single vector which a softmax layer then uses to make a label prediction.
    """
    VOCABULARY_SIZE = 20000
    TRAIN_EMBEDDINGS = False
    RNN_UNITS = (64,)
    RNN_TYPE = "gru"
    BIDIRECTIONAL = False
    DROPOUT = 0.5
    LEARNING_RATE = 0.001
    LANGUAGE_MODEL = "en"

    @classmethod
    def custom_command_line_options(cls):
        return {
            "sequence_length": {"help": "Maximum number of tokens per text (default use longest in the data)",
                                "type": int,
                                "metavar": "LENGTH"},
            "vocabulary_size": {"help": "number of words in the vocabulary (default %d)" % cls.VOCABULARY_SIZE,
                                "metavar": "SIZE"},
            "train_embeddings": {"help": "train word embeddings? (default %s)" % cls.TRAIN_EMBEDDINGS},
            "rnn_type": {"choices": ["gru", "lstm"], "help": "RNN type (default %s)" % cls.RNN_TYPE},
            "rnn_units": {
                "help": "number of units in stacked RNN layers (default one layer with %d units)" % cls.RNN_UNITS[0],
                "metavar": "UNITS"},
            "bidirectional": {"help": "bidirectional RNN? (default %s)" % cls.BIDIRECTIONAL},
            "dropout": {"help": "dropout rate (default %0.2f)" % cls.DROPOUT},
            "learning_rate": {"metavar": "RATE", "help": "learning rate (default %0.5f)" % cls.LEARNING_RATE},
            "language_model": {"help": "The spaCy language model to use (default '%s')" % cls.LANGUAGE_MODEL,
                               "metavar": "NAME"}
        }

    def __init__(self, training,
                 sequence_length=None, vocabulary_size=VOCABULARY_SIZE, train_embeddings=TRAIN_EMBEDDINGS,
                 language_model=LANGUAGE_MODEL, rnn_type=RNN_TYPE, rnn_units=RNN_UNITS, bidirectional=BIDIRECTIONAL,
                 dropout=DROPOUT, learning_rate=LEARNING_RATE):
        from keras.models import Sequential
        from keras.layers import Bidirectional, Dense, Dropout, GRU, LSTM
        from keras.optimizers import Adam

        label_names = training[2]
        if sequence_length is None:
            sequence_length = longest_text(training[0], language_model)

        embedder = TextSequenceEmbedder(vocabulary_size, sequence_length, language_model)
        model = Sequential()
        model.add(embedder.embedding_layer_factory()(input_length=sequence_length, mask_zero=True,
                                                     trainable=train_embeddings, name="embedding"))
        rnn_class = {"lstm": LSTM, "gru": GRU}[rnn_type]
        for i, units in enumerate(rnn_units, 1):
            name = "rnn-%d" % i
            return_sequences = i < len(rnn_units)
            if bidirectional:
                rnn = Bidirectional(rnn_class(units, return_sequences=return_sequences), name=name)
            else:
                rnn = rnn_class(units, return_sequences=return_sequences, name=name)
            model.add(rnn)
            model.add(Dropout(dropout, name="dropout-%d" % i))
        model.add(Dense(len(label_names), activation="softmax", name="softmax"))
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        self.rnn_units = rnn_units
        self.bidirectional = bidirectional
        self.dropout = dropout
        super().__init__(model, embedder, label_names)

    def __repr__(self):
        if self.bidirectional:
            bidi = " bidirectional,"
        else:
            bidi = ""
        return "Neural text sequence classifier: %d labels, RNN units %s,%s dropout rate %0.2f\n%s" % (
            self.num_labels, self.rnn_units, bidi, self.dropout, self.embedder)


class ConvolutionNetClassifier(TextEmbeddingClassifier):
    """
    This model uses GloVe vectors to embed the text into matrices of size sequence length × 300, clipping or padding
    the first dimension for each individual text as needed. A 1-dimensional convolutional/max-pooling converts these
    embeddings to a single vector which a softmax layer then uses to make a label prediction.
    """
    VOCABULARY_SIZE = 20000
    DROPOUT = 0.5
    FILTERS = 100
    KERNEL_SIZE = 3
    POOL_FACTOR = 4
    LEARNING_RATE = 0.001
    LANGUAGE_MODEL = "en"

    @classmethod
    def custom_command_line_options(cls):
        return {
            "sequence_length": {"help": "Maximum number of tokens per text (default use longest in the data)",
                                "type": int,
                                "metavar": "LENGTH"},
            "vocabulary_size": {"help": "number of words in the vocabulary (default %d)" % cls.VOCABULARY_SIZE,
                                "metavar": "SIZE"},
            "dropout": {"help": "Dropout rate (default %0.2f)" % cls.DROPOUT},
            "filters": {"help": "Number of filters (default %d)" % cls.FILTERS, "metavar": "FILTERS"},
            "kernel_size": {"help": "Size of kernel (default %d)" % cls.KERNEL_SIZE, "metavar": "SIZE"},
            "pool_factor": {"help": "Pooling downscale factor (default %d)" % cls.POOL_FACTOR, "metavar": "FACTOR"},
            "learning_rate": {"metavar": "RATE", "help": "learning rate (default %0.5f)" % cls.LEARNING_RATE},
            "language_model": {"help": "Language model (default %s)" % cls.LANGUAGE_MODEL, "metavar": "MODEL"}
        }

    def __init__(self, training,
                 sequence_length=None, vocabulary_size=VOCABULARY_SIZE, dropout=DROPOUT, filters=FILTERS,
                 kernel_size=KERNEL_SIZE, pool_factor=POOL_FACTOR, learning_rate=LEARNING_RATE,
                 language_model=LANGUAGE_MODEL):
        from keras.layers import Dropout, Conv1D, Flatten, MaxPooling1D, Dense
        from keras.models import Sequential
        from keras.optimizers import Adam

        label_names = training[2]
        if sequence_length is None:
            sequence_length = longest_text(training[0], language_model)
        embedder = TextSequenceEmbedder(vocabulary_size, sequence_length, language_model)

        model = Sequential()
        model.add(embedder.embedding_layer_factory()(input_length=sequence_length, trainable=False, name="embedding"))
        model.add(Conv1D(filters, kernel_size, padding="valid", activation="relu", strides=1, name="convolution"))
        model.add(MaxPooling1D(pool_size=pool_factor, name="pooling"))
        model.add(Flatten(name="flatten"))
        model.add(Dropout(dropout, name="dropout"))
        model.add(Dense(len(label_names), activation="softmax", name="softmax"))
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_factor = pool_factor
        self.dropout = dropout
        super().__init__(model, embedder, label_names)

    def __repr__(self):
        return "Convolutional text sequence classifier: " + \
               "%d labels, %d filters, kernel size %d, pool factor %d, dropout rate %0.2f\n%s" % (
                   self.num_labels, self.filters, self.kernel_size, self.pool_factor, self.dropout, self.embedder)


class BagOfWordsClassifier(TextEmbeddingClassifier):
    """
    A softmax layer uses the average of the GloVe token embeddings to make a label prediction.
    """
    LEARNING_RATE = 0.001
    LANGUAGE_MODEL = "en"

    @classmethod
    def custom_command_line_options(cls):
        return {
            "learning_rate": {"metavar": "RATE", "help": "learning rate (default %0.5f)" % cls.LEARNING_RATE},
            "language_model": {"help": "the spaCy language model to use (default '%s')" % cls.LANGUAGE_MODEL,
                               "metavar": "NAME"}
        }

    def __init__(self, training, learning_rate=LEARNING_RATE, language_model=LANGUAGE_MODEL):
        from keras.layers import Dense
        from keras.models import Sequential
        from keras.optimizers import Adam
        from .text import BagOfWordsEmbedder

        label_names = training[2]
        embedder = BagOfWordsEmbedder(language_model)
        model = Sequential()
        model.add(Dense(len(label_names), input_shape=(embedder.embedding_size,), activation="softmax", name="softmax"))
        optimizer = Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        super().__init__(model, embedder, label_names)

    def __repr__(self):
        return "Neural bag of words classifier: %d labels\n%s" % (self.num_labels, self.embedder)
