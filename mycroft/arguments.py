"""
Convenience functions for creating shared command line arguments.
"""


# Model configuration
def model_group_name(parser):
    return parser.add_argument_group("model", description="Arguments for specifying the model configuration:")


def dropout_argument(parser, default=0.5):
    parser.add_argument("--dropout", metavar="RATE", type=float, default=default,
                        help="Dropout rate (default %0.2f)" % default)


# Data
def data_group_name(parser):
    return parser.add_argument_group("data", description="Arguments for specifying the data to train on:")


def training_data_argument(parser):
    parser.add_argument("training", help="training data file")


def limit_argument(parser):
    parser.add_argument("--limit", type=int, help="only train on this many samples (default use all the data)")


def validation_argument(parser):
    parser.add_argument("--validation", metavar="PORTION", type=float,
                        help="portion of data to use for validation (default none)")


def text_name_argument(parser, default="text"):
    parser.add_argument("--text-name", metavar="NAME", default=default,
                        help="name of the text column (default '%s')" % default)


def label_name_argument(parser, default="label"):
    parser.add_argument("--label-name", metavar="NAME", default=default,
                        help="name of the label column (default '%s')" % default)


def omit_labels(parser):
    parser.add_argument("--omit-labels", metavar="LABEL", nargs="*", help="omit samples with these label values")


def data_group(parser):
    data_group = data_group_name(parser)
    training_data_argument(data_group)
    limit_argument(data_group)
    validation_argument(data_group)
    text_name_argument(data_group)
    label_name_argument(data_group)
    omit_labels(data_group)
    return data_group


# Training parameters
def training_group_name(parser):
    return parser.add_argument_group("training", description="Arguments for controlling the training procedure:")


def epochs_argument(parser, default=10):
    parser.add_argument("--epochs", type=int, default=default, help="number of training epochs (default %d)" % default)


def batch_size_argument(parser, default=32):
    parser.add_argument("--batch-size", metavar="SIZE", type=int, default=default,
                        help="batch size (default %d)" % default)


def model_directory_argument(parser):
    parser.add_argument("--model-directory", metavar="DIRECTORY",
                        help="directory in which to store the model (default do not store a model)")


def logging_argument(parser, default="epoch"):
    parser.add_argument("--logging", choices=["none", "progress", "epoch"], default=default,
                        help="no logging, a progress bar, one line per epoch (default %s)" % default)


def training_group(parser):
    train_group = training_group_name(parser)
    epochs_argument(train_group)
    batch_size_argument(train_group)
    model_directory_argument(train_group)
    logging_argument(train_group)
    return train_group


# Language processing parameters
def langauge_group_name(parser):
    return parser.add_argument_group("language", description="Arguments for controlling language processing:")


def vocabulary_size_argument(parser, default=20000):
    parser.add_argument("--vocabulary-size", metavar="SIZE", default=default,
                        help="number of words in the vocabulary (default %d)" % default)


def language_model_argument(parser, default="en"):
    parser.add_argument("--language-model", metavar="MODEL", default=default,
                        help="the spaCy language model to use (default '%s')" % default)
