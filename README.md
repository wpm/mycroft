# Mycroft

Mycroft classifies English text.

## Installation

Install Mycroft by running `python setup.py install`.

Mycroft depends on the [spaCy](https://spacy.io/) text natural language processing toolkit.
It may be necessary to install spaCy's English language text model with a command like `python -m spacy download en` 
before running.
See spaCy's [models documentation](https://spacy.io/docs/usage/models) for more information.


## Running

Run Mycroft with the command `mycroft`.
Subcommands enable you to train models and use them to make predictions on unlabeled data sets.
Run `mycroft --help` for details about specific commands.

Input data takes the form of comma-separated-value documents.
Training data has columns containing the text and the labels.
Test data takes the same form minus the labels column.


## Classifier Model

[GloVe](https://nlp.stanford.edu/projects/glove/) vectors are used to embed the text into matrices of size
_maximum tokens × 300_, clipping or padding the first dimension for each individual text as needed.
An LSTM converts these embeddings to single vectors which are them mapped to a softmax prediction over the labels.
