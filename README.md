# Mycroft

Mycroft classifies English text.

## Installation

Install Mycroft by running `python setup.py install`.

Mycroft depends uses the [spaCy](https://spacy.io/) text natural language processing toolkit to parse text.
By default it installs spaCy's English language text model.
See spaCy's [models documentation](https://spacy.io/docs/usage/models) for more information.


## Running

Run Mycroft with the command `mycroft`.
Subcommands enable you to train models and use them to make predictions on unlabeled data sets.
Run `mycroft --help` for details about specific commands.

The training data is a comma- or tab-delimited file with column of text and a column of labels.
The test data is in the same format without the labels.

## Classifier Model

[GloVe](https://nlp.stanford.edu/projects/glove/) vectors are used to embed the text into matrices of size
_maximum tokens × 300_, clipping or padding the first dimension for each individual text as needed.
An LSTM converts these embeddings to single vectors which are then used for a softmax prediction over the labels.
