# Mycroft

Mycroft is a toolkit for doing text classification with word embeddings.
It provides a command line interface for training and evaluating different kinds of neural network classifiers and a
programmatic interface for incorporating these classifiers into other programs.

## Installation

Install Mycroft by running `python setup.py install`.

Mycroft uses the [spaCy](https://spacy.io/) text natural language processing toolkit to process text.
The built-in models just use spaCy for tokenization, but models built on Mycroft's programmatic interface may use
spaCy's more advanced features like part-of-speech tagging and syntactic parsing.
By default it installs spaCy's English language text model, though you may specify other language models from the
command line.
See spaCy's [models documentation](https://spacy.io/docs/usage/models) for more information.


## Running from the Command Line

Run Mycroft with the command `mycroft`.
Subcommands enable you to train models and use them to make predictions on unlabeled data sets.
Run `mycroft --help` for details about specific commands.

The training data is a comma- or tab-delimited file with column of text and a column of labels.
The test data is in the same format without the labels.

Run `mycroft demo` to see a quick example of the command line syntax and data formats.

## Classifier Models

Mycroft implements two kinds of word-embedding models.

* __Neural bag of words__

  300-dimensional [GloVe](https://nlp.stanford.edu/projects/glove/) vectors are used to embed the tokens in the text.
  A softmax layer uses the average of the token embeddings to make a label prediction.

* __Neural text sequence__

  The same GloVe vectors are used to embed the text into matrices of size _sequence length × 300_, clipping or padding
  the first dimension for each individual text as needed.
  A recursive neural network (either an LSTM or GRU) converts these embeddings to a single vector which a softmax layer
  then uses to make a label prediction.

As a comparison baseline Mycroft also implements one non-word embedding model.

* __Word count__

  This model trains an SVM over TF-IDF of words.


## Programmatic Interface

You can write your own Keras-based text-embedding classfiers by extending the `TextEmbeddingClassifier` base class in
`mycroft.model`.
See the built-in `BagOfWordsEmbeddingClassifier` and `TextSequenceEmbeddingClassifier` classes for examples on how to
do this.
This allows you to focus on just writing the model while reusing Mycroft's pandas data processing, incorporation of
word embedding vectors, the prediction and evaluation code which remains the same regardless of model, and a command
line interface.
