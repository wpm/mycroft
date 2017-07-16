import io
import os

import six


def to_lines(filename):
    def non_empty_lines():
        return list(filter(None, [line.strip() for line in f.readlines()]))

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    if six.PY3:
        with open(filename) as f:
            lines = non_empty_lines()
    else:
        with io.open(filename, encoding="UTF8") as f:
            lines = non_empty_lines()
    return lines
