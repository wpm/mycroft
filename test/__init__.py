import os


def to_lines(filename):
    def non_empty_lines():
        return list(filter(None, [line.strip() for line in f.readlines()]))

    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(filename) as f:
        return non_empty_lines()
