from subprocess import check_call

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

import mycroft


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        check_call("python -m spacy download en".split())
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        check_call("python -m spacy download en".split())
        install.run(self)


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name="mycroft",
    version=mycroft.__version__,
    packages=["mycroft"],
    url="https://github.com/wpm/mycroft",
    license="MIT",
    keywords="lstm keras spacy machine-learning natural-language-processing rnn word-embeddings",
    python_requires=">=3.2",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    entry_points={
        "console_scripts": ["mycroft=mycroft.console:default_main"],
    },
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Text classifier",
    long_description=readme(),
    install_requires=["cytoolz", "keras", "numpy", "pandas", "scikit-learn", "spacy"],
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand
    }
)
