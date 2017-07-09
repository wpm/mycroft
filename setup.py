from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

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


setup(
    name="mycroft",
    version=mycroft.__version__,
    packages=["mycroft"],
    url="https://github.com/wpm/mycroft",
    license="",
    entry_points={
        "console_scripts": ["mycroft=mycroft.console:main"],
    },
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Text classifier", install_requires=["h5py", "keras", "numpy", "pandas", "spacy", "scikit-learn"],
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand
    }
)
