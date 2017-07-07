from setuptools import setup

import mycroft

setup(
    name="mycroft",
    version=mycroft.__version__,
    packages=["mycroft"],
    url="https://github.com/wpm/mycroft",
    license="",
    entry_points={
        "console_scripts": ["mycroft=mycroft:main"],
    },
    author="W.P. McNeill",
    author_email="billmcn@gmail.com",
    description="Text classifier", install_requires=["h5py", "keras", "numpy", "pandas", "spacy"]
)
