import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pylrpredictor",
    version = "0.1",
    author = "Tobias Domhan",
    author_email = "tdomhan@gmail.com",
    install_requires = ['numpy', 'docutils>=0.3', 'setuptools', 'matplotlib'],
    description = ("Predicting learning curves in python"),
    license = "BSD",
    keywords = "python learning curves, prediction",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=find_packages(),#['pylrpredictor'],
    long_description="",
)
