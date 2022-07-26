'''
Setup script, to make package pip installable
'''
from setuptools import setup


setup(name = 'wordery',
      version = '1.0.0',
      description = 'Python package for general preprocessing of work order type data for Natural Language Processing',
      author = 'Ali Barlas and Paula Soares',
      author_email = 'TBC',
      packages = ['wordery'],
      url = 'TBC',
      install_requires = ['numpy','pandas','nltk','autocorrect'],
)