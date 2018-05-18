"""A setuptools based setup module adapted from PyPa's sample project.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='g2ti',
    version='0.0.1',
    description='Scripts for g2ti',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/OGGM/g2ti',
    author='Fabien Maussion',
    author_email='',
    classifiers=[],
    keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[],
    extras_require={
        'test': ['pytest'],
    },
    package_data={},
    data_files={},
    entry_points={},
    project_urls={},
)
