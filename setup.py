import setuptools
import os
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="woods",
    version='0.1',
    author="Jean-Christophe Gagnon-Audet",
    author_email="jcgagnon74@gmail.com",
    url="https://woods-benchmarks.github.io/",
    description="A set of Out-of-Distribution Generalization Benchmarks for Time Series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [],
    license='MIT',
    packages=setuptools.find_packages(exclude=['woods.scripts']),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.7',
)