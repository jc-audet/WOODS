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
    description="A set of Out-of-Distribution Generalization Benchmarks for Sequential Prediction Tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        'academictorrents==2.3.3',
        'better_apidoc==0.3.2',
        'Braindecode==0.5.1',
        'gdown==3.13.0',
        'h5py==3.6.0',
        'matplotlib==3.4.1',
        'mne==0.23.2',
        'moabb==0.4.4',
        'numpy==1.21.2',
        'Pillow==8.4.0',
        'pptree==3.1',
        'prettytable==2.1.0',
        'pyEDFlib==0.1.22',
        'pytorchvideo==0.1.3',
        'scikit_learn==1.0.1',
        'scipy==1.7.2',
        'torch==1.8.1',
        'torchvision==0.9.1',
        'tqdm==4.62.2',
        'xlrd==2.0.1',
    ],
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