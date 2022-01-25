#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fedshuffle",
    version="0.0.1",
    author="FAIR",
    author_email="samuel.horvath@kaust.edu.sa",
    description="Simulated Federated Learning Experiments"
                "based on PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fairinternal/fedshuffle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
