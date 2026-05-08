#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "NAMpy"
DESCRIPTION = "A python package for neural additive modelling"
HOMEPAGE = "tbd"
DOCS = "tbd"
EMAIL = "anton.thielmann@tu-clausthal.de"
AUTHOR = "Anton Thielmann"
REQUIRES_PYTHON = ">=3.10"

# Load the package's verison file and its content.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "nampy"

with open(PACKAGE_DIR / "__version__.py") as f:
    VERSION = f.readlines()[-1].split()[-1].strip("\"'")

# Read the README file for long description
try:
    with open(ROOT_DIR / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    # extras_require=extras_reqs,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    project_urls={"Documentation": DOCS},
    url=HOMEPAGE,
)
