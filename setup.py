#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from pathlib import Path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "NAMpy"
DESCRIPTION = "A python package for neural additive modelling"
HOMEPAGE = "tbd"
DOCS = "tbd"
EMAIL = "anton.thielmann@tu-clausthal.de"
AUTHOR = "Anton Thielmann"
REQUIRES_PYTHON = ">=3.6, <=3.12.3"

# Load the package's verison file and its content.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "nampy"

with open(PACKAGE_DIR / "__version__.py") as f:
    VERSION = f.readlines()[-1].split()[-1].strip("\"'")

# ger install_reqs from requirements file, used for setup function later
with open(os.path.join(ROOT_DIR, "requirements.txt")) as f:
    # next(f)
    install_reqs = [
        line.rstrip()
        for line in f.readlines()
        if not line.startswith("#") and not line.startswith("git+")
    ]


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    install_requires=install_reqs,
    # extras_require=extras_reqs,
    license="Copyright (c) 2024 BASF SE",  # adapt based on your needs
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
