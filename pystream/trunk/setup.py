#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script for pystream
"""

from setuptools import setup, find_packages

setup(
    name = "pystream",
    version = "0.1",
    packages = find_packages(),
    
    zip_safe = False,

    author = "Brian Granger, Tech-X Corporation",
    author_email = "ellisonbg@gmail.com",
    description = "Stream and GPU computing in Python",
    license = "BSD",
    keywords = "gpu python ctypes cuda",
)
