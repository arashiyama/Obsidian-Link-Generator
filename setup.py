#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Auto Link Obsidian.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="auto_link_obsidian",
    version="0.1.0",
    author="Jonathan Care",
    author_email="jonc@lacunae.org",
    description="A tool for enhancing Obsidian notes with auto-tagging and linking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonathancare/auto_link_obsidian",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "pyyaml>=6.0",
        "tqdm>=4.61.0",
        "tenacity>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "obsidian-enhance=main:main",
        ],
    },
)