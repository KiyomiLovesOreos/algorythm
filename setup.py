"""
Setup script for Algorythm library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="algorythm",
    version="0.1.0",
    author="KiyomiLovesOreos",
    description="A Python Library for Algorithmic Music - Manim-inspired declarative audio synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KiyomiLovesOreos/algorythm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "export": [
            "soundfile>=0.10.0",  # For FLAC export
            "pydub>=0.25.0",      # For MP3/OGG export
        ],
    },
    entry_points={
        "console_scripts": [
            "algorythm=algorythm.cli:main",
        ],
    },
)
