"""
Setup script for Algorythm library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="algorythm",
    version="0.4.0",  # Updated version with visualization enhancements
    author="KiyomiLovesOreos",
    description="A Python Library for Algorithmic Music - Manim-inspired declarative audio synthesis with video visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KiyomiLovesOreos/algorythm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Video :: Display",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pydub>=0.25.1",   # Required for MP3/OGG/FLAC audio file loading
        "Pillow>=8.0.0",   # Required for MP4 export fallback
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
        "video": [
            "opencv-python>=4.5.0",  # For fast video export
            "matplotlib>=3.3.0",     # Alternative video backend
        ],
        "playback": [
            "pyaudio>=0.2.11",    # For real-time audio playback
        ],
        "gui": [
            "pyaudio>=0.2.11",    # For audio playback in GUI
        ],
        "all": [
            "soundfile>=0.10.0",
            "pydub>=0.25.0",
            "opencv-python>=4.5.0",
            "matplotlib>=3.3.0",
            "pyaudio>=0.2.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "algorythm=algorythm.cli:main",
            "algorythm-live=algorythm.live_gui:launch",
        ],
    },
)
