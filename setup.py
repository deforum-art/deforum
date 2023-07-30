"""
setup.py script for the 'deforum' package.

This script gathers package metadata and dependencies for the
'deforum' package and sets it up for distribution and installation.

The package's metadata (like name, version, author, etc.), 
as well as its dependencies are specified in the call to `setup()`. 
"""
from setuptools import setup, find_packages

setup(
    name="deforum",
    version="0.1.6",
    author="deforum",
    author_email="deforum.art@gmail.com",
    description="diffusion animation toolkit",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deforum-art/deforum",
    packages=find_packages(),
    install_requires=[
        "diffusers>0.19.0",
        "torch>=2.0.0",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
        "opencv-python-headless", # helpful for headless servers
        "imageio",
        "imageio-ffmpeg",
        "natsort",
        "nltk",
        "pydash",
        "pydantic<2.0.0",
        "loguru",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
