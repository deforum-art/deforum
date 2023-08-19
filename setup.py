from setuptools import find_packages, setup

setup(
    name="deforum",
    version="0.1.7",
    author="deforum",
    author_email="deforum.art@gmail.com",
    description="diffusion animation toolkit",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/deforum-art/deforum",
    packages=find_packages(),
    package_data={'deforum': ['data/*.txt']},  # Include any .txt files in deforum/data/
    install_requires=[
        "accelerate",
        "diffusers>0.19.0",
        "einops",
        "imageio",
        "imageio-ffmpeg",
        "loguru",
        "natsort",
        "nltk",
        "opencv-python-headless",
        "pydantic<2.0.0",
        "pydash",
        "safetensors",
        "scipy",
        "sortedcontainers",
        "torch>=2.0.0",
        "torchvision",
        "transformers",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)