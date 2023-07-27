from setuptools import setup, find_packages

setup(
    name='deforum',
    version='0.1.0',
    author='deforum',
    author_email='deforum.art@gmail.com',
    description='diffusion animation toolkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/deforum-art/deforum',  
    packages=find_packages(),
    install_requires=[
        'diffusers==0.19.0',
        'torch==2.0.1',
        'transformers==4.31.0',
        'accelerate==0.21.0',
        'safetensors==0.3.1',
        'opencv-python==4.8.0.74',
        'imageio==2.31.1',
        'imageio-ffmpeg==0.4.8',
        'natsort==8.4.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)