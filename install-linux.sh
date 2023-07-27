#!/bin/bash

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements via pip
pip install -r requirements.txt