@echo off

REM Create Python virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Install requirements via pip
pip install -r requirements.txt
