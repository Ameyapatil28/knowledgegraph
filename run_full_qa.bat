@echo off
set PYTHON_EXE=C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe

if not exist "%PYTHON_EXE%" (
    echo Python 3.10 not found at %PYTHON_EXE%
    python run_full_qa_demo.py %*
) else (
    "%PYTHON_EXE%" run_full_qa_demo.py %*
)
