@echo off
set PYTHON_EXE=C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe

if not exist "%PYTHON_EXE%" (
    echo Launching FastAPI backend using system python...
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload %*
) else (
    echo Launching FastAPI backend using Python 3.10 (%PYTHON_EXE%)...
    "%PYTHON_EXE%" -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload %*
)
