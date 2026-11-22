$pythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"

if (Test-Path $pythonExe) {
    Write-Host "Launching FastAPI backend using Python 3.10 ($pythonExe)..." -ForegroundColor Cyan
    & $pythonExe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload $args
} else {
    Write-Host "Launching FastAPI backend using system python..." -ForegroundColor Cyan
    python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload $args
}
