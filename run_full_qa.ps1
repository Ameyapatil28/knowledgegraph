$pythonExe = "C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe"

if (Test-Path $pythonExe) {
    & $pythonExe run_full_qa_demo.py $args
} else {
    python run_full_qa_demo.py $args
}
