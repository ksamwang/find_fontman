$ErrorActionPreference = "Stop"

winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wheel
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r python_service\requirements.txt
.\.venv\Scripts\python -c "import PIL, numpy, cv2, paddle, paddleocr; print('python env ok')"
go test ./...
