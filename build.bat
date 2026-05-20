@echo off
setlocal EnableExtensions

set "RELEASE_NAME=2026-01-02_01"
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "RELEASE_ROOT=%ROOT%\release"
set "OUT=%RELEASE_ROOT%\%RELEASE_NAME%"

if "%~1"=="/?" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage

echo [fontman] Building release: %OUT%

where go >nul 2>nul
if errorlevel 1 (
    echo [fontman] ERROR: go was not found in PATH.
    exit /b 1
)

if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)
if "%FONTMAN_PIP_INDEX%"=="" (
    set "FONTMAN_PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple"
)

if exist "%OUT%" (
    echo [fontman] Removing old release directory...
    rmdir /s /q "%OUT%"
)
mkdir "%OUT%" || exit /b 1

echo [fontman] Building Go web shell...
go build -o "%OUT%\findfontman.exe" .\cmd\findfontman
if errorlevel 1 exit /b 1

echo [fontman] Building Python runtime...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\scripts\build_runtime_windows.ps1" -Python "%PYTHON%" -PipIndexUrl "%FONTMAN_PIP_INDEX%"
if errorlevel 1 exit /b 1

if not exist "%ROOT%\dist\fontman-runtime\fontman-service.exe" (
    echo [fontman] ERROR: fontman runtime was not created.
    exit /b 1
)

echo [fontman] Copying runtime...
xcopy "%ROOT%\dist\fontman-runtime" "%OUT%\fontman-runtime\" /E /I /Y >nul
if errorlevel 1 exit /b 1

echo [fontman] Copying SDK and docs...
xcopy "%ROOT%\sdk" "%OUT%\sdk\" /E /I /Y >nul
if errorlevel 1 exit /b 1
copy /Y "%ROOT%\README.md" "%OUT%\README.md" >nul
if errorlevel 1 exit /b 1

echo [fontman] Preparing data directories...
if not exist "%OUT%\data" mkdir "%OUT%\data"
if exist "%ROOT%\data\font_ai" (
    if not exist "%OUT%\data\font_ai" mkdir "%OUT%\data\font_ai"
    for %%F in (font_embedding.pt font_index.npz font_index_meta.json train_config.json) do (
        if exist "%ROOT%\data\font_ai\%%F" copy /Y "%ROOT%\data\font_ai\%%F" "%OUT%\data\font_ai\%%F" >nul
    )
) else (
    echo [fontman] WARNING: data\font_ai was not found; release will not include trained model data.
)
if not exist "%OUT%\data\previews" mkdir "%OUT%\data\previews"
if not exist "%OUT%\data\uploads" mkdir "%OUT%\data\uploads"
if not exist "%OUT%\data\crops" mkdir "%OUT%\data\crops"

echo [fontman] Preparing empty fonts directory. Actual fonts are intentionally excluded.
if not exist "%OUT%\fonts" mkdir "%OUT%\fonts"
>"%OUT%\fonts\PUT_FONTS_HERE.txt" echo Put licensed font files or font folders here. Actual fonts are not included in this release package.

echo [fontman] Writing launch scripts...
>"%OUT%\start_web.bat" echo @echo off
>>"%OUT%\start_web.bat" echo setlocal
>>"%OUT%\start_web.bat" echo cd /d "%%~dp0"
>>"%OUT%\start_web.bat" echo set "ADDR=:8080"
>>"%OUT%\start_web.bat" echo set "VISION_ADDR=127.0.0.1:9091"
>>"%OUT%\start_web.bat" echo start "" "http://localhost:8080"
>>"%OUT%\start_web.bat" echo ".\findfontman.exe"

>"%OUT%\start_runtime.bat" echo @echo off
>>"%OUT%\start_runtime.bat" echo setlocal
>>"%OUT%\start_runtime.bat" echo cd /d "%%~dp0"
>>"%OUT%\start_runtime.bat" echo ".\fontman-runtime\fontman-service.exe" --addr 127.0.0.1:9091 --root . --fonts .\fonts --data .\data --previews .\data\previews

>"%OUT%\RELEASE_NOTES.txt" echo Find Fontman release %RELEASE_NAME%
>>"%OUT%\RELEASE_NOTES.txt" echo.
>>"%OUT%\RELEASE_NOTES.txt" echo Included:
>>"%OUT%\RELEASE_NOTES.txt" echo - findfontman.exe
>>"%OUT%\RELEASE_NOTES.txt" echo - fontman-runtime\fontman-service.exe and runtime dependencies
>>"%OUT%\RELEASE_NOTES.txt" echo - data\font_ai model/index resources when present
>>"%OUT%\RELEASE_NOTES.txt" echo - Go and C++ single-file SDKs under sdk\
>>"%OUT%\RELEASE_NOTES.txt" echo - embedding matching runtime; OCR dependencies are not bundled
>>"%OUT%\RELEASE_NOTES.txt" echo.
>>"%OUT%\RELEASE_NOTES.txt" echo Not included:
>>"%OUT%\RELEASE_NOTES.txt" echo - actual font files under fonts\
>>"%OUT%\RELEASE_NOTES.txt" echo.
>>"%OUT%\RELEASE_NOTES.txt" echo Run start_web.bat for the debug web UI, or start_runtime.bat for SDK-only service usage.

echo [fontman] Release built successfully:
echo %OUT%

endlocal
exit /b 0

:usage
echo Usage: build.bat
echo.
echo Builds release\%RELEASE_NAME% with the Go web shell, Python runtime bundle,
echo trained data\font_ai resources, SDKs, docs, and an empty fonts directory.
echo Actual font files are intentionally excluded.
endlocal
exit /b 0
