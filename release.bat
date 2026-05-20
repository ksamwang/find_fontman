@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "RELEASE_NAME=2026-01-02_01"
set "TAG_NAME=v2026.01.02-01"
set "TITLE=Find Fontman 2026-01-02_01"
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "RELEASE_ROOT=%ROOT%\release"
set "OUT=%RELEASE_ROOT%\%RELEASE_NAME%"
set "ZIP=%RELEASE_ROOT%\find_fontman_%RELEASE_NAME%.zip"
set "SKIP_BUILD=0"
set "DRAFT=0"
set "PRERELEASE=0"

:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="/?" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="--skip-build" (
    set "SKIP_BUILD=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--draft" (
    set "DRAFT=1"
    shift
    goto :parse_args
)
if /I "%~1"=="--prerelease" (
    set "PRERELEASE=1"
    shift
    goto :parse_args
)
echo [fontman] ERROR: unknown argument: %~1
goto :usage_error

:args_done

where gh >nul 2>nul
if errorlevel 1 (
    echo [fontman] ERROR: GitHub CLI gh was not found in PATH.
    exit /b 1
)

gh auth status >nul 2>nul
if errorlevel 1 (
    echo [fontman] ERROR: gh is not logged in. Run: gh auth login
    exit /b 1
)

git status --short > "%TEMP%\fontman_git_status.txt"
for %%A in ("%TEMP%\fontman_git_status.txt") do if %%~zA NEQ 0 (
    echo [fontman] ERROR: git working tree is not clean.
    type "%TEMP%\fontman_git_status.txt"
    del "%TEMP%\fontman_git_status.txt" >nul 2>nul
    exit /b 1
)
del "%TEMP%\fontman_git_status.txt" >nul 2>nul

if "%SKIP_BUILD%"=="0" (
    echo [fontman] Building release package first...
    call "%ROOT%\build.bat"
    if errorlevel 1 exit /b 1
) else (
    echo [fontman] Skipping build step.
)

if not exist "%OUT%" (
    echo [fontman] ERROR: release directory does not exist: %OUT%
    exit /b 1
)

echo [fontman] Creating zip: %ZIP%
if exist "%ZIP%" del /f /q "%ZIP%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '%OUT%\*' -DestinationPath '%ZIP%' -Force"
if errorlevel 1 exit /b 1

git rev-parse "%TAG_NAME%" >nul 2>nul
if errorlevel 1 (
    echo [fontman] Creating git tag: %TAG_NAME%
    git tag "%TAG_NAME%"
    if errorlevel 1 exit /b 1
    git push origin "%TAG_NAME%"
    if errorlevel 1 exit /b 1
) else (
    echo [fontman] Tag already exists: %TAG_NAME%
)

set "NOTES=Windows release build. Includes findfontman.exe, fontman runtime, Go/C++ SDKs, trained font_ai model/index data, launch scripts, and an empty fonts directory. Actual font files are not included."
set "FLAGS="
if "%DRAFT%"=="1" set "FLAGS=!FLAGS! --draft"
if "%PRERELEASE%"=="1" set "FLAGS=!FLAGS! --prerelease"

gh release view "%TAG_NAME%" >nul 2>nul
if errorlevel 1 (
    echo [fontman] Creating GitHub release: %TAG_NAME%
    gh release create "%TAG_NAME%" "%ZIP%" --title "%TITLE%" --notes "%NOTES%" --target main !FLAGS!
    if errorlevel 1 exit /b 1
) else (
    echo [fontman] Release already exists; uploading asset with --clobber.
    gh release upload "%TAG_NAME%" "%ZIP%" --clobber
    if errorlevel 1 exit /b 1
)

echo [fontman] GitHub release is ready:
echo https://github.com/ksamwang/find_fontman/releases/tag/%TAG_NAME%

endlocal
exit /b 0

:usage
echo Usage: release.bat [--skip-build] [--draft] [--prerelease]
echo.
echo Builds release\%RELEASE_NAME%, zips it, creates tag %TAG_NAME%,
echo and creates or updates the GitHub Release asset.
echo.
echo Options:
echo   --skip-build   Reuse an existing release\%RELEASE_NAME% directory.
echo   --draft        Create the GitHub Release as a draft when it does not exist.
echo   --prerelease   Mark the GitHub Release as a prerelease when it does not exist.
endlocal
exit /b 0

:usage_error
echo Run release.bat --help for usage.
endlocal
exit /b 1
