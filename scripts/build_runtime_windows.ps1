param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$PipIndexUrl = "",
    [switch]$IncludeData,
    [switch]$IncludeFonts
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Python)) {
    $Python = "python"
}

$pipArgs = @("-m", "pip", "install", "pyinstaller")
if (![string]::IsNullOrWhiteSpace($PipIndexUrl)) {
    $pipArgs += @("-i", $PipIndexUrl)
}
& $Python @pipArgs

& $Python -m PyInstaller --clean --noconfirm .\scripts\fontman_runtime.spec

$runtime = Join-Path (Get-Location) "dist\fontman-runtime"
if (!(Test-Path $runtime)) {
    throw "Runtime output was not created: $runtime"
}

if ($IncludeData -and (Test-Path ".\data\font_ai")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $runtime "data") | Out-Null
    Copy-Item -Recurse -Force ".\data\font_ai" (Join-Path $runtime "data\font_ai")
}

if ($IncludeFonts -and (Test-Path ".\fonts")) {
    Copy-Item -Recurse -Force ".\fonts" (Join-Path $runtime "fonts")
}

$readme = @"
# Fontman Runtime

Run:

````powershell
.\fontman-service.exe --addr 127.0.0.1:9091 --root . --fonts .\fonts --data .\data --previews .\data\previews
````

The Go and C++ SDKs call `http://127.0.0.1:9091` by default.
"@
$readme | Set-Content -Encoding UTF8 (Join-Path $runtime "README.md")

Write-Host "Runtime built at $runtime"
