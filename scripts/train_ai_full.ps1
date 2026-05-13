param(
    [int]$SamplesPerFont = 1000,
    [int]$Epochs = 8,
    [int]$BatchSize = 32,
    [int]$GradAccum = 1,
    [int]$Workers = 20,
    [int]$TorchThreads = 40,
    [int]$TorchInteropThreads = 4,
    [int]$IndexSamplesPerFont = 16,
    [int]$LogEvery = 25,
    [int]$ScanLogEvery = 100,
    [int]$LimitFonts = 0,
    [string]$Fonts = "fonts",
    [switch]$Resume,
    [switch]$SkipIndex
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path ".").Path
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$fontPath = (Resolve-Path $Fonts).Path
$env:PYTHONUNBUFFERED = "1"
$env:OMP_NUM_THREADS = "$TorchThreads"
$env:MKL_NUM_THREADS = "$TorchThreads"

Write-Host "Find Fontman full CPU training"
Write-Host "Root: $root"
Write-Host "Fonts: $fontPath"
Write-Host "Limit fonts: $LimitFonts (0 means all)"
Write-Host "Samples/font: $SamplesPerFont"
Write-Host "Epochs: $Epochs"
Write-Host "Batch size: $BatchSize"
Write-Host "Workers: $Workers"
Write-Host "Torch threads: $TorchThreads"
Write-Host "Index samples/font: $IndexSamplesPerFont"
Write-Host ""

$trainArgs = @(
    "-m", "python_service.font_ai.train",
    "--root", $root,
    "--fonts", $fontPath,
    "--limit-fonts", "$LimitFonts",
    "--samples-per-font", "$SamplesPerFont",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--grad-accum", "$GradAccum",
    "--workers", "$Workers",
    "--torch-threads", "$TorchThreads",
    "--torch-interop-threads", "$TorchInteropThreads",
    "--device", "cpu",
    "--log-every", "$LogEvery",
    "--scan-log-every", "$ScanLogEvery"
)

if ($Resume) {
    $trainArgs += "--resume"
}

& $python @trainArgs

if (-not $SkipIndex) {
    & $python -m python_service.font_ai.build_index `
        --root $root `
        --fonts $fontPath `
        --samples-per-font $IndexSamplesPerFont `
        --device cpu `
        --log-every 25
}
