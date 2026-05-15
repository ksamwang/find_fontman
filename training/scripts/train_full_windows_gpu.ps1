param(
    [int]$SamplesPerFont = 1000,
    [int]$Epochs = 8,
    [int]$BatchSize = 64,
    [int]$GradAccum = 1,
    [int]$Workers = 8,
    [int]$TorchThreads = 0,
    [int]$TorchInteropThreads = 0,
    [int]$IndexSamplesPerFont = 16,
    [int]$LogEvery = 25,
    [int]$ScanLogEvery = 100,
    [int]$CheckpointEvery = 500,
    [int]$LimitFonts = 0,
    [double]$Lr = 3e-4,
    [double]$HardNegativeRatio = 0.25,
    [double]$GradClip = 5.0,
    [int]$MaxSkippedSteps = 20,
    [ValidateSet("auto", "bfloat16", "float16", "none")]
    [string]$AmpDtype = "auto",
    [string]$Fonts = "fonts",
    [string]$Output = "training\output",
    [string]$Texts = "training\texts\zh_common.txt",
    [string]$PipIndexUrl = "https://pypi.org/simple",
    [string[]]$PipFallbackIndexUrls = @("https://mirrors.aliyun.com/pypi/simple", "https://pypi.tuna.tsinghua.edu.cn/simple", "https://mirrors.cloud.tencent.com/pypi/simple"),
    [string]$TorchCudaIndexUrl = "https://download.pytorch.org/whl/cu126",
    [string]$TorchCpuIndexUrl = "https://download.pytorch.org/whl/cpu",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$Resume,
    [switch]$SkipIndex,
    [switch]$ExportToMainData,
    [switch]$SkipInstall,
    [switch]$UpgradePip,
    [switch]$SkipVCRedistInstall
)

$ErrorActionPreference = "Stop"

function Find-Python {
    $candidates = @("py -3.12", "py -3.11", "python")
    foreach ($candidate in $candidates) {
        $parts = $candidate.Split(" ")
        $exe = $parts[0]
        $args = @()
        if ($parts.Length -gt 1) {
            $args = $parts[1..($parts.Length - 1)]
        }
        try {
            $output = & $exe @args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            $exitCode = $LASTEXITCODE
            if ($exitCode -eq 0 -and ($output -eq "3.11" -or $output -eq "3.12")) {
                return @{ Exe = $exe; Args = $args }
            }
        } catch {
        }
    }
    return $null
}

function Install-Python {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        throw "Python 3.11/3.12 was not found and winget is not available. Install Python 3.11 or 3.12, then rerun this script."
    }
    Write-Host "Installing Python 3.11 with winget..."
    winget install --id Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
}

function Invoke-Python {
    param(
        [hashtable]$PythonCommand,
        [string[]]$Arguments
    )
    & $PythonCommand.Exe @($PythonCommand.Args + $Arguments)
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$Label
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Invoke-PipInstallWithFallback {
    param(
        [string[]]$PipArguments,
        [string]$Label
    )
    $indexes = @($PipIndexUrl) + $PipFallbackIndexUrls
    $lastExit = 0
    foreach ($index in $indexes) {
        Write-Host "$Label using $index"
        $arguments = @("-m", "pip", "install", "--index-url", $index, "--prefer-binary") + $PipArguments
        & $python @arguments
        $lastExit = $LASTEXITCODE
        if ($lastExit -eq 0) {
            return
        }
        Write-Warning "$Label failed with $index, trying next mirror..."
    }
    throw "$Label failed with exit code $lastExit"
}

function Install-VCRedist {
    if ($SkipVCRedistInstall) {
        Write-Host "Skipping VC++ Redistributable installation because -SkipVCRedistInstall was provided."
        return
    }
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        Write-Warning "winget is not available; cannot auto-install Microsoft VC++ Redistributable."
        return
    }
    Write-Host "Ensuring Microsoft Visual C++ Redistributable is installed..."
    winget install --id Microsoft.VCRedist.2015+.x64 -e --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "VC++ Redistributable install/repair returned exit code $LASTEXITCODE. Continuing, but PyTorch DLL loading may fail."
    }
}

function Show-NvidiaInfo {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        Write-Warning "nvidia-smi was not found."
        return
    }
    Write-Host "NVIDIA driver status:"
    & nvidia-smi
}

function Verify-TorchCuda {
    Show-NvidiaInfo
    & $python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); raise SystemExit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) {
        throw @"
PyTorch CUDA verification failed.

Common Windows causes:
- Microsoft Visual C++ Redistributable is missing or broken.
- NVIDIA driver is too old for the selected PyTorch CUDA wheel.
- The current venv has a broken or mixed torch install.

Suggested fixes:
1. Reboot once if VC++ Redistributable was just installed or repaired.
2. Update the NVIDIA driver, then rerun this script.
3. Delete training\.venv and rerun this script to reinstall torch cleanly.
4. For a CPU-only smoke test, rerun with -Device cpu.

You can skip VC++ auto-install with -SkipVCRedistInstall if you manage it manually.
"@
    }
}

$root = (Resolve-Path ".").Path
$trainingRoot = Join-Path $root "training"
$venv = Join-Path $trainingRoot ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
$fontPath = (Resolve-Path $Fonts).Path
$outputPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $root $Output }
$requirements = Join-Path $trainingRoot "requirements.txt"
$lrText = $Lr.ToString([Globalization.CultureInfo]::InvariantCulture)
$hardNegativeRatioText = $HardNegativeRatio.ToString([Globalization.CultureInfo]::InvariantCulture)
$gradClipText = $GradClip.ToString([Globalization.CultureInfo]::InvariantCulture)

if (-not (Test-Path $venvPython)) {
    $pythonCommand = Find-Python
    if ($null -eq $pythonCommand) {
        Install-Python
        $pythonCommand = Find-Python
    }
    if ($null -eq $pythonCommand) {
        throw "Python 3.11/3.12 is still not available after installation attempt."
    }

    Write-Host "Creating venv at $venv"
    Invoke-Python -PythonCommand $pythonCommand -Arguments @("-m", "venv", $venv)
}

$python = $venvPython
$env:PYTHONUNBUFFERED = "1"
if ($TorchThreads -gt 0) {
    $env:OMP_NUM_THREADS = "$TorchThreads"
    $env:MKL_NUM_THREADS = "$TorchThreads"
}

if (-not $SkipInstall) {
    if ($UpgradePip) {
        Write-Host "Upgrading pip tooling..."
        Invoke-PipInstallWithFallback -PipArguments @("--upgrade", "pip", "wheel", "setuptools") -Label "pip tooling install"
    } else {
        Write-Host "Keeping existing pip. Use -UpgradePip if you want to upgrade pip/wheel/setuptools."
    }

    Write-Host "Installing standalone training requirements..."
    Invoke-PipInstallWithFallback -PipArguments @("-r", $requirements) -Label "training requirements install"

    if ($Device -eq "cuda") {
        $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($null -eq $nvidiaSmi) {
            throw "nvidia-smi was not found. Install/repair the NVIDIA driver, or rerun with -Device cpu."
        }
        Install-VCRedist
        Write-Host "Installing PyTorch CUDA wheel..."
        Invoke-Checked -FilePath $python -Arguments @("-m", "pip", "install", "torch", "torchvision", "--index-url", $TorchCudaIndexUrl) -Label "PyTorch CUDA install"
        Verify-TorchCuda
    } else {
        Write-Host "Installing PyTorch CPU wheel..."
        Invoke-Checked -FilePath $python -Arguments @("-m", "pip", "install", "torch", "torchvision", "--index-url", $TorchCpuIndexUrl) -Label "PyTorch CPU install"
    }
} else {
    Write-Host "Skipping dependency installation because -SkipInstall was provided."
}

Write-Host "Find Fontman standalone training"
Write-Host "Root: $root"
Write-Host "Fonts: $fontPath"
Write-Host "Output: $outputPath"
Write-Host "Device: $Device"
Write-Host "Limit fonts: $LimitFonts (0 means all)"
Write-Host "Samples/font: $SamplesPerFont"
Write-Host "Epochs: $Epochs"
Write-Host "Batch size: $BatchSize"
Write-Host "Workers: $Workers"
Write-Host "Learning rate: $lrText"
Write-Host "Hard negative ratio: $hardNegativeRatioText"
Write-Host "Gradient clip: $gradClipText"
Write-Host "Max skipped optimizer steps: $MaxSkippedSteps"
Write-Host "AMP dtype: $AmpDtype"
Write-Host ""

$trainArgs = @(
    "-m", "training.font_ai.train",
    "--root", $root,
    "--fonts", $fontPath,
    "--output", $outputPath,
    "--texts", $Texts,
    "--limit-fonts", "$LimitFonts",
    "--samples-per-font", "$SamplesPerFont",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--grad-accum", "$GradAccum",
    "--workers", "$Workers",
    "--torch-threads", "$TorchThreads",
    "--torch-interop-threads", "$TorchInteropThreads",
    "--lr", $lrText,
    "--device", $Device,
    "--log-every", "$LogEvery",
    "--scan-log-every", "$ScanLogEvery",
    "--checkpoint-every", "$CheckpointEvery",
    "--hard-negative-ratio", $hardNegativeRatioText,
    "--grad-clip", $gradClipText,
    "--max-skipped-steps", "$MaxSkippedSteps",
    "--amp-dtype", $AmpDtype
)

if ($Resume) {
    $trainArgs += "--resume"
}

Invoke-Checked -FilePath $python -Arguments $trainArgs -Label "font AI training"

if (-not $SkipIndex) {
    Invoke-Checked -FilePath $python -Arguments @(
        "-m", "training.font_ai.build_index",
        "--root", $root,
        "--fonts", $fontPath,
        "--output", $outputPath,
        "--samples-per-font", "$IndexSamplesPerFont",
        "--device", $Device,
        "--log-every", "25"
    ) -Label "font AI index build"
}

if ($ExportToMainData) {
    $source = Join-Path $outputPath "font_ai"
    $target = Join-Path $root "data\font_ai"
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Copy-Item -Force (Join-Path $source "font_embedding.pt") $target
    Copy-Item -Force (Join-Path $source "font_index.npz") $target
    Copy-Item -Force (Join-Path $source "font_index_meta.json") $target
    Write-Host "Exported model artifacts to $target"
}
