# Standalone Font AI Training

This directory is a self-contained training project for Find Fontman font embeddings.
It does not install or require the Go service, OCR, PaddleOCR, or OpenCV.

## Windows GPU Training

Run from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\scripts\train_full_windows_gpu.ps1
```

Defaults:

- fonts: `fonts/`
- output: `training/output/font_ai/`
- samples per font: `1000`
- epochs: `8`
- device: `cuda`
- text source: built-in common word list plus seeded templates for names, companies, industries, promos, numbers, and Chinese/English mixed text
- sampling: balanced by font family/style/weight, with hard-negative tail samples controlled by `-HardNegativeRatio` / `--hard-negative-ratio`

Quick smoke test:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\scripts\train_full_windows_gpu.ps1 -LimitFonts 20 -SamplesPerFont 5 -Epochs 1
```

Resume training:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\scripts\train_full_windows_gpu.ps1 -Resume
```

Resume is epoch-granular. A checkpoint saved after a complete epoch resumes from the next epoch; a mid-epoch checkpoint restores weights but replays the current epoch so synthetic sampling stays deterministic and balanced.

Export artifacts to the main service:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\scripts\train_full_windows_gpu.ps1 -Resume -SkipIndex -ExportToMainData
```

The main service reads:

- `data/font_ai/font_embedding.pt`
- `data/font_ai/font_index.npz`
- `data/font_ai/font_index_meta.json`

The script can install Python 3.11 with `winget` when Python 3.11/3.12 is missing.
It does not install or upgrade NVIDIA drivers. If CUDA is unavailable, rerun with `-Device cpu`.

The script keeps the bundled pip by default because some mirrors may reject the latest pip wheel.
Use `-UpgradePip` only when you need it. If PyPI is slow from your network, switch mirrors:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\scripts\train_full_windows_gpu.ps1 -PipIndexUrl https://mirrors.aliyun.com/pypi/simple
```

If CUDA verification fails with `c10.dll` or `WinError 1114`, it is usually a Windows runtime or driver issue.
The script tries to install/repair Microsoft VC++ Redistributable with `winget`.
After that, reboot once, update the NVIDIA driver if needed, delete `training\.venv`, and rerun the script.
