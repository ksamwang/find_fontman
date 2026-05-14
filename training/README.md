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

Quick smoke test:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\scripts\train_full_windows_gpu.ps1 -LimitFonts 20 -SamplesPerFont 5 -Epochs 1
```

Resume training:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\training\scripts\train_full_windows_gpu.ps1 -Resume
```

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
