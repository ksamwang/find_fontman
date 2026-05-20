# find_fontman

Image font recognition MVP. Go provides the debug web/API shell; Python is the main vision and font recognition SDK.

## Current Architecture

- Go web service: upload image, select text region, proxy OCR/match calls, stream progress with SSE.
- Python vision service: OCR crop handling, renderer matcher fallback, font AI training/inference.
- Font AI path: synthetic font dataset -> ArcFace embedding model -> NumPy cosine index -> online TopK retrieval.
- Runtime data stays under `data/`; local fonts stay under `fonts/`; neither is committed.

## Setup

```powershell
winget install -e --id Python.Python.3.11 --accept-package-agreements --accept-source-agreements
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple wheel
.\.venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r python_service\requirements.txt
```

Font AI requires PyTorch. Install a CPU or CUDA build separately for your machine:

```powershell
.\.venv\Scripts\python -m pip install torch torchvision
```

## Run

```powershell
go run .\cmd\findfontman
```

Open:

```text
http://localhost:8080
```

## Environment Variables

- `ADDR`: Go web address, default `:8080`
- `VISION_ADDR`: Python service address, default `127.0.0.1:9091`
- `PYTHON`: Python interpreter path
- `SKIP_PYTHON_SERVICE=1`: do not auto-start Python service
- `FONTMAN_MAX_CANDIDATES`: renderer fallback coarse candidate limit, `0` means all
- `FONTMAN_FINE_CANDIDATES`: renderer fallback fine candidate limit, `0` means all
- `FONTMAN_MATCH_WORKERS`: renderer fallback worker threads

## Font AI Workflow

Smoke train a small ArcFace model and build an index:

```powershell
.\scripts\train_ai_smoke.ps1 -LimitFonts 100 -SamplesPerFont 10 -Epochs 1
```

Full Chinese simplified training:

```powershell
.\.venv\Scripts\python -m python_service.font_ai.train --fonts "fonts\1中文简体" --samples-per-font 100 --epochs 8 --batch-size 16 --grad-accum 2
.\.venv\Scripts\python -m python_service.font_ai.build_index --fonts "fonts\1中文简体" --samples-per-font 8
```

When `data/font_ai/font_embedding.pt` and `data/font_ai/font_index.npz` exist, `/match` uses embedding retrieval. Otherwise it falls back to the renderer matcher.

## Benchmark

```powershell
.\scripts\benchmark.ps1 -SampleSize 20
```

Reports:

- `data/benchmark/report.json`
- `data/benchmark/report.html`

## SDK / Windows Runtime

Go and C++ callers should use the single-file SDKs under `sdk/` and call the local Fontman runtime service. Build a Windows runtime bundle that does not require Python on the target machine:

```powershell
.\scripts\build_runtime_windows.ps1 -IncludeData -IncludeFonts
```

See `sdk/README.md` for runtime startup and Go/C++ examples.
