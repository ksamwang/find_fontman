$ErrorActionPreference = "Stop"

param(
    [int]$SampleSize = 20,
    [string]$Texts = "data\benchmark_texts.txt"
)

.\.venv\Scripts\python -m python_service.vision.benchmark --texts $Texts --sample-size $SampleSize
