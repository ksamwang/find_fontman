$ErrorActionPreference = "Stop"

param(
    [int]$LimitFonts = 100,
    [int]$SamplesPerFont = 10,
    [int]$Epochs = 1
)

.\.venv\Scripts\python -m python_service.font_ai.train --limit-fonts $LimitFonts --samples-per-font $SamplesPerFont --epochs $Epochs --batch-size 8 --grad-accum 2
.\.venv\Scripts\python -m python_service.font_ai.build_index --samples-per-font 2
