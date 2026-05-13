param(
    [int]$LimitFonts = 100,
    [int]$SamplesPerFont = 10,
    [int]$Epochs = 1,
    [string]$Fonts = "fonts\1中文简体"
)

$ErrorActionPreference = "Stop"

.\.venv\Scripts\python -m python_service.font_ai.train --fonts $Fonts --limit-fonts $LimitFonts --samples-per-font $SamplesPerFont --epochs $Epochs --batch-size 8 --grad-accum 2
.\.venv\Scripts\python -m python_service.font_ai.build_index --fonts $Fonts --samples-per-font 2
