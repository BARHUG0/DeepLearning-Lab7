param([int]$TrainingProcessId = 0)

$ErrorActionPreference = 'Stop'
$Repository = Split-Path -Parent $PSScriptRoot
$Python = Join-Path (Split-Path -Parent $Repository) '.venv\Scripts\python.exe'
Set-Location -LiteralPath $Repository

if ($TrainingProcessId -ne 0) {
    if (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
        Write-Output "Waiting for embedding training process $TrainingProcessId."
        Wait-Process -Id $TrainingProcessId
    }
    if (-not (Test-Path -LiteralPath 'informe/results/gensim.json')) {
        throw 'Embedding training did not finish. Inspect informe/artifacts/training.log.'
    }
} else {
    foreach ($Stage in @('prepare', 'pilot', 'embeddings')) {
        & $Python -u -m src.run $Stage
        if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed." }
    }
}

foreach ($Stage in @('arithmetic', 'classification', 'final')) {
    Write-Output "Starting $Stage."
    & $Python -u -m src.run $Stage
    if ($LASTEXITCODE -ne 0) { throw "Stage $Stage failed." }
}

& $Python -u -m src.deliverables
if ($LASTEXITCODE -ne 0) { throw 'Notebook and report generation failed.' }

quarto render informe/report.md --to pdf
if ($LASTEXITCODE -ne 0) { throw 'PDF rendering failed.' }

& $Python -u -m src.deliverables --validate
if ($LASTEXITCODE -ne 0) { throw 'Final deliverable validation failed.' }

Write-Output 'All experiments and deliverable checks completed successfully.'
