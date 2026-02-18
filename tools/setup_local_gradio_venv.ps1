param(
    [string]$ProjectRoot = "",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$gradioRoot = Join-Path $ProjectRoot "Gradio"
$venvPath = Join-Path $gradioRoot ".venv"
$py = Join-Path $venvPath "Scripts\python.exe"
$req = Join-Path $gradioRoot "requirements.txt"

if (-not (Test-Path -LiteralPath $gradioRoot)) {
    throw "Missing project folder: $gradioRoot"
}

if ($Recreate -and (Test-Path -LiteralPath $venvPath)) {
    Remove-Item -LiteralPath $venvPath -Recurse -Force
}

if (-not (Test-Path -LiteralPath $py)) {
    python -m venv $venvPath
}

& $py -m pip install --upgrade pip
& $py -m pip install -r $req

# Gradio 4.44 requires an older huggingface-hub API (HfFolder)
& $py -m pip install "huggingface-hub==0.25.2"

Write-Host "Local Gradio venv ready:"
Write-Host "  $py"
