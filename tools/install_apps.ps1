param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$Recreate,
    [switch]$SkipGradio,
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"

$rootReq = Join-Path $ProjectRoot "requirements.txt"
$rootVenv = Join-Path $ProjectRoot ".venv"
$rootPy = Join-Path $rootVenv "Scripts\python.exe"
$gradioSetup = Join-Path $ProjectRoot "tools\setup_local_gradio_venv.ps1"
$gradioPy = Join-Path $ProjectRoot "Gradio\.venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $rootReq)) {
    throw "Missing requirements.txt in project root: $ProjectRoot"
}

if ($Recreate -and (Test-Path -LiteralPath $rootVenv)) {
    Write-Host "Removing existing root venv: $rootVenv"
    Remove-Item -LiteralPath $rootVenv -Recurse -Force
}

if (-not (Test-Path -LiteralPath $rootPy)) {
    Write-Host "Creating root venv..."
    python -m venv $rootVenv
}

Write-Host "Installing main app dependencies..."
& $rootPy -m pip install --upgrade pip
& $rootPy -m pip install -r $rootReq

if (-not $SkipGradio) {
    if (-not (Test-Path -LiteralPath $gradioSetup)) {
        throw "Missing setup script: $gradioSetup"
    }
    Write-Host "Installing Gradio app dependencies..."
    if ($Recreate) {
        & powershell -ExecutionPolicy Bypass -File $gradioSetup -ProjectRoot $ProjectRoot -Recreate
    }
    else {
        & powershell -ExecutionPolicy Bypass -File $gradioSetup -ProjectRoot $ProjectRoot
    }
}

if ($SmokeTest) {
    Write-Host "Running smoke checks..."
    & $rootPy -c "import streamlit, pandas, numpy, faster_whisper; print('main venv: OK')"
    if ((-not $SkipGradio) -and (Test-Path -LiteralPath $gradioPy)) {
        & $gradioPy -c "import gradio, faster_whisper; print('gradio venv: OK')"
    }
}

Write-Host ""
Write-Host "Install complete."
Write-Host "Main Streamlit:"
Write-Host "  powershell -ExecutionPolicy Bypass -File tools\run_main_streamlit.ps1 -Port 8501"
Write-Host "Mini Streamlit:"
Write-Host "  powershell -ExecutionPolicy Bypass -File tools\run_mini_streamlit.ps1 -Port 8502"
Write-Host "Gradio:"
Write-Host "  powershell -ExecutionPolicy Bypass -File tools\run_gradio.ps1 -Port 7860"
