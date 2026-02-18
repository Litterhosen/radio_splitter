param(
    [string]$ProjectRoot = "",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$env:RADIO_SPLITTER_RUNTIME_ROOT = "C:\radio_splitter_runtime"
$env:RADIO_SPLITTER_OUTPUT_ROOT = "C:\radio_splitter_runtime\output"
$env:RADIO_SPLITTER_GRADIO_OUTPUT_ROOT = "C:\radio_splitter_runtime\output_gradio"

$gradioRoot = Join-Path $ProjectRoot "Gradio"
$py = Join-Path $gradioRoot ".venv\Scripts\python.exe"
$app = Join-Path $gradioRoot "app_gradio.py"

if (-not (Test-Path -LiteralPath $py)) {
    throw "Missing venv python. Run: powershell -ExecutionPolicy Bypass -File tools\setup_local_gradio_venv.ps1"
}

Push-Location $gradioRoot
try {
    $env:GRADIO_SERVER_PORT = "$Port"
    & $py -c "import sys,streamlit; print(sys.executable); print('streamlit', streamlit.__version__)"
    & $py $app
}
finally {
    Pop-Location
}
