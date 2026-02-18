param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

$env:RADIO_SPLITTER_RUNTIME_ROOT = "C:\radio_splitter_runtime"
$env:RADIO_SPLITTER_OUTPUT_ROOT = "C:\radio_splitter_runtime\output"
$env:RADIO_SPLITTER_GRADIO_OUTPUT_ROOT = "C:\radio_splitter_runtime\output_gradio"
$env:RADIO_SPLITTER_MINI_OUTPUT_ROOT = "C:\radio_splitter_runtime\mini_output"

$app = Join-Path $ProjectRoot "app_main.py"
$venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$py = if (Test-Path $venvPy) { $venvPy } else { "python" }

Push-Location $ProjectRoot
try {
    & $py -m streamlit run $app --server.port $Port
}
finally {
    Pop-Location
}
