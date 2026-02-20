param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$Port = 8502,
    [string]$Address = "127.0.0.1",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$env:RADIO_SPLITTER_RUNTIME_ROOT = "C:\radio_splitter_runtime"
$env:RADIO_SPLITTER_OUTPUT_ROOT = "C:\radio_splitter_runtime\output"
$env:RADIO_SPLITTER_GRADIO_OUTPUT_ROOT = "C:\radio_splitter_runtime\output_gradio"
$env:RADIO_SPLITTER_MINI_OUTPUT_ROOT = "C:\radio_splitter_runtime\mini_output"

$app = Join-Path $ProjectRoot "mini_splitter\app_mini.py"
$venvPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$py = if (Test-Path $venvPy) { $venvPy } else { "python" }

Push-Location $ProjectRoot
try {
    $headlessValue = if ($Headless.IsPresent) { "true" } else { "false" }
    & $py -m streamlit run $app --server.port $Port --server.address $Address --server.headless $headlessValue
}
finally {
    Pop-Location
}
