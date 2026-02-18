param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

$env:RADIO_SPLITTER_RUNTIME_ROOT = "C:\radio_splitter_runtime"
$env:RADIO_SPLITTER_OUTPUT_ROOT = "C:\radio_splitter_runtime\output"
$env:RADIO_SPLITTER_GRADIO_OUTPUT_ROOT = "C:\radio_splitter_runtime\output_gradio"
$env:RADIO_SPLITTER_MINI_OUTPUT_ROOT = "C:\radio_splitter_runtime\mini_output"
$env:GRADIO_SERVER_PORT = "$Port"

$gradioRoot = Join-Path $ProjectRoot "Gradio"
$app = Join-Path $gradioRoot "app_gradio.py"

$candidates = @(
    (Join-Path $gradioRoot ".venv\Scripts\python.exe"),
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
    (Join-Path $gradioRoot "venv\Scripts\python.exe")
)

$py = $null
foreach ($cand in $candidates) {
    if (-not (Test-Path $cand)) {
        continue
    }
    $venvRoot = Split-Path (Split-Path $cand -Parent) -Parent
    $sitePackages = Join-Path $venvRoot "Lib\site-packages"
    $hasGradio = Test-Path (Join-Path $sitePackages "gradio")
    $hasPil = Test-Path (Join-Path $sitePackages "PIL")
    if ($hasGradio -and $hasPil) {
        $py = $cand
        break
    }
}

if (-not $py) {
    $py = "python"
}

Push-Location $ProjectRoot
try {
    Write-Host "Using python: $py"
    if ($py -match "iCloudDrive") {
        Write-Warning "Detected python from iCloud path. Prefer tools/setup_local_gradio_venv.ps1 and tools/run_gradio_local.ps1."
    }
    & $py $app
}
finally {
    Pop-Location
}
