param(
    [string]$DestinationRoot = "C:\radio_splitter_runtime",
    [string]$ProjectRoot = "",
    [string]$LocalCloneRoot = "",
    [int]$Threads = 32,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Section([string]$msg) {
    Write-Host ""
    Write-Host "== $msg ==" -ForegroundColor Cyan
}

function Ensure-Dir([string]$path) {
    if (-not (Test-Path -LiteralPath $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

function Move-TreeFast([string]$SourceDir, [string]$TargetDir, [int]$Mt, [bool]$WhatIf) {
    if (-not (Test-Path -LiteralPath $SourceDir)) {
        return
    }

    $items = Get-ChildItem -LiteralPath $SourceDir -Force -ErrorAction SilentlyContinue
    if (-not $items -or $items.Count -eq 0) {
        return
    }

    Ensure-Dir $TargetDir

    $args = @(
        "`"$SourceDir`"",
        "`"$TargetDir`"",
        "/E",
        "/MOVE",
        "/R:1",
        "/W:1",
        "/MT:$Mt",
        "/NFL",
        "/NDL",
        "/NP",
        "/NJH",
        "/NJS"
    )

    if ($WhatIf) {
        Write-Host "[DRY-RUN] robocopy $($args -join ' ')" -ForegroundColor Yellow
        return
    }

    & robocopy @args | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -ge 8) {
        throw "robocopy failed for '$SourceDir' -> '$TargetDir' (exit code $rc)"
    }
}

Write-Section "Preparing destination"
Ensure-Dir $DestinationRoot
$destOutput = Join-Path $DestinationRoot "output"
$destOutputGradio = Join-Path $DestinationRoot "output_gradio"
$destLegacy = Join-Path $DestinationRoot "legacy_output"
Ensure-Dir $destOutput
Ensure-Dir $destOutputGradio
Ensure-Dir $destLegacy

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($LocalCloneRoot)) {
    if (Test-Path -LiteralPath "C:\Users\brian\Programmering\radio_splitter2") {
        $LocalCloneRoot = "C:\Users\brian\Programmering\radio_splitter2"
    }
    elseif (Test-Path -LiteralPath "C:\Users\brian\Programmering\radio_splitter2_local") {
        $LocalCloneRoot = "C:\Users\brian\Programmering\radio_splitter2_local"
    }
    else {
        $LocalCloneRoot = "C:\dev\radio_splitter2_local"
    }
}

$appDataRoot = "C:\Users\brian\AppData\Local\radio_splitter2"

$sources = @(
    @{ Label = "project-output"; Path = (Join-Path $ProjectRoot "output"); Target = (Join-Path $destLegacy "from_project_output") },
    @{ Label = "project-output-gradio"; Path = (Join-Path $ProjectRoot "output_gradio"); Target = (Join-Path $destLegacy "from_project_output_gradio") },
    @{ Label = "project-gradio-output-gradio"; Path = (Join-Path $ProjectRoot "Gradio\output_gradio"); Target = (Join-Path $destLegacy "from_project_gradio_output_gradio") },
    @{ Label = "local-clone-output"; Path = (Join-Path $localCloneRoot "output"); Target = (Join-Path $destLegacy "from_local_clone_output") },
    @{ Label = "local-clone-output-gradio"; Path = (Join-Path $localCloneRoot "output_gradio"); Target = (Join-Path $destLegacy "from_local_clone_output_gradio") },
    @{ Label = "local-clone-gradio-output-gradio"; Path = (Join-Path $localCloneRoot "Gradio\output_gradio"); Target = (Join-Path $destLegacy "from_local_clone_gradio_output_gradio") },
    @{ Label = "appdata-output"; Path = (Join-Path $appDataRoot "output"); Target = $destOutput },
    @{ Label = "appdata-output-gradio"; Path = (Join-Path $appDataRoot "output_gradio"); Target = $destOutputGradio }
)

Write-Section "Moving output trees"
foreach ($s in $sources) {
    $src = $s.Path
    if (-not (Test-Path -LiteralPath $src)) {
        continue
    }
    Write-Host "Source: $($s.Label) -> $src"
    Write-Host "Target: $($s.Target)"
    Move-TreeFast -SourceDir $src -TargetDir $s.Target -Mt $Threads -WhatIf:$DryRun
}

Write-Section "Done"
if ($DryRun) {
    Write-Host "Dry run complete. Re-run without -DryRun to execute."
} else {
    Write-Host "Move complete. Legacy output is now under: $destLegacy"
    Write-Host "Active runtime output folders:"
    Write-Host "  $destOutput"
    Write-Host "  $destOutputGradio"
}
