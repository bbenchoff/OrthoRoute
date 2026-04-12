# copy_to_kicad.ps1
# Fast dev sync: copies Python sources and config directly to the local KiCad plugin folder.
# Use instead of `python build.py --deploy` when you only changed .py files or orthoroute.json.
#
# Usage:  .\copy_to_kicad.ps1
#         .\copy_to_kicad.ps1 -Verbose   # show each copied file
#         .\copy_to_kicad.ps1 -Validate  # run smoke test after copy

[CmdletBinding()]
param(
    [switch]$Validate
)

$src = $PSScriptRoot

# Resolve the KiCad plugin directory portably.
# KiCad 9.0 stores 3rd-party plugins under Documents\KiCad\9.0\3rdparty\plugins\.
# [Environment]::GetFolderPath('MyDocuments') follows OneDrive/corporate folder
# redirection automatically, so this works on any machine.
$kicadVersion = "9.0"
$pluginName   = "com_github_bbenchoff_orthoroute"
$docs = [System.Environment]::GetFolderPath('MyDocuments')
$dst  = Join-Path $docs "KiCad\$kicadVersion\3rdparty\plugins\$pluginName"

if (-not (Test-Path $dst)) {
    Write-Error "Plugin directory not found: $dst`nRun 'python build.py --deploy' first to create it."
    exit 1
}

$script:copied = 0
$script:errors = 0

function Sync-File {
    param([string]$SrcPath, [string]$DstPath)
    $dstDir = Split-Path $DstPath -Parent
    if (-not (Test-Path $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    try {
        Copy-Item -Path $SrcPath -Destination $DstPath -Force
        Write-Verbose "  copied  $($SrcPath.Replace($src, '').TrimStart('\'))"
        $script:copied++
    } catch {
        Write-Warning "  FAILED  $SrcPath : $_"
        $script:errors++
    }
}

Write-Host "Syncing to: $dst"

# --- main.py ---------------------------------------------------------------
Sync-File "$src\main.py" "$dst\main.py"

# --- swig_init.py -> __init__.py (KiCad SWIG ActionPlugin entry point) ------
if (Test-Path "$src\swig_init.py") {
    Sync-File "$src\swig_init.py" "$dst\__init__.py"
}

# --- orthoroute.json (runtime config) --------------------------------------
if (Test-Path "$src\orthoroute.json") {
    Sync-File "$src\orthoroute.json" "$dst\orthoroute.json"
}

# --- graphics/kicad_theme.json (PCB viewer color theme) --------------------
if (Test-Path "$src\graphics\kicad_theme.json") {
    Sync-File "$src\graphics\kicad_theme.json" "$dst\graphics\kicad_theme.json"
}

# --- plugin.json and metadata.json (KiCad plugin registration) ------------
# Copy from build/ directory if available, otherwise create minimal versions
$buildDir = Join-Path $src "build\com_github_bbenchoff_orthoroute"
if (Test-Path "$buildDir\plugin.json") {
    Sync-File "$buildDir\plugin.json" "$dst\plugin.json"
}
if (Test-Path "$buildDir\metadata.json") {
    Sync-File "$buildDir\metadata.json" "$dst\metadata.json"
}

# --- Icon files (toolbar icon support) -------------------------------------
if (Test-Path "$buildDir\icon-24.png") {
    Sync-File "$buildDir\icon-24.png" "$dst\icon-24.png"
}
if (Test-Path "$buildDir\icon-64.png") {
    Sync-File "$buildDir\icon-64.png" "$dst\icon-64.png"
}

# --- orthoroute/ package (all .py files, preserving sub-directory structure)
Get-ChildItem -Path "$src\orthoroute" -Recurse -Filter "*.py" | ForEach-Object {
    $rel = $_.FullName.Substring("$src\orthoroute".Length)   # e.g. \algorithms\manhattan\unified_pathfinder.py
    Sync-File $_.FullName "$dst\orthoroute$rel"
}

# --- Summary ---------------------------------------------------------------
$status = if ($script:errors -eq 0) { "OK" } else { "ERRORS" }
Write-Host "[$status] $($script:copied) file(s) copied, $($script:errors) error(s)"
if ($script:errors -gt 0) { exit 1 }

# --- Optional validation step ---
if ($Validate) {
    Write-Host ""
    Write-Host "=== Running Quick Validation ===" -ForegroundColor Cyan
    Write-Host "  Smoke test (100 nets, <30s) to check for breakage..."
    Write-Host ""
    
    $validateScript = Join-Path $src "scripts\optimize_and_validate.ps1"
    if (Test-Path $validateScript) {
        try {
            & $validateScript -SkipDeploy -TestBoard smoke -Compare "tests/regression/smoke_metrics.json"
            $validationExit = $LASTEXITCODE
            
            Write-Host ""
            if ($validationExit -eq 0) {
                Write-Host "✅ VALIDATION PASSED: Changes safe to use" -ForegroundColor Green
            } elseif ($validationExit -eq 2) {
                Write-Host "⚠️ VALIDATION WARNING: Performance regression detected" -ForegroundColor Yellow
            } else {
                Write-Host "❌ VALIDATION FAILED: Routing errors detected (exit code $validationExit)" -ForegroundColor Red
                exit $validationExit
            }
        } catch {
            Write-Host "❌ Validation script error: $_" -ForegroundColor Red
            exit 3
        }
    } else {
        Write-Warning "Validation script not found: $validateScript"
        Write-Host "  Skipping validation (script creation in progress?)"
    }
}
