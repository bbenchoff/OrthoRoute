<#
.SYNOPSIS
    Launch KiCad with OrthoRoute debug logging enabled.

.DESCRIPTION
    Sets ORTHO_DEBUG and optional screenshot env vars in the current process,
    then starts KiCad so it inherits them. Env vars are scoped to this session
    only — they are removed after KiCad exits.

.PARAMETER KiCadVersion
    KiCad major version to launch. Default: 9.0

.PARAMETER ScreenshotFreq
    Save a PCB screenshot every N routing iterations. Default: 5 (0 = every iteration).

.PARAMETER ScreenshotScale
    PNG resolution multiplier (e.g. 8 = 8× native). Default: 8

.PARAMETER NoScreenshots
    Skip screenshots entirely (log only, no PNG output).

.PARAMETER RunValidation
    Run automated smoke test validation after KiCad exits.
    Calls scripts/optimize_and_validate.ps1 with smoke test.

.EXAMPLE
    .\launch_kicad_debug.ps1
    .\launch_kicad_debug.ps1 -KiCadVersion 10.0
    .\launch_kicad_debug.ps1 -ScreenshotFreq 10 -ScreenshotScale 4
    .\launch_kicad_debug.ps1 -NoScreenshots
    .\launch_kicad_debug.ps1 -RunValidation  # Run smoke test after KiCad exits
#>

param(
    [string]$KiCadVersion   = "9.0",
    [int]   $ScreenshotFreq  = 5,
    [int]   $ScreenshotScale = 8,
    [switch]$NoScreenshots,
    [switch]$RunValidation
)

$kicadExe = "C:\Program Files\KiCad\$KiCadVersion\bin\kicad.exe"

if (-not (Test-Path $kicadExe)) {
    Write-Error "KiCad $KiCadVersion not found at: $kicadExe"
    Write-Host  "Available versions:"
    Get-ChildItem "C:\Program Files\KiCad" -Directory | Select-Object -ExpandProperty Name
    exit 1
}

# --- Set env vars (process scope only) ---
$env:ORTHO_DEBUG = '1'

if ($NoScreenshots) {
    Remove-Item Env:ORTHO_SCREENSHOT_FREQ  -ErrorAction SilentlyContinue
    Remove-Item Env:ORTHO_SCREENSHOT_SCALE -ErrorAction SilentlyContinue
} else {
    $env:ORTHO_SCREENSHOT_FREQ  = "$ScreenshotFreq"
    $env:ORTHO_SCREENSHOT_SCALE = "$ScreenshotScale"
}

# --- Print config ---
Write-Host ""
Write-Host "=== OrthoRoute Debug Launch ===" -ForegroundColor Cyan
Write-Host "  KiCad         : $kicadExe"
Write-Host "  ORTHO_DEBUG   : $env:ORTHO_DEBUG   (full DEBUG log + [ITER N] timing)"
if ($NoScreenshots) {
    Write-Host "  Screenshots   : disabled"
} else {
    Write-Host "  SCREENSHOT_FREQ  : every $ScreenshotFreq iteration(s)"
    Write-Host "  SCREENSHOT_SCALE : $ScreenshotScale x"
}
Write-Host ""
$docs      = [System.Environment]::GetFolderPath('MyDocuments')
$pluginDir = Join-Path $docs "KiCad\$KiCadVersion\3rdparty\plugins\com_github_bbenchoff_orthoroute"

Write-Host "  Plugin folder : $pluginDir"
Write-Host "  Plugin log    : $pluginDir\logs\latest.log"
Write-Host ""
Write-Host "  IPC pre-flight checklist:" -ForegroundColor Yellow
Write-Host "    1. Open a .kicad_pcb file in KiCad BEFORE running the plugin"
Write-Host "    2. KiCad menu: Preferences -> Plugins -> Enable Python API  (must be checked)"
Write-Host "    3. Restart KiCad after changing the Python API setting"
Write-Host ""
Write-Host "  Press Ctrl+C here to kill KiCad and clean up env vars." -ForegroundColor Yellow
Write-Host ""
# --- Sync plugin sources before launching ---
Write-Host "Syncing plugin sources..." -ForegroundColor DarkCyan
& "$PSScriptRoot\copy_to_kicad.ps1"
Write-Host ""
try {
    # Start KiCad — it inherits env vars from this process
    & $kicadExe
} finally {
    # Clean up regardless of how KiCad exits
    Remove-Item Env:ORTHO_DEBUG, Env:ORTHO_SCREENSHOT_FREQ, Env:ORTHO_SCREENSHOT_SCALE -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "KiCad exited. Debug env vars cleared." -ForegroundColor Green

    # Show log tail for convenience
    $pluginLog = Join-Path ([System.Environment]::GetFolderPath('MyDocuments')) `
        "KiCad\$KiCadVersion\3rdparty\plugins\com_github_bbenchoff_orthoroute\logs\latest.log"
    if (Test-Path $pluginLog) {
        Write-Host ""
        Write-Host "--- Last 40 lines of latest.log ---" -ForegroundColor Cyan
        Get-Content $pluginLog | Select-Object -Last 40
    } else {
        Write-Host "  No log found at: $pluginLog" -ForegroundColor Yellow
    }
    
    # --- Optional validation step ---
    if ($RunValidation) {
        Write-Host ""
        Write-Host "=== Running Automated Validation ===" -ForegroundColor Cyan
        Write-Host "  Smoke test (100 nets, <30s) to verify routing still works..."
        Write-Host ""
        
        $validateScript = Join-Path $PSScriptRoot "scripts\optimize_and_validate.ps1"
        if (Test-Path $validateScript) {
            try {
                & $validateScript -SkipDeploy -TestBoard smoke
                $validationExit = $LASTEXITCODE
                
                Write-Host ""
                if ($validationExit -eq 0) {
                    Write-Host "✅ VALIDATION PASSED: Routing successful" -ForegroundColor Green
                } elseif ($validationExit -eq 2) {
                    Write-Host "⚠️ VALIDATION WARNING: Performance regression detected" -ForegroundColor Yellow
                } else {
                    Write-Host "❌ VALIDATION FAILED: Routing errors detected (exit code $validationExit)" -ForegroundColor Red
                }
            } catch {
                Write-Host "❌ Validation script error: $_" -ForegroundColor Red
            }
        } else {
            Write-Warning "Validation script not found: $validateScript"
        }
    }
}
