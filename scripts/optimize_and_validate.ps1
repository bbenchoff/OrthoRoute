<#
.SYNOPSIS
    Automated optimization workflow: deploy, test, and validate OrthoRoute changes.

.DESCRIPTION
    Streamlines the optimization cycle by automating:
    1. Deploy code to KiCad plugin folder (unless -SkipDeploy)
    2. Run regression test (smoke or backplane)
    3. Parse log to extract metrics
    4. Validate against golden thresholds (if -Compare specified)
    5. Report results with clear exit codes

    Use this script after making code changes to quickly validate that:
    - Routing still succeeds (all nets routed, converged)
    - Performance hasn't regressed significantly
    - No critical errors introduced

.PARAMETER TestBoard
    Which test to run:
      - 'smoke'     : 100 nets, <30s, fast validation (default)
      - 'backplane' : 512 nets, 11-18 min, full golden run

.PARAMETER SkipDeploy
    Skip copy_to_kicad.ps1 sync step. Use when plugin folder is already up to date.

.PARAMETER ProfileMode
    Enable ORTHO_DEBUG=1 for detailed profiling logs. Increases log size ~10-20×.

.PARAMETER Compare
    Path to golden metrics file for validation (e.g., tests/regression/smoke_metrics.json).
    If specified, script will compare results and fail/warn on regressions.

.PARAMETER ShowLog
    Display the full log file after test completes (useful for debugging).

.PARAMETER Json
    Output results as JSON instead of human-readable format.

.EXAMPLE
    .\scripts\optimize_and_validate.ps1
    # Quick smoke test with default settings (100 nets, no profiling)

.EXAMPLE
    .\scripts\optimize_and_validate.ps1 -ProfileMode -Compare tests/regression/smoke_metrics.json
    # Full validation: debug logs + golden comparison

.EXAMPLE
    .\scripts\optimize_and_validate.ps1 -TestBoard backplane -SkipDeploy
    # Full backplane test, assume code already deployed

.EXAMPLE
    .\scripts\optimize_and_validate.ps1 -Json > results.json
    # Export metrics as JSON for automation/CI

.NOTES
    Exit codes:
      0 = Success (routing completed, all validations passed)
      1 = Routing failed (nets not routed, convergence failed, or hard errors)
      2 = Performance regression detected (soft warnings from golden comparison)
      3 = Script/environment error (prerequisites missing, files not found)
#>

[CmdletBinding()]
param(
    [ValidateSet('smoke', 'backplane')]
    [string]$TestBoard = 'smoke',
    
    [switch]$SkipDeploy,
    [switch]$ProfileMode,
    
    [string]$Compare = '',
    [switch]$ShowLog,
    [switch]$Json
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================

function Test-Prerequisites {
    param([string]$Root)
    
    $issues = @()
    
    # Check Python
    try {
        $null = python --version 2>&1
    } catch {
        $issues += "Python not found in PATH"
    }
    
    # Check pytest
    try {
        $null = pytest --version 2>&1
    } catch {
        $issues += "pytest not found (run: pip install -r requirements.txt)"
    }
    
    # Check analyze_log.py
    $analyzeScript = Join-Path $Root "scripts\analyze_log.py"
    if (-not (Test-Path $analyzeScript)) {
        $issues += "scripts/analyze_log.py not found"
    }
    
    # Check test files
    $testFile = if ($TestBoard -eq 'smoke') {
        Join-Path $Root "tests\regression\test_smoke.py"
    } else {
        Join-Path $Root "tests\regression\test_backplane.py"
    }
    
    if (-not (Test-Path $testFile)) {
        $issues += "Test file not found: $testFile"
    }
    
    # Check golden metrics if comparison requested
    if ($Compare -and -not (Test-Path $Compare)) {
        $issues += "Golden metrics file not found: $Compare"
    }
    
    if ($issues.Count -gt 0) {
        Write-Host "❌ Prerequisites check failed:" -ForegroundColor Red
        $issues | ForEach-Object { Write-Host "   - $_" -ForegroundColor Yellow }
        exit 3
    }
}

# =============================================================================
# DEPLOYMENT
# =============================================================================

function Invoke-Deployment {
    param([string]$Root)
    
    Write-Host ""
    Write-Host "📦 DEPLOYMENT" -ForegroundColor Cyan
    Write-Host ("=" * 80)
    
    $copyScript = Join-Path $Root "copy_to_kicad.ps1"
    
    if ($SkipDeploy) {
        Write-Host "  Skipped (--SkipDeploy)" -ForegroundColor DarkGray
        return $true
    }
    
    if (-not (Test-Path $copyScript)) {
        Write-Warning "copy_to_kicad.ps1 not found, skipping deployment"
        return $true
    }
    
    try {
        & $copyScript
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Deployment failed (exit code $LASTEXITCODE)" -ForegroundColor Red
            return $false
        }
        Write-Host "✅ Deployment successful" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ Deployment error: $_" -ForegroundColor Red
        return $false
    }
}

# =============================================================================
# TESTING
# =============================================================================

function Invoke-RegressionTest {
    param(
        [string]$Root,
        [string]$Board,
        [bool]$EnableProfiling
    )
    
    Write-Host ""
    Write-Host "🧪 TESTING" -ForegroundColor Cyan
    Write-Host ("=" * 80)
    Write-Host "  Test board:    $Board"
    Write-Host "  Profiling:     $(if ($EnableProfiling) { 'ENABLED (ORTHO_DEBUG=1)' } else { 'disabled' })"
    
    if ($Board -eq 'smoke') {
        $testTarget = "tests/regression/test_smoke.py::TestSmokeRouting::test_smoke_routing_pipeline"
        $expectedTime = "~30s"
    } else {
        $testTarget = "tests/regression/test_backplane.py::TestHeadlessRouting::test_headless_routing_pipeline"
        $expectedTime = "~11-18 min"
    }
    
    Write-Host "  Expected time: $expectedTime"
    Write-Host ""
    
    # Set environment for profiling
    if ($EnableProfiling) {
        $env:ORTHO_DEBUG = '1'
    } else {
        Remove-Item Env:ORTHO_DEBUG -ErrorAction SilentlyContinue
    }
    
    try {
        Push-Location $Root
        
        Write-Host "Running: pytest $testTarget -v"
        Write-Host ""
        
        $startTime = Get-Date
        pytest $testTarget -v --tb=short
        $exitCode = $LASTEXITCODE
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        Write-Host ""
        Write-Host "Test completed in $([math]::Round($duration, 1))s (exit code: $exitCode)"
        
        return @{
            Success = ($exitCode -eq 0)
            ExitCode = $exitCode
            Duration = $duration
        }
        
    } finally {
        Pop-Location
        Remove-Item Env:ORTHO_DEBUG -ErrorAction SilentlyContinue
    }
}

# =============================================================================
# LOG ANALYSIS
# =============================================================================

function Get-LatestLogPath {
    param([string]$Root)
    
    $logsDir = Join-Path $Root "logs"
    if (-not (Test-Path $logsDir)) {
        return $null
    }
    
    $latestLog = Join-Path $logsDir "latest.log"
    if (Test-Path $latestLog) {
        return $latestLog
    }
    
    # Fallback: find most recent timestamped log
    $logs = Get-ChildItem $logsDir -Filter "*.log" | Sort-Object LastWriteTime -Descending
    if ($logs.Count -gt 0) {
        return $logs[0].FullName
    }
    
    return $null
}

function Invoke-LogAnalysis {
    param(
        [string]$Root,
        [string]$LogPath,
        [string]$GoldenPath,
        [bool]$JsonOutput
    )
    
    Write-Host ""
    Write-Host "📊 LOG ANALYSIS" -ForegroundColor Cyan
    Write-Host ("=" * 80)
    
    if (-not $LogPath -or -not (Test-Path $LogPath)) {
        Write-Host "❌ Log file not found: $LogPath" -ForegroundColor Red
        return $null
    }
    
    Write-Host "  Log file: $LogPath"
    
    $analyzeScript = Join-Path $Root "scripts\analyze_log.py"
    $args = @("--log-file", $LogPath)
    
    if ($GoldenPath) {
        $args += @("--compare", $GoldenPath)
        Write-Host "  Golden:   $GoldenPath"
    }
    
    if ($JsonOutput) {
        $args += "--json"
    }
    
    Write-Host ""
    
    try {
        $output = & python $analyzeScript @args
        $exitCode = $LASTEXITCODE
        
        # Display output
        $output | ForEach-Object { Write-Host $_ }
        
        # Parse JSON if in JSON mode to extract status
        if ($JsonOutput) {
            $result = $output | ConvertFrom-Json
            return @{
                Success = ($exitCode -eq 0)
                ExitCode = $exitCode
                Metrics = $result
            }
        } else {
            return @{
                Success = ($exitCode -eq 0)
                ExitCode = $exitCode
                Metrics = $null
            }
        }
        
    } catch {
        Write-Host "❌ Log analysis error: $_" -ForegroundColor Red
        return $null
    }
}

# =============================================================================
# MAIN
# =============================================================================

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  OrthoRoute Optimization & Validation Workflow" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Check prerequisites
Test-Prerequisites -Root $repoRoot

# Step 1: Deploy
$deploySuccess = Invoke-Deployment -Root $repoRoot
if (-not $deploySuccess) {
    Write-Host ""
    Write-Host "❌ WORKFLOW FAILED: Deployment error" -ForegroundColor Red
    exit 3
}

# Step 2: Run test
$testResult = Invoke-RegressionTest -Root $repoRoot -Board $TestBoard -EnableProfiling $ProfileMode
if (-not $testResult.Success) {
    Write-Host ""
    Write-Host "❌ WORKFLOW FAILED: Test failed (exit code $($testResult.ExitCode))" -ForegroundColor Red
    Write-Host "   Routing did not complete successfully or test encountered errors" -ForegroundColor Yellow
    exit 1
}

# Step 3: Analyze log
$logPath = Get-LatestLogPath -Root $repoRoot
$analysisResult = Invoke-LogAnalysis -Root $repoRoot -LogPath $logPath -GoldenPath $Compare -JsonOutput $Json

if ($null -eq $analysisResult) {
    Write-Host ""
    Write-Host "⚠️  WORKFLOW WARNING: Log analysis failed" -ForegroundColor Yellow
    Write-Host "   Test passed but could not parse metrics" -ForegroundColor Yellow
    exit 0  # Test succeeded, analysis is bonus
}

# Step 4: Display log if requested
if ($ShowLog -and $logPath -and (Test-Path $logPath)) {
    Write-Host ""
    Write-Host "📄 FULL LOG" -ForegroundColor Cyan
    Write-Host ("=" * 80)
    Get-Content $logPath
}

# Step 5: Final status
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════════════════" -ForegroundColor Cyan

if ($Compare) {
    # With golden comparison
    if ($analysisResult.ExitCode -eq 0) {
        Write-Host "✅ WORKFLOW COMPLETE: All validations passed" -ForegroundColor Green
        Write-Host "   Test passed ✓  Golden comparison passed ✓" -ForegroundColor Green
        exit 0
    } elseif ($analysisResult.ExitCode -eq 1 -and $analysisResult.Metrics) {
        # Check if it's a soft warning or hard failure
        $comparison = $analysisResult.Metrics.comparison
        if ($comparison -and $comparison.overall_status -eq 'WARN') {
            Write-Host "⚠️  WORKFLOW COMPLETE: Performance regression detected" -ForegroundColor Yellow
            Write-Host "   Test passed ✓  Golden comparison shows warnings (soft regression)" -ForegroundColor Yellow
            exit 2  # Soft regression
        } else {
            Write-Host "❌ WORKFLOW FAILED: Golden comparison failed" -ForegroundColor Red
            Write-Host "   Test passed but metrics outside acceptable thresholds" -ForegroundColor Red
            exit 1  # Hard failure
        }
    } else {
        Write-Host "❌ WORKFLOW FAILED: Log analysis error" -ForegroundColor Red
        exit 1
    }
} else {
    # No comparison, just test success
    Write-Host "✅ WORKFLOW COMPLETE: Test passed" -ForegroundColor Green
    Write-Host "   (No golden comparison performed - use -Compare to validate performance)" -ForegroundColor DarkGray
    exit 0
}
