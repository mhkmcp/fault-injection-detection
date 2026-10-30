# Windows stand-in for the Makefile:  .\tasks.ps1 <task>
# If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\tasks.ps1 <task>
param([Parameter(Position = 0)][string]$Task = "help")

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"   # UTF-8 for every file and pipe, whatever the Windows code page
Set-Location $PSScriptRoot

function Run([string[]]$Cmd) {
    Write-Host "> $($Cmd -join ' ')" -ForegroundColor Cyan
    $rest = if ($Cmd.Length -gt 1) { $Cmd[1..($Cmd.Length - 1)] } else { @() }
    & $Cmd[0] @rest
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Task) {
    "setup"     { Run @("uv", "sync", "--group", "dev", "--group", "ui", "--group", "deploy") }
    "gpu"       { Run @("uv", "run", "python", "-m", "fidnn.devices") }
    "test"      { Run @("uv", "run", "pytest", "-q") }
    "lint"      { Run @("uv", "run", "ruff", "check", "src", "tests") }
    "m0"        { Run @("uv", "run", "python", "-m", "fidnn", "bench", "m0") }
    "m0-quick"  { Run @("uv", "run", "python", "-m", "fidnn", "bench", "m0", "--quick", "--out", "artifacts/m0_quick", "--report", "artifacts/m0_quick/M0_throughput.md") }
    "m0-report" { Run @("uv", "run", "python", "-m", "fidnn", "bench", "m0", "--report-only") }
    "reproduce" {
        Run @("uv", "run", "python", "-m", "fidnn", "eval", "m2", "--precision", "int8")
        Run @("uv", "run", "python", "-m", "fidnn", "report")
    }
    default {
        Write-Host "tasks: setup, gpu, test, lint, m0, m0-quick, m0-report, reproduce"
    }
}
