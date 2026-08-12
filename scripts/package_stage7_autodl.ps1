param(
    [string]$Output = "G:\tiaozhanbei\newrag_stage74_after_fix_code.tar.gz"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Output = [System.IO.Path]::GetFullPath($Output)

if (-not $ProjectRoot.StartsWith("G:\tiaozhanbei\newrag")) {
    throw "Unexpected project root: $ProjectRoot"
}

if (Test-Path -LiteralPath $Output) {
    Remove-Item -LiteralPath $Output -Force
}

Push-Location $ProjectRoot
try {
    & python scripts/validate_stage74_release.py --out artifacts/stage7/stage74_release_audit.json
    if ($LASTEXITCODE -ne 0) {
        throw "Stage7.4 release audit failed"
    }
    # tar.gz preserves Chinese Markdown filenames more reliably than zip when
    # moving from Windows to Linux. Models and Qdrant are intentionally omitted.
    & tar.exe -czf $Output `
        --exclude="./.git" `
        --exclude="./models" `
        --exclude="./artifacts/qdrant_local" `
        --exclude="./artifacts/pytest_tmp" `
        --exclude="./artifacts/pytest_*" `
        --exclude="./artifacts/stage1/test_*" `
        --exclude="./artifacts/stage2" `
        --exclude="./artifacts/stage3/test_*" `
        --exclude="./artifacts/stage4" `
        --exclude="./artifacts/stage5" `
        --exclude="./artifacts/stage6" `
        --exclude="./.pytest_cache" `
        --exclude="./src/*.egg-info" `
        --exclude="./stage1_artifacts.tar.gz" `
        .
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Get-Item -LiteralPath $Output | Select-Object FullName, Length, LastWriteTime
