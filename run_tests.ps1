$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=================================================================" -ForegroundColor Magenta
Write-Host "   TRANSFORMER ARCHITECTURE & INFERENCE VERIFICATION SUITE       " -ForegroundColor Magenta
Write-Host "=================================================================" -ForegroundColor Magenta

Write-Host "`n>>> Running Architectural Unit Tests <<<" -ForegroundColor Cyan
python tests/test_transformer_architecture.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Transformer Architecture Tests Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n>>> Running GPU KV-Cache Inference Benchmark <<<" -ForegroundColor Green
python benchmark_kv_cache.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "KV Cache Benchmark Failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[SUCCESS] All Transformer Tests and Benchmarks Passed!" -ForegroundColor Green
$global:LASTEXITCODE = 0
exit 0
