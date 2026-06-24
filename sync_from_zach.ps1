# sync_from_zach.ps1
# Pull the latest files from zach-ai:D:\StockModel and push to GitHub.
# Usage: .\sync_from_zach.ps1 [-Message "custom commit message"]
param(
    [string]$Message = "Sync from zach-ai: update scripts and run outputs"
)

$repo   = $PSScriptRoot
$remote = "zach-ai"
$src    = "D:/StockModel"

Write-Host "Pulling scripts from $remote ..." -ForegroundColor Cyan
scp "${remote}:${src}/pipeline.py"        "$repo/scripts/pipeline.py"
scp "${remote}:${src}/pipeline_full.py"   "$repo/scripts/pipeline_full.py"
scp "${remote}:${src}/pipeline_course.py" "$repo/scripts/pipeline_course.py"

Write-Host "Pulling outputs from $remote ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force "$repo/outputs" | Out-Null
scp "${remote}:${src}/*.png" "$repo/outputs/"
scp "${remote}:${src}/*.csv" "$repo/outputs/"
scp "${remote}:${src}/run_log.txt" "$repo/outputs/"

Write-Host "Committing and pushing ..." -ForegroundColor Cyan
Set-Location $repo
git add scripts/ outputs/
git status --short
git commit -m $Message
git push origin main

Write-Host "Done." -ForegroundColor Green
