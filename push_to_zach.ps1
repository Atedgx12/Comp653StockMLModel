# push_to_zach.ps1
# Push the latest repo files to zach-ai:D:\StockModel and run the pipeline.
# Usage: .\push_to_zach.ps1
param(
    [string]$Remote  = "zach-ai",
    [string]$Dst     = "D:/StockModel",
    [switch]$RunPipeline,
    [switch]$RunProbe
)

$repo = $PSScriptRoot

Write-Host "Pushing scripts to ${Remote}:${Dst} ..." -ForegroundColor Cyan
scp "$repo/scripts/pipeline.py"        "${Remote}:${Dst}/pipeline.py"
scp "$repo/scripts/pipeline_full.py"   "${Remote}:${Dst}/pipeline_full.py"
scp "$repo/scripts/pipeline_course.py" "${Remote}:${Dst}/pipeline_course.py"
scp "$repo/scripts/cross_section.py"   "${Remote}:${Dst}/cross_section.py"

if ($RunPipeline) {
    Write-Host "Running pipeline_course.py on $Remote ..." -ForegroundColor Cyan
    ssh $Remote "python D:\StockModel\pipeline_course.py"
}

if ($RunProbe) {
    Write-Host "Running cross_section.py on $Remote ..." -ForegroundColor Cyan
    ssh $Remote "python D:\StockModel\cross_section.py > D:\StockModel\cross_section_out.txt 2>&1"
    scp "${Remote}:D:/StockModel/cross_section_out.txt" "$repo/outputs/cross_section_out.txt"
    Write-Host "Probe output saved to outputs/cross_section_out.txt" -ForegroundColor Green
}

Write-Host "Done." -ForegroundColor Green
