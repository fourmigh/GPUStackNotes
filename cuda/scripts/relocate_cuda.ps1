# Move CUDA 12.5 from C: to F: and create a directory junction
$ErrorActionPreference = "Stop"

$src = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
$dst = "F:\CUDA\v12.5"

if (-not (Test-Path $src)) {
    throw "Source not found: $src"
}
if ((Get-Item $src).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    Write-Host "Already a junction, nothing to do."
    exit 0
}

New-Item -ItemType Directory -Force -Path "F:\CUDA" | Out-Null
if (Test-Path $dst) {
    throw "Destination already exists: $dst"
}

Write-Host "Measuring source size..."
$size = (Get-ChildItem $src -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
Write-Host ("Source size: {0:N2} GB" -f ($size / 1GB))

Write-Host "Moving to $dst ..."
Move-Item -Path $src -Destination $dst -Force

Write-Host "Creating junction ..."
cmd /c mklink /J "$src" "$dst"
if ($LASTEXITCODE -ne 0) { throw "mklink failed" }

Write-Host "Done."
Get-Item $src | Select-Object FullName, Attributes, LinkType, Target | Format-List
