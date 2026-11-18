# Generic CUDA build script: compile any .cu file and run it.
#
# Usage (run from the repo root):
#   powershell -File cuda/scripts/build.ps1                    # build & run cuda/examples/demo.cu
#   powershell -File cuda/scripts/build.ps1 cuda/examples/demo.cu
#   powershell -File cuda/scripts/build.ps1 foo.cu -NoRun      # build only
#   powershell -File cuda/scripts/build.ps1 foo.cu -Out a.exe -Arch sm_61 -NoClean
param(
    [string]$Src = "",          # positional: source .cu file (relative to repo root; defaults to cuda/examples/demo.cu)
    [string]$Out = "",          # output exe (defaults to <src>.exe)
    [string]$Arch = "sm_61",    # GPU architecture
    [switch]$NoRun,             # compile only, do not run
    [switch]$NoClean            # keep .lib/.exp intermediate artifacts
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path   # cuda\scripts
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir) # repo root

# 1. Source file (relative paths are resolved against the repo root)
if (-not $Src) { $Src = "cuda\examples\demo.cu" }
if (-not [IO.Path]::IsPathRooted($Src)) { $Src = Join-Path $repoRoot $Src }
if (-not (Test-Path $Src)) { throw "Source not found: $Src" }
$Src = (Resolve-Path $Src).Path

# 2. Output exe
if (-not $Out) { $Out = [IO.Path]::ChangeExtension($Src, ".exe") }
$Out = [IO.Path]::GetFullPath($Out)

# 3. Locate CUDA Toolkit (nvcc)
$cudaCandidates = @(
    $env:CUDA_PATH,
    "F:\CUDA\v12.5",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
) | Where-Object { $_ }
$nvcc = $null
foreach ($c in $cudaCandidates) {
    if (Test-Path "$c\bin\nvcc.exe") { $nvcc = "$c\bin\nvcc.exe"; break }
}
if (-not $nvcc) { throw "CUDA Toolkit (nvcc.exe) not found." }

# 4. Locate MSVC (vcvarsall.bat)
$vcvarsCandidates = @(
    "F:\VSBuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
)
$vcvars = $null
foreach ($v in $vcvarsCandidates) {
    if (Test-Path $v) { $vcvars = $v; break }
}
if (-not $vcvars) { throw "vcvarsall.bat (MSVC) not found." }

# 5. Compile
$cmd = "call `"$vcvars`" x64 && `"$nvcc`" -arch=$Arch --cudart static -allow-unsupported-compiler -Xcompiler=/utf-8 -o `"$Out`" `"$Src`""
Write-Host "==> $cmd"
cmd /c $cmd
if ($LASTEXITCODE -ne 0) { throw "nvcc failed (exit $LASTEXITCODE)" }

# 6. Clean intermediate artifacts
if (-not $NoClean) {
    Remove-Item ([IO.Path]::ChangeExtension($Out, ".lib")), ([IO.Path]::ChangeExtension($Out, ".exp")) -Force -ErrorAction SilentlyContinue
}

# 7. Run
if (-not $NoRun) {
    Write-Host "`n===== Running $Out ====="
    & $Out
}
