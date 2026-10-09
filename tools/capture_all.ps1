# Regression test: runs the cases in tools/captures.txt (every app, plus variants)
# with --capture and compares each image with its reference in tests/reference/
# (4x downscaled PNGs). A case fails if the app exits non-zero, which includes any
# logged error or Vulkan validation message, or if its image drifts from the
# reference. Images and logs land in build/captures/.
#
#   pwsh tools/capture_all.ps1 [-Bin build/vs2026/bin] [-Frames 60] [-Only 'scene*'] [-Suffix -debug] [-Update]
#
# -Bin defaults to the first of build/vs2026/bin and build/vs2022/bin that has been built.
#
# -Suffix runs <app><suffix>.exe: -debug for the Debug build (with validation).
# -Update writes the references from this run: after an intended change, look at
# the new images before committing them. References depend on the GPU and driver;
# on another machine, run with -Update once first.
param(
    [string] $Bin = "",
    [int] $Frames = 60,
    [string] $Only = "*",
    [string] $Suffix = "",
    [switch] $Update
)

$root = Split-Path $PSScriptRoot -Parent
if (-not $Bin) {
    $Bin = @("build/vs2026/bin", "build/vs2022/bin") | Where-Object { Test-Path (Join-Path $root $_) } | Select-Object -First 1
    if (-not $Bin) { throw "No build found: build with a preset first (README, Build), or pass -Bin." }
}
$binDir = Resolve-Path (Join-Path $root $Bin)
$outDir = Join-Path $root "build/captures"
$refDir = Join-Path $root "tests/reference"
$stats = Join-Path $PSScriptRoot "png_stats.py"
New-Item -ItemType Directory -Force $outDir, $refDir | Out-Null

$cases = @()
foreach ($line in Get-Content (Join-Path $PSScriptRoot "captures.txt")) {
    $words = @($line.Trim() -split '\s+' | Where-Object { $_ })
    if ($words.Count -lt 2 -or $words[0].StartsWith('#')) { continue }
    $cases += [pscustomobject]@{ Name = $words[0]; App = $words[1]; Args = @($words | Select-Object -Skip 2) }
}
# Every app under apps/ without a case gets one with no arguments (if there's an apps/).
if (Test-Path (Join-Path $root "apps")) {
    Get-ChildItem (Join-Path $root "apps") -Directory |
        Where-Object { (Test-Path (Join-Path $_.FullName "CMakeLists.txt")) -and ($cases.App -notcontains $_.Name) } |
        ForEach-Object { $cases += [pscustomobject]@{ Name = $_.Name; App = $_.Name; Args = @() } }
}
$cases = @($cases | Where-Object { $_.Name -like $Only })

$failed = @()
foreach ($case in $cases) {
    $exe = Join-Path $binDir "$($case.App)$Suffix.exe"
    if (-not (Test-Path $exe)) {
        Write-Output ("{0,-18} MISSING  {1}" -f $case.Name, $exe)
        $failed += $case.Name
        continue
    }

    $png = Join-Path $outDir "$($case.Name).png"
    $log = Join-Path $outDir "$($case.Name).log"
    Remove-Item $png -ErrorAction SilentlyContinue
    Push-Location $binDir
    & $exe @($case.Args) --capture $png --frames $Frames *> $log
    $code = $LASTEXITCODE
    Pop-Location

    if ($code -ne 0 -or -not (Test-Path $png)) {
        Write-Output ("{0,-18} FAILED   exit {1}, see {2}" -f $case.Name, $code, $log)
        $failed += $case.Name
        continue
    }
    $ref = Join-Path $refDir "$($case.Name).png"
    if ($Update -or -not (Test-Path $ref)) {
        python $stats --thumb $png $ref
        Write-Output ("{0,-18} ok       reference written" -f $case.Name)
        continue
    }
    $result = python $stats --check $ref $png
    if ($LASTEXITCODE -eq 0) {
        Write-Output ("{0,-18} ok       {1}" -f $case.Name, $result)
    } else {
        Write-Output ("{0,-18} DIFFERS  {1}; compare {2} with {3}" -f $case.Name, $result, $png, $ref)
        $failed += $case.Name
    }
}

if ($failed.Count -gt 0) {
    Write-Output "$($failed.Count) of $($cases.Count) failed: $($failed -join ', ')"
    exit 1
}
Write-Output "all $($cases.Count) passed"
