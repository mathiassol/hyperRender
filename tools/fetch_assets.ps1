# Downloads test scenes from the Khronos glTF Sample Assets, and CC0 HDR
# environments from Poly Haven, into assets/downloaded/ (gitignored). Each model
# keeps its own licence file; see the model's folder.
#
#   pwsh tools/fetch_assets.ps1 [-Models Sponza,DamagedHelmet] [-Hdris name_1k] [-Force]
param(
    [string[]] $Models = @("Sponza", "DamagedHelmet", "AlphaBlendModeTest", "CesiumMan", "EnvironmentTest"),
    [string[]] $Hdris = @("kloofendal_48d_partly_cloudy_puresky_1k"),
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$dest = Join-Path $root "assets/downloaded"
$repo = "KhronosGroup/glTF-Sample-Assets"
$branch = "main"

function Get-Tree($path) {
    # Recursively list files under a repo path via the GitHub contents API.
    $items = Invoke-RestMethod "https://api.github.com/repos/$repo/contents/$path`?ref=$branch" -Headers @{ "User-Agent" = "hyperRender" }
    foreach ($item in $items) {
        if ($item.type -eq "dir") { Get-Tree $item.path } else { $item }
    }
}

foreach ($model in $Models) {
    $modelDir = Join-Path $dest $model
    if ((Test-Path $modelDir) -and -not $Force) {
        Write-Output "${model}: already present (use -Force to re-download)"
        continue
    }

    # Only the plain glTF variant plus the model's metadata/licence files.
    $files = @(Get-Tree "Models/$model/glTF") + @(Get-Tree "Models/$model" | Where-Object { $_.path -notmatch "/glTF" -and $_.name -match "LICENSE|README|metadata" })
    Write-Output "${model}: $($files.Count) files"
    foreach ($f in $files) {
        $relative = $f.path.Substring("Models/$model/".Length)
        $target = Join-Path $modelDir $relative
        New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
        Invoke-WebRequest $f.download_url -OutFile $target
    }
}
$hdriDir = Join-Path $dest "hdri"
New-Item -ItemType Directory -Force $hdriDir | Out-Null
foreach ($hdri in $Hdris) {
    $target = Join-Path $hdriDir "$hdri.hdr"
    if ((Test-Path $target) -and -not $Force) {
        Write-Output "${hdri}: already present"
        continue
    }
    $resolution = ($hdri -split "_")[-1]
    Write-Output "${hdri}: Poly Haven (CC0)"
    Invoke-WebRequest "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/$resolution/$hdri.hdr" -OutFile $target
}
Write-Output "done: $dest"
