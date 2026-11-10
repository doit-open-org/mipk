# Downloads a prebuilt onnxruntime into third_party\onnxruntime\win-x64.
#   powershell -ExecutionPolicy Bypass -File scripts\fetch_onnxruntime.ps1
$ErrorActionPreference = 'Stop'
$Version = if ($env:ONNXRUNTIME_VERSION) { $env:ONNXRUNTIME_VERSION } else { '1.23.2' }
$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root 'third_party\onnxruntime\win-x64'
if (Test-Path (Join-Path $Dest 'include\onnxruntime_c_api.h')) {
    Write-Host "onnxruntime already present: $Dest"
    exit 0
}
$Name = "onnxruntime-win-x64-$Version"
$Url = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$Name.zip"
$Zip = Join-Path $env:TEMP "$Name.zip"
Write-Host "downloading $Url"
Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
$Parent = Join-Path $Root 'third_party\onnxruntime'
New-Item -ItemType Directory -Force -Path $Parent | Out-Null
Expand-Archive -Path $Zip -DestinationPath $Parent -Force
Move-Item (Join-Path $Parent $Name) $Dest
Remove-Item $Zip
Write-Host "installed: $Dest"
