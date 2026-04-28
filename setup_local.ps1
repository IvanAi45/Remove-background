$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Clear inherited Python environment variables so a broken virtualenv
# cannot interfere with creating a fresh one.
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue

$pythonPath = "C:\Users\Ivan\AppData\Local\Programs\Python\Python312\python.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$desktopModelDir = "C:\Users\Ivan\Desktop\Remove-background-main\backend\models\fashn-human-parser"
$localModelDir = Join-Path $projectRoot "backend\models\fashn-human-parser"
$downloadScriptPath = Join-Path $projectRoot ".tmp\download_clip_model.py"

if (-not (Test-Path $pythonPath)) {
    throw "Python 3.12 was not found at $pythonPath"
}

if (Test-Path ".venv") {
    $venvLooksBroken =
        -not (Test-Path $venvPython) -or
        -not (Test-Path ".venv\Lib\site-packages\pip") -or
        -not (Test-Path ".venv\Lib\site-packages\_distutils_hack")

    if ($venvLooksBroken) {
        Remove-Item ".venv" -Recurse -Force
    }
}

if (-not (Test-Path $venvPython)) {
    & $pythonPath -m venv .venv
}

& $venvPython -m pip install -r backend\requirements.txt

if (-not (Test-Path $localModelDir) -and (Test-Path $desktopModelDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $localModelDir) | Out-Null
    Copy-Item $desktopModelDir $localModelDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $downloadScriptPath) | Out-Null
@'
from transformers import CLIPModel, CLIPProcessor

CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP model ready")
'@ | Set-Content -Path $downloadScriptPath -Encoding UTF8

& $venvPython $downloadScriptPath

Write-Host ""
Write-Host "Local environment is ready."
Write-Host "Start the backend with:"
Write-Host ".\.venv\Scripts\python.exe backend\app.py"
