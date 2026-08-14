param(
    [string]$ComfyRoot = "N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI"
)

$ErrorActionPreference = "Stop"
$env:HF_HOME = "N:\Comfy-Desktop\HF_Cache"
$env:HF_HUB_CACHE = "N:\Comfy-Desktop\HF_Cache\hub"
$env:TRANSFORMERS_CACHE = "N:\Comfy-Desktop\HF_Cache\transformers"
$pythonPath = Join-Path $ComfyRoot ".venv\Scripts\python.exe"
$installerPath = Join-Path $PSScriptRoot "install_models.py"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "未找到 ComfyUI Python: $pythonPath"
}

& $pythonPath $installerPath
if ($LASTEXITCODE -ne 0) {
    throw "KBL 模型安装失败，退出码: $LASTEXITCODE"
}

