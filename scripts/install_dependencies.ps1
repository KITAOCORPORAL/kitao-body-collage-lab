param(
    [string]$ComfyRoot = "N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI"
)

$ErrorActionPreference = "Stop"
$pythonPath = Join-Path $ComfyRoot ".venv\Scripts\python.exe"
$requirementsPath = Join-Path (Split-Path $PSScriptRoot -Parent) "requirements.txt"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "未找到 ComfyUI Python: $pythonPath"
}

& $pythonPath -m pip install -r $requirementsPath
if ($LASTEXITCODE -ne 0) {
    throw "依赖安装失败，pip 退出码: $LASTEXITCODE"
}

Write-Output "KBL 基础依赖安装完成。请完全退出并重新启动 Comfy Desktop。"

