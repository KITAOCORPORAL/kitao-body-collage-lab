[CmdletBinding()]
param(
    [string]$ComfyUIRoot,
    [string]$ComfyUserName = "default"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$workflowSourceDir = Join-Path $repoRoot "workflows"
$releaseWorkflows = @(
    "Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json",
    "Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json"
)

function Resolve-ComfyUIRoot {
    param([string]$RequestedRoot)

    if (-not [string]::IsNullOrWhiteSpace($RequestedRoot)) {
        return [System.IO.Path]::GetFullPath($RequestedRoot)
    }

    if (-not [string]::IsNullOrWhiteSpace($env:COMFYUI_ROOT)) {
        return [System.IO.Path]::GetFullPath($env:COMFYUI_ROOT)
    }

    # When this script is run from an installed custom node, infer the ComfyUI
    # root from <ComfyUI>\custom_nodes\Kitao_Body_Collage_Lab.
    $repoParent = Split-Path $repoRoot -Parent
    if ((Split-Path $repoParent -Leaf) -ieq "custom_nodes") {
        return [System.IO.Path]::GetFullPath((Split-Path $repoParent -Parent))
    }

    # When run from a source checkout, use Comfy Desktop's installation record
    # instead of assuming a fixed drive or user-data location.
    $installationsPath = Join-Path $env:APPDATA "Comfy Desktop\installations.json"
    if (Test-Path -LiteralPath $installationsPath) {
        $installations = Get-Content -LiteralPath $installationsPath -Raw | ConvertFrom-Json
        foreach ($installation in @($installations)) {
            if ($installation.status -ne "installed" -or [string]::IsNullOrWhiteSpace($installation.installPath)) {
                continue
            }

            $candidate = Join-Path $installation.installPath "ComfyUI"
            if (Test-Path -LiteralPath (Join-Path $candidate "folder_paths.py")) {
                return [System.IO.Path]::GetFullPath($candidate)
            }
        }
    }

    throw "Could not locate ComfyUI. Pass -ComfyUIRoot or set COMFYUI_ROOT."
}

$resolvedComfyUIRoot = Resolve-ComfyUIRoot -RequestedRoot $ComfyUIRoot
$pythonExe = Join-Path $resolvedComfyUIRoot ".venv\Scripts\python.exe"
$folderPathsModule = Join-Path $resolvedComfyUIRoot "folder_paths.py"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "ComfyUI Python was not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $folderPathsModule)) {
    throw "folder_paths.py was not found: $folderPathsModule"
}

Push-Location $resolvedComfyUIRoot
try {
    $userDirectoryOutput = & $pythonExe -s -c "import os, folder_paths; print(os.path.abspath(folder_paths.get_user_directory()))"
    if ($LASTEXITCODE -ne 0) {
        throw "ComfyUI Python could not resolve folder_paths.get_user_directory()."
    }
}
finally {
    Pop-Location
}

$comfyUserDirectory = [string](@($userDirectoryOutput) | Select-Object -Last 1)
$comfyUserDirectory = $comfyUserDirectory.Trim()
if ([string]::IsNullOrWhiteSpace($comfyUserDirectory)) {
    throw "ComfyUI returned an empty user directory."
}

$workflowLibrary = Join-Path (Join-Path $comfyUserDirectory $ComfyUserName) "workflows"
$targetDirectory = Join-Path $workflowLibrary "Kitao_Body_Collage_Lab"
New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null

$results = @()
foreach ($workflowName in $releaseWorkflows) {
    $sourcePath = Join-Path $workflowSourceDir $workflowName
    $destinationPath = Join-Path $targetDirectory $workflowName

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Release workflow was not found: $sourcePath"
    }

    $sourceHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
    $status = "installed"

    if (Test-Path -LiteralPath $destinationPath) {
        $existingHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
        if ($existingHash -ne $sourceHash) {
            throw "Refusing to overwrite an existing user workflow with different content: $destinationPath"
        }
        $status = "already-current"
    }
    else {
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath
    }

    $installedHash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
    if ($installedHash -ne $sourceHash) {
        throw "SHA256 verification failed for: $destinationPath"
    }

    $results += [pscustomobject]@{
        Status = $status
        Source = $sourcePath
        Installed = $destinationPath
        SHA256 = $installedHash
    }
}

Write-Host "COMFY_USER_DIR=$comfyUserDirectory"
Write-Host "COMFY_WORKFLOW_DIR=$workflowLibrary"
Write-Host "KBL_WORKFLOW_DIR=$targetDirectory"
$results | Format-List
