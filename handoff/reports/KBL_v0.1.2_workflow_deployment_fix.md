# KBL v0.1.2 Workflow Deployment Fix

Date: 2026-08-15
Repository branch: `main`
Starting commit: `4dc2695 fix: repair v0.1.2 workflow metadata`

## Required paths

SOURCE_WORKFLOW=`C:\Users\Admin\Documents\灵图\Kitao_Body_Collage_Lab\workflows\Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json`; `C:\Users\Admin\Documents\灵图\Kitao_Body_Collage_Lab\workflows\Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json`

CUSTOM_NODE_WORKFLOW=`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab\workflows\Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json`; `N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab\workflows\Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json`

COMFY_USER_DIR=`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\user`

COMFY_WORKFLOW_DIR=`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows`

INSTALLED_CLEAN_OBJECTS=`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\Kitao_Body_Collage_Lab\Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json`

INSTALLED_COLLAGE_PARTS=`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\Kitao_Body_Collage_Lab\Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json`

SOURCE_SHA256=`CLEAN_OBJECTS:C476E966C4C895EBEDA7C235F083AB609E405EF551DE81D13DB65CFE6BAA18EC; COLLAGE_PARTS:AC364ADE00D8A9C3BCAA431131A85D8DDD5B8AFA4954490C453BFFE7C91C8260`

INSTALLED_SHA256=`CLEAN_OBJECTS:C476E966C4C895EBEDA7C235F083AB609E405EF551DE81D13DB65CFE6BAA18EC; COLLAGE_PARTS:AC364ADE00D8A9C3BCAA431131A85D8DDD5B8AFA4954490C453BFFE7C91C8260`

ROOT_CAUSE=`workflows were committed to the KBL repository but were never deployed into the active ComfyUI user workflow directory; additionally, the custom_nodes deployment was still v0.1.1 and did not contain the v0.1.2 workflow files`

UI_VISIBLE_AFTER_FIX=`RESTART_REQUIRED`

## Diagnosis

- `folder_paths.get_user_directory()` returned `N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\user`.
- The active single-user workflow library is `user\default\workflows`.
- Neither v0.1.2 workflow existed in that library before the fix.
- Neither v0.1.2 workflow existed in the deployed custom node before the fix.
- Deployed `version.py` reported `0.1.1`, and deployed runtime hashes differed from the v0.1.2 source repository.
- Both Desktop and ComfyUI were running during deployment. No process was killed.

## Changes applied

1. Created `user\default\workflows\Kitao_Body_Collage_Lab` without deleting or overwriting any existing user workflow.
2. Copied both v0.1.2 workflow JSON files with `Copy-Item`.
3. Synchronized the changed v0.1.2 runtime files to the deployed custom node:
   - `version.py`
   - `nodes\export_nodes.py`
   - `nodes\oneclick_nodes.py`
   - `nodes\scene_nodes.py`
   - `nodes\utils\scene_utils.py`
4. Copied both v0.1.2 workflow JSON files into the deployed custom node's `workflows` directory.
5. Added `scripts\install_workflows.ps1` to locate the active ComfyUI user directory through ComfyUI's own Python and `folder_paths`, install the release workflows safely, verify SHA256, refuse to overwrite different user content, and support repeat execution.
6. Updated README Quick Start with the installed Workflows entry and manual repository Load alternatives.

## Validation

- `git pull --ff-only origin main`: already up to date before modifications.
- Installer script first run: both files `installed`.
- Installer script second run: both files `already-current`.
- PowerShell parser: PASS, zero errors.
- Source, deployed custom-node copy, and installed user copy: SHA256 identical for both workflows.
- JSON parse: PASS for all six checked copies.
- Top-level IDs are valid UUID v4:
  - Clean Objects: `21b68d83-4b55-4dc8-b03d-5f241f44b53b`
  - Collage Parts: `5a73346e-68d8-42d5-8305-dbfb0ff9b1ee`
- Clean Objects nodes: `KBL_Load_Image_Picker`, `KBL_OneClick_Decompose_Export`, `PreviewImage` present.
- Clean Objects default `output_mode`: `object_clean`.
- Collage Parts default `output_mode`: `collage_parts`.
- Deployed `version.py`: `KBL_VERSION = "0.1.2"`.
- Deployed `KBL_OneClick_Decompose_Export.INPUT_TYPES` source contains `output_mode` choices `object_clean` and `collage_parts`, defaulting to `object_clean`.
- `python -m unittest Kitao_Body_Collage_Lab.tests.test_stage_f`: 22 tests passed.

## UI state and restart requirement

While the existing backend was still running, its `/userdata?dir=workflows&recurse=true` response immediately listed:

- `Kitao_Body_Collage_Lab/Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json`
- `Kitao_Body_Collage_Lab/Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json`

However, `/object_info/KBL_OneClick_Decompose_Export` still exposed the old v0.1.1 in-memory inputs (`quality`, `include_body_parts`, `keep_unlabeled_objects`) because Python custom nodes are loaded at backend startup. The backend later stopped without being killed by this repair. A full ComfyUI restart is therefore required so the Workflows panel and the v0.1.2 node schema are loaded together. No Desktop, Python, or ComfyUI process was force-terminated.
