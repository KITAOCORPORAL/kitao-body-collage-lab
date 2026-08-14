# Kitao Body Collage Lab

Photographic body and element decomposition toolkit for ComfyUI.

## What it does

```text
Image
→ Element Detection
→ Person Segmentation
→ Pose Estimation
→ Body Part Split
→ Alpha Refinement
→ Collage Asset Package
```

KBL keeps the source photograph's pixels. It exports masks, cropped transparent PNG assets, previews, metadata, and a portable Manifest instead of repainting the person.

## Current version

`0.1.0` — stable release baseline. Manifest schema: `0.1`.

## Quick Start

1. From the KBL project directory, install the local model files:

   ```powershell
   .\scripts\install_models.ps1
   ```

2. Restart ComfyUI completely.

3. Load:

   ```text
   workflows/Kitao_Body_Collage_Lab_v0.1_full_export.json
   ```

4. Select a local photograph in `KBL Load Image`.

5. Run the workflow.

6. Find the exported project under:

   ```text
   N:\ComfyUI\output\Kitao_Body_Collage_Lab\
   ```

Run the release-readiness check at any time:

```powershell
python scripts\kbl_doctor.py
```

## Main nodes

- `KBL Load Image`
- `KBL Element Detector`
- `KBL Pose Estimator`
- `KBL Body Splitter`
- `KBL Mask Refiner`
- `KBL Cutout Exporter`

## Output

Each KBL project can contain:

- cropped RGBA body-part and generic-element PNG files;
- raw, refined, and alpha masks;
- overview, mask, contact-sheet, and exploded previews;
- `manifest.json` using the frozen KBL Manifest v0.1 contract;
- `pipeline_diagnostics.json` with timings, counts, validation, and refine-area warnings.

The Project Reader can load an exported package without running GroundingDINO, SAM2, or DWPose:

```python
from Kitao_Body_Collage_Lab.nodes.utils.project_reader import load_kbl_project

project = load_kbl_project(r"N:\ComfyUI\output\Kitao_Body_Collage_Lab\MY_PROJECT")
```

## Models

| Component | Purpose | v0.1.0 status |
| --- | --- | --- |
| GroundingDINO | text-guided detection | required |
| SAM2 | person and object segmentation | required |
| DWPose | body, hand, and foot keypoints | required; CPU fallback supported |
| BiRefNet | optional refinement experiment | optional / not required |

Models stay outside this repository under `N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab`. KBL does not silently download weights or transmit photographs or diagnostics.

## Known limitations

- `guided` is the stable default; `auto` and `hybrid` remain experimental.
- BiRefNet is optional and is not required for safe/soft refinement.
- Advanced multi-person body splitting is not supported in v0.1.0.
- Severe occlusion or cropped limbs can be reported as missing, uncertain, or partial; KBL does not invent hidden anatomy.

## Documentation

- Chinese usage guide: `docs/usage_zh.md`
- Installation: `docs/install_zh.md`
- Troubleshooting: `docs/troubleshooting_zh.md`
- Frozen data contract: `docs/data_contract_v0.1.md`
- GPT ↔ Codex status: `handoff/latest_status.md`

Large model weights, source photographs, generated validation outputs, caches, and credentials are intentionally excluded from Git.
