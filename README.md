# Kitao Body Collage Lab

One-click photographic body and scene-element decomposition toolkit for ComfyUI.

## What it does

```text
Image
→ Florence-2 Scene Inventory
→ Universal Element Segmentation
→ Person Segmentation
→ Pose Estimation
→ Body Part Split
→ Alpha Refinement
→ Collage Asset Package
```

KBL keeps the source photograph's pixels. It exports masks, cropped transparent PNG assets, previews, metadata, and a portable Manifest instead of repainting the person.

## Current version

`0.1.2` — One-Click Clean Objects release. Manifest schema remains `0.1`.

## Quick Start

1. From the KBL project directory, install the local model files:

   ```powershell
   .\scripts\install_models.ps1
   ```

2. Install the release workflows into the current ComfyUI user library:

   ```powershell
   .\scripts\install_workflows.ps1
   ```

3. Restart ComfyUI completely.

4. Open the recommended workflow using either method:

   **A. Installed ComfyUI Workflows entry (recommended)**

   ```text
   Workflows
   └─ Kitao_Body_Collage_Lab
      └─ Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS
   ```

   The material-mining alternative appears beside it as
   `Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS`.

   **B. Manual Load from the repository**

   ```text
   workflows/Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json
   ```

   The repository `workflows` directory is not the ComfyUI user Workflows directory. Committing a
   JSON file to the repository does not make it appear in the Workflows panel; run
   `scripts\install_workflows.ps1` to deploy release workflows into the active user library.

5. Click `KBL 选择图片` and choose or upload a photograph from ComfyUI Input.

6. Run the workflow.

7. Find the exported project under:

   ```text
   N:\ComfyUI\output\Kitao_Body_Collage_Lab\
   ```

The default `object_clean` mode exports whole people and whole semantic objects. It suppresses
hair/body sub-elements, region proposals, background-shaped masks, and broken fragments. For
material-mining and body-part layers, use
`workflows/Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json` instead.

Run the release-readiness check at any time:

```powershell
python scripts\kbl_doctor.py
```

## Main nodes

- `KBL Load Image`
- `KBL Load Image Picker`
- `KBL Scene Inventory`
- `KBL Universal Element Detector`
- `KBL One-Click Decompose Export`
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

| Component | Purpose | v0.1.2 status |
| --- | --- | --- |
| Florence-2 base-ft | prompt-free scene inventory and region proposals | required; pinned Microsoft revision, local-only |
| GroundingDINO | text-guided detection | required |
| SAM2 | person and object segmentation | required |
| DWPose | body, hand, and foot keypoints | required; CPU fallback supported |
| BiRefNet | optional refinement experiment | optional / not required |

Models stay outside this repository under `N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab`. KBL does not silently download weights or transmit photographs or diagnostics.

## Known limitations

- `object_clean` intentionally prefers a clean false negative over a background-shaped false object.
- Whole-object quality still depends on Florence-2 locating a coherent semantic object and SAM2 producing a coherent mask.
- The legacy `guided` workflow remains supported; its `auto` and `hybrid` modes remain experimental.
- BiRefNet is optional and is not required for safe/soft refinement.
- Multiple people are exported as generic person layers, while body splitting selects the largest person only.
- Severe occlusion or cropped limbs can be reported as missing, uncertain, or partial; KBL does not invent hidden anatomy.

## Documentation

- Chinese usage guide: `docs/usage_zh.md`
- Installation: `docs/install_zh.md`
- Troubleshooting: `docs/troubleshooting_zh.md`
- Frozen data contract: `docs/data_contract_v0.1.md`
- GPT ↔ Codex status: `handoff/latest_status.md`

Large model weights, source photographs, generated validation outputs, caches, and credentials are intentionally excluded from Git.
