# Changelog

## 0.1.1

### Added

- Local-only Florence-2 scene inventory with OD, dense caption, region proposal, and caption grounding
- `KBL_Load_Image_Picker`, `KBL_Scene_Inventory`, `KBL_Universal_Element_Detector`, and `KBL_OneClick_Decompose_Export`
- Semantic-first deduplication, background-like region filtering, unknown-region preservation, and Windows-safe asset names
- One-click and advanced all-elements ComfyUI workflows
- Scene inventory preview, all-elements Contact Sheet, Exploded View, and Stage F diagnostics
- Automatic largest-person body routing while preserving every detected person as a generic layer

### Validation

- 50/50 unit and compatibility tests: PASS
- Four Shared Input real-scene runs: PASS
- 43/43 exported assets: RGBA, transparent, and source-RGB exact where alpha is nonzero
- Florence-2 peak CUDA allocation: 717,854,208 bytes on RTX 5060 Ti 16 GB
- Legacy Stage B/C/D/E real-photo regression and Manifest validator: PASS

### Known limitations

- Small, occluded props can still be missed in cluttered scenes; Advanced workflow and `extra_prompt` remain the tuning path.
- Multiple people are exported independently, but body splitting still targets the largest person only.

## 0.1.0

### Added

- GroundingDINO guided object detection
- SAM2 object and person segmentation
- DWPose body pose estimation with CPU provider fallback
- Anatomical body-part splitting
- Hand and foot extraction
- Conservative safe/soft mask refinement
- Cropped straight-alpha RGBA export
- Anchor, orientation, bbox, joint, and quality metadata
- Portable Manifest v0.1 and project validator
- Model-free KBL Project Reader
- Pipeline diagnostics with per-stage timings and refine-area warnings
- Explicit image/folder regression runner
- Unified `kbl_doctor.py` release-readiness check

### Validation

- Stage B/C/D full real-photo regression: PASS
- TEST C3 crossed-leg sample: PASS
- 4672×7008 guided person + SAM2 original-coordinate regression: PASS
- Manifest v0.1 compatibility and real Stage D Project Reader tests: PASS

### Known limitations

- `auto` and `hybrid` detection modes remain experimental; `guided` is the stable default.
- BiRefNet is optional and not installed in the validated baseline.
- Advanced multi-person body splitting is not supported.
- Hidden anatomy is not generated; severe occlusion or cropped limbs may be marked missing, uncertain, or partial.
