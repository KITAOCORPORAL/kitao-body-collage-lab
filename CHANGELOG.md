# Changelog

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
