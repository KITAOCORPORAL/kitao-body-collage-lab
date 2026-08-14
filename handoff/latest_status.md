# KBL Current Status

Current version:
v0.1.0

Current stage:
Stage E — PASS

Release baseline:
v0.1.0

Last completed:
Stage E validation, frozen data contracts, Project Reader, Manifest compatibility, diagnostics, Doctor, README, and release preparation.

Pipeline:
GroundingDINO
→ SAM2
→ DWPose
→ Body Splitter
→ Mask Refiner
→ Cutout Exporter

Validation:
- Stage B/C/D full real-photo regression: PASS
- R1 simple full body: PASS
- R2 bent arms near body: PASS
- R3 / TEST C3 crossed legs: PASS
- R4 4672×7008 original-coordinate person segmentation: PASS
- Unit and compatibility tests: 35 / 35 PASS
- KBL Doctor: READY

Known issues:
- BiRefNet is optional and not installed
- auto/hybrid remain experimental
- advanced multi-person body splitting is not supported

Next:
v0.2 Collage Board

Latest report:
`handoff/reports/回传给GPT_Kitao_Body_Collage_Lab_v0.1_阶段E收尾报告.md`
