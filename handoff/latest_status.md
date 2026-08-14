# KBL Current Status

Current version:
v0.1.1

Current stage:
Stage F — PASS

Release baseline:
v0.1.1

Current goal:
One-click ComfyUI all-elements decomposition workflow

Pipeline:
Florence-2 Scene Inventory
→ SAM2 Universal Element Segmentation
→ Background Filter / Deduplication
→ Largest-Person DWPose / Body Splitter
→ Mask Refiner
→ Cutout Exporter

Validation:
- Stage B/C/D/E regression: PASS
- Unit and compatibility tests: 50 / 50 PASS
- Workflow graph validation: 5 / 5 PASS
- Shared Input F1–F4 real inference: PASS
- Real RGBA/source-RGB validation: 43 / 43 PASS
- KBL Doctor: READY after deployment sync

Known issues:
- BiRefNet is optional and not installed
- Complex scenes may miss small or heavily occluded props
- Body Split selects the largest person only

Next after v0.1.1:
Real-world user testing and segmentation quality tuning

Latest report:
`handoff/reports/回传给GPT_Kitao_Body_Collage_Lab_v0.1.1_StageF_一键全元素拆解报告.md`
