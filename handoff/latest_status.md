# KBL Current Status

Current version:
v0.1.2

Current stage:
Stage F — One-Click Clean Objects PASS

Release baseline:
v0.1.2

Current goal:
One-click whole-object decomposition with separate clean-object and collage-parts modes

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

Next:
v0.1.3 whole-object recall tuning

Latest report:
`handoff/reports/回传给GPT_Kitao_Body_Collage_Lab_v0.1.2_完整元素模式修复报告.md`
