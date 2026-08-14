# Kitao Body Collage Lab v0.1｜阶段 E 收尾报告

## 1. 总状态

**PASS**

Stage E 已完成验证、契约冻结、Project Reader、诊断、用户文档和发布准备。未开发 Collage Board 或其他 v0.2 功能。

## 2. 当前版本

- KBL release：`0.1.0`
- Manifest contract：`0.1`
- 单一版本源：`version.py`

## 3. Git commit SHA

- Release implementation commit：`74b5aff6e59ffa940b567069334ed2bd3291d3a4`
- Commit message：`release: finalize Kitao Body Collage Lab v0.1.0`
- Branch：`main`
- Push：`PASS`

## 4. Git tag

`v0.1.0`

Annotated tag message：`Kitao Body Collage Lab v0.1.0`

## 5. 测试总数

标准库 `unittest`：**35 / 35 PASS**。

测试覆盖 Stage B、Stage C、Stage D、Manifest v0.1 compatibility、Project Reader 和 Stage E 的版本、路径安全、folder 非递归发现、refine warning/fallback 与 pipeline diagnostics。

此外完成：

- Python compileall：PASS
- 3 个 ComfyUI workflow JSON 解析：PASS
- sample Manifest JSON 解析：PASS
- 实际部署包 import：PASS，版本 `0.1.0`，节点数量 7

## 6. R1 / R2 / R3 / R4

### R1 简单全身 — PASS

- 样本：`2022_10_23_11_46_38_IMG_2227.JPG`
- 分辨率：1084 × 1444
- person：1
- body parts：14
- missing / uncertain：0 / 0
- body subset person：true
- overlap_after：0
- Manifest validator：PASS

### R2 手臂贴身体 / 弯曲 — PASS

使用同一阶段 C 真实样本重新执行完整链路。双臂弯曲并靠近躯干，14 部位齐全，重叠归属后为 0。

### R3 交叉腿 — PASS

- 样本：`ScreenShot_2026-06-18_234639_674.png`
- 分辨率：2111 × 1935
- 场景：双腿交叉，一腿遮挡另一腿
- body parts：14
- 左右 thigh / calf / foot：全部存在
- anatomical left/right：按人物自身左右输出
- missing / uncertain：0 / 0
- body subset person：true
- overlap_after：0
- exploded preview：已生成并完成视觉检查
- Manifest validator：PASS
- 完整耗时：11.452330 秒

遮挡侧只输出真实可见像素，没有补造隐藏身体。

### R4 高分辨率 — PASS

- 样本：`DSC00080_1.jpg`
- 分辨率：4672 × 7008
- 子流程：GroundingDINO guided person → SAM2
- person mask：保持 4672 × 7008 原图尺寸
- bbox / mask：保持原图坐标系
- 模式：memory-bounded `high_resolution_person_only`
- 耗时：4.663526 秒

R4 采用指令允许的合理高分辨率子流程，不建立 14 张全尺寸浮点 body mask；因此该项不运行 Manifest exporter，`manifest_validator = NOT_RUN_PERSON_ONLY`。

## 7. TEST C3

**PASS**

只检查了用户允许的 `N:\Comfy-Desktop\ComfyUI-Shared\input`，未扫描其他硬盘或私人图库。C3 输出位于：

`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_e\c3`

## 8. Project Reader

**PASS**

新增 `nodes/utils/project_reader.py`。`load_kbl_project(project_path)` 只读取 Manifest、pipeline diagnostics 和 PNG header，不运行 GroundingDINO、SAM2、DWPose 或任何 AI 模型。

真实读取：

`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_d\KBL_STAGE_D_TEST`

结果：Manifest PASS，14 body parts、8 generic elements、local_anchor、orientation_deg 和 RGBA PNG 路径全部解析成功。Reader 同时拒绝绝对路径、反斜杠和 `..` traversal。

## 9. Manifest Compatibility

**PASS**

新增 `tests/test_manifest_compatibility.py`，读取 `examples/sample_manifest.json` 并验证：

- Manifest version / pipeline version
- source / image
- body_parts / elements
- POSIX 相对文件路径
- original/local anchors
- orientation
- joints
- quality

Manifest 字段语义冻结为 v0.1；破坏性结构变化必须升级 Manifest version 或进入 v0.2。

## 10. Doctor

**READY**

- Python 3.13.12：PASS
- Torch 2.10.0+cu130：PASS
- CUDA / NVIDIA GeForce RTX 5060 Ti：PASS
- GroundingDINO：PASS
- SAM2：PASS
- DWPose：PASS — CPUExecutionProvider
- BiRefNet：OPTIONAL / NOT INSTALLED
- Output directory：PASS
- ComfyUI custom node deployment：PASS

未安装或升级 Torch、CUDA、Transformers、ONNX Runtime 或 BiRefNet。

## 11. Data Contract

**PASS**

新增 `docs/data_contract_v0.1.md`，冻结：

- `KBL_ELEMENTS v0.1`
- `KBL_BODY_PARTS v0.1`
- `KBL_MANIFEST v0.1`

文档定义字段、类型、原图坐标系、half-open bbox、anchor / local_anchor、orientation_deg、quality_flag、joint 和 POSIX 相对路径规则。

## 12. Pipeline Diagnostics / Safe Refine

完整导出新增 `pipeline_diagnostics.json`，记录版本、source filename、resolution、分阶段 timings、counts、validation 和逐 mask 面积诊断，不写入图片内容。

Stage B/C/D 完整真实回归：

- raw area total：405,239
- refined area total：418,842
- 总面积变化：+3.357%
- 最大单对象变化：+19.249%
- `body_right_upper_arm`：记录 `REFINE_AREA_WARNING`
- >20% 自动 fallback 保护：保留并通过测试
- fallback count：1

## 13. Git 安全审计

**PASS**

- 审计仓库文件：64
- 总大小：280,749 bytes
- `*.onnx` / `*.safetensors` / `*.pt` / `*.pth` / `*.ckpt`：0
- 真实摄影 JPG / RAW：0
- validation 输出：0
- >50 MiB 文件：0
- `.env` / credential / private key 文件名：0
- token / credential 特征命中：0

模型、真实照片、HF cache、验证输出和凭据均未提交。

## 14. 修改文件完整清单

新增：

- `CHANGELOG.md`
- `version.py`
- `docs/data_contract_v0.1.md`
- `nodes/utils/project_reader.py`
- `scripts/kbl_doctor.py`
- `scripts/regression_test.py`
- `tests/test_manifest_compatibility.py`
- `tests/test_stage_e.py`
- `handoff/reports/回传给GPT_Kitao_Body_Collage_Lab_v0.1_阶段E收尾报告.md`

修改：

- `README.md`
- `__init__.py`
- `docs/install_zh.md`
- `docs/troubleshooting_zh.md`
- `docs/usage_zh.md`
- `examples/sample_manifest.json`
- `handoff/latest_status.md`
- `nodes/detection_nodes.py`
- `nodes/export_nodes.py`
- `nodes/refine_nodes.py`
- `nodes/utils/manifest_utils.py`
- `nodes/utils/path_config.py`
- `tests/test_stage_d.py`
- `workflows/Kitao_Body_Collage_Lab_v0.1_body_split.json`
- `workflows/Kitao_Body_Collage_Lab_v0.1_element_detection.json`
- `workflows/Kitao_Body_Collage_Lab_v0.1_full_export.json`

## 15. 已知限制

1. `auto` / `hybrid` 仍为 experimental；v0.1.0 稳定默认是 `guided`。
2. BiRefNet 是 optional / not installed；safe/soft 主流程不依赖它。
3. 不支持高级多人 body split。
4. 不生成不可见人体；严重遮挡或出画部位会如实标记 missing / uncertain / partial。
5. DWPose 使用 CPUExecutionProvider，属于 PASS fallback，不是失败。

## 16. Mock / placeholder

生产实现与 R1–R4 真实验收中：**不存在 mock / placeholder**。

单元测试使用合成 mask 和 fake backend fixture 验证边界条件，不参与真实回归结论。BiRefNet 的不可用状态明确标记为 optional / not installed，不冒充已实现。

## 17. 是否建议进入 v0.2 Collage Board

**建议进入，但本次未进入。**

v0.1.0 已形成稳定 Project/Manifest/PNG 读取边界。v0.2 Collage Board 可以只依赖 KBL Project Reader，无需认识 GroundingDINO、SAM2 或 DWPose。
