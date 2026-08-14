# Kitao Body Collage Lab v0.1 阶段 B 元素检测报告

## 1. 阶段 B 完成情况

代码开发、节点注册、工作流、最小测试和文档已完成。生产节点已直接调用 Transformers 的真实 GroundingDINO 与 SAM2 类，没有用 mock 替换生产推理。当前本机模型目录为空，因此没有完成真实权重加载和摄影图片出图验收；这项不能描述为“本机模型已接通并跑通”。

## 2. 实际实现功能

- `KBL_Load_Image`：加载 JPG/JPEG/PNG/WEBP，应用 EXIF Orientation，RGB/RGBA 转 RGB，保持原始尺寸。
- `KBL_Element_Detector`：`auto`、`guided`、`hybrid`。
- GroundingDINO：多个英文关键词规范化后执行文本到 bbox，保留 label、score、box。
- SAM2 guided：按 GroundingDINO bbox 生成每个对象的独立 mask。
- SAM2 auto：规则点网格作为正提示，使用真实 SAM2 多 mask 输出产生候选区域。
- hybrid：合并 guided/auto，按 IoU、面积、同类包含关系和上限去重。
- 每个元素保留 id、label、mask、bbox、confidence、sam_score、area、source。
- 输出 `[N,H,W]` 独立 MASK、boxes/labels/scores、叠加预览和 mask contact sheet。
- 模型 load/infer/cleanup，GroundingDINO 释放后再加载 SAM2；CUDA OOM 有中文操作提示。
- 所有模型只允许 N 盘本地目录，`local_files_only=True`，不会隐式下载。

## 3. 修改文件完整清单

- `Kitao_Body_Collage_Lab/requirements.txt`
- `Kitao_Body_Collage_Lab/README.md`
- `Kitao_Body_Collage_Lab/nodes/__init__.py`
- `Kitao_Body_Collage_Lab/nodes/input_nodes.py`
- `Kitao_Body_Collage_Lab/nodes/detection_nodes.py`
- `Kitao_Body_Collage_Lab/nodes/model_backends.py`
- `Kitao_Body_Collage_Lab/nodes/utils/image_io.py`
- `Kitao_Body_Collage_Lab/nodes/utils/bbox_utils.py`
- `Kitao_Body_Collage_Lab/nodes/utils/mask_utils.py`
- `Kitao_Body_Collage_Lab/nodes/utils/path_config.py`
- `Kitao_Body_Collage_Lab/nodes/utils/preview_utils.py`
- `Kitao_Body_Collage_Lab/tests/__init__.py`
- `Kitao_Body_Collage_Lab/tests/test_stage_b.py`
- `Kitao_Body_Collage_Lab/workflows/Kitao_Body_Collage_Lab_v0.1_element_detection.json`
- `Kitao_Body_Collage_Lab/docs/install_zh.md`
- `Kitao_Body_Collage_Lab/docs/usage_zh.md`
- `Kitao_Body_Collage_Lab/docs/troubleshooting_zh.md`
- `Kitao_Body_Collage_Lab/回传给GPT_Kitao_Body_Collage_Lab_v0.1_阶段B元素检测报告.md`

## 4. 新增模型依赖

- Python：`transformers>=5.8.0`。本机已经安装 `5.8.0`，没有重新安装或覆盖 Torch。
- GroundingDINO 推荐仓库：`IDEA-Research/grounding-dino-tiny`。
- SAM2 推荐仓库：`facebook/sam2.1-hiera-small`。

## 5. 模型期待文件名

GroundingDINO 至少需要：

- `config.json`
- `preprocessor_config.json`
- `tokenizer_config.json`
- `tokenizer.json`
- `model.safetensors`（允许 Transformers 分片 safetensors）

SAM2 至少需要：

- `config.json`
- `preprocessor_config.json`
- `processor_config.json`
- `model.safetensors`（允许 Transformers 分片 safetensors）

## 6. 模型实际路径

```text
N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\grounding_dino
N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\sam2
```

当前两目录存在但为空。

## 7. ComfyUI 节点名称

- 内部名：`KBL_Load_Image`；显示名：`KBL 加载图片`
- 内部名：`KBL_Element_Detector`；显示名：`KBL 元素检测与分割`

## 8. 工作流 JSON 路径

```text
Kitao_Body_Collage_Lab\workflows\Kitao_Body_Collage_Lab_v0.1_element_detection.json
```

## 9. 如何测试

1. 手动把完整 Transformers 模型仓库放到第 6 节路径。
2. 把项目部署到当前 ComfyUI `custom_nodes` 并重启 Comfy Desktop。
3. 加载阶段 B 工作流。
4. 在 `KBL 加载图片` 填写真实 JPG/PNG/WEBP 绝对路径。
5. 首先运行 `guided + portrait_basic`，确认 bbox 与 mask；再运行 `hybrid` 检查候选去重。
6. 检查两个 PreviewImage：叠加预览和独立 mask contact sheet。

自动测试命令：

```powershell
$env:PYTHONPATH='C:\Users\Admin\Documents\灵图'
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' -m unittest -v Kitao_Body_Collage_Lab.tests.test_stage_b
```

## 10. 已知问题

- 本机缺少权重，真实模型兼容性和实际显存峰值尚未在本机图片上验证。
- auto 使用 SAM2 规则点网格，不是 Meta 官方独立 `SAM2AutomaticMaskGenerator` 类；它仍是真实 SAM2 推理，但候选覆盖率与速度需要拿真实图片调参。
- `auto_object` 没有语义类别；只有 guided 结果带 GroundingDINO 标签。
- 高分辨率原图输出 mask 保持原尺寸，但模型内部按处理器配置缩放输入，这是 SAM2/GroundingDINO 正常推理机制。
- 高分辨率图会按总 mask 像素量动态降低实际候选上限并打印日志，避免 64 张全尺寸浮点 MASK 耗尽系统内存；不改变原图或输出 mask 尺寸。
- 单次节点执行只处理 ComfyUI IMAGE 批次中的第一张图，符合本期单图范围。

## 11. 是否存在临时代码 / mock / placeholder

- 生产推理代码：没有 mock 或随机/阈值分割替代品。
- 测试代码：有 `FakeDinoBackend`、`FakeSamBackend`，只用于验证 bbox→SAM 数据契约、独立 mask、空结果和预览，不进入生产节点注册。
- `body_split_nodes.py`、`refine_nodes.py`、`export_nodes.py` 仍是阶段 C/D 预留空模块，不属于阶段 B 已完成功能。
- 因本机没有模型权重，不能宣称模型实际出图已跑通。

## 12. 阶段 C 开发前还需解决什么

1. 安装两个完整本地模型目录。
2. 用至少一张真实人物摄影图完成 guided/hybrid 出图验收。
3. 根据实际 GPU 显存和结果调节自动点网格、阈值及候选上限。
4. 冻结 `KBL_ELEMENTS` 数据契约后，再让 DWPose/人体拆分器直接消费 person 元素。
