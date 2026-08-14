# Kitao Body Collage Lab v0.1 阶段 B.5 真实模型验收报告

## 1. 总状态

**PASS**

阶段 B.5 成功标准全部满足。GroundingDINO、SAM2、真实摄影图片、透明 cutout、ComfyUI KBL 节点、Preview、显存记录和 `KBL_ELEMENTS` 契约均已真实验收。本轮没有进入阶段 C。

## 2. GroundingDINO

实际模型：`IDEA-Research/grounding-dino-tiny`

路径：

```text
N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\grounding_dino
```

实际根目录文件：

```text
.gitattributes
README.md
added_tokens.json
config.json
model.safetensors
preprocessor_config.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
vocab.txt
```

- `model.safetensors`：689,359,096 bytes
- SHA256：`1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`
- `pytorch_model.bin`：不存在
- 本地加载：成功，`local_files_only=True`
- device：`cuda`
- dtype：`torch.bfloat16`
- 真实 smoke 推理：1.9616 秒
- `torch.cuda.max_memory_allocated()`：1,616,224,256 bytes，约 1.51 GiB

## 3. SAM2

实际模型：`facebook/sam2.1-hiera-small`

路径：

```text
N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\sam2
```

实际根目录文件：

```text
.gitattributes
README.md
config.json
model.safetensors
preprocessor_config.json
processor_config.json
sam2.1_hiera_s.yaml
video_preprocessor_config.json
```

- `model.safetensors`：184,305,280 bytes
- SHA256：`0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60`
- `.pt`：不存在
- 本地加载：成功，`local_files_only=True`
- device：`cuda`
- dtype：`torch.bfloat16`
- 真实 smoke 推理：0.7835 秒
- `torch.cuda.max_memory_allocated()`：348,371,968 bytes，约 0.32 GiB

当前仓库配置是 `sam2_video`。阶段 B.5 已改为从 `Sam2VideoConfig` 明确构造 image-only `Sam2Config` 后加载 `Sam2Model`；357 个 image 模型权重全部匹配，missing/unexpected/mismatched 均为 0，原先的模型类型警告已消失。

GroundingDINO 推理完成并清理后才加载 SAM2。每个后端结束执行 `model.to("cpu")`、`del`、`gc.collect()` 和 `torch.cuda.empty_cache()`，两个大模型不长期同时驻留显存。

## 4. 测试图片

仅使用现有 ComfyUI input 中的真实摄影图片，没有复制进代码仓库：

- `DSC00080_1.jpg`：4672 × 7008，用于纯 Python person smoke test和首轮 ComfyUI guided 测试。
- `2022_10_23_11_46_38_IMG_2227.JPG`：1084 × 1444，用于 ComfyUI 多关键词 guided、auto、hybrid 测试。

共享 input 是 `N:\Comfy-Desktop\ComfyUI-Shared\input` 到 `N:\comfyui\input` 的 junction，报告不记录或复制其他私人素材。

## 5. GroundingDINO 检测结果

纯 Python smoke test，提示 `person.`：

```json
{
  "label": "person",
  "confidence": 0.8807970285415649,
  "bbox": [2141.05, 2854.87, 3614.27, 5788.33]
}
```

ComfyUI 多关键词 `person. clothing. shoe. bag.` 真实结果：

- `person_01`：0.84
- `clothing_01`：0.54
- `bag_01`：0.46
- `shoe_01`：0.75
- `shoe_02`：0.74

对应 bbox 和 mask 在验证目录的 guided preview 中可见。

## 6. SAM2 结果

纯 Python person bbox → SAM2：

- mask：4672 × 7008，与原图一致
- mask area：1,332,326 pixels
- bbox area ratio：0.308292
- SAM score：0.953125
- NaN：无
- Inf：无
- alpha 非零像素：1,332,326
- 原摄影像素：保留，只将 SAM2 mask 写入 alpha

`KBL_ELEMENTS` v0.1 已冻结。每个 element 在生产边界强制要求：

```text
id
label
mask
bbox
confidence
area
source
```

同时校验 `mask.shape == (原图高度, 原图宽度)`、bbox 为四坐标、`area == mask 非零像素数`。阶段 C 可直接消费 `person` 元素。

## 7. 验证文件路径

```text
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\grounding_preview.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\sam2_person_mask.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\sam2_person_cutout.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\smoke_test_result.json
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\comfy_guided_element_preview.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\comfy_guided_mask_preview.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\comfy_auto_element_preview.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\comfy_auto_mask_preview.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\comfy_hybrid_element_preview.png
N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation\comfy_hybrid_mask_preview.png
```

## 8. ComfyUI 实测

- 实际实例：`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI`
- 实际 Python：`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe`
- `KBL_Load_Image`：成功注册并真实执行
- `KBL_Element_Detector`：成功注册并真实执行
- `PreviewImage`：叠加预览和 contact sheet 均成功生成并人工目视检查
- 最终重启后 guided 多关键词节点图：3.48 秒，状态 `success`
- 最终部署默认：`guided`

未使用 Browser、WebView、Chrome、Computer Use 或 localhost 页面；通过 PowerShell/Python HTTP API 提交 ComfyUI prompt并读取 history。

## 9. auto / guided / hybrid

### guided

- 状态：PASS，v0.1 默认
- 32.7MP person 节点图：6.52 秒
- 1084×1444 多关键词：2.62 秒；最终代码复验 3.48 秒
- 优点：人物、服装、鞋、包可按指定类别稳定分割

### auto

- 状态：PASS（实验功能）
- 1084×1444，`max_candidates=16`：1.12 秒
- 12 个原始候选经 IoU/包含/面积去重后保留 3 个，证明去重实际生效
- 问题：保留 1 个正确 person mask，同时有 2 个大面积背景 mask
- 显存稳定，没有 OOM；点网格已分块处理

### hybrid

- 状态：PASS（实验功能）
- 1084×1444：3.03 秒
- 7 个 guided + 12 个 auto 候选，共 19 个，最终保留 7 个
- 问题：guided 语义元素正确，但仍混入 auto 的大面积背景 mask

基于真实结果，v0.1 默认从 `hybrid` 改为 `guided`。auto/hybrid 没有删除，但不再作为用户首跑默认。

## 10. 修改文件完整清单

- `Kitao_Body_Collage_Lab/README.md`
- `Kitao_Body_Collage_Lab/docs/install_zh.md`
- `Kitao_Body_Collage_Lab/docs/usage_zh.md`
- `Kitao_Body_Collage_Lab/nodes/detection_nodes.py`
- `Kitao_Body_Collage_Lab/nodes/model_backends.py`
- `Kitao_Body_Collage_Lab/nodes/utils/mask_utils.py`
- `Kitao_Body_Collage_Lab/scripts/install_models.py`
- `Kitao_Body_Collage_Lab/scripts/install_models.ps1`
- `Kitao_Body_Collage_Lab/scripts/model_integrity.py`
- `Kitao_Body_Collage_Lab/scripts/smoke_test_models.py`
- `Kitao_Body_Collage_Lab/tests/test_stage_b.py`
- `Kitao_Body_Collage_Lab/workflows/Kitao_Body_Collage_Lab_v0.1_element_detection.json`
- `Kitao_Body_Collage_Lab/回传给GPT_Kitao_Body_Collage_Lab_v0.1_阶段B5真实模型验收报告.md`

## 11. 测试命令

模型安装：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' 'C:\Users\Admin\Documents\灵图\Kitao_Body_Collage_Lab\scripts\install_models.py'
```

模型完整性检查：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' 'C:\Users\Admin\Documents\灵图\Kitao_Body_Collage_Lab\scripts\model_integrity.py'
```

纯 Python smoke test：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' 'C:\Users\Admin\Documents\灵图\Kitao_Body_Collage_Lab\scripts\smoke_test_models.py' --image 'N:\comfyui\input\DSC00080_1.jpg'
```

单元测试：

```powershell
$env:PYTHONPATH='C:\Users\Admin\Documents\灵图'
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' -m unittest -v Kitao_Body_Collage_Lab.tests.test_stage_b
```

ComfyUI 真实测试使用 `/prompt` 和 `/history/<prompt_id>` API，prompt/history 证据保存在 validation 目录的 `comfy_*_history.json`。

## 12. 已知问题

1. auto/hybrid 的点网格会产生大面积背景 mask，所以保留为实验功能。
2. auto 不是 Meta 原生 `SAM2AutomaticMaskGenerator`，是 SAM2 multimask + 规则点网格 + KBL 去重。
3. 32.7MP 图片若输出 64 个全尺寸浮点 mask，系统 RAM 代价很高；现有代码按总 mask 像素量动态降低候选上限，不改变 mask 尺寸。
4. 当前 ComfyUI 环境的其他自定义节点 `PatchTritonVAE` 在启动时报告缺少 Triton；KBL 节点仍成功注册和运行，与 KBL 无关。
5. 报告中的近似峰值是纯 Python真实推理期间 `torch.cuda.max_memory_allocated()`；没有引入外部显存分析器。

## 13. mock / placeholder / 未验证代码

- 生产路径：无 mock、无随机 mask、无阈值分割、无生成模型。
- 测试文件仍有 `FakeDinoBackend`/`FakeSamBackend`，只用于数据契约单元测试，不进入节点注册或生产执行。
- guided：真实验证完成。
- auto：真实验证完成。
- hybrid：真实验证完成。
- `body_split_nodes.py`、`refine_nodes.py`、`export_nodes.py` 仍为未来阶段空模块，本轮没有开发或宣称完成。

## 14. 是否建议进入阶段 C

**建议进入阶段 C。**

阶段 B.5 已 PASS，且 person element 的真实原尺寸 mask 和冻结数据契约可直接交给后续 `KBL_Body_Splitter`。是否实际进入仍等待用户审查批准。

