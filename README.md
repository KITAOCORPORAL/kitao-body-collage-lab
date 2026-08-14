# Kitao Body Collage Lab v0.1

面向真实摄影素材的人体/元素分解 ComfyUI 自定义节点项目。v0.1 只做分割、部位拆分、透明 PNG 与检查文件导出，不使用生成模型重绘人物。

## 当前进度

阶段 D 已完成可移植拼贴素材包。完整流程把 `KBL_ELEMENTS` 与 `KBL_BODY_PARTS` 经过保守 safe/soft 边缘精修，逐对象导出原摄影 RGB + straight alpha 的 RGBA PNG、raw/refined/alpha mask、预览和 Manifest v0.1；Exporter 不会重跑模型。阶段 B/C 数据契约保持冻结。

## 本机安装位置

正在运行的 Comfy Desktop 实例实际位于：

```text
N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI
```

因此项目安装到：

```text
N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab
```

这和通用示例 `N:\ComfyUI\custom_nodes` 不同，是 Comfy Desktop 的实例目录机制导致的，不能把节点放到共享模型目录。

## 路径规范

| 用途 | 本机默认路径 |
| --- | --- |
| 自定义节点 | `N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab` |
| 共享模型根目录 | `N:\Comfy-Desktop\ComfyUI-Shared\models` |
| KBL 输出根目录 | `N:\ComfyUI\output\Kitao_Body_Collage_Lab` |
| SAM2 | `<模型根目录>\Kitao_Body_Collage_Lab\sam2` |
| GroundingDINO | `<模型根目录>\Kitao_Body_Collage_Lab\grounding_dino` |
| DWPose | `<模型根目录>\Kitao_Body_Collage_Lab\dwpose` |
| BiRefNet（可选） | `<模型根目录>\Kitao_Body_Collage_Lab\birefnet` |

可在启动 ComfyUI 前设置 `KBL_MODEL_ROOT` 和 `KBL_EXPORT_ROOT` 覆盖后两项。项目不会自动下载模型，也不会向外发送图片或诊断数据。

## 阶段 D 使用

1. 完全退出并重新启动 Comfy Desktop。
2. 如模型尚未安装，运行 `scripts\install_models.ps1`；脚本只写 N 盘。
3. 加载 `workflows/Kitao_Body_Collage_Lab_v0.1_full_export.json`。
4. 在 `KBL 加载图片` 中填写本地图片绝对路径并运行。

命令行真实验收：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' scripts\smoke_test_export.py --image 'N:\comfyui\input\your_photo.jpg'
```

默认输出为 cropped RGBA、24px padding、版本化目录，结果写入 `N:\ComfyUI\output\Kitao_Body_Collage_Lab`。BiRefNet 为 `NOT INSTALLED` 的实验接口；safe/soft 不依赖它。

完整模型文件名、安装与排错步骤见 `docs/install_zh.md` 和 `docs/troubleshooting_zh.md`。

## Development status

Kitao Body Collage Lab is a ComfyUI custom-node project for photographic element decomposition and body-part collage workflows.

Current pipeline:

```text
Image
→ GroundingDINO
→ SAM2
→ DWPose
→ Body Splitter
→ Mask Refiner
→ Cutout Exporter
```

Stages B–D have passed local unit, real-photo, manifest, and ComfyUI API validation. Large model weights, source photographs, generated outputs, caches, and credentials are not stored in this repository.

For GPT ↔ Codex continuity, read `handoff/latest_status.md` first.
