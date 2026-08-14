# 安装说明（KBL v0.1.1）

## 1. 项目位置

本机正在运行的 Comfy Desktop 使用以下实例：

```text
N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI
```

项目必须位于：

```text
N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab
```

## 2. 安装 Python 基础依赖

关闭正在运行的 Comfy Desktop，然后在 PowerShell 执行：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab\scripts\install_dependencies.ps1'
```

脚本只调用该 ComfyUI 实例自己的 `.venv`，不会使用系统 Python。v0.1.0 safe/soft 精修与 PNG/Manifest 导出只使用现有 NumPy、OpenCV、Pillow 和标准库，不新增依赖，也不会升级 Torch、CUDA 或 ONNX Runtime。

## 3. 模型目录

本机 `shared_model_paths.yaml` 指向 N 盘共享模型目录。请预留：

```text
N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\
├─ sam2\
├─ grounding_dino\
├─ dwpose\
├─ florence2\
└─ birefnet\
```

阶段 B 使用当前 ComfyUI 已安装的 `transformers>=5.8.0` 原生后端。推荐模型仓库：

- GroundingDINO：`IDEA-Research/grounding-dino-tiny`
- SAM2：`facebook/sam2.1-hiera-small`
- DWPose：`yzd-v/DWPose`（只取 `yolox_l.onnx` 与 `dw-ll_ucoco_384.onnx`）
- Florence-2：`microsoft/Florence-2-base-ft`，固定 revision `f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`

必须手动把完整仓库文件放到对应目录根部，不要只放 `.pt`：

```text
N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\grounding_dino\
├─ config.json
├─ preprocessor_config.json
├─ tokenizer_config.json
├─ tokenizer.json
├─ vocab.txt
└─ model.safetensors

N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\sam2\
├─ config.json
├─ preprocessor_config.json
├─ processor_config.json
└─ model.safetensors

N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\dwpose\
├─ yolox_l.onnx
└─ dw-ll_ucoco_384.onnx
```

KBL 调用 `from_pretrained(..., local_files_only=True)`，并设置离线模式，不会静默下载。若用户自行使用 Hugging Face CLI 下载，请显式指定上述 N 盘 `--local-dir`。

Florence-2 在当前 Transformers 5.8 环境使用 Microsoft 官方固定 revision 中的本地模型代码。安装后生产推理始终 `local_files_only`；动态模块缓存、HF cache 和 Transformers cache 都固定在 N 盘。不要单独升级 Torch、CUDA 或 Transformers。

也可关闭 Comfy Desktop 后运行一键安装：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab\scripts\install_models.ps1'
```

脚本设置 `HF_HOME`、`HF_HUB_CACHE`、`TRANSFORMERS_CACHE` 到 `N:\Comfy-Desktop\HF_Cache`，下载完整 repository snapshot，忽略 `.pt` 与 `pytorch_model.bin`，并逐文件校验。

如外部安装器强制使用 Hugging Face 缓存，可在执行前设置：

```powershell
$env:HF_HOME = 'N:\ComfyUI\.cache\huggingface'
$env:TORCH_HOME = 'N:\ComfyUI\.cache\torch'
```

## 4. 输出目录

默认：

```text
N:\ComfyUI\output\Kitao_Body_Collage_Lab
```

可在启动前覆盖：

```powershell
$env:KBL_EXPORT_ROOT = 'N:\ComfyUI\output\Kitao_Body_Collage_Lab'
```

## 5. 工作流与验证加载

重启 Comfy Desktop，加载：

```text
workflows\Kitao_Body_Collage_Lab_v0.1.1_ONE_CLICK_ALL_ELEMENTS.json
```

在 `KBL 选择图片` 节点选择或上传图片。默认使用 `complete`、身体部件开启、保留未标注 Region Proposal、24px padding 和 `version` 覆盖策略。

完成后可运行统一检查：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' scripts\kbl_doctor.py
```
