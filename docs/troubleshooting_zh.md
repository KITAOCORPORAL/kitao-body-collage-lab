# 常见问题

## 搜不到 KBL 节点

1. 确认目录是 `N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab`。
2. 确认不是多套了同名目录，例如 `Kitao_Body_Collage_Lab\Kitao_Body_Collage_Lab\__init__.py`。
3. 完全退出 Comfy Desktop 后重新启动。
4. 查看终端是否有 `IMPORT FAILED` 或缺少 Python 包的报错。

## 路径输出不是 N 盘

启动前设置 `KBL_MODEL_ROOT`、`KBL_EXPORT_ROOT`。不要把大模型复制到 C 盘的用户缓存。

## 提示缺少 GroundingDINO 或 SAM2

模型目录必须是完整 Transformers 本地仓库，不能只有 `.pt`。检查 `config.json`、预处理器/处理器配置、tokenizer（GroundingDINO）和 `model.safetensors` 是否直接位于报错给出的目录。

## 提示缺少 Florence-2 或 Florence 加载失败

运行 `scripts\model_integrity.py`，确认 `florence2` 目录同时包含配置、tokenizer、processor、`model.safetensors`、Microsoft 官方 `modeling_florence2.py` / `configuration_florence2.py` / `processing_florence2.py` 和 revision 记录。不要只复制权重，也不要为此升级整个 ComfyUI。生产推理必须 local-only；缓存应位于 N 盘。

## 提示缺少 DWPose 模型

确认 `yolox_l.onnx` 与 `dw-ll_ucoco_384.onnx` 直接位于 `N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\dwpose`。安装脚本会逐文件检查，不以空目录冒充完成。

## DWPose 显示 CPUExecutionProvider

这是允许的 fallback，表示当前 ComfyUI 的 ONNX Runtime 未提供 CUDA Provider。KBL 会在日志明确打印 Provider；不要为了提速直接升级 `onnxruntime-gpu`，以免破坏其他节点环境。

## CUDA 显存不足

先关闭其他模型工作流，降低 `max_elements`、`max_regions` 或旧工作流的 `max_detections`。KBL 按 Florence → GroundingDINO（仅 extra prompt）→ SAM2 → DWPose 的顺序释放模型，不会让它们长期同时驻留显存。

## One-Click 漏掉小道具或出现背景块

先使用 `complete`，并在 Advanced 工作流中提高 `max_regions`。Florence-2 负责对象清点，不保证识别每个极小或严重遮挡的道具；可用旧版 GroundingDINO `extra_prompt` 补充明确名词。未标注区域会保留为 `region_xxx.png`，但接触多边且占画面过大的候选会被背景过滤。建筑、桌子等明确语义对象即使较大也不会仅按面积删除。

高分辨率图片的独立全尺寸 MASK 本身会占用大量内存。KBL 会按总 mask 像素量给 `max_candidates` 加安全上限，并在终端打印实际值；不会缩放最终 mask 或原图。

## guided 没有结果

降低 `confidence_threshold` 至 `0.20-0.25`，使用简短英文名词并以点号分隔。GroundingDINO 只产生 bbox；没有 bbox 时 guided 模式会稳定输出空结果。

## auto 出现重复区域

适当降低 `mask_iou_threshold`（例如 `0.75`）或提高 `min_mask_area`。auto 是 SAM2 点网格候选生成，并非 GroundingDINO 类别检测，因此标签为 `auto_object`。

## 手、脚或肢体显示 missing / uncertain

先检查 pose preview。完全遮挡或关键点低于阈值时，KBL 按设计不输出该部位；可小幅降低 `joint_confidence_threshold`，但不要把不可见部位当作成功。`include_face`/`include_hair` 在阶段 C 仍是实验接口，不会输出伪 mask。

## 部位粘连或覆盖躯干

保持 `resolve_overlap=true`，它使用各部位骨架/anchor 距离场处理局部重叠。Body Splitter 诊断中的 `overlap_pixels_after` 应为 0；如异常，请保留原图路径、pose preview、body mask sheet 与 diagnostics 复现。

## BiRefNet 显示 NOT INSTALLED

v0.1.0 的 BiRefNet 仅为实验接口，不是主流程依赖。选择 `safe` 或 `soft` 即可完成精修和导出；选择 `birefnet` 时节点会明确报错，不会伪装成已执行。

## Manifest validator FAILED

不要手工把临时目录改名为正式项目。查看 validator 的具体错误：缺文件、非 RGBA、空 alpha、非法 bbox/anchor 或 raw mask 面积不一致。Exporter 失败时会清除 `.kbl_tmp`，不会设置 `export_complete=true` 的残缺项目。

## 项目名已存在

默认 `version` 会创建 `_v002`、`_v003`。`skip` 会明确终止；`replace` 会在新项目验证通过后替换旧目录。不会静默覆盖。
