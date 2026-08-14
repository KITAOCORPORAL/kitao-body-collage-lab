# 使用说明

## 0.1.1 一键全元素工作流

普通用户加载 `workflows/Kitao_Body_Collage_Lab_v0.1.1_ONE_CLICK_ALL_ELEMENTS.json`。在 `KBL 选择图片` 节点点击上传/选择图片，然后运行队列；不需要填写绝对路径，也不需要预先描述图片内容。

默认 `complete` 会依次运行 Florence-2 OD、Dense Region Caption、Region Proposal、SAM2、背景过滤、去重、人物路由、Mask Refiner 和 Exporter。检测到人物时，默认只对最大人物执行 14 部位拆分；多人仍会分别导出 `person_01.png`、`person_02.png`。无人场景不会调用 DWPose，也不会报错。

输出项目位于 `N:\ComfyUI\output\Kitao_Body_Collage_Lab\`，包含透明 RGBA、原始/精修/alpha mask、`scene_inventory_preview.png`、Contact Sheet、Exploded View、Manifest 0.1 和 `pipeline_diagnostics.json`。PNG 的有效 RGB 直接来自源照片，只修改 alpha，不生成或重画对象。

质量模式：

- `fast`：Florence OD → SAM2 → Export，不拆身体部位。
- `balanced`：OD + Dense Caption → SAM2 → 可选身体拆分 → Export。
- `complete`：再加入 Region Proposal、背景过滤和去重，默认推荐。

需要逐节点调参时加载 `workflows/Kitao_Body_Collage_Lab_v0.1.1_ALL_ELEMENTS_ADVANCED.json`。旧版 `KBL 加载图片` 绝对路径入口和 v0.1 工作流继续保留。

## 元素检测工作流

加载 `workflows/Kitao_Body_Collage_Lab_v0.1_element_detection.json`，在 `KBL 加载图片` 填写 JPG/JPEG/PNG/WEBP 绝对路径。

`KBL 元素检测与分割` 提供三种模式，v0.1.0 默认 `guided`：

- `guided`：GroundingDINO 文本检测 bbox，再由 SAM2 切分。
- `auto`：SAM2 使用规则点网格提出候选 mask，不加载 GroundingDINO。
- `hybrid`：合并两类结果并按 IoU、面积和同类包含关系去重。

真实验收发现 auto/hybrid 能稳定执行和去重，但点网格仍可能提出大面积背景 mask；因此把稳定的文本指定元素抠图设为默认，auto/hybrid 保留为实验选项。

默认 `mixed_scene` 预设。使用 `custom` 时，`text_prompt_override` 必须填写；支持逗号、句号、分号或换行分隔，内部会规范成 GroundingDINO 推荐的点号短语格式。

输出 `candidate_masks` 保持 `[N,H,W]` 独立批次，不合并。`elements` 是阶段 C 使用的 `KBL_ELEMENTS` 对象，逐项保留 id、label、mask、bbox、confidence、sam_score、area 和 source。

## 人体姿态与部位拆分

加载 `workflows/Kitao_Body_Collage_Lab_v0.1_body_split.json`。主链路为：加载图片 → 元素检测与分割 → 人体姿态识别 → 人体部位拆分 → 三类预览。

`KBL 人体姿态识别` 只从 `KBL_ELEMENTS` 选择 `label == person` 的对象，支持 `largest`、`center` 和 `index`。DWPose 内部可缩放推理，但 keypoint、bbox 与预览均恢复到原图坐标。

`standard` 输出 head、torso、左右 upper arm/forearm/hand/thigh/calf/foot；`basic` 合并为左右 arm/leg；`fine` 当前等同 standard，face/hair 参数只作为实验接口。每个有效部位保存原尺寸 mask、bbox、anchor、original_anchor、quality；四肢同时保存 joint_start、joint_end 与 orientation_deg。

部位候选由动态关节走廊、手/足关键点云、主动 torso/head 区域与人物 mask 共同构成。启用 `sam_refine_parts` 时，同一个 SAM2 实例批量处理所有 ROI；结果仍与候选及人物 mask 相交。不可见或置信度不足的部位写入 `missing_parts`，不生成假 mask。

真实命令行验证：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' scripts\smoke_test_body_split.py --image 'N:\comfyui\input\your_photo.jpg'
```

## Mask 精修与素材包导出

加载 `workflows/Kitao_Body_Collage_Lab_v0.1_full_export.json`。`safe` 清理小孔、小岛并执行有限 closing/expand；单个 mask 若初次相对面积漂移超过 20%，自动撤销 expand。最终变化绝对值超过 15% 时，`pipeline_diagnostics.json` 会记录 `REFINE_AREA_WARNING`。`soft` 保留同一 binary mask，只在边界 band 生成 0–1 alpha。raw mask 永远保留，body binary/alpha 永远受原 person mask 约束。

真实照片回归只处理明确给出的图片或文件夹；文件夹模式不递归，也不会扫描其他磁盘：

```powershell
python scripts\regression_test.py --image "N:\明确路径\sample.jpg"
python scripts\regression_test.py --folder "N:\明确目录"
python scripts\regression_test.py --cross-leg-image "N:\明确路径\cross_leg.jpg"
```

读取已经导出的 KBL Project 不会运行任何模型：

```python
from Kitao_Body_Collage_Lab.nodes.utils.project_reader import load_kbl_project
project = load_kbl_project(r"N:\ComfyUI\output\Kitao_Body_Collage_Lab\PROJECT")
```

Exporter 默认 `all + cropped + padding 24 + version`。每个对象逐一裁切、写 RGBA 后释放临时图像，不会同时构造全部 full-canvas RGBA。项目内部路径写为相对路径，cropped 素材保存 `crop_origin`、`original_anchor` 与 `local_anchor`。Exporter 不加载 GroundingDINO、SAM2 或 DWPose。

完整真实验收：

```powershell
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' scripts\smoke_test_export.py --image 'N:\comfyui\input\your_photo.jpg'
& 'N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe' scripts\validate_manifest.py 'N:\ComfyUI\output\Kitao_Body_Collage_Lab\PROJECT\manifest.json'
```

项目先写入同盘隐藏 `.kbl_tmp` 目录，所有 PNG、mask、preview 和 Manifest validator 通过后才改名为正式项目。`replace` 也先备份旧项目，失败时恢复；默认 `version` 不覆盖旧结果。
