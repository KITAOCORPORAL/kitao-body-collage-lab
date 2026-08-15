# Kitao Body Collage Lab v0.1.2 完整元素模式修复报告

## 结论

v0.1.2 `One-Click Clean Objects` 已完成。默认入口改为 `object_clean`，真实黑白复杂图 `DSC00080_1.jpg` 的最终 Contact Sheet 只保留完整 person；hair/face、body parts 和破碎的 `object_002` 均未导出。Manifest schema 继续使用 0.1，Project Reader 兼容测试通过。

## 1. output_mode

`KBL_OneClick_Decompose_Export` 已新增 `object_clean`（默认）与 `collage_parts`。One-Click 节点只暴露 image、project_name、output_mode、padding、export_root；`image_meta` 保持可连接的可选输入。v0.1.1 随附工作流改由兼容适配节点加载，旧 JSON 未删除。

## 2. object_clean 内部规则

- quality = balanced
- inventory_strategy = semantic_only
- 只执行 Florence-2 `<OD>`
- region proposal = false
- include_body_parts = false
- keep_unlabeled_objects = false
- allow_person_subelements = false
- prefer_whole_objects = true
- Contact Sheet 只接收最终过滤后的完整元素

`collage_parts` 保留 dense caption、region proposal、匿名区域和人体部位素材挖矿路径。

## 3. Parent-Child Suppression

已新增。完整 person 成立后，位于其内部的 hair、head、face、arm、hand、leg、foot、clothing、shoe、body region 和 region 局部默认拒绝。Florence 的 `human face` 现在优先归一为 face，不再被 `human` 错归一为 person。

## 4. Background Rejection

已新增 object-clean 专用 background-like rejection。对 `object_xxx`、`region_xxx` 和非稳定语义候选使用更严格规则，综合 mask/bbox 面积、接触画面边界数量、bbox 覆盖、内部包含的语义对象数进行拒绝。新增 diagnostics：`background_like_rejected_count` 及对应 label 列表。

## 5. Fragment Filter / Whole-Object Priority

已新增 `fragment_score`、`whole_object_score`、细长/稀疏/小面积内嵌/高重叠局部残片过滤、semantic object 对 generic region 的优先级，以及同标签重叠候选的完整对象选择。person 优先选择面积最大的完整 mask。两个分数作为可选字段写入 Manifest element，不改变 Manifest 0.1 必需字段。

## 6. 新工作流

- `workflows/Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json`
- `workflows/Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json`

README 默认推荐 CLEAN_OBJECTS；v0.1.1 工作流仍保留，并使用 `KBL_OneClick_Decompose_Export_v011` 兼容节点。

## 7. 黑白复杂图专项回归

样本：`N:\comfyui\input\DSC00080_1.jpg`，4672×7008，CUDA 实跑 Florence-2 + SAM2。

旧版实测：person、hair、脏的大块 object_002。

新版最终结果：

- element_count = 1
- person = 1，完整 1506×2951 人物 cutout
- hair/face 单独导出 = 0
- body_part_count = 0
- unlabeled_elements = 0
- object_002 导出 = 0
- background_like_rejected_count = 1（object_002）
- parent_child_suppressed_count = 1（face）
- inventory tasks = [`<OD>`]
- Contact Sheet 仅 1 个完整人物元素

专项项目：`C:\Users\Admin\Documents\灵图\output\kbl-v0.1.2-regression\KBL_V012_DSC00080_CLEAN_v002`

## 8. 默认推荐

推荐普通用户默认使用 `object_clean`。当完整物体无法可靠识别时，宁可少导出，也不保留大块背景伪对象。需要身体局部、匿名区域或拼贴碎片时再使用 `collage_parts`。

## 9. 已知限制

- `object_clean` 不能补画被遮挡或超出画面的对象。
- Florence-2 `<OD>` 未定位到的主要道具不会凭空出现；默认模式优先 precision，不保证 recall。
- SAM2 若没有产生完整 mask，规则层只能拒绝错误候选，不能重建完整对象。
- 同一标签、彼此高度重叠的多个真实对象仍可能被当作重复候选压制；非重叠同类对象可保留。

## 验证

- 单元、Manifest、Project Reader、Stage B/C/D/E/F：57/57 PASS
- 两个 v0.1.2 workflow JSON：解析 PASS
- 真实模型专项回归：PASS
- 真实结果视觉检查：PASS
