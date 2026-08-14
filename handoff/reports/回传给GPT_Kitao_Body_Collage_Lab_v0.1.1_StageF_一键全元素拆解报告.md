# Kitao Body Collage Lab v0.1.1 Stage F 一键全元素拆解报告

## 结论

Stage F：PASS。KBL 已具备 ComfyUI 图片选择器、无提示词 Scene Inventory、SAM2 全元素分割、背景过滤、语义优先去重、未知区域保留、人物自动路由、透明 PNG 导出、Contact Sheet、Exploded View、Manifest 0.1 和诊断输出。未开发 Collage Board、独立软件或网页。

“All Elements”按 object-level decomposition 验收，不等同于每个像素或每个极小道具都必须成为独立图层。复杂遮挡场景中的小物体召回仍是后续真实用户测试的调优项。

## 实现基线

- 版本：`KBL_VERSION = 0.1.1`
- Manifest：继续使用 `0.1`，新增字段均为可选字段，旧 Reader 兼容
- Florence-2：`microsoft/Florence-2-base-ft`
- 固定 revision：`f6c1a25888ffc1d945ee8a1a77ac833c7303d46e`
- 模型目录：`N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\florence2`
- 生产加载：本地 Microsoft snapshot、`local_files_only`、N 盘 HF/Transformers 动态模块缓存
- 实测设备：RTX 5060 Ti 16 GB；Florence-2 使用 CUDA + BF16
- Florence-2 最大 CUDA allocation：717,854,208 bytes
- SAM2 最大 CUDA allocation：628,922,368 bytes
- DWPose：CPUExecutionProvider fallback，符合既定基线

当前 Transformers 5.8 原生 Florence 类与该旧 checkpoint 的参数命名不兼容，因此 backend 使用固定 Microsoft revision 的官方本地代码，并在加载后绑定 BART shared weights；未升级 Torch、CUDA 或 Transformers。生产推理仍完全离线。

## 新增节点与工作流

- `KBL_Load_Image_Picker` / `KBL 选择图片`
- `KBL_Scene_Inventory` / `KBL 场景元素清点`
- `KBL_Universal_Element_Detector` / `KBL 全元素分割`
- `KBL_OneClick_Decompose_Export` / `KBL 一键全元素拆解`
- `workflows/Kitao_Body_Collage_Lab_v0.1.1_ONE_CLICK_ALL_ELEMENTS.json`
- `workflows/Kitao_Body_Collage_Lab_v0.1.1_ALL_ELEMENTS_ADVANCED.json`

One-Click 节点只负责串联已有 public nodes/backends，没有复制第二套 Florence、GroundingDINO、SAM2、DWPose、Body Split、Refiner 或 Exporter 实现。

## 自动清点与导出

Complete 模式实际执行 `<OD>`、`<DENSE_REGION_CAPTION>`、`<REGION_PROPOSAL>`，并从详细 caption 中提取已知概念做本地 open-vocabulary grounding。Region Proposal 即使无语义也可保留为 `region_xxx.png`。背景过滤实际覆盖大面积、接触多边、包含多个语义对象的未标注候选；去重优先保留 semantic labeled mask。

所有资产均以源图 RGB 加 mask alpha 写入，不做生成、重画或对象补全。导出文件名过滤 Windows 禁止字符并限制长度。

## 真实样例验收

全部输入仅来自 ComfyUI Shared Input。

### F1：人物、包、鞋、服装与身体部件

- 项目：`N:\ComfyUI\output\Kitao_Body_Collage_Lab\KBL_STAGE_F_F1_RELEASE_v002`
- Scene candidates：18（semantic 11 / region 7）
- 最终 generic elements：12（semantic 10 / unlabeled 2）
- 背景拒绝：1；重复移除：5
- 人物：1；Body Split：`person_01`
- 身体部件：14；总 PNG：26
- 总耗时：17.7546 秒
- 肉眼结果：person、两件 clothing、两只 shoe、bag 和 14 个身体部件均有效；SAM2 alpha 边缘整体可用。建筑也被当作有语义场景元素保留，仍有少量建筑区域图层，属于当前精度边界。

### F2：多道具拼贴场景

- 项目：`N:\ComfyUI\output\Kitao_Body_Collage_Lab\KBL_STAGE_F_F2_RELEASE_v002`
- Scene candidates：17（semantic 11 / region 6）
- 最终 elements：11；背景拒绝：1；重复移除：5
- 人物：3，按配置不执行 Body Split；总 PNG：11
- 总耗时：7.4929 秒
- 肉眼结果：桌面、电话、卡片、椅子、玻璃杯/眼镜、脚和人物图层可见；三个人物均作为 generic PNG 保留。

### F3：无人场景安全路由

- 项目：`N:\ComfyUI\output\Kitao_Body_Collage_Lab\KBL_STAGE_F_F3_RELEASE_v002`
- Scene candidates：3；背景拒绝：1；重复移除：1
- `people_detected = 0`；`body_split_person = null`；身体部件：0
- generic PNG：1；总耗时：9.7057 秒
- 运行日志没有 DWPose 调用，无人物路径正常完成导出。源图为白底 `HUMAN MASK` 文字测试图，最终保留一个文字/画布对象；它验证路由安全，不代表复杂无人道具召回质量。

### F4：复杂杂乱场景

- 项目：`N:\ComfyUI\output\Kitao_Body_Collage_Lab\KBL_STAGE_F_F4_RELEASE_v002`
- Scene candidates：12（semantic 7 / region 5）
- SAM masks：12；背景拒绝：1；重复移除：6
- 最终 elements：5；人物：2；总 PNG：5
- 总耗时：11.8958 秒
- 肉眼结果：两个人物、红色台面/布料和帘布形成可用独立图层，没有同一对象五份近似输出。小礼盒和装饰球没有稳定成为独立图层，是本版明确记录的召回限制。

## 自动验证

- Unit + compatibility：50 / 50 PASS（旧阶段 35 + Stage F 15）
- 5 份 workflow JSON：解析与节点/连线双向一致性 PASS
- Python compileall：PASS
- Model integrity：GroundingDINO / SAM2 / DWPose / Florence-2 全部 PASS
- 旧 Stage B/C/D/E 真实照片回归：15 个资产、14 个标准身体部件、Manifest validator PASS
- F1–F4 Manifest 0.1：4 / 4 PASS
- 实际导出资产：43 / 43 为 RGBA，43 / 43 含透明度
- 对每个资产逐像素比较 `alpha > 0` 区域：43 / 43 的 RGB 与源图 crop 完全一致
- Scene Inventory Preview / Contact Sheet / Exploded View / Diagnostics：4 / 4 齐全

## 发布与后续

发布分支：`feat/one-click-all-elements`。发布时合并到 `main`，保留 `v0.1.0`，新增 `v0.1.1`。下一步仅为真实用户测试与 segmentation quality tuning，不进入 Collage Board。
