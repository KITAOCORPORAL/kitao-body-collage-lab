# Kitao Body Collage Lab v0.1｜阶段 C 人体部位拆分报告

## 1. 总状态

**PASS**

阶段 C 已完成真实 GroundingDINO person → SAM2 person mask → 自有 DWPose ONNX backend → 人体结构候选 → 一次加载的局部 SAM2 refine → 骨架距离场重叠归属。未使用生成人体、inpainting、重绘或遮挡补全，也未进入阶段 D。

强制 TEST C1 已通过。测试图同时包含弯曲且贴近躯干的手臂，覆盖 TEST C2。当前未使用额外交叉腿样本进行 TEST C3，按批准书记为待后续样本验证，不影响 C1 强制验收。

## 2. DWPose

模型仓库：`yzd-v/DWPose`

模型文件：

- `N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\dwpose\yolox_l.onnx`，216,746,733 bytes
- `N:\Comfy-Desktop\ComfyUI-Shared\models\Kitao_Body_Collage_Lab\dwpose\dw-ll_ucoco_384.onnx`，134,399,116 bytes

Provider：`CPUExecutionProvider`

原因：当前 ComfyUI 环境的 `onnxruntime 1.27.0` 仅提供 CPU/Azure Provider。未强行安装或升级 `onnxruntime-gpu`。

真实 DWPose 运行时间：**0.9203 秒**。完整真实链路为 5.3007 秒；ComfyUI API Prompt 执行为 5.99 秒。

KBL 使用自有 `DWPoseBackend`，不依赖 `comfyui_controlnet_aux` 执行。关键点映射按当前模型真实 133 点输出建立：COCO body 17、foot 6、face 68、left hand 21、right hand 21；所有点转换回原图坐标。confidence 在 schema 边界钳制为 0–1。

## 3. 测试图片

- 文件名：`2022_10_23_11_46_38_IMG_2227.JPG`
- 原路径：`N:\comfyui\input\2022_10_23_11_46_38_IMG_2227.JPG`
- 分辨率：1084 × 1444
- 原图未复制进项目仓库
- 场景：单人全身摄影；头、双臂、双手、双腿、双脚可见；双臂弯曲并靠近躯干

## 4. DWPose 识别结果

- 识别人数：1
- 与阶段 B `KBL_ELEMENTS` 中 `person_01` 通过 bbox IoU 匹配
- 阈值 0.30 下有效点：133 / 133（含 body、foot、face、双手）
- anatomical left/right：通过；使用人物自身左右标签，不按画面左右推断
- pose preview：肉眼检查通过，肩、肘、腕、髋、膝、踝、双手关键点均落在真实人体上

## 5. Body Split 结果

成功部位共 14 个：

`head`、`torso`、`left_upper_arm`、`left_forearm`、`left_hand`、`right_upper_arm`、`right_forearm`、`right_hand`、`left_thigh`、`left_calf`、`left_foot`、`right_thigh`、`right_calf`、`right_foot`。

- `missing_parts`：`[]`
- `uncertain_parts`：`[]`
- 左手 / 右手：成功 / 成功
- 左脚 / 右脚：成功 / 成功
- 每个 mask：1084 × 1444，与原图一致
- 每个 body part mask：经程序断言均为 person mask 子集
- person mask area：144,986
- body part union area：128,961
- `body_part_union_vs_person`：0.889472（88.9472%）
- overlap before：23,292 pixels
- overlap after：0 pixels

## 6. 每个部位明细

| label | area | bbox `[x1,y1,x2,y2]` | confidence | SAM score | quality | source |
| --- | ---: | --- | ---: | ---: | --- | --- |
| head | 11,849 | `[372,462,478,608]` | 0.994060 | 0 | ok | dwpose+person_mask |
| torso | 29,745 | `[350,601,508,860]` | 0.592785 | 0 | ok | dwpose+person_mask |
| left_upper_arm | 5,607 | `[480,594,552,744]` | 0.758764 | 0 | ok | dwpose+person_mask |
| left_forearm | 2,667 | `[496,729,551,835]` | 0.797460 | 0.902344 | ok | dwpose+sam2 |
| left_hand | 5,927 | `[414,813,515,901]` | 0.870558 | 0 | ok | dwpose+person_mask |
| right_upper_arm | 1,653 | `[340,611,384,752]` | 0.778552 | 0.820312 | ok | dwpose+sam2 |
| right_forearm | 1,630 | `[286,742,363,825]` | 0.810837 | 0.882812 | ok | dwpose+sam2 |
| right_hand | 2,352 | `[269,807,302,912]` | 0.892423 | 0.925781 | ok | dwpose+sam2 |
| left_thigh | 15,246 | `[433,853,553,1096]` | 0.592785 | 0.914062 | ok | dwpose+sam2 |
| left_calf | 14,815 | `[477,1074,580,1312]` | 0.809955 | 0.949219 | ok | dwpose+sam2 |
| left_foot | 3,331 | `[510,1281,567,1360]` | 0.645952 | 0.757812 | ok | dwpose+sam2 |
| right_thigh | 17,863 | `[334,835,424,1094]` | 0.607273 | 0.902344 | ok | dwpose+sam2 |
| right_calf | 13,653 | `[358,1075,440,1304]` | 0.771172 | 0.921875 | ok | dwpose+sam2 |
| right_foot | 2,623 | `[362,1282,436,1353]` | 0.608954 | 0.910156 | ok | dwpose+sam2 |

每个部位均保存 `id`、`label`、`category`、原尺寸 `mask`、`bbox`、`area`、`confidence`、`pose_confidence`、`sam_score`、`quality_flag`、`source_person_id`、`anchor`、`original_anchor`、`orientation_deg` 和 `source`。四肢另存 `joint_start`、`joint_end`。

SAM score 为 0 表示该部位的批量 SAM2 候选未满足保留条件，最终使用 DWPose 几何区域与真实 person mask 的交集；并非 mock 或生成像素。

## 7. 拆分方法与禁止补全

- upper arm / forearm / thigh / calf：按关节段构造动态宽度 corridor，宽度同时参考关节距离与人物宽度；不是固定矩形或固定粗线。
- hand：使用 wrist + DWPose 21 点 hand cloud 的凸包与扩展，再约束于 person mask；不足时标记 uncertain 或 missing。
- foot：使用 ankle + big toe / small toe / heel 的几何区域，再约束于 person mask；出画时允许 partial。
- head：使用 face point cloud、nose 与 shoulder 比例主动建立区域。
- torso：使用双肩、双髋主动建立扩展 polygon，不采用简单的 person 减肢体残差。
- SAM2：同一人物的全部 part prompts 一次加载、批量推理、一次 cleanup；不是 load × 14。
- overlap：使用 joint segment / anchor 距离场分配重叠像素，处理贴身和弯曲肢体。
- low confidence / 空关键点：单元测试确认不会生成假部位。

## 8. Validation 文件

根目录：`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_c`

正式 smoke test：

- `pose_preview.png`
- `body_parts_preview.png`
- `body_mask_sheet.png`
- `body_exploded_preview.png`
- `body_split_result.json`

ComfyUI API 实际执行证据：

- `comfyui_api_pose_preview.png`
- `comfyui_api_body_parts_preview.png`
- `comfyui_api_body_mask_sheet.png`
- `comfyui_api_body_exploded_preview.png`
- `comfyui_stage_c.log`
- `comfyui_stage_c.err.log`

## 9. ComfyUI 工作流真实执行状态

**PASS**

- 工作流文件：`workflows\Kitao_Body_Collage_Lab_v0.1_body_split.json`
- 实际部署：`N:\comfyui\ComfyUI-Installs\ComfyUI\ComfyUI\custom_nodes\Kitao_Body_Collage_Lab`
- ComfyUI 成功注册：`KBL_Pose_Estimator`、`KBL_Body_Splitter`
- API Prompt ID：`1c80c498-2777-4bd1-90d3-b52486411a19`
- `node_errors`：`{}`
- 状态：`success` / `completed: true`
- 4 个 Preview 节点均返回真实图像
- 日志：`[KBL] detected=8 masks=8 kept=8 mode=guided`
- 日志：`[KBL] DWPose provider: CPUExecutionProvider`
- 日志：`Prompt executed in 5.99 seconds`

验证仅使用隐藏 ComfyUI 后端、HTTP API 和文件检查；未打开浏览器或 localhost 页面。

## 10. 测试

标准库 `unittest`：**10 / 10 PASS**。

覆盖：person 传入 Pose Estimator、无 person 安全、多 person largest/center/index、原图坐标保持、anatomical left/right、原尺寸 mask、person mask 子集、空/低置信度点、missing 不伪造、overlap resolution、anchor、orientation、confidence 边界、ComfyUI 节点注册。

`pytest` 未安装，因此未污染环境安装测试框架；测试文件可由 `python -m unittest` 直接执行。

## 11. 阶段 C 修改文件完整清单

新增：

- `nodes\backends\__init__.py`
- `nodes\backends\dwpose_backend.py`
- `nodes\pose_nodes.py`
- `nodes\body_split_nodes.py`
- `nodes\utils\pose_utils.py`
- `scripts\smoke_test_body_split.py`
- `tests\test_stage_c.py`
- `workflows\Kitao_Body_Collage_Lab_v0.1_body_split.json`
- `回传给GPT_Kitao_Body_Collage_Lab_v0.1_阶段C人体部位拆分报告.md`

修改：

- `README.md`
- `requirements.txt`
- `docs\install_zh.md`
- `docs\usage_zh.md`
- `docs\troubleshooting_zh.md`
- `nodes\__init__.py`
- `nodes\model_backends.py`
- `nodes\utils\path_config.py`
- `scripts\install_models.py`
- `scripts\model_integrity.py`

部署副本已同步到实际 ComfyUI `custom_nodes` 目录，关键文件哈希一致。

## 12. Python / Torch / CUDA / ONNX Runtime

- 新增依赖声明：`onnxruntime>=1.17`
- 当前已安装：`onnxruntime 1.27.0`
- 未执行依赖升级或重装
- Torch：未修改，保持 `2.10.0+cu130`
- CUDA：未修改
- Transformers：未修改，保持 `5.8.0`
- ONNX Runtime：未修改

## 13. 已知问题

1. 当前 ONNX Runtime 没有 CUDA Provider，DWPose 使用 CPU fallback；功能通过，速度约 0.92 秒/图。
2. TEST C3（交叉或明显弯曲腿）尚无专门样本验收，待后续提供或确认样本后回归。
3. `fine` 当前等同 `standard`；face / hair 参数是实验接口，diagnostics 会明确记录请求但不输出伪部位。
4. 人体 mask 包含衣物，Body Splitter 的语义部位像素相应包含实际可见服装；这是摄影人体拼贴的预期，不追求裸露皮肤分割。
5. ComfyUI 启动日志中其他第三方节点存在 `triton`、Impact Pack SAM2 可选依赖警告，与 KBL 无关，未影响 KBL Prompt 成功执行。

## 14. Mock / placeholder

推理、mask、预览和 ComfyUI 验收中 **不存在 mock / placeholder**。

仅 `fine` 的 face/hair 保留实验参数，代码明确不输出它们；这不是伪实现。单元测试使用合成 pose fixture 和 fake backend 仅测试数据契约，不参与真实验收结论。

## 15. 阶段 D 前需要解决的问题

无阻断项。建议在阶段 D 的非阻断回归清单中加入一张明确交叉腿/大幅弯腿样本（TEST C3），并继续保留 mask 子集、左右语义、anchor/orientation 与重叠为 0 的断言。

## 16. 是否建议进入阶段 D

**建议进入阶段 D，但本次未进入。**

阶段 C 的节点、数据契约、真实模型、真实摄影图、真实 ComfyUI API 工作流与验证文件均已通过；TEST C3 属于批准书允许的后续样本验证项。
