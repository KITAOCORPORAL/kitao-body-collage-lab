# Kitao Body Collage Lab v0.1｜阶段 D 素材导出报告

## 1. 总状态

**PASS**

阶段 D 已把阶段 B/C 的真实 `KBL_ELEMENTS` 与 `KBL_BODY_PARTS` 转换为可独立移动的摄影拼贴素材包：原图副本、14 个人体部位 RGBA PNG、8 个 generic element RGBA PNG、raw/refined/alpha masks、五类正式预览和可移植 Manifest v0.1。未开发 Collage Board，也未使用生成、inpainting 或重绘。

## 2. 实际测试图片

- 文件：`N:\comfyui\input\2022_10_23_11_46_38_IMG_2227.JPG`
- 文件名：`2022_10_23_11_46_38_IMG_2227.JPG`
- 分辨率：1084 × 1444
- 原图未复制进源码仓库；测试项目中的 `source/original.jpg` 由 `shutil.copy2` 直接复制原文件，未重新编码 JPEG

## 3. 导出项目目录

命令行完整真实验收项目：

`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_d\KBL_STAGE_D_TEST`

真实 ComfyUI 工作流项目：

`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_d\KBL_STAGE_D_COMFYUI_TEST`

项目内部包含 `source/`、`body/`、分类后的 `elements/`、`masks/raw|refined|alpha/`、`preview/` 和 `manifest.json`。项目先写同盘 `.kbl_tmp`，validator 通过后才原子改名；默认 `version` 不覆盖旧项目。

## 4. 导出数量

- body part：14
- generic element：8
- 合计：22
- generic labels：person ×1、face ×1、foot_shoe ×2、hair ×1、clothing ×1、hand ×2
- 复合标签 `foot_shoe` 通过简单 token 字典归入 `elements/clothing/`

## 5. 每个 Body PNG

所有 PNG 均为 cropped、RGBA、8-bit、straight alpha，content 周围默认保留 24px padding。

| label / file | PNG 尺寸 | PNG 内 alpha bbox | raw area | refined area | quality |
| --- | ---: | --- | ---: | ---: | --- |
| `body_head.png` | 155×195 | `[24,24,131,171]` | 11,845 | 12,047 | ok |
| `body_torso.png` | 208×309 | `[24,24,184,285]` | 29,744 | 30,586 | ok |
| `body_left_upper_arm.png` | 121×194 | `[24,24,97,170]` | 5,604 | 5,879 | ok |
| `body_left_forearm.png` | 105×155 | `[24,24,81,131]` | 2,666 | 2,812 | ok |
| `body_left_hand.png` | 151×138 | `[24,24,127,114]` | 5,927 | 6,224 | ok |
| `body_right_upper_arm.png` | 93×191 | `[24,24,69,167]` | 1,652 | 1,970 | ok |
| `body_right_forearm.png` | 127×131 | `[24,24,103,107]` | 1,629 | 1,837 | ok |
| `body_right_hand.png` | 82×154 | `[24,24,58,130]` | 2,347 | 2,375 | ok |
| `body_left_thigh.png` | 169×289 | `[24,24,145,265]` | 15,244 | 15,539 | ok |
| `body_left_calf.png` | 153×285 | `[24,24,129,261]` | 14,813 | 15,337 | ok |
| `body_left_foot.png` | 106×129 | `[24,24,82,105]` | 3,331 | 3,530 | ok |
| `body_right_thigh.png` | 139×309 | `[24,24,115,285]` | 17,858 | 18,203 | ok |
| `body_right_calf.png` | 131×279 | `[24,24,107,255]` | 13,651 | 14,027 | ok |
| `body_right_foot.png` | 123×121 | `[24,24,99,97]` | 2,623 | 2,800 | ok |

## 6. 手 / 脚单独验收

对 `left_hand`、`right_hand`、`left_foot`、`right_foot` 逐张执行：

- PNG mode = RGBA：PASS
- alpha 非空：PASS
- alpha bbox 合法：PASS
- cropped 尺寸和 24px padding：PASS
- `local_anchor` 位于 PNG 坐标系：PASS
- 任取有效 alpha 像素，其 RGB 与原摄影图对应坐标逐字节相同：PASS
- 无白底 / 黑底或预乘 alpha RGB 污染：PASS

同样检查了 head 和 torso，六个关键部位全部通过。

## 7. Safe refine

- 模式：safe
- record 数量：22
- raw area total：405,239
- refined binary area total：418,842
- 总面积变化：+3.357%
- 单对象最大绝对变化：19.249%
- 保护机制：若 1px expand 导致单对象变化超过 20%，自动撤销 expand，仅保留孔洞/小岛/closing 清理
- raw mask 未修改：PASS
- body refined mask 仍为原 person mask 子集：PASS
- 小孔、小岛测试：PASS

## 8. Soft refine

- binary mask 与 safe 的保守拓扑处理一致，raw mask 仍保留
- 0 < alpha < 1 的边界像素：120,925
- 主体内部 alpha = 1：PASS
- 远离边界的外部 alpha = 0：PASS
- body alpha 超出 person mask：0 pixels

## 9. BiRefNet

**NOT INSTALLED**

节点保留 `birefnet` 选项和 N 盘本地模型接口，但选择时会明确报告 `EXPERIMENTAL / NOT INSTALLED`，不会伪装执行或取代原始 SAM2 / Body Splitter mask。safe/soft 主流程不依赖 BiRefNet。

## 10. Manifest

- 版本：`KBL_MANIFEST_VERSION = "0.1"`
- 路径：`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_d\KBL_STAGE_D_TEST\manifest.json`
- 内部资源路径：POSIX 风格相对路径
- source：filename、绝对 source_path、相对 copied_source_path、SHA256
- image：width、height、format、orientation、color_mode
- body：文件与三类 mask、original/content/crop bbox、crop origin、原/局部 anchor、orientation、joint、confidence、quality、source
- generic：同样保存相对文件、mask、bbox、crop、置信度、面积与 original_position
- `export_complete = true` 只在全部文件完成后写入

## 11. Manifest validator

**PASS**

- JSON 可解析
- 检查资产：22
- errors：`[]`
- 所有 RGBA PNG 存在且 alpha 非空
- raw/refined/alpha mask 路径存在
- bbox / crop_origin / local_anchor / orientation 合法
- manifest area 与 raw mask 面积一致
- 独立脚本：`scripts\validate_manifest.py`

## 12. 正式预览

目录：`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_d\KBL_STAGE_D_TEST\preview`

- `overview.png`：原图位置上的真实轮廓、编号和 label
- `body_parts_preview.png`：原位置 body 检查图
- `mask_sheet.png`：refined/alpha mask 清点
- `contact_sheet.png`：真实透明 PNG + 棋盘格 + label + PNG 尺寸
- `exploded_view.png`：按人体部位顺序排列的正式素材清点图，generic elements 接在 body 后

ComfyUI 返回预览证据：

`N:\ComfyUI\output\Kitao_Body_Collage_Lab\validation_stage_d\comfyui_full_export_preview.png`

## 13. ComfyUI full workflow

**PASS**

- 工作流：`workflows\Kitao_Body_Collage_Lab_v0.1_full_export.json`
- 节点：加载 → 元素分割 → 姿态 → Body Split → Mask 精修 → 素材导出 → Preview
- 默认值：guided / standard / safe / all / cropped / padding 24 / version
- 最终 Prompt ID：`56804b44-e312-426c-97d2-c9b7af67af56`
- `node_errors = {}`
- 状态：`success`，`completed = true`
- 实际生成项目：`KBL_STAGE_D_COMFYUI_TEST`
- 该项目 Manifest validator：PASS，checked_assets = 22
- 未打开浏览器或 localhost 页面，仅使用隐藏后端和 HTTP API

## 14. 实际运行时间

- 命令行完整真实流程：9.1325 秒
- ComfyUI 最终执行：约 8.45 秒（execution_start → execution_success）
- DWPose Provider：CPUExecutionProvider

## 15. 内存 / 显存

- 无 CUDA OOM
- 无系统内存异常
- 无磁盘空间异常
- Exporter 逐对象构建 crop RGBA、写盘并释放；未一次性建立 22 个 full-canvas RGBA
- 当前测试为 1084×1444；阶段 B 的 4672×7008 mask 坐标契约未改变
- ComfyUI 其他第三方节点的 triton/Impact Pack 可选依赖警告与 KBL 无关

## 16. 阶段 C 回归

**PASS**

- B/C/D 合计 unittest：28 / 28 PASS
- 阶段 C 专项：10 / 10 PASS
- body mask ⊆ person mask：PASS
- anatomical left/right：PASS
- anchor / original_anchor：PASS
- orientation / joint：PASS
- overlap_after = 0：PASS
- Exporter `models_rerun = false`：PASS

## 17. TEST C3

**not yet validated**

未扫描其他硬盘，也未把未确认照片当作交叉腿验收样本。它仍是非阻断回归项。

## 18. 原子性与错误处理

- 非法项目名：明确拒绝
- 不可写目录 / 磁盘不足：写入前检查
- source 不存在：按 copy_source 语义明确处理
- mask 尺寸不一致 / 空 alpha：拒绝导出
- PNG / Manifest 写入或 validator 失败：删除精确 `.kbl_tmp` 目录
- `replace`：先把旧项目移动到精确 backup，成功后清除；失败时恢复
- `version`：稳定生成 `_v002`、`_v003`
- `skip`：已有项目时明确终止
- 原子失败单元测试：PASS，没有正式项目或伪完整临时项目残留

## 19. 修改文件完整清单

新增：

- `tests\test_stage_d.py`
- `scripts\smoke_test_export.py`
- `scripts\validate_manifest.py`
- `workflows\Kitao_Body_Collage_Lab_v0.1_full_export.json`
- `回传给GPT_Kitao_Body_Collage_Lab_v0.1_阶段D素材导出报告.md`

正式实现 / 修改：

- `nodes\refine_nodes.py`
- `nodes\export_nodes.py`
- `nodes\utils\alpha_utils.py`
- `nodes\utils\manifest_utils.py`
- `nodes\__init__.py`
- `nodes\utils\path_config.py`
- `README.md`
- `docs\install_zh.md`
- `docs\usage_zh.md`
- `docs\troubleshooting_zh.md`
- `examples\sample_manifest.json`

## 20. 新增依赖

**无。**

阶段 D 使用项目和 ComfyUI 已有的 NumPy、Pillow、OpenCV、Torch 张量接口以及 Python 标准库。

## 21. 环境是否修改

- Torch：未修改，`2.10.0+cu130`
- CUDA：未修改
- Transformers：未修改，`5.8.0`
- ONNX Runtime：未修改，`1.27.0`
- NumPy：未修改，`2.4.4`
- Pillow：未修改，`12.2.0`
- OpenCV headless：未修改，`4.13.0.92`

## 22. Mock / placeholder

真实推理、refine、PNG、preview、Manifest、validator 和 ComfyUI 验收中 **不存在 mock / placeholder**。

单元测试使用合成 mask fixture 测试边界条件，不参与真实验收结论。BiRefNet 是明确标注的 `NOT INSTALLED` 实验接口，不冒充完成。

## 23. 已知问题

1. BiRefNet 未安装；不影响 safe/soft 主体 PASS。
2. TEST C3 仍未验收。
3. GroundingDINO 有时产生复合标签（如 `foot_shoe`）；Exporter 采用简单 token 字典归类，不做 AI 再分类。
4. Generic element 的语义质量取决于阶段 B 检测结果；Exporter 忠实导出已有 mask，不重新推理或修正语义。
5. `full_canvas` / `both` 已实现，但主验收按默认 `cropped` 完成；超大图建议继续使用 cropped，避免单个 full-canvas PNG 的额外内存和磁盘开销。

## 24. 是否建议进入阶段 E / v0.1 收尾

**建议进入阶段 E / v0.1 收尾，但本次未进入。**

阶段 D 已形成明确工程边界：后续软件只需读取 Manifest v0.1，即可获得 RGBA PNG、raw/refined/alpha mask、bbox、anchor、orientation、quality 与 source 元数据，无需关心 GroundingDINO、SAM2 或 DWPose。
