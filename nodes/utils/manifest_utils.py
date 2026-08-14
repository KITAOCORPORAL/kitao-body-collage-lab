"""KBL manifest v0.1 validation and portable-path helpers."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ...version import KBL_MANIFEST_VERSION


def load_manifest_data(manifest_path):
    """Read a KBL manifest without loading models or image pixels."""
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def validate_manifest_schema(manifest):
    """Validate the frozen v0.1 structure independently of project files."""
    errors = []
    for key in ("manifest_version", "pipeline_version", "project_name", "source", "image", "elements", "body_parts", "diagnostics", "export_complete"):
        if key not in manifest:
            errors.append(f"缺少顶层字段：{key}")
    if manifest.get("manifest_version") != KBL_MANIFEST_VERSION:
        errors.append(f"manifest_version 必须为 {KBL_MANIFEST_VERSION}")
    if not isinstance(manifest.get("source"), dict):
        errors.append("source 必须是 object")
    if not isinstance(manifest.get("image"), dict):
        errors.append("image 必须是 object")
    for collection in ("elements", "body_parts"):
        if not isinstance(manifest.get(collection), list):
            errors.append(f"{collection} 必须是 array")
    return errors


def relative_posix(path, root):
    return Path(path).relative_to(root).as_posix()


def _valid_bbox(value, width, height):
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(number, (int, float)) and math.isfinite(number) for number in value):
        return False
    x1, y1, x2, y2 = value
    return 0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height


def validate_manifest(manifest_path):
    path = Path(manifest_path)
    root = path.parent
    errors = []
    try:
        manifest = load_manifest_data(path)
    except Exception as exc:
        return {"status": "FAILED", "errors": [f"JSON 无法解析：{exc}"], "checked_assets": 0}
    errors.extend(validate_manifest_schema(manifest))
    if manifest.get("export_complete") is not True:
        errors.append("export_complete 不是 true")
    image_meta = manifest.get("image", {})
    width, height = int(image_meta.get("width", 0)), int(image_meta.get("height", 0))
    checked = 0
    for kind in ("elements", "body_parts"):
        for index, item in enumerate(manifest.get(kind, [])):
            prefix = f"{kind}[{index}]"
            file_value = item.get("file")
            if not file_value:
                errors.append(f"{prefix} 缺少 file")
                continue
            asset_path = root / Path(file_value)
            if not asset_path.is_file():
                errors.append(f"{prefix} 文件不存在：{file_value}")
                continue
            checked += 1
            try:
                with Image.open(asset_path) as png:
                    if png.mode != "RGBA":
                        errors.append(f"{prefix} PNG 不是 RGBA")
                    alpha = np.asarray(png.getchannel("A"))
                    if int(np.count_nonzero(alpha)) == 0:
                        errors.append(f"{prefix} alpha 为空")
                    png_width, png_height = png.size
            except Exception as exc:
                errors.append(f"{prefix} PNG 无法读取：{exc}")
                continue
            for field in ("original_bbox", "content_bbox", "crop_bbox"):
                if not _valid_bbox(item.get(field), width, height):
                    errors.append(f"{prefix} {field} 非法")
            origin = item.get("crop_origin")
            if not isinstance(origin, list) or len(origin) != 2:
                errors.append(f"{prefix} crop_origin 非法")
            anchor = item.get("local_anchor")
            if not isinstance(anchor, list) or len(anchor) != 2 or not (-1 <= anchor[0] <= png_width + 1 and -1 <= anchor[1] <= png_height + 1):
                errors.append(f"{prefix} local_anchor 不在 PNG 坐标系内")
            if not isinstance(item.get("orientation_deg", 0), (int, float)):
                errors.append(f"{prefix} orientation_deg 不是数字")
            for mask_field in ("raw_mask_file", "refined_mask_file", "alpha_mask_file"):
                value = item.get(mask_field)
                if value and not (root / Path(value)).is_file():
                    errors.append(f"{prefix} {mask_field} 不存在：{value}")
            raw_value = item.get("raw_mask_file")
            if raw_value and (root / Path(raw_value)).is_file():
                with Image.open(root / Path(raw_value)) as raw_png:
                    raw_area = int(np.count_nonzero(np.asarray(raw_png)))
                tolerance = max(2, int(item.get("area", 0) * 0.001))
                if abs(raw_area - int(item.get("area", 0))) > tolerance:
                    errors.append(f"{prefix} area 与 raw mask 不一致")
    return {"status": "PASS" if not errors else "FAILED", "errors": errors, "checked_assets": checked}
