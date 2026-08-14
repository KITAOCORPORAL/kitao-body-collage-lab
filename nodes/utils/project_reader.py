"""Read a portable KBL project without importing or running AI models."""

import json
from pathlib import Path, PurePosixPath

from PIL import Image

from .manifest_utils import load_manifest_data, validate_manifest


def _asset_path(project_root, relative_path):
    value = str(relative_path or "")
    posix = PurePosixPath(value)
    if not value or posix.is_absolute() or ".." in posix.parts or "\\" in value:
        raise ValueError(f"KBL project asset path 必须是安全的 POSIX 相对路径：{value!r}")
    target = (project_root / Path(*posix.parts)).resolve()
    if project_root != target and project_root not in target.parents:
        raise ValueError(f"KBL project asset path 越界：{value!r}")
    return target


def _read_assets(project_root, records):
    assets = []
    for record in records:
        target = _asset_path(project_root, record.get("file"))
        with Image.open(target) as image:
            png = {"width": image.width, "height": image.height, "mode": image.mode}
        item = dict(record)
        item["resolved_file"] = str(target)
        item["png"] = png
        assets.append(item)
    return assets


def load_kbl_project(project_path):
    """Load validated Manifest v0.1 metadata and PNG headers from a KBL project."""
    root = Path(project_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"KBL project directory 不存在：{root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"KBL manifest 不存在：{manifest_path}")
    validation = validate_manifest(manifest_path)
    if validation["status"] != "PASS":
        raise ValueError("KBL manifest validation FAILED: " + "; ".join(validation["errors"]))
    manifest = load_manifest_data(manifest_path)
    diagnostics_path = root / "pipeline_diagnostics.json"
    pipeline_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.is_file() else None
    return {
        "project": {
            "path": str(root),
            "manifest_path": str(manifest_path),
            "name": manifest["project_name"],
            "manifest_version": manifest["manifest_version"],
            "pipeline_version": manifest["pipeline_version"],
        },
        "source": dict(manifest["source"]),
        "elements": _read_assets(root, manifest["elements"]),
        "body_parts": _read_assets(root, manifest["body_parts"]),
        "diagnostics": {
            "manifest": dict(manifest["diagnostics"]),
            "pipeline": pipeline_diagnostics,
            "validation": validation,
        },
    }


__all__ = ["load_kbl_project"]
