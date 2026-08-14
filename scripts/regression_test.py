"""Explicit, real-image KBL regression runner.

The folder mode is deliberately non-recursive and never discovers directories on
its own. Source images are not copied into diagnostics JSON.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from Kitao_Body_Collage_Lab.nodes.body_split_nodes import KBLBodySplitter, STANDARD_PARTS
from Kitao_Body_Collage_Lab.nodes.detection_nodes import KBLElementDetector
from Kitao_Body_Collage_Lab.nodes.export_nodes import KBLCutoutExporter
from Kitao_Body_Collage_Lab.nodes.input_nodes import KBLLoadImage
from Kitao_Body_Collage_Lab.nodes.pose_nodes import KBLPoseEstimator
from Kitao_Body_Collage_Lab.nodes.refine_nodes import KBLMaskRefiner
from Kitao_Body_Collage_Lab.nodes.utils.manifest_utils import validate_manifest
from Kitao_Body_Collage_Lab.version import KBL_VERSION

try:
    from .model_integrity import require_models
except ImportError:  # Direct execution: python scripts/regression_test.py
    from model_integrity import require_models


MODEL_ROOT = Path("N:/Comfy-Desktop/ComfyUI-Shared/models/Kitao_Body_Collage_Lab")
DEFAULT_OUTPUT_ROOT = Path("N:/ComfyUI/output/Kitao_Body_Collage_Lab/validation_stage_e")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _round_times(values):
    return {key: round(float(value), 6) for key, value in values.items()}


def _project_name(prefix, image_path):
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", image_path.stem).strip("_") or "IMAGE"
    return f"{prefix}_{stem}"[:80]


def explicit_folder_images(folder):
    """Return supported immediate children of one explicitly supplied folder."""
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"KBL regression folder 不存在：{root}")
    images = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"显式目录没有支持的图片：{root}")
    return images


def _person_mask(elements):
    people = [item for item in elements if item.get("label") == "person"]
    return max(people, key=lambda item: int(item.get("area", 0)), default=None)


def _body_validation(body_parts, person):
    if person is None:
        return False, 0
    person_mask = np.asarray(person["mask"], dtype=bool)
    subset = all(not np.any(np.asarray(item["mask"], dtype=bool) & ~person_mask) for item in body_parts)
    overlap = int((np.stack([np.asarray(item["mask"], dtype=bool) for item in body_parts]).sum(axis=0) > 1).sum()) if body_parts else 0
    return subset, overlap


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_image(image_path, output_root, prefix="KBL_STAGE_E", person_only=False, require_standard_parts=False):
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise FileNotFoundError(f"KBL regression image 不存在或格式不支持：{image_path}")
    require_models(MODEL_ROOT)
    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    timings = {"grounding_dino": 0.0, "sam2": 0.0, "dwpose": 0.0, "body_split": 0.0, "refine": 0.0, "export": 0.0, "total": 0.0}

    image, image_meta, _, width, height, filename = KBLLoadImage().load(str(image_path))
    detector = KBLElementDetector()
    elements, *_ = detector.detect(image, "guided", "custom", "person.", 0.25, 10, 0.85, 256, 16)
    timings.update(detector.last_timings)
    person = _person_mask(elements)
    if person is None:
        raise RuntimeError(f"没有检测到 person：{filename}")
    if np.asarray(person["mask"]).shape != (height, width):
        raise RuntimeError(f"person mask 未保持原图尺寸：{np.asarray(person['mask']).shape} != {(height, width)}")

    if person_only:
        timings["total"] = time.perf_counter() - total_started
        run_root = output_root / _project_name(prefix, image_path)
        run_root.mkdir(parents=True, exist_ok=True)
        diagnostics = {
            "pipeline_version": KBL_VERSION,
            "source": filename,
            "resolution": [width, height],
            "timings": _round_times(timings),
            "counts": {"elements": len(elements), "body_parts": 0, "missing_parts": 0, "uncertain_parts": 0},
            "validation": {"body_subset_person": True, "overlap_after": 0, "manifest_valid": None, "person_mask_original_size": True},
        }
        _write_json(run_root / "pipeline_diagnostics.json", diagnostics)
        return {
            "status": "PASS", "filename": filename, "resolution": [width, height],
            "person_count": sum(item.get("label") == "person" for item in elements),
            "element_count": len(elements), "body_part_count": 0,
            "missing_parts": [], "uncertain_parts": [],
            "pipeline_time": diagnostics["timings"]["total"],
            "manifest_validator": {"status": "NOT_RUN_PERSON_ONLY", "errors": [], "checked_assets": 0},
            "project": str(run_root), "pipeline_diagnostics": str(run_root / "pipeline_diagnostics.json"),
            "validation": diagnostics["validation"], "mode": "high_resolution_person_only",
        }

    started = time.perf_counter()
    pose_data, _, _ = KBLPoseEstimator().estimate(image, elements, "largest", 0, 0.30)
    timings["dwpose"] = time.perf_counter() - started
    started = time.perf_counter()
    body_parts, _, body_diagnostics_json, _, _, _ = KBLBodySplitter().split(
        image, elements, pose_data, "standard", 0.30, True, True, True, True, True, False, False
    )
    timings["body_split"] = time.perf_counter() - started
    body_diagnostics = json.loads(body_diagnostics_json)
    started = time.perf_counter()
    refined, _, _, refine_diagnostics_json = KBLMaskRefiner().refine(image, elements, body_parts, "safe", 1, 0, 0.75, 16, True, True)
    timings["refine"] = time.perf_counter() - started
    started = time.perf_counter()
    export_result = KBLCutoutExporter().export(
        image, image_meta, elements, body_parts, refined, _project_name(prefix, image_path), str(output_root),
        "all", False, True, True, True, True, True, "cropped", 24, "version", 64,
    )
    timings["export"] = time.perf_counter() - started
    timings["total"] = time.perf_counter() - total_started
    project = Path(export_result[0])
    manifest_validator = validate_manifest(export_result[1])
    subset, overlap_after = _body_validation(body_parts, person)
    missing = list(body_diagnostics.get("missing_parts", []))
    uncertain = list(body_diagnostics.get("uncertain_parts", []))
    present = {item["label"] for item in body_parts}
    standard_parts_present = set(STANDARD_PARTS).issubset(present)
    passed = subset and overlap_after == 0 and manifest_validator["status"] == "PASS"
    if require_standard_parts:
        passed = passed and standard_parts_present
    diagnostics = {
        "pipeline_version": KBL_VERSION,
        "source": filename,
        "resolution": [width, height],
        "timings": _round_times(timings),
        "counts": {"elements": len(elements), "body_parts": len(body_parts), "missing_parts": len(missing), "uncertain_parts": len(uncertain)},
        "validation": {"body_subset_person": subset, "overlap_after": overlap_after, "manifest_valid": manifest_validator["status"] == "PASS", "required_body_parts_present": standard_parts_present},
        "refine": json.loads(refine_diagnostics_json),
    }
    _write_json(project / "pipeline_diagnostics.json", diagnostics)
    return {
        "status": "PASS" if passed else "FAILED", "filename": filename, "resolution": [width, height],
        "person_count": sum(item.get("label") == "person" for item in elements),
        "element_count": len(elements), "body_part_count": len(body_parts),
        "missing_parts": missing, "uncertain_parts": uncertain,
        "pipeline_time": diagnostics["timings"]["total"], "manifest_validator": manifest_validator,
        "project": str(project), "pipeline_diagnostics": str(project / "pipeline_diagnostics.json"),
        "exploded_preview": str(project / "preview" / "exploded_view.png"),
        "validation": diagnostics["validation"], "mode": "full",
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Run KBL regression on only explicitly supplied images.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Explicit image path")
    source.add_argument("--folder", help="Explicit folder; immediate image files only, never recursive")
    source.add_argument("--cross-leg-image", help="Explicit TEST C3 image path; writes under validation_stage_e/c3")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--project-prefix", default="KBL_STAGE_E")
    parser.add_argument("--person-only", action="store_true", help="Memory-bounded guided person + SAM2 high-resolution regression")
    parser.add_argument("--require-standard-parts", action="store_true", help="Fail unless all 14 standard body parts are present")
    return parser.parse_args()


def main():
    args = _parse_args()
    output_root = Path(args.output_root)
    prefix = args.project_prefix
    require_standard = args.require_standard_parts
    if args.cross_leg_image:
        images = [Path(args.cross_leg_image)]
        output_root = output_root / "c3"
        prefix = "KBL_STAGE_E_C3"
        require_standard = True
    elif args.image:
        images = [Path(args.image)]
    else:
        images = explicit_folder_images(args.folder)
    results = []
    for image_path in images:
        try:
            result = run_image(image_path, output_root, prefix, args.person_only, require_standard)
        except Exception as exc:
            result = {
                "status": "FAILED", "filename": image_path.name, "resolution": None,
                "person_count": 0, "element_count": 0, "body_part_count": 0,
                "missing_parts": [], "uncertain_parts": [], "pipeline_time": 0.0,
                "manifest_validator": {"status": "NOT_RUN", "errors": [str(exc)], "checked_assets": 0},
                "error": str(exc),
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    summary = {"pipeline_version": KBL_VERSION, "count": len(results), "results": results}
    _write_json(output_root / "regression_summary.json", summary)
    return 0 if all(item["status"] == "PASS" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
