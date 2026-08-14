"""Real photo: GroundingDINO -> SAM2 person -> DWPose -> body parts."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from Kitao_Body_Collage_Lab.nodes.body_split_nodes import KBLBodySplitter
from Kitao_Body_Collage_Lab.nodes.model_backends import GroundingDinoBackend, Sam2Backend
from Kitao_Body_Collage_Lab.nodes.pose_nodes import KBLPoseEstimator
from Kitao_Body_Collage_Lab.nodes.utils.image_io import comfy_image_to_pil, load_image_file, pil_to_comfy_image

from model_integrity import require_models

MODEL_ROOT = Path("N:/Comfy-Desktop/ComfyUI-Shared/models/Kitao_Body_Collage_Lab")
OUTPUT_ROOT = Path("N:/ComfyUI/output/Kitao_Body_Collage_Lab/validation_stage_c")


def serializable_part(part):
    return {key: value for key, value in part.items() if key != "mask"}


def save_preview(tensor, name):
    path = OUTPUT_ROOT / name
    comfy_image_to_pil(tensor).save(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="真实全身人物 JPG/PNG/WEBP 绝对路径")
    parser.add_argument("--joint-threshold", type=float, default=0.30)
    parser.add_argument("--no-sam-refine", action="store_true")
    args = parser.parse_args()

    require_models(MODEL_ROOT)
    image, meta = load_image_file(args.image)
    image_tensor = pil_to_comfy_image(image)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    dino = GroundingDinoBackend(MODEL_ROOT / "grounding_dino", 0.25)
    detections = dino.detect(image, "person.", 10)
    persons = [item for item in detections if "person" in item["label"]]
    if not persons:
        raise RuntimeError("FAILED: GroundingDINO 未检测到 person")
    detection = max(persons, key=lambda item: item["confidence"])

    sam = Sam2Backend(MODEL_ROOT / "sam2")
    guided, _ = sam.segment(image, [detection], False, 1, 1)
    if len(guided) != 1:
        raise RuntimeError(f"FAILED: SAM2 person mask 数量异常：{len(guided)}")
    person = guided[0]
    person.update({"id": "person_01", "label": "person", "category": "person"})
    person["mask"] = np.asarray(person["mask"], dtype=bool)
    if person["mask"].shape != (image.height, image.width):
        raise RuntimeError("FAILED: person mask 不等于原图尺寸")

    pose_node = KBLPoseEstimator()
    pose_data, pose_preview, _selected = pose_node.estimate(image_tensor, [person], "largest", 0, args.joint_threshold)
    if not pose_data.get("selected_pose"):
        raise RuntimeError("FAILED: DWPose 未识别到与 person 匹配的人体")

    body_node = KBLBodySplitter()
    body_started = time.perf_counter()
    parts, _masks, diagnostics_json, body_preview, mask_sheet, exploded = body_node.split(
        image_tensor, [person], pose_data, "standard", args.joint_threshold,
        not args.no_sam_refine, True, True, True, True, False, False,
    )
    body_seconds = time.perf_counter() - body_started
    diagnostics = json.loads(diagnostics_json)
    if not parts:
        raise RuntimeError("FAILED: Body Splitter 没有输出有效部位")
    for part in parts:
        mask = np.asarray(part["mask"], dtype=bool)
        if mask.shape != person["mask"].shape or np.any(mask & ~person["mask"]):
            raise RuntimeError(f"FAILED: {part['label']} mask 尺寸错误或超出 person mask")

    outputs = {
        "pose_preview": save_preview(pose_preview, "pose_preview.png"),
        "body_parts_preview": save_preview(body_preview, "body_parts_preview.png"),
        "body_mask_sheet": save_preview(mask_sheet, "body_mask_sheet.png"),
        "body_exploded_preview": save_preview(exploded, "body_exploded_preview.png"),
    }
    result = {
        "status": "PASS",
        "image": {"filename": meta["filename"], "width": meta["width"], "height": meta["height"], "source_path": meta["source_path"]},
        "models": {"dwpose": str(MODEL_ROOT / "dwpose"), "provider": pose_data["metrics"].get("provider")},
        "timings_seconds": {
            "grounding_dino": dino.last_metrics.get("elapsed_seconds"),
            "sam2_person": sam.last_metrics.get("elapsed_seconds"),
            "dwpose": pose_data["metrics"].get("elapsed_seconds"),
            "body_split": round(body_seconds, 4),
            "total": round(time.perf_counter() - started, 4),
        },
        "dwpose_people": len(pose_data.get("people", [])),
        "diagnostics": diagnostics,
        "parts": [serializable_part(part) for part in parts],
        "outputs": outputs,
        "mock_or_placeholder": False,
    }
    result_path = OUTPUT_ROOT / "body_split_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
