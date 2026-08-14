"""Real GroundingDINO bbox -> SAM2 mask validation using one user-selected photo."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from Kitao_Body_Collage_Lab.nodes.model_backends import GroundingDinoBackend, Sam2Backend
from Kitao_Body_Collage_Lab.nodes.utils.image_io import load_image_file

from model_integrity import require_models

MODEL_ROOT = Path("N:/Comfy-Desktop/ComfyUI-Shared/models/Kitao_Body_Collage_Lab")
OUTPUT_ROOT = Path("N:/ComfyUI/output/Kitao_Body_Collage_Lab/validation")


def save_grounding_preview(image, detection, path):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    box = tuple(int(round(value)) for value in detection["bbox"])
    draw.rectangle(box, outline=(255, 64, 32), width=max(3, image.width // 500))
    draw.text((box[0] + 4, max(0, box[1] - 16)), f'{detection["label"]} {detection["confidence"]:.3f}', fill=(255, 64, 32), font=font)
    preview.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="真实人物 JPG/PNG/WEBP 的绝对路径")
    args = parser.parse_args()
    require_models(MODEL_ROOT)
    image, meta = load_image_file(args.image)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    dino = GroundingDinoBackend(MODEL_ROOT / "grounding_dino", 0.25)
    detections = dino.detect(image, "person.", 10)
    if not detections:
        raise RuntimeError("FAILED: GroundingDINO 对 person 没有返回 bbox")
    person = max(
        (item for item in detections if "person" in item["label"]),
        key=lambda item: item["confidence"],
        default=detections[0],
    )
    print("GroundingDINO loaded")
    print(json.dumps({"image": meta, "metrics": dino.last_metrics, "detections": detections}, ensure_ascii=False, indent=2))
    save_grounding_preview(image, person, OUTPUT_ROOT / "grounding_preview.png")

    sam = Sam2Backend(MODEL_ROOT / "sam2")
    guided, _ = sam.segment(image, [person], False, 1, 1)
    if len(guided) != 1:
        raise RuntimeError(f"FAILED: SAM2 返回 mask 数量异常: {len(guided)}")
    element = guided[0]
    mask = np.asarray(element["mask"])
    if mask.shape != (image.height, image.width):
        raise RuntimeError(f"FAILED: mask 尺寸 {mask.shape} != 原图 {(image.height, image.width)}")
    if not np.isfinite(mask.astype(np.float32)).all() or int(mask.sum()) <= 0:
        raise RuntimeError("FAILED: SAM2 mask 为空或包含 NaN/Inf")
    bbox_area = max(1.0, (person["bbox"][2] - person["bbox"][0]) * (person["bbox"][3] - person["bbox"][1]))
    area_ratio = float(mask.sum() / bbox_area)
    if not 0.05 <= area_ratio <= 2.5:
        raise RuntimeError(f"FAILED: SAM2 mask 面积相对 bbox 不合理: {area_ratio:.4f}")

    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(OUTPUT_ROOT / "sam2_person_mask.png")
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255, mode="L"))
    rgba.save(OUTPUT_ROOT / "sam2_person_cutout.png")
    result = {
        "status": "PASS",
        "image": {"filename": meta["filename"], "width": meta["width"], "height": meta["height"]},
        "detection": person,
        "sam2": {
            "mask_width": image.width,
            "mask_height": image.height,
            "mask_area": int(mask.sum()),
            "bbox_area_ratio": round(area_ratio, 6),
            "sam_score": element["sam_score"],
            "metrics": sam.last_metrics,
        },
        "outputs": [str(OUTPUT_ROOT / name) for name in ("grounding_preview.png", "sam2_person_mask.png", "sam2_person_cutout.png")],
    }
    (OUTPUT_ROOT / "smoke_test_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAM2 loaded")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del sam, dino
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

