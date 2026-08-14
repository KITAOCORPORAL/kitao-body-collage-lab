"""GroundingDINO + SAM2 high-level element detector."""

import json
import time
from pathlib import Path

from .model_backends import GroundingDinoBackend, Sam2Backend
from .utils.image_io import comfy_image_to_pil
from .utils.mask_utils import assign_element_ids, deduplicate_elements, masks_to_batch, validate_elements
from .utils.path_config import get_path_config
from .utils.preview_utils import render_element_preview, render_mask_contact_sheet

CATEGORY_PRESETS = {
    "portrait_basic": ["person", "hair", "face", "hand", "foot", "clothing", "shoe"],
    "body_parts_focus": ["person", "head", "face", "hair", "arm", "hand", "leg", "foot"],
    "props_focus": ["chair", "table", "flower", "glass", "mirror", "rope", "fabric", "cloth", "jewelry", "bag", "shoe"],
    "mixed_scene": ["person", "chair", "table", "flower", "glass", "mirror", "fabric", "cloth", "plant", "shoe", "jewelry"],
}


def resolve_prompt(category_preset, text_prompt_override):
    if category_preset == "custom":
        prompt = text_prompt_override.strip()
        if not prompt:
            raise ValueError("KBL category_preset=custom 时 text_prompt_override 不能为空")
        return prompt
    return ". ".join(CATEGORY_PRESETS[category_preset]) + "."


class KBLElementDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_mode": (["guided", "auto", "hybrid"], {"default": "guided"}),
                "category_preset": (["portrait_basic", "body_parts_focus", "props_focus", "mixed_scene", "custom"], {"default": "mixed_scene"}),
                "text_prompt_override": ("STRING", {"default": "person. chair. flower.", "multiline": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.95, "step": 0.01}),
                "max_detections": ("INT", {"default": 32, "min": 1, "max": 128}),
                "mask_iou_threshold": ("FLOAT", {"default": 0.85, "min": 0.10, "max": 0.99, "step": 0.01}),
                "min_mask_area": ("INT", {"default": 256, "min": 1, "max": 10000000}),
                "max_candidates": ("INT", {"default": 64, "min": 1, "max": 256}),
            }
        }

    RETURN_TYPES = ("KBL_ELEMENTS", "MASK", "STRING", "STRING", "STRING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("elements", "candidate_masks", "boxes", "labels", "scores", "element_preview", "mask_preview")
    FUNCTION = "detect"
    CATEGORY = "Kitao Body Collage/分割"
    DESCRIPTION = "GroundingDINO 文本框选 + SAM2 guided/auto/hybrid 独立元素分割。"

    def __init__(self):
        self.dino_backend_class = GroundingDinoBackend
        self.sam_backend_class = Sam2Backend
        self.last_timings = {"grounding_dino": 0.0, "sam2": 0.0}

    def detect(
        self,
        image,
        detection_mode,
        category_preset,
        text_prompt_override,
        confidence_threshold,
        max_detections,
        mask_iou_threshold,
        min_mask_area,
        max_candidates,
    ):
        pil_image = comfy_image_to_pil(image)
        height, width = pil_image.height, pil_image.width
        safe_candidate_limit = max(1, 384_000_000 // max(1, width * height))
        effective_max_candidates = min(int(max_candidates), safe_candidate_limit)
        if effective_max_candidates < int(max_candidates):
            print(
                f"[KBL] high-resolution safety cap: max_candidates {max_candidates} -> "
                f"{effective_max_candidates} for {width}x{height}"
            )
        paths = get_path_config()
        model_root = Path(paths["model_root"]) / "Kitao_Body_Collage_Lab"

        detections = []
        if detection_mode in {"guided", "hybrid"}:
            prompt = resolve_prompt(category_preset, text_prompt_override)
            dino = self.dino_backend_class(model_root / "grounding_dino", confidence_threshold)
            started = time.perf_counter()
            detections = dino.detect(pil_image, prompt, max_detections)
            self.last_timings["grounding_dino"] = time.perf_counter() - started
        else:
            self.last_timings["grounding_dino"] = 0.0

        if detection_mode == "guided" and not detections:
            guided, automatic = [], []
            self.last_timings["sam2"] = 0.0
        else:
            sam = self.sam_backend_class(model_root / "sam2")
            started = time.perf_counter()
            guided, automatic = sam.segment(
                pil_image,
                detections if detection_mode in {"guided", "hybrid"} else [],
                detection_mode in {"auto", "hybrid"},
                effective_max_candidates,
                min_mask_area,
            )
            self.last_timings["sam2"] = time.perf_counter() - started
        candidates = guided + automatic
        elements = deduplicate_elements(
            candidates,
            mask_iou_threshold,
            min_mask_area,
            effective_max_candidates,
        )
        assign_element_ids(elements)
        validate_elements(elements, height, width)
        print(f"[KBL] detected={len(detections)} masks={len(candidates)} kept={len(elements)} mode={detection_mode}")

        boxes = json.dumps([item["bbox"] for item in elements], ensure_ascii=False)
        labels = json.dumps([item["label"] for item in elements], ensure_ascii=False)
        scores = json.dumps([round(float(item["confidence"]), 6) for item in elements])
        return (
            elements,
            masks_to_batch(elements, height, width),
            boxes,
            labels,
            scores,
            render_element_preview(pil_image, elements),
            render_mask_contact_sheet(elements),
        )

NODE_CLASS_MAPPINGS = {"KBL_Element_Detector": KBLElementDetector}
NODE_DISPLAY_NAME_MAPPINGS = {"KBL_Element_Detector": "KBL 元素检测与分割"}
