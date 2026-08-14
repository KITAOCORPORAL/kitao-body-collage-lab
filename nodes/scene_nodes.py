"""Florence scene inventory and universal SAM2 element segmentation nodes."""

import json
from pathlib import Path
from PIL import ImageDraw, ImageFont

from .backends.florence2_backend import Florence2Backend
from .model_backends import GroundingDinoBackend, Sam2Backend
from .utils.image_io import comfy_image_to_pil, pil_to_comfy_image
from .utils.mask_utils import assign_element_ids, deduplicate_elements, masks_to_batch, validate_elements
from .utils.path_config import get_path_config
from .utils.preview_utils import render_element_preview
from .utils.scene_utils import background_candidate_filter, merge_scene_candidates, normalize_scene_label


def _preview(image, candidates):
    out = image.copy().convert("RGB"); draw = ImageDraw.Draw(out); font = ImageFont.load_default()
    for index, item in enumerate(candidates, 1):
        box = tuple(int(round(x)) for x in item["bbox"]); draw.rectangle(box, outline="#00e0b8", width=max(2, image.width // 700))
        draw.text((box[0] + 3, max(0, box[1] - 13)), f"{index:02d} {item['label']}", fill="#00e0b8", font=font)
    return pil_to_comfy_image(out)


class KBLSceneInventory:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "inventory_mode": (["semantic", "complete"], {"default": "complete"}),
                "max_regions": ("INT", {"default": 64, "min": 1, "max": 256}),
                "include_region_proposals": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("KBL_SCENE_CANDIDATES", "STRING", "IMAGE")
    RETURN_NAMES = ("scene_candidates", "diagnostics", "scene_inventory_preview")
    FUNCTION = "inventory"; CATEGORY = "Kitao Body Collage/全元素"
    def __init__(self): self.backend_class = Florence2Backend
    def inventory(self, image, inventory_mode, max_regions, include_region_proposals):
        pil = comfy_image_to_pil(image); root = Path(get_path_config()["model_root"]) / "Kitao_Body_Collage_Lab"
        tasks = ["<OD>"]
        if inventory_mode in {"semantic", "complete"}: tasks.append("<DENSE_REGION_CAPTION>")
        if inventory_mode == "complete" and include_region_proposals: tasks.append("<REGION_PROPOSAL>")
        backend = self.backend_class(root / "florence2")
        candidates = merge_scene_candidates(backend.inventory(pil, tasks, max_regions), max_regions)
        diagnostics = {"scene_candidates": len(candidates), "semantic_candidates": sum(bool(x["semantic"]) for x in candidates),
                       "region_candidates": sum(not x["semantic"] for x in candidates), "tasks": tasks, "metrics": backend.last_metrics}
        return candidates, json.dumps(diagnostics, ensure_ascii=False), _preview(pil, candidates)


class KBLUniversalElementDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "scene_candidates": ("KBL_SCENE_CANDIDATES",),
                "extra_prompt": ("STRING", {"default": "", "multiline": True}),
                "max_elements": ("INT", {"default": 64, "min": 1, "max": 128}),
                "min_area": ("INT", {"default": 256, "min": 1, "max": 10000000}),
                "max_area_ratio": ("FLOAT", {"default": 0.90, "min": 0.1, "max": 1.0, "step": 0.01}),
                "dedup_iou": ("FLOAT", {"default": 0.85, "min": 0.1, "max": 0.99, "step": 0.01}),
                "keep_unlabeled_regions": ("BOOLEAN", {"default": True})}}
    RETURN_TYPES = ("KBL_ELEMENTS", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("elements", "masks", "diagnostics", "element_preview")
    FUNCTION = "segment"; CATEGORY = "Kitao Body Collage/全元素"
    def __init__(self): self.dino_backend_class = GroundingDinoBackend; self.sam_backend_class = Sam2Backend
    def segment(self, image, scene_candidates, extra_prompt, max_elements, min_area, max_area_ratio, dedup_iou, keep_unlabeled_regions):
        pil = comfy_image_to_pil(image); root = Path(get_path_config()["model_root"]) / "Kitao_Body_Collage_Lab"
        detections = []
        for item in scene_candidates:
            if item.get("semantic") or keep_unlabeled_regions:
                detections.append({**item, "discovery_source": item.get("source", "florence"), "confidence": item.get("confidence") or 0.5})
        if extra_prompt.strip():
            dino = self.dino_backend_class(root / "grounding_dino", 0.25)
            for item in dino.detect(pil, extra_prompt, max_elements):
                item.update({"raw_label": item["label"], "canonical_label": normalize_scene_label(item["label"], item["label"]),
                             "semantic": True, "discovery_source": "grounding_dino_extra"})
                item["label"] = item["canonical_label"]; detections.append(item)
        sam = self.sam_backend_class(root / "sam2")
        guided, _ = sam.segment(
            pil, detections[: int(max_elements) * 2], False, max_elements, min_area
        )
        for item in guided:
            item["source"] = item.get("discovery_source", "guided")
        filtered, rejected = background_candidate_filter(guided, pil.size, max_area_ratio)
        elements = deduplicate_elements(filtered, dedup_iou, min_area, max_elements)
        assign_element_ids(elements); validate_elements(elements, pil.height, pil.width)
        diagnostics = {"sam_masks": len(guided), "background_rejected": len(rejected),
                       "duplicates_removed": len(filtered)-len(elements), "generic_elements_kept": len(elements),
                       "semantic_elements": sum(bool(x.get("semantic")) for x in elements),
                       "unlabeled_elements": sum(not x.get("semantic") for x in elements), "sam_metrics": sam.last_metrics}
        return elements, masks_to_batch(elements, pil.height, pil.width), json.dumps(diagnostics, ensure_ascii=False), render_element_preview(pil, elements)

NODE_CLASS_MAPPINGS = {"KBL_Scene_Inventory": KBLSceneInventory, "KBL_Universal_Element_Detector": KBLUniversalElementDetector}
NODE_DISPLAY_NAME_MAPPINGS = {"KBL_Scene_Inventory": "KBL 场景元素清点", "KBL_Universal_Element_Detector": "KBL 全元素分割"}
