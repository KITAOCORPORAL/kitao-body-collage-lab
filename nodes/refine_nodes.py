"""Conservative mask refinement without modifying source RGB pixels."""

import json
from pathlib import Path

import numpy as np
import torch

from .utils.alpha_utils import mask_bbox, refine_binary_mask, soft_alpha_from_binary
from .utils.image_io import comfy_image_to_pil
from .utils.path_config import get_path_config
from .utils.preview_utils import render_mask_contact_sheet


def _person_masks(elements):
    return {
        item["id"]: np.asarray(item["mask"], dtype=bool)
        for item in elements
        if item.get("label") == "person" and item.get("id")
    }


def _refine_item(item, kind, mode, person_masks, settings):
    raw = np.asarray(item["mask"], dtype=bool)
    raw_copy = raw.copy()
    if mode == "none":
        binary = raw.copy()
    else:
        binary = refine_binary_mask(
            raw,
            expand_px=settings["expand_px"], erode_px=settings["erode_px"],
            min_component_area=settings["min_component_area"],
            fill_small_holes=settings["fill_small_holes"],
            remove_small_islands=settings["remove_small_islands"],
        )
        raw_area = int(raw.sum())
        if raw_area and abs(int(binary.sum()) - raw_area) / raw_area > 0.20:
            binary = refine_binary_mask(
                raw,
                expand_px=0, erode_px=settings["erode_px"],
                min_component_area=settings["min_component_area"],
                fill_small_holes=settings["fill_small_holes"],
                remove_small_islands=settings["remove_small_islands"],
            )
    if kind == "body":
        constraint = person_masks.get(item.get("source_person_id"))
        if constraint is None and person_masks:
            constraint = max(person_masks.values(), key=lambda value: int(value.sum()))
        if constraint is not None:
            binary &= constraint
    alpha = soft_alpha_from_binary(binary, settings["feather_px"]) if mode == "soft" else binary.astype(np.float32)
    if kind == "body":
        constraint = person_masks.get(item.get("source_person_id"))
        if constraint is None and person_masks:
            constraint = max(person_masks.values(), key=lambda value: int(value.sum()))
        if constraint is not None:
            alpha *= constraint.astype(np.float32)
    return {
        "id": item["id"],
        "label": item["label"],
        "kind": kind,
        "raw_mask": raw_copy,
        "binary_mask": binary,
        "alpha_mask": alpha.astype(np.float32),
        "raw_bbox": mask_bbox(raw),
        "refined_bbox": mask_bbox(alpha > 0),
        "raw_area": int(raw.sum()),
        "refined_area": int(binary.sum()),
        "area_change_ratio": float((binary.sum() - raw.sum()) / max(1, raw.sum())),
        "quality_flag": item.get("quality_flag", "ok"),
        "refine_mode": mode,
    }


class KBLMaskRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "elements": ("KBL_ELEMENTS",),
            "body_parts": ("KBL_BODY_PARTS",),
            "refine_mode": (["safe", "soft", "none", "birefnet"], {"default": "safe"}),
            "expand_px": ("INT", {"default": 1, "min": 0, "max": 16}),
            "erode_px": ("INT", {"default": 0, "min": 0, "max": 16}),
            "feather_px": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 32.0, "step": 0.25}),
            "min_component_area": ("INT", {"default": 16, "min": 0, "max": 100000}),
            "fill_small_holes": ("BOOLEAN", {"default": True}),
            "remove_small_islands": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("KBL_REFINED_MASKS", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("refined_masks", "alpha_masks", "mask_preview", "diagnostics")
    FUNCTION = "refine"
    CATEGORY = "Kitao Body Collage/导出"

    def refine(self, image, elements, body_parts, refine_mode, expand_px, erode_px, feather_px, min_component_area, fill_small_holes, remove_small_islands):
        pil = comfy_image_to_pil(image)
        if refine_mode == "birefnet":
            model_dir = Path(get_path_config()["model_root"]) / "Kitao_Body_Collage_Lab" / "birefnet"
            raise RuntimeError(f"KBL BiRefNet: EXPERIMENTAL / NOT INSTALLED。模型接口目录：{model_dir}。请改用 safe 或 soft。")
        settings = {
            "expand_px": int(expand_px), "erode_px": int(erode_px),
            "feather_px": float(feather_px), "min_component_area": int(min_component_area),
            "fill_small_holes": bool(fill_small_holes), "remove_small_islands": bool(remove_small_islands),
        }
        people = _person_masks(elements)
        records = []
        for item in elements:
            records.append(_refine_item(item, "element", refine_mode, people, settings))
        for item in body_parts:
            records.append(_refine_item(item, "body", refine_mode, people, settings))
        expected = {"head", "torso", "left_upper_arm", "left_forearm", "left_hand", "right_upper_arm", "right_forearm", "right_hand", "left_thigh", "left_calf", "left_foot", "right_thigh", "right_calf", "right_foot"}
        body_labels = {item.get("label") for item in body_parts}
        result = {
            "version": "0.1", "mode": refine_mode, "settings": settings,
            "records": records, "missing_body_parts": sorted(expected - body_labels),
            "image_size": [pil.width, pil.height], "birefnet_status": "NOT INSTALLED",
        }
        alpha_batch = torch.stack([torch.from_numpy(item["alpha_mask"]) for item in records]) if records else torch.zeros((1, pil.height, pil.width), dtype=torch.float32)
        preview_records = [{"id": item["id"], "mask": item["binary_mask"], "area": item["refined_area"]} for item in records]
        diagnostics = {
            "mode": refine_mode, "record_count": len(records),
            "raw_area_total": sum(item["raw_area"] for item in records),
            "refined_area_total": sum(item["refined_area"] for item in records),
            "max_abs_area_change_ratio": max((abs(item["area_change_ratio"]) for item in records), default=0.0),
            "missing_body_parts": result["missing_body_parts"], "raw_masks_preserved": True,
            "birefnet": "NOT INSTALLED",
        }
        return result, alpha_batch, render_mask_contact_sheet(preview_records), json.dumps(diagnostics, ensure_ascii=False)


NODE_CLASS_MAPPINGS = {"KBL_Mask_Refiner": KBLMaskRefiner}
NODE_DISPLAY_NAME_MAPPINGS = {"KBL_Mask_Refiner": "KBL Mask 精修"}
