"""Thin orchestration node for the public KBL pipeline nodes."""

import json
import time
from pathlib import Path

from .backends.florence2_backend import Florence2Backend
from .body_split_nodes import KBLBodySplitter
from .export_nodes import KBLCutoutExporter
from .pose_nodes import KBLPoseEstimator
from .refine_nodes import KBLMaskRefiner
from .scene_nodes import KBLSceneInventory, KBLUniversalElementDetector, _preview
from .utils.image_io import comfy_image_to_pil
from .utils.path_config import get_path_config
from .utils.scene_utils import merge_scene_candidates


class KBLOneClickDecomposeExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "project_name": ("STRING", {"default": "KBL_PROJECT"}),
            "quality": (["fast", "balanced", "complete"], {"default": "complete"}),
            "include_body_parts": ("BOOLEAN", {"default": True}),
            "keep_unlabeled_objects": ("BOOLEAN", {"default": True}),
            "padding": ("INT", {"default": 24, "min": 0, "max": 512}),
            "export_root": ("STRING", {"default": "N:\\ComfyUI\\output\\Kitao_Body_Collage_Lab"}),
        }, "optional": {"image_meta": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("project_directory", "manifest_path", "element_count", "body_part_count", "total_png_count", "contact_sheet", "exploded_view", "diagnostics")
    FUNCTION = "run"; CATEGORY = "Kitao Body Collage/全元素"; OUTPUT_NODE = True

    def run(self, image, project_name, quality, include_body_parts, keep_unlabeled_objects, padding, export_root, image_meta=""):
        started = time.perf_counter(); pil = comfy_image_to_pil(image)
        if quality == "fast":
            root = Path(get_path_config()["model_root"]) / "Kitao_Body_Collage_Lab"
            backend = Florence2Backend(root / "florence2")
            candidates = merge_scene_candidates(backend.inventory(pil, ["<OD>"], 64), 64)
            inventory_diag = {"scene_candidates": len(candidates), "semantic_candidates": len(candidates), "region_candidates": 0, "metrics": backend.last_metrics}
            inventory_preview = _preview(pil, candidates)
        else:
            mode = "complete" if quality == "complete" else "semantic"
            candidates, inventory_json, inventory_preview = KBLSceneInventory().inventory(image, mode, 64, quality == "complete")
            inventory_diag = json.loads(inventory_json)
        elements, _, universal_json, _ = KBLUniversalElementDetector().segment(
            image, candidates, "", 64, 256, 0.90, 0.85, keep_unlabeled_objects
        )
        universal_diag = json.loads(universal_json); people = [item for item in elements if item.get("label") == "person"]
        body_parts = []
        if people and include_body_parts and quality != "fast":
            pose, _, _ = KBLPoseEstimator().estimate(image, people, "largest", 0, 0.30)
            body_parts, _, _, _, _, _ = KBLBodySplitter().split(
                image, elements, pose, "standard", 0.30, True, True, True, True, True, False, False
            )
        refined, _, _, _ = KBLMaskRefiner().refine(image, elements, body_parts, "soft", 1, 0, 0.75, 16, True, True)
        exported = KBLCutoutExporter().export(
            image, image_meta, elements, body_parts, refined, project_name, export_root, "all",
            True, True, True, True, True, True, "cropped", padding, "version", 64,
        )
        project, manifest, total, body_count, element_count, contact, _, exploded = exported
        comfy_image_to_pil(inventory_preview).save(Path(project, "preview", "scene_inventory_preview.png"))
        diagnostics = {**inventory_diag, **universal_diag, "people_detected": len(people),
                       "body_split_person": max(people, key=lambda x: x.get("area", 0))["id"] if body_parts else None,
                       "body_parts": body_count, "total_png": total,
                       "elapsed_seconds": round(time.perf_counter()-started, 4), "status": "PASS"}
        diagnostics_path = Path(project, "pipeline_diagnostics.json")
        existing_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        existing_diagnostics["stage_f"] = diagnostics
        diagnostics_path.write_text(json.dumps(existing_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[KBL ONE CLICK]\n\n" + "\n\n".join([
            f"Scene candidates:\n{len(candidates)}", f"Semantic elements:\n{universal_diag['semantic_elements']}",
            f"Unlabeled elements:\n{universal_diag['unlabeled_elements']}", f"Rejected backgrounds:\n{universal_diag['background_rejected']}",
            f"Body parts:\n{body_count}", f"PNG exported:\n{total}", f"Project:\n{project}", f"Manifest:\n{manifest}", "Status:\nPASS"
        ]))
        return project, manifest, element_count, body_count, total, contact, exploded, json.dumps(diagnostics, ensure_ascii=False)

NODE_CLASS_MAPPINGS = {"KBL_OneClick_Decompose_Export": KBLOneClickDecomposeExport}
NODE_DISPLAY_NAME_MAPPINGS = {"KBL_OneClick_Decompose_Export": "KBL 一键全元素拆解"}
