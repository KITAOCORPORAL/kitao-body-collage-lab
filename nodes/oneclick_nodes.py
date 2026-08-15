"""Thin orchestration node for the public KBL pipeline nodes."""

import json
import time
from pathlib import Path

from .body_split_nodes import KBLBodySplitter
from .export_nodes import KBLCutoutExporter
from .pose_nodes import KBLPoseEstimator
from .refine_nodes import KBLMaskRefiner
from .scene_nodes import KBLSceneInventory, KBLUniversalElementDetector
from .utils.image_io import comfy_image_to_pil


OUTPUT_POLICIES = {
    "object_clean": {
        "quality": "balanced",
        "inventory_strategy": "semantic_only",
        "include_region_proposals": False,
        "include_body_parts": False,
        "keep_unlabeled_objects": False,
        "allow_person_subelements": False,
        "prefer_whole_objects": True,
    },
    "collage_parts": {
        "quality": "complete",
        "inventory_strategy": "semantic_plus_regions",
        "include_region_proposals": True,
        "include_body_parts": True,
        "keep_unlabeled_objects": True,
        "allow_person_subelements": True,
        "prefer_whole_objects": False,
    },
}


def resolve_output_policy(output_mode):
    if output_mode not in OUTPUT_POLICIES:
        raise ValueError(f"未知 output_mode: {output_mode}")
    return dict(OUTPUT_POLICIES[output_mode])


class KBLOneClickDecomposeExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "project_name": ("STRING", {"default": "KBL_PROJECT"}),
            "output_mode": (["object_clean", "collage_parts"], {"default": "object_clean"}),
            "padding": ("INT", {"default": 24, "min": 0, "max": 512}),
            "export_root": ("STRING", {"default": "N:\\ComfyUI\\output\\Kitao_Body_Collage_Lab"}),
        }, "optional": {"image_meta": ("STRING", {"default": ""})}}
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("project_directory", "manifest_path", "element_count", "body_part_count", "total_png_count", "contact_sheet", "exploded_view", "diagnostics")
    FUNCTION = "run"; CATEGORY = "Kitao Body Collage/全元素"; OUTPUT_NODE = True

    def run(self, image, project_name, output_mode="object_clean", padding=24,
            export_root="N:\\ComfyUI\\output\\Kitao_Body_Collage_Lab", image_meta=""):
        started = time.perf_counter()
        policy = resolve_output_policy(output_mode)
        candidates, inventory_json, inventory_preview = KBLSceneInventory().inventory(
            image, policy["inventory_strategy"], 64, policy["include_region_proposals"]
        )
        inventory_diag = json.loads(inventory_json)
        elements, _, universal_json, _ = KBLUniversalElementDetector().segment(
            image, candidates, "", 64, 256, 0.90, 0.85,
            policy["keep_unlabeled_objects"], output_mode,
            policy["prefer_whole_objects"], policy["allow_person_subelements"],
        )
        universal_diag = json.loads(universal_json); people = [item for item in elements if item.get("label") == "person"]
        body_parts = []
        if people and policy["include_body_parts"]:
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
        diagnostics = {**inventory_diag, **universal_diag, "output_mode": output_mode,
                       "effective_policy": policy, "people_detected": len(people),
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
            f"Clean background rejects:\n{universal_diag.get('background_like_rejected_count', 0)}",
            f"Suppressed children:\n{universal_diag.get('parent_child_suppressed_count', 0)}",
            f"Body parts:\n{body_count}", f"PNG exported:\n{total}", f"Output mode:\n{output_mode}",
            f"Project:\n{project}", f"Manifest:\n{manifest}", "Status:\nPASS"
        ]))
        return project, manifest, element_count, body_count, total, contact, exploded, json.dumps(diagnostics, ensure_ascii=False)


class KBLOneClickDecomposeExportV011(KBLOneClickDecomposeExport):
    """Compatibility adapter for the bundled v0.1.1 workflow widget layout."""

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

    def run(self, image, project_name, quality="complete", include_body_parts=True,
            keep_unlabeled_objects=True, padding=24,
            export_root="N:\\ComfyUI\\output\\Kitao_Body_Collage_Lab", image_meta=""):
        output_mode = "collage_parts" if include_body_parts or keep_unlabeled_objects else "object_clean"
        return super().run(image, project_name, output_mode, padding, export_root, image_meta)


NODE_CLASS_MAPPINGS = {
    "KBL_OneClick_Decompose_Export": KBLOneClickDecomposeExport,
    "KBL_OneClick_Decompose_Export_v011": KBLOneClickDecomposeExportV011,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KBL_OneClick_Decompose_Export": "KBL 一键全元素拆解",
    "KBL_OneClick_Decompose_Export_v011": "KBL 一键全元素拆解（v0.1.1 兼容）",
}
