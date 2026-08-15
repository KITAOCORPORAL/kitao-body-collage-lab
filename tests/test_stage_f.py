import json
import unittest
from pathlib import Path

import numpy as np

from Kitao_Body_Collage_Lab.nodes import NODE_CLASS_MAPPINGS
from Kitao_Body_Collage_Lab.nodes.backends.florence2_backend import parse_florence_result
from Kitao_Body_Collage_Lab.nodes.export_nodes import _element_filename, category_folder
from Kitao_Body_Collage_Lab.nodes.oneclick_nodes import KBLOneClickDecomposeExport, resolve_output_policy
from Kitao_Body_Collage_Lab.nodes.pose_nodes import KBLPoseEstimator
from Kitao_Body_Collage_Lab.nodes.utils.mask_utils import assign_element_ids, deduplicate_elements
from Kitao_Body_Collage_Lab.nodes.utils.pose_utils import select_person
from Kitao_Body_Collage_Lab.nodes.utils.scene_utils import (
    background_candidate_filter,
    filter_clean_objects,
    merge_scene_candidates,
    normalize_scene_label,
    safe_asset_name,
    scene_inventory_tasks,
)
from Kitao_Body_Collage_Lab.nodes.utils.manifest_utils import validate_manifest_schema


def _element(label, mask, semantic=True, source="florence_od", raw_label=None, confidence=.9):
    ys, xs = np.nonzero(mask)
    return {
        "id": "", "label": label, "raw_label": raw_label or label,
        "canonical_label": label, "mask": mask, "area": int(mask.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "semantic": semantic, "source": source, "discovery_source": source,
        "confidence": confidence, "sam_score": .9,
    }


class StageFTests(unittest.TestCase):
    def test_florence_candidate_parsing(self):
        result = {"<OD>": {"bboxes": [[1, 2, 30, 40]], "labels": ["a red chair"]}}
        item = parse_florence_result(result, "<OD>", 100, 100)[0]
        self.assertTrue(item["semantic"]); self.assertEqual(item["source"], "florence_od")

    def test_canonical_label(self):
        self.assertEqual(normalize_scene_label("woman wearing black dress"), "person")
        self.assertEqual(normalize_scene_label("human face"), "face")
        self.assertEqual(normalize_scene_label("a red chair"), "chair")

    def test_region_fallback_naming(self):
        result = {"<REGION_PROPOSAL>": {"bboxes": [[0, 0, 10, 10], [20, 20, 30, 30]]}}
        items = merge_scene_candidates(parse_florence_result(result, "<REGION_PROPOSAL>", 50, 50), 10)
        self.assertEqual([x["label"] for x in items], ["region_001", "region_002"])
        assign_element_ids(items)
        self.assertEqual([x["id"] for x in items], ["region_001", "region_002"])

    def test_background_rejection(self):
        giant = np.ones((100, 100), bool)
        kept, rejected = background_candidate_filter([{"mask": giant, "semantic": False}], (100, 100))
        self.assertFalse(kept); self.assertEqual(len(rejected), 1)

    def test_inventory_strategies(self):
        self.assertEqual(scene_inventory_tasks("semantic_only", False), ["<OD>"])
        self.assertEqual(scene_inventory_tasks("semantic_plus_dense", False), ["<OD>", "<DENSE_REGION_CAPTION>"])
        self.assertEqual(scene_inventory_tasks("semantic_plus_regions", True)[-1], "<REGION_PROPOSAL>")
        self.assertEqual(scene_inventory_tasks("complete", True)[-1], "<REGION_PROPOSAL>")

    def test_object_clean_suppresses_hair_and_background_blob(self):
        person = np.zeros((100, 100), bool); person[8:96, 25:76] = True
        hair = np.zeros((100, 100), bool); hair[10:34, 33:69] = True
        dirty = np.ones((100, 100), bool); dirty[20:80, 20:80] = False
        elements = [
            _element("person", person),
            _element("hair", hair, source="florence_dense_caption"),
            _element("object_002", dirty, semantic=True, source="florence_dense_caption", raw_label="scene"),
        ]
        kept, diagnostics = filter_clean_objects(elements, (100, 100), False, True)
        self.assertEqual([item["label"] for item in kept], ["person"])
        self.assertEqual(diagnostics["parent_child_suppressed_count"], 1)
        self.assertEqual(diagnostics["background_like_rejected_count"], 1)

    def test_clean_mode_keeps_unknown_od_object_without_person(self):
        excavator = np.zeros((100, 100), bool); excavator[20:80, 15:85] = True
        kept, diagnostics = filter_clean_objects([
            _element("object_001", excavator, source="florence_od", raw_label="excavator")
        ], (100, 100), False, True)
        self.assertEqual(len(kept), 1)
        self.assertEqual(diagnostics["background_like_rejected_count"], 0)

    def test_clean_mode_rejects_generic_full_width_structure(self):
        structure = np.zeros((100, 100), bool); structure[:78, :] = True
        kept, diagnostics = filter_clean_objects([
            _element("object_002", structure, source="florence_od", raw_label="land vehicle")
        ], (100, 100), False, True)
        self.assertEqual(kept, [])
        self.assertEqual(diagnostics["background_like_rejected_count"], 1)

    def test_clean_mode_keeps_multiple_non_overlapping_semantic_objects(self):
        chair_a = np.zeros((100, 100), bool); chair_a[10:40, 10:35] = True
        chair_b = np.zeros((100, 100), bool); chair_b[55:90, 65:92] = True
        kept, _ = filter_clean_objects([
            _element("chair", chair_a), _element("chair", chair_b)
        ], (100, 100), False, True)
        self.assertEqual(len(kept), 2)

    def test_semantic_beats_unlabeled_dedup(self):
        mask = np.ones((20, 20), bool)
        items = [{"label": "region_001", "mask": mask, "area": 400, "semantic": False, "confidence": 1.0},
                 {"label": "chair", "mask": mask, "area": 400, "semantic": True, "confidence": 0.1}]
        self.assertEqual(deduplicate_elements(items, .8, 1, 10)[0]["label"], "chair")

    def test_multiple_same_labels_numbering(self):
        items = [{"label": "chair"}, {"label": "chair"}, {"label": "bag"}]
        self.assertEqual([x["id"] for x in assign_element_ids(items)], ["chair_01", "chair_02", "bag_01"])

    def test_windows_safe_filename(self):
        self.assertEqual(safe_asset_name('red:/glass*?"<>| 01'), "red_glass_01")
        self.assertEqual(_element_filename({"id": "chair_01"}, "furniture"), "chair_01.png")

    def test_no_person_route(self):
        import torch
        image = torch.zeros((1, 16, 16, 3))
        data, _, selected = KBLPoseEstimator().estimate(image, [], "largest", 0, .3)
        self.assertEqual(data["people"], []); self.assertEqual(selected, [])

    def test_multi_person_generic_ids(self):
        items = [{"label": "person"}, {"label": "person"}, {"label": "person"}]
        assign_element_ids(items)
        self.assertEqual(items[-1]["id"], "person_03")

    def test_body_split_selects_largest_only(self):
        people = [{"id": "person_01", "label": "person", "area": 10, "bbox": [0,0,2,5]}, {"id": "person_02", "label": "person", "area": 30, "bbox": [0,0,5,6]}]
        self.assertEqual(select_person(people, "largest", (10, 10), 0)["id"], "person_02")

    def test_one_click_registration(self):
        self.assertIn("KBL_Scene_Inventory", NODE_CLASS_MAPPINGS)
        self.assertIn("KBL_Universal_Element_Detector", NODE_CLASS_MAPPINGS)
        self.assertIn("KBL_OneClick_Decompose_Export", NODE_CLASS_MAPPINGS)
        self.assertIn("KBL_OneClick_Decompose_Export_v011", NODE_CLASS_MAPPINGS)
        self.assertIn("KBL_Load_Image_Picker", NODE_CLASS_MAPPINGS)

    def test_one_click_clean_defaults_and_simple_surface(self):
        inputs = KBLOneClickDecomposeExport.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            ["image", "project_name", "output_mode", "padding", "export_root"],
        )
        self.assertEqual(inputs["required"]["output_mode"][1]["default"], "object_clean")
        policy = resolve_output_policy("object_clean")
        self.assertFalse(policy["include_body_parts"])
        self.assertFalse(policy["keep_unlabeled_objects"])
        self.assertEqual(policy["inventory_strategy"], "semantic_only")

    def test_one_click_persists_scene_inventory_preview(self):
        source = Path(__file__).resolve().parents[1] / "nodes" / "oneclick_nodes.py"
        self.assertIn('"scene_inventory_preview.png"', source.read_text(encoding="utf-8"))

    def test_old_manifest_compatibility(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "examples/sample_manifest.json").read_text(encoding="utf-8"))
        manifest["pipeline_version"] = "0.1.0"
        self.assertEqual(validate_manifest_schema(manifest), [])

    def test_legacy_and_new_workflow_json_valid(self):
        root = Path(__file__).resolve().parents[1] / "workflows"
        for path in root.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8")); self.assertEqual(data["version"], .4)

    def test_v012_workflows_exist_with_expected_modes(self):
        root = Path(__file__).resolve().parents[1] / "workflows"
        clean = json.loads((root / "Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_CLEAN_OBJECTS.json").read_text(encoding="utf-8"))
        parts = json.loads((root / "Kitao_Body_Collage_Lab_v0.1.2_ONE_CLICK_COLLAGE_PARTS.json").read_text(encoding="utf-8"))
        self.assertEqual(clean["nodes"][1]["widgets_values"][1], "object_clean")
        self.assertEqual(parts["nodes"][1]["widgets_values"][1], "collage_parts")

    def test_unlabeled_folder(self):
        self.assertEqual(category_folder({"label": "region_001", "semantic": False}), "unlabeled")


if __name__ == "__main__": unittest.main()
