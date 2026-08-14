import json
import unittest
from pathlib import Path

import numpy as np

from Kitao_Body_Collage_Lab.nodes import NODE_CLASS_MAPPINGS
from Kitao_Body_Collage_Lab.nodes.backends.florence2_backend import parse_florence_result
from Kitao_Body_Collage_Lab.nodes.export_nodes import _element_filename, category_folder
from Kitao_Body_Collage_Lab.nodes.pose_nodes import KBLPoseEstimator
from Kitao_Body_Collage_Lab.nodes.utils.mask_utils import assign_element_ids, deduplicate_elements
from Kitao_Body_Collage_Lab.nodes.utils.pose_utils import select_person
from Kitao_Body_Collage_Lab.nodes.utils.scene_utils import background_candidate_filter, merge_scene_candidates, normalize_scene_label, safe_asset_name
from Kitao_Body_Collage_Lab.nodes.utils.manifest_utils import validate_manifest_schema


class StageFTests(unittest.TestCase):
    def test_florence_candidate_parsing(self):
        result = {"<OD>": {"bboxes": [[1, 2, 30, 40]], "labels": ["a red chair"]}}
        item = parse_florence_result(result, "<OD>", 100, 100)[0]
        self.assertTrue(item["semantic"]); self.assertEqual(item["source"], "florence_od")

    def test_canonical_label(self):
        self.assertEqual(normalize_scene_label("woman wearing black dress"), "person")
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
        self.assertIn("KBL_Load_Image_Picker", NODE_CLASS_MAPPINGS)

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

    def test_unlabeled_folder(self):
        self.assertEqual(category_folder({"label": "region_001", "semantic": False}), "unlabeled")


if __name__ == "__main__": unittest.main()
