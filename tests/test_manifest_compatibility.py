import json
import os
import unittest
from pathlib import Path, PurePosixPath

from Kitao_Body_Collage_Lab.nodes.utils.manifest_utils import validate_manifest_schema
from Kitao_Body_Collage_Lab.nodes.utils.project_reader import load_kbl_project
from Kitao_Body_Collage_Lab.version import KBL_MANIFEST_VERSION, KBL_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_D_PROJECT = Path("N:/ComfyUI/output/Kitao_Body_Collage_Lab/validation_stage_d/KBL_STAGE_D_TEST")


class ManifestCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample = json.loads((PROJECT_ROOT / "examples" / "sample_manifest.json").read_text(encoding="utf-8"))

    def test_frozen_versions_and_top_level_contract(self):
        self.assertEqual(validate_manifest_schema(self.sample), [])
        self.assertEqual(self.sample["manifest_version"], KBL_MANIFEST_VERSION)
        self.assertEqual(self.sample["pipeline_version"], KBL_VERSION)
        self.assertIsInstance(self.sample["source"], dict)
        self.assertIsInstance(self.sample["image"], dict)
        self.assertIsInstance(self.sample["body_parts"], list)
        self.assertIsInstance(self.sample["elements"], list)

    def test_paths_anchors_orientation_joints_and_quality_remain_parseable(self):
        records = self.sample["body_parts"] + self.sample["elements"]
        self.assertTrue(self.sample["body_parts"])
        self.assertTrue(self.sample["elements"])
        for item in records:
            for field in ("file", "raw_mask_file", "refined_mask_file", "alpha_mask_file"):
                value = item[field]
                self.assertFalse(PurePosixPath(value).is_absolute())
                self.assertNotIn("\\", value)
            self.assertEqual(len(item["original_anchor"]), 2)
            self.assertEqual(len(item["local_anchor"]), 2)
            self.assertIsInstance(item["orientation_deg"], (int, float))
            self.assertIn(item["quality_flag"], {"ok", "uncertain", "partial", "missing"})
        limb = next(item for item in self.sample["body_parts"] if item["joint_start"] is not None)
        self.assertEqual(len(limb["joint_start"]), 2)
        self.assertEqual(len(limb["joint_end"]), 2)


class ProjectReaderTests(unittest.TestCase):
    def test_loads_real_stage_d_project(self):
        project = Path(os.environ.get("KBL_STAGE_D_PROJECT", str(DEFAULT_STAGE_D_PROJECT)))
        if not project.is_dir():
            self.skipTest(f"real Stage D project not available: {project}")
        data = load_kbl_project(project)
        self.assertEqual(data["diagnostics"]["validation"]["status"], "PASS")
        self.assertEqual(len(data["body_parts"]), 14)
        self.assertEqual(len(data["elements"]), 8)
        for part in data["body_parts"]:
            self.assertEqual(len(part["local_anchor"]), 2)
            self.assertIsInstance(part["orientation_deg"], (int, float))
            self.assertTrue(Path(part["resolved_file"]).is_file())
            self.assertEqual(part["png"]["mode"], "RGBA")


if __name__ == "__main__":
    unittest.main()
