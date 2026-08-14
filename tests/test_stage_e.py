import tempfile
import unittest
from pathlib import Path

import numpy as np

from Kitao_Body_Collage_Lab.nodes.refine_nodes import _refine_item
from Kitao_Body_Collage_Lab.nodes.utils.path_config import get_path_config
from Kitao_Body_Collage_Lab.nodes.utils.project_reader import _asset_path
from Kitao_Body_Collage_Lab.scripts.regression_test import explicit_folder_images
from Kitao_Body_Collage_Lab.version import KBL_MANIFEST_VERSION, KBL_VERSION


class StageETests(unittest.TestCase):
    def test_version_is_single_release_source(self):
        self.assertEqual(KBL_VERSION, "0.1.1")
        self.assertEqual(KBL_MANIFEST_VERSION, "0.1")
        self.assertEqual(get_path_config()["version"], KBL_VERSION)

    def test_refine_warning_and_fallback_diagnostics(self):
        raw = np.zeros((30, 30), dtype=bool)
        raw[10:20, 10:20] = True
        settings = {"expand_px": 1, "erode_px": 0, "min_component_area": 1, "fill_small_holes": True, "remove_small_islands": True, "feather_px": 0.75}
        fallback = _refine_item({"id": "mask_01", "label": "mask", "mask": raw}, "element", "safe", {}, settings)
        self.assertTrue(fallback["fallback_applied"])
        noisy = np.zeros((30, 30), dtype=bool)
        noisy[10:13, 10:13] = True
        noisy[2, 2] = True
        noisy[25, 25] = True
        settings["expand_px"] = 0
        settings["min_component_area"] = 3
        warning = _refine_item({"id": "mask_02", "label": "mask", "mask": noisy}, "element", "safe", {}, settings)
        self.assertEqual(warning["warning"], "REFINE_AREA_WARNING")
        self.assertGreater(abs(warning["area_change_percent"]), 15.0)

    def test_folder_mode_is_explicit_and_non_recursive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.jpg").write_bytes(b"not decoded by discovery")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "hidden.png").write_bytes(b"not discovered")
            self.assertEqual([path.name for path in explicit_folder_images(root)], ["a.jpg"])

    def test_project_reader_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self.assertRaisesRegex(ValueError, "相对路径"):
                _asset_path(root, "../outside.png")


if __name__ == "__main__":
    unittest.main()
