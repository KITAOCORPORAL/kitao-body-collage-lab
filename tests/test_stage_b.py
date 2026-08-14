import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from Kitao_Body_Collage_Lab.nodes.detection_nodes import KBLElementDetector
from Kitao_Body_Collage_Lab.nodes.input_nodes import KBLLoadImage
from Kitao_Body_Collage_Lab.nodes.model_backends import validate_model_directory
from Kitao_Body_Collage_Lab.nodes.utils.image_io import pil_to_comfy_image
from Kitao_Body_Collage_Lab.nodes.utils.mask_utils import deduplicate_elements, validate_elements
from Kitao_Body_Collage_Lab.nodes.utils.preview_utils import render_element_preview, render_mask_contact_sheet


class FakeDinoBackend:
    last_prompt = None

    def __init__(self, model_dir, confidence_threshold):
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold

    def detect(self, image, text_prompt, max_detections):
        FakeDinoBackend.last_prompt = text_prompt
        return [
            {"label": "person", "bbox": [1.0, 1.0, 7.0, 7.0], "confidence": 0.93},
            {"label": "chair", "bbox": [8.0, 2.0, 14.0, 7.0], "confidence": 0.82},
        ][:max_detections]


class FakeSamBackend:
    received_detections = None

    def __init__(self, model_dir):
        self.model_dir = model_dir

    def segment(self, image, detections, include_auto, max_candidates, min_mask_area):
        FakeSamBackend.received_detections = detections
        guided = []
        for item in detections:
            x1, y1, x2, y2 = [int(value) for value in item["bbox"]]
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[y1:y2, x1:x2] = True
            guided.append({**item, "mask": mask, "area": int(mask.sum()), "sam_score": 0.91, "source": "guided"})
        return guided, []


class EmptyDinoBackend(FakeDinoBackend):
    def detect(self, image, text_prompt, max_detections):
        return []


class EmptySamBackend(FakeSamBackend):
    def segment(self, image, detections, include_auto, max_candidates, min_mask_area):
        return [], []


class StageBTests(unittest.TestCase):
    def test_jpg_and_png_load_at_original_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            Image.new("RGB", (31, 17), "red").save(root / "photo.jpg")
            Image.new("RGBA", (19, 23), (0, 255, 0, 100)).save(root / "photo.png")
            loader = KBLLoadImage()
            jpg = loader.load(str(root / "photo.jpg"))
            png = loader.load(str(root / "photo.png"))
            self.assertEqual(jpg[0].shape, (1, 17, 31, 3))
            self.assertEqual((jpg[3], jpg[4]), (31, 17))
            self.assertEqual(png[0].shape, (1, 23, 19, 3))
            self.assertEqual(json.loads(png[1])["filename"], "photo.png")

    def test_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "oriented.jpg"
            image = Image.new("RGB", (10, 20), "blue")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            result = KBLLoadImage().load(str(path))
            self.assertEqual((result[3], result[4]), (20, 10))

    def test_missing_model_error_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, r"\[缺少模型\]"):
                validate_model_directory(Path(temp_dir) / "sam2", "SAM2")

    def test_grounding_results_reach_sam_and_masks_stay_independent(self):
        node = KBLElementDetector()
        node.dino_backend_class = FakeDinoBackend
        node.sam_backend_class = FakeSamBackend
        result = node.detect(
            pil_to_comfy_image(Image.new("RGB", (16, 10), "white")),
            "guided", "custom", "person, chair", 0.3, 8, 0.85, 1, 64,
        )
        elements, masks = result[0], result[1]
        self.assertEqual([item["label"] for item in FakeSamBackend.received_detections], ["person", "chair"])
        self.assertEqual(len(elements), 2)
        self.assertEqual(masks.shape, (2, 10, 16))
        self.assertFalse(torch.equal(masks[0], masks[1]))
        self.assertEqual(result[5].shape[1:3], (10, 16))
        self.assertEqual(result[6].ndim, 4)

    def test_empty_detection_does_not_crash(self):
        node = KBLElementDetector()
        node.dino_backend_class = EmptyDinoBackend
        node.sam_backend_class = EmptySamBackend
        result = node.detect(
            pil_to_comfy_image(Image.new("RGB", (12, 8), "black")),
            "guided", "portrait_basic", "", 0.3, 8, 0.85, 1, 64,
        )
        self.assertEqual(result[0], [])
        self.assertEqual(result[1].shape, (1, 8, 12))
        self.assertEqual(json.loads(result[2]), [])

    def test_dedup_and_preview_helpers(self):
        mask_a = np.zeros((10, 10), dtype=bool)
        mask_b = np.zeros((10, 10), dtype=bool)
        mask_a[1:7, 1:7] = True
        mask_b[1:7, 1:7] = True
        elements = [
            {"label": "person", "mask": mask_a, "area": 36, "confidence": 0.9, "source": "guided", "bbox": [1, 1, 7, 7], "id": "person_01"},
            {"label": "person", "mask": mask_b, "area": 36, "confidence": 0.7, "source": "auto", "bbox": [1, 1, 7, 7], "id": "person_02"},
        ]
        kept = deduplicate_elements(elements, 0.8, 1, 64)
        self.assertEqual(len(kept), 1)
        self.assertEqual(render_element_preview(Image.new("RGB", (10, 10)), kept).shape, (1, 10, 10, 3))
        self.assertEqual(render_mask_contact_sheet(kept).ndim, 4)

    def test_kbl_elements_contract(self):
        mask = np.zeros((8, 12), dtype=bool)
        mask[2:6, 3:9] = True
        elements = [{
            "id": "person_01", "label": "person", "mask": mask,
            "bbox": [3, 2, 9, 6], "confidence": 0.9,
            "area": 24, "source": "guided",
        }]
        self.assertIs(validate_elements(elements, 8, 12), elements)
        with self.assertRaisesRegex(ValueError, "尺寸"):
            validate_elements(elements, 9, 12)


if __name__ == "__main__":
    unittest.main()
