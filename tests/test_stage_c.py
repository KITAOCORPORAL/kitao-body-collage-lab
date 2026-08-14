import unittest
from unittest.mock import patch

import numpy as np
import torch

from Kitao_Body_Collage_Lab.nodes.body_split_nodes import _candidate_parts, _resolve_overlaps
from Kitao_Body_Collage_Lab.nodes.body_split_nodes import NODE_CLASS_MAPPINGS as BODY_NODE_MAPPINGS
from Kitao_Body_Collage_Lab.nodes.backends.dwpose_backend import DWPoseBackend
from Kitao_Body_Collage_Lab.nodes.pose_nodes import KBLPoseEstimator
from Kitao_Body_Collage_Lab.nodes.utils.pose_utils import orientation, select_person


def kp(x, y, confidence=0.9):
    return {"x": x, "y": y, "confidence": confidence, "visible": confidence >= 0.3}


def pose_fixture():
    points = {
        "nose": kp(50, 14), "left_shoulder": kp(62, 30), "right_shoulder": kp(38, 30),
        "left_elbow": kp(70, 48), "right_elbow": kp(30, 48), "left_wrist": kp(72, 64), "right_wrist": kp(28, 64),
        "left_hip": kp(58, 62), "right_hip": kp(42, 62), "left_knee": kp(60, 84), "right_knee": kp(40, 84),
        "left_ankle": kp(62, 108), "right_ankle": kp(38, 108),
        "left_big_toe": kp(68, 114), "left_small_toe": kp(66, 116), "left_heel": kp(60, 112),
        "right_big_toe": kp(32, 114), "right_small_toe": kp(34, 116), "right_heel": kp(40, 112),
    }
    face = [kp(42 + i % 8 * 2, 10 + i // 8 * 2) for i in range(68)]
    left_hand = [kp(70 + i % 5, 62 + i // 5) for i in range(21)]
    right_hand = [kp(30 - i % 5, 62 + i // 5) for i in range(21)]
    return {"bbox": [20, 4, 80, 118], "keypoints": points, "face_points": face, "left_hand_points": left_hand, "right_hand_points": right_hand}


class StageCTests(unittest.TestCase):
    def test_body_splitter_is_registered(self):
        self.assertIn("KBL_Body_Splitter", BODY_NODE_MAPPINGS)

    def test_dwpose_confidence_schema_is_bounded(self):
        value = DWPoseBackend._point(np.array([[12.0, 34.0]]), np.array([1.25]), 0)
        self.assertEqual(value["confidence"], 1.0)
        self.assertTrue(value["visible"])

    def test_person_element_is_passed_and_pose_coordinates_stay_original(self):
        expected_pose = pose_fixture()

        class FakeBackend:
            received_shape = None

            def __init__(self, _root):
                self.last_metrics = {"provider": "test", "elapsed_seconds": 0.0}

            def infer(self, image):
                FakeBackend.received_shape = image.shape
                return [expected_pose]

        image = torch.zeros((1, 120, 100, 3), dtype=torch.float32)
        person_mask = np.ones((120, 100), dtype=bool)
        person = {"id": "person_07", "label": "person", "area": int(person_mask.sum()), "bbox": [20, 4, 80, 118], "mask": person_mask}
        node = KBLPoseEstimator()
        node.backend_class = FakeBackend
        data, _preview, selected = node.estimate(image, [person], "largest", 0, 0.3)
        self.assertEqual(FakeBackend.received_shape, (120, 100, 3))
        self.assertIs(selected[0], person)
        self.assertIs(data["selected_person"], person)
        self.assertEqual(data["selected_pose"]["keypoints"]["left_wrist"]["x"], 72)
        self.assertEqual(data["selected_pose"]["keypoints"]["left_wrist"]["y"], 64)

    def test_pose_estimator_no_person_is_safe(self):
        image = torch.zeros((1, 120, 100, 3), dtype=torch.float32)
        data, preview, selected = KBLPoseEstimator().estimate(image, [], "largest", 0, 0.3)
        self.assertIsNone(data["selected_pose"])
        self.assertEqual(selected, [])
        self.assertIs(preview, image)

    def test_no_person_and_largest_center_index_selection(self):
        self.assertIsNone(select_person([], "largest", (100, 120)))
        mask_a = np.ones((120, 100), bool); mask_b = np.ones((120, 100), bool)
        people = [
            {"id": "person_01", "label": "person", "area": 100, "bbox": [0, 0, 10, 10], "mask": mask_a},
            {"id": "person_02", "label": "person", "area": 200, "bbox": [40, 40, 70, 90], "mask": mask_b},
        ]
        self.assertEqual(select_person(people, "largest", (100, 120))["id"], "person_02")
        self.assertEqual(select_person(people, "center", (100, 120))["id"], "person_02")
        self.assertEqual(select_person(people, "index", (100, 120), 0)["id"], "person_01")

    def test_anatomical_left_right_mapping(self):
        pose = pose_fixture()
        self.assertGreater(pose["keypoints"]["left_shoulder"]["x"], pose["keypoints"]["right_shoulder"]["x"])

    def test_parts_are_original_shape_and_subset(self):
        person = np.zeros((120, 100), bool); person[4:119, 20:81] = True
        parts, missing = _candidate_parts(pose_fixture(), person, 0.3, True, True, True)
        self.assertEqual(missing, [])
        self.assertEqual(len(parts), 14)
        for part in parts:
            self.assertEqual(part["mask"].shape, person.shape)
            self.assertFalse(np.any(part["mask"] & ~person))
            self.assertEqual(part["area"], int(part["mask"].sum()))

    def test_empty_and_low_confidence_points_do_not_fake_parts(self):
        person = np.ones((120, 100), bool)
        pose = pose_fixture()
        for key in ("left_elbow", "left_wrist"):
            pose["keypoints"][key]["confidence"] = 0.01
        pose["left_hand_points"] = []
        parts, missing = _candidate_parts(pose, person, 0.3, True, True, True)
        labels = [part["label"] for part in parts]
        self.assertNotIn("left_forearm", labels)
        self.assertNotIn("left_hand", labels)
        self.assertIn("left_forearm", missing)
        self.assertIn("left_hand", missing)

    def test_overlap_resolution_anchor_and_orientation(self):
        person = np.ones((120, 100), bool)
        parts, _ = _candidate_parts(pose_fixture(), person, 0.3, True, True, True)
        before, after = _resolve_overlaps(parts)
        self.assertGreater(before, 0)
        self.assertEqual(after, 0)
        forearm = next(part for part in parts if part["label"] == "left_forearm")
        self.assertEqual(forearm["anchor"], forearm["joint_start"])
        self.assertAlmostEqual(forearm["orientation_deg"], orientation(forearm["joint_start"], forearm["joint_end"]), places=5)

    def test_orientation_cardinal_directions(self):
        self.assertAlmostEqual(orientation((0, 0), (1, 0)), 0.0)
        self.assertAlmostEqual(orientation((0, 0), (0, 1)), 90.0)


if __name__ == "__main__":
    unittest.main()
