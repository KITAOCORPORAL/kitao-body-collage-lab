"""Pose selection, geometry and preview helpers."""

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .bbox_utils import mask_bbox
from .image_io import pil_to_comfy_image

SKELETON_EDGES = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def select_person(elements, selector, image_size, index=0):
    people = [item for item in elements if item.get("label") == "person"]
    if not people:
        return None
    if selector == "index":
        return people[max(0, min(int(index), len(people) - 1))]
    if selector == "center":
        cx, cy = image_size[0] / 2, image_size[1] / 2
        return min(people, key=lambda item: ((item["bbox"][0] + item["bbox"][2]) / 2 - cx) ** 2 + ((item["bbox"][1] + item["bbox"][3]) / 2 - cy) ** 2)
    return max(people, key=lambda item: int(item.get("area", 0)))


def bbox_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union


def point(pose, name, threshold=0.3):
    value = pose.get("keypoints", {}).get(name)
    if not value or float(value.get("confidence", 0)) < threshold:
        return None
    return np.array([value["x"], value["y"]], dtype=np.float32)


def orientation(start, end):
    return float(math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])))


def corridor(shape, start, end, width):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.line(mask, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)), 255, max(3, int(width)), cv2.LINE_AA)
    return mask > 0


def polygon_mask(shape, points, expand=0):
    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(np.rint(np.asarray(points)).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))
        mask = cv2.dilate(mask, kernel)
    return mask > 0


def ellipse_mask(shape, center, axes):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, tuple(np.rint(center).astype(int)), tuple(max(2, int(v)) for v in axes), 0, 0, 360, 255, -1)
    return mask > 0


def part_record(label, mask, pose_confidence, source_person_id, anchor, joint_start=None, joint_end=None, quality="ok", sam_score=0.0):
    bbox = mask_bbox(mask)
    record = {
        "id": f"body_{label}", "label": label, "category": "body", "mask": mask,
        "bbox": bbox, "area": int(mask.sum()), "confidence": float(pose_confidence),
        "pose_confidence": float(pose_confidence), "sam_score": float(sam_score),
        "quality_flag": quality, "source_person_id": source_person_id,
        "anchor": [float(anchor[0]), float(anchor[1])], "original_anchor": [float(anchor[0]), float(anchor[1])],
        "source": "dwpose+sam2" if sam_score else "dwpose+person_mask",
    }
    if joint_start is not None and joint_end is not None:
        record["joint_start"] = [float(joint_start[0]), float(joint_start[1])]
        record["joint_end"] = [float(joint_end[0]), float(joint_end[1])]
        record["orientation_deg"] = orientation(joint_start, joint_end)
    else:
        record["orientation_deg"] = 0.0
    return record


def render_pose_preview(image, pose, threshold=0.3):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for a, b in SKELETON_EDGES:
        pa, pb = point(pose, a, threshold), point(pose, b, threshold)
        if pa is not None and pb is not None:
            draw.line((*pa, *pb), fill=(0, 255, 255), width=max(3, image.width // 400))
    collections = [pose.get("left_hand_points", []), pose.get("right_hand_points", [])]
    for points in collections:
        for item in points:
            if item["confidence"] >= threshold:
                x, y = item["x"], item["y"]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 80, 40))
    for name, item in pose.get("keypoints", {}).items():
        if item["confidence"] >= threshold and name in {v for edge in SKELETON_EDGES for v in edge}:
            x, y = item["x"], item["y"]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 0))
    return pil_to_comfy_image(preview)

