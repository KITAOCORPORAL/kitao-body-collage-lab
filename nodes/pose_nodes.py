"""DWPose estimator node."""

from pathlib import Path

import numpy as np

from .backends.dwpose_backend import DWPoseBackend
from .utils.image_io import comfy_image_to_pil
from .utils.path_config import get_path_config
from .utils.pose_utils import bbox_iou, render_pose_preview, select_person


class KBLPoseEstimator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",), "person_elements": ("KBL_ELEMENTS",),
            "person_selector": (["largest", "center", "index"], {"default": "largest"}),
            "person_index": ("INT", {"default": 0, "min": 0, "max": 32}),
            "confidence_threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.95, "step": 0.01}),
        }}

    RETURN_TYPES = ("KBL_POSE", "IMAGE", "KBL_ELEMENTS")
    RETURN_NAMES = ("pose_data", "pose_preview", "selected_person")
    FUNCTION = "estimate"
    CATEGORY = "Kitao Body Collage/人体"

    def __init__(self):
        self.backend_class = DWPoseBackend

    def estimate(self, image, person_elements, person_selector, person_index, confidence_threshold):
        pil = comfy_image_to_pil(image)
        selected = select_person(person_elements, person_selector, pil.size, person_index)
        if selected is None:
            return {"people": [], "selected_pose": None, "selected_person": None, "metrics": {}}, image, []
        root = Path(get_path_config()["model_root"]) / "Kitao_Body_Collage_Lab" / "dwpose"
        backend = self.backend_class(root)
        people = backend.infer(np.asarray(pil))
        selected_pose = max(people, key=lambda pose: bbox_iou(pose["bbox"], selected["bbox"]), default=None)
        data = {"people": people, "selected_pose": selected_pose, "selected_person": selected, "metrics": backend.last_metrics, "confidence_threshold": float(confidence_threshold)}
        preview = render_pose_preview(pil, selected_pose, confidence_threshold) if selected_pose else image
        return data, preview, [selected]

NODE_CLASS_MAPPINGS = {"KBL_Pose_Estimator": KBLPoseEstimator}
NODE_DISPLAY_NAME_MAPPINGS = {"KBL_Pose_Estimator": "KBL 人体姿态识别"}

