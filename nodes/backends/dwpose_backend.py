"""Self-contained DWPose ONNX inference for KBL."""

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

BODY_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
FOOT_NAMES = ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]


def _nms(boxes, scores, threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


def _yolox_preprocess(image, size=(640, 640)):
    padded = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    ratio = min(size[0] / image.shape[0], size[1] / image.shape[1])
    resized = cv2.resize(image, (int(image.shape[1] * ratio), int(image.shape[0] * ratio)))
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None], ratio


def _decode_yolox(output, size=(640, 640)):
    strides = (8, 16, 32)
    grids, expanded = [], []
    for stride in strides:
        h, w = size[0] // stride, size[1] // stride
        xv, yv = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded.append(np.full((*grid.shape[:2], 1), stride))
    grid = np.concatenate(grids, axis=1)
    stride = np.concatenate(expanded, axis=1)
    output[..., :2] = (output[..., :2] + grid) * stride
    output[..., 2:4] = np.exp(output[..., 2:4]) * stride
    return output


def _affine_crop(image, bbox, input_size):
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * 1.25
    aspect = input_size[0] / input_size[1]
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    src_dir = np.array([0, -scale[0] * 0.5], dtype=np.float32)
    dst_dir = np.array([0, -input_size[0] * 0.5], dtype=np.float32)
    src = np.array([center, center + src_dir, center + [-src_dir[1], src_dir[0]]], dtype=np.float32)
    dst_center = np.array([input_size[0] * 0.5, input_size[1] * 0.5], dtype=np.float32)
    dst = np.array([dst_center, dst_center + dst_dir, dst_center + [-dst_dir[1], dst_dir[0]]], dtype=np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(image, matrix, input_size)
    crop = (crop.astype(np.float32) - np.array([123.675, 116.28, 103.53])) / np.array([58.395, 57.12, 57.375])
    return crop, center, scale


class DWPoseBackend:
    def __init__(self, model_dir):
        root = Path(model_dir)
        self.det_path = root / "yolox_l.onnx"
        self.pose_path = root / "dw-ll_ucoco_384.onnx"
        missing = [path.name for path in (self.det_path, self.pose_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"[缺少 DWPose 模型]\n期待目录：{root}\n缺少：\n" + "\n".join(f"- {name}" for name in missing))
        available = ort.get_available_providers()
        self.provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
        self.last_metrics = {}

    def infer(self, image_rgb):
        started = time.perf_counter()
        det = ort.InferenceSession(str(self.det_path), providers=[self.provider])
        pose = ort.InferenceSession(str(self.pose_path), providers=[self.provider])
        print(f"[KBL] DWPose provider: {self.provider}")
        tensor, ratio = _yolox_preprocess(image_rgb)
        output = det.run(None, {det.get_inputs()[0].name: tensor})[0]
        predictions = _decode_yolox(output.copy())[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4] * predictions[:, 5]
        xyxy = np.column_stack((boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2, boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2)) / ratio
        valid = scores > 0.30
        xyxy, scores = xyxy[valid], scores[valid]
        xyxy = xyxy[_nms(xyxy, scores)] if len(xyxy) else xyxy
        people = []
        input_shape = pose.get_inputs()[0].shape
        input_size = (int(input_shape[3]), int(input_shape[2]))
        for bbox in xyxy:
            crop, center, scale = _affine_crop(image_rgb, bbox, input_size)
            pose_input = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32)
            simcc_x, simcc_y = pose.run(None, {pose.get_inputs()[0].name: pose_input})
            x_idx, y_idx = simcc_x[0].argmax(axis=-1), simcc_y[0].argmax(axis=-1)
            conf = np.minimum(simcc_x[0].max(axis=-1), simcc_y[0].max(axis=-1))
            points = np.column_stack((x_idx, y_idx)).astype(np.float32) / 2.0
            points = points / np.array(input_size) * scale + center - scale / 2
            people.append(self._schema(points, conf, bbox))
        self.last_metrics = {"provider": self.provider, "elapsed_seconds": round(time.perf_counter() - started, 4), "people": len(people)}
        del det, pose
        return people

    @staticmethod
    def _point(points, scores, index):
        confidence = float(np.clip(scores[index], 0.0, 1.0))
        return {"x": float(points[index, 0]), "y": float(points[index, 1]), "confidence": confidence, "visible": bool(confidence >= 0.30)}

    def _schema(self, points, scores, bbox):
        keypoints = {name: self._point(points, scores, index) for index, name in enumerate(BODY_NAMES)}
        keypoints.update({name: self._point(points, scores, 17 + index) for index, name in enumerate(FOOT_NAMES)})
        face = [self._point(points, scores, index) for index in range(23, 91)]
        left_hand = [self._point(points, scores, index) for index in range(91, 112)]
        right_hand = [self._point(points, scores, index) for index in range(112, 133)]
        return {"bbox": [float(v) for v in bbox], "keypoints": keypoints, "face_points": face, "left_hand_points": left_hand, "right_hand_points": right_hand}
