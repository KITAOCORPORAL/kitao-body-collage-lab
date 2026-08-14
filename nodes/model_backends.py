"""Local-only GroundingDINO and SAM2 inference backends."""

import gc
import math
import os
import re
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import (
    GroundingDinoForObjectDetection,
    GroundingDinoProcessor,
    Sam2Config,
    Sam2Model,
    Sam2Processor,
    Sam2VideoConfig,
)

from .utils.bbox_utils import clamp_box, mask_bbox

MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def configure_local_caches(model_root):
    cache_root = Path(model_root).parent / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "transformers"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def validate_model_directory(model_dir, model_name):
    path = Path(model_dir)
    required = ["config.json", "preprocessor_config.json"]
    if model_name == "GroundingDINO":
        required.extend(["tokenizer_config.json", "tokenizer.json"])
    elif model_name == "SAM2":
        required.append("processor_config.json")
    elif model_name == "DWPose":
        required = ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]
    missing = [filename for filename in required if not (path / filename).is_file()]
    has_weights = model_name == "DWPose" or any((path / filename).is_file() for filename in MODEL_WEIGHT_FILES)
    has_sharded_weights = any(path.glob("model-*-of-*.safetensors")) or any(path.glob("pytorch_model-*-of-*.bin"))
    if missing or not (has_weights or has_sharded_weights):
        expected = ", ".join(required) + " 以及 model.safetensors（或分片权重）"
        missing_files = list(missing)
        if not (has_weights or has_sharded_weights):
            missing_files.append("model.safetensors")
        raise FileNotFoundError(
            "[缺少模型]\n"
            f"模型：{model_name}\n"
            f"目录：{path}\n"
            "缺少：\n"
            + "\n".join(f"- {name}" for name in missing_files)
            + f"\n期待文件：{expected}\nKBL 已禁止联网自动下载。"
        )
    return path


def normalize_prompt(prompt):
    terms = [term.strip().lower() for term in re.split(r"[,.，;；\n]+", prompt) if term.strip()]
    return ". ".join(dict.fromkeys(terms)) + ("." if terms else "")


def normalize_label(label):
    cleaned = re.sub(r"[^a-z0-9_]+", "_", str(label).strip().lower()).strip("_")
    return cleaned or "object"


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_dtype(device):
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _move_batch(batch, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=_compute_dtype(device))
    return nullcontext()


def cleanup_model(model):
    if model is not None:
        model.to("cpu")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _oom_error(component):
    return RuntimeError(
        f"KBL {component} 推理时 CUDA 显存不足。请降低 max_detections/max_candidates 或自动点网格密度，"
        "关闭其他占用显存的模型后重试。"
    )


class GroundingDinoBackend:
    def __init__(self, model_dir, confidence_threshold):
        self.model_dir = validate_model_directory(model_dir, "GroundingDINO")
        self.confidence_threshold = float(confidence_threshold)
        self.last_metrics = {}

    def detect(self, image, text_prompt, max_detections):
        prompt = normalize_prompt(text_prompt)
        if not prompt:
            return []
        configure_local_caches(self.model_dir.parent)
        device = _device()
        model = None
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            processor = GroundingDinoProcessor.from_pretrained(self.model_dir, local_files_only=True)
            dtype = _compute_dtype(device)
            model = GroundingDinoForObjectDetection.from_pretrained(
                self.model_dir,
                local_files_only=True,
                torch_dtype=dtype,
            ).to(device).eval()
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            model_inputs = _move_batch(inputs, device)
            with torch.inference_mode(), _autocast(device):
                outputs = model(**model_inputs)
            result = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=model_inputs["input_ids"],
                threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            detections = []
            for box, score, label in zip(result["boxes"], result["scores"], result["text_labels"]):
                detections.append(
                    {
                        "label": normalize_label(label),
                        "bbox": clamp_box(box.detach().float().cpu().tolist(), image.width, image.height),
                        "confidence": float(score.detach().float().cpu()),
                    }
                )
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            self.last_metrics = {
                "device": str(device),
                "dtype": str(dtype),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            }
            return detections[: int(max_detections)]
        except torch.OutOfMemoryError as exc:
            raise _oom_error("GroundingDINO") from exc
        finally:
            cleanup_model(model)


class Sam2Backend:
    def __init__(self, model_dir):
        self.model_dir = validate_model_directory(model_dir, "SAM2")
        self.last_metrics = {}

    def _load(self):
        configure_local_caches(self.model_dir.parent)
        device = _device()
        processor = Sam2Processor.from_pretrained(self.model_dir, local_files_only=True)
        video_config = Sam2VideoConfig.from_pretrained(self.model_dir, local_files_only=True)
        image_config = Sam2Config(
            vision_config=video_config.vision_config.to_dict(),
            prompt_encoder_config=video_config.prompt_encoder_config.to_dict(),
            mask_decoder_config=video_config.mask_decoder_config.to_dict(),
        )
        model = Sam2Model.from_pretrained(
            self.model_dir,
            config=image_config,
            local_files_only=True,
            torch_dtype=_compute_dtype(device),
        ).to(device).eval()
        return processor, model, device

    def _predict(self, processor, model, device, image, boxes=None, points=None):
        kwargs = {"images": image, "return_tensors": "pt"}
        if boxes is not None:
            kwargs["input_boxes"] = [boxes]
        if points is not None:
            kwargs["input_points"] = [[[[float(x), float(y)]] for x, y in points]]
            kwargs["input_labels"] = [[[1] for _ in points]]
        inputs = processor(**kwargs)
        original_sizes = inputs["original_sizes"]
        model_inputs = _move_batch({key: value for key, value in inputs.items() if key != "original_sizes"}, device)
        with torch.inference_mode(), _autocast(device):
            outputs = model(**model_inputs, multimask_output=True)
        masks = processor.post_process_masks(outputs.pred_masks.detach().float().cpu(), original_sizes)[0]
        scores = outputs.iou_scores.detach().float().cpu()[0]
        return masks, scores

    def segment(self, image, detections, include_auto, max_candidates, min_mask_area):
        """Run guided and auto prompts while keeping one SAM2 instance for the image."""
        processor = model = None
        try:
            device = _device()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            processor, model, device = self._load()
            dtype = _compute_dtype(device)
            guided = []
            if detections:
                masks, scores = self._predict(
                    processor,
                    model,
                    device,
                    image,
                    boxes=[item["bbox"] for item in detections],
                )
                for index, detection in enumerate(detections):
                    best = int(torch.argmax(scores[index]).item())
                    mask = masks[index, best].numpy().astype(bool)
                    guided.append(
                        {
                            **detection,
                            "mask": mask,
                            "area": int(mask.sum()),
                            "sam_score": float(scores[index, best]),
                            "source": "guided",
                        }
                    )
            automatic = []
            if include_auto:
                automatic = self._automatic_with_loaded_model(
                    processor, model, device, image, max_candidates, min_mask_area
                )
            self.last_metrics = {
                "device": str(device),
                "dtype": str(dtype),
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            }
            return guided, automatic
        except torch.OutOfMemoryError as exc:
            raise _oom_error("SAM2") from exc
        finally:
            cleanup_model(model)

    def refine_part_prompts(self, image, prompts):
        """Refine all body-part boxes/points with one SAM2 load."""
        if not prompts:
            return []
        processor = model = None
        try:
            processor, model, device = self._load()
            masks, scores = self._predict(processor, model, device, image, boxes=[item["bbox"] for item in prompts])
            results = []
            for index in range(len(prompts)):
                best = int(torch.argmax(scores[index]).item())
                results.append({"mask": masks[index, best].numpy().astype(bool), "sam_score": float(scores[index, best])})
            return results
        finally:
            cleanup_model(model)

    def _automatic_with_loaded_model(self, processor, model, device, image, max_candidates, min_mask_area):
        grid_side = max(4, min(8, int(math.ceil(math.sqrt(max_candidates)))))
        points = [
            ((column + 0.5) * image.width / grid_side, (row + 0.5) * image.height / grid_side)
            for row in range(grid_side)
            for column in range(grid_side)
        ]
        elements = []
        masks_per_point = 3
        target_postprocess_pixels = 96_000_000
        chunk_size = max(1, min(8, target_postprocess_pixels // max(1, image.width * image.height * masks_per_point)))
        for start in range(0, len(points), chunk_size):
            chunk = points[start : start + chunk_size]
            masks, scores = self._predict(processor, model, device, image, points=chunk)
            for point_index in range(len(chunk)):
                for mask_index in range(scores.shape[-1]):
                    score = float(scores[point_index, mask_index])
                    if score < 0.72:
                        continue
                    mask = masks[point_index, mask_index].numpy().astype(bool)
                    area = int(mask.sum())
                    if area < int(min_mask_area):
                        continue
                    elements.append(
                        {
                            "label": "auto_object",
                            "bbox": mask_bbox(mask),
                            "confidence": score,
                            "sam_score": score,
                            "mask": mask,
                            "area": area,
                            "source": "auto",
                        }
                    )
        return elements
