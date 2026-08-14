"""Pinned, local-only Microsoft Florence-2 scene inventory backend."""

import gc
import json
import re
import time
from pathlib import Path

import torch

from ..model_backends import cleanup_model, configure_local_caches, validate_model_directory

TASK_SOURCES = {
    "<OD>": "florence_od",
    "<DENSE_REGION_CAPTION>": "florence_dense_caption",
    "<REGION_PROPOSAL>": "florence_region_proposal",
    "<OPEN_VOCABULARY_DETECTION>": "florence_caption_grounding",
}


def parse_florence_result(result, task, width, height):
    """Convert Florence post-processing output to the Stage-F candidate contract."""
    payload = result.get(task, result) if isinstance(result, dict) else {}
    boxes = payload.get("bboxes", []) if isinstance(payload, dict) else []
    labels = payload.get("labels", []) if isinstance(payload, dict) else []
    scores = payload.get("scores", []) if isinstance(payload, dict) else []
    source = TASK_SOURCES[task]
    candidates = []
    region_index = 0
    for index, box in enumerate(boxes):
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        bbox = [max(0.0, min(x1, width)), max(0.0, min(y1, height)),
                max(0.0, min(x2, width)), max(0.0, min(y2, height))]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        raw_label = str(labels[index]).strip() if index < len(labels) else ""
        semantic = bool(raw_label) and task != "<REGION_PROPOSAL>"
        if semantic:
            label = raw_label
        else:
            region_index += 1
            label = f"region_{region_index:03d}"
        candidates.append({
            "id": "", "label": label, "raw_label": raw_label, "bbox": bbox,
            "source": source, "semantic": semantic,
            "confidence": float(scores[index]) if index < len(scores) else None,
        })
    return candidates


def _transformers_5_compatibility():
    """Bridge APIs removed after the pinned Microsoft implementation shipped."""
    from transformers import PretrainedConfig, PreTrainedTokenizerBase
    for name in ("forced_bos_token_id", "forced_eos_token_id", "decoder_start_token_id",
                 "bos_token_id", "eos_token_id", "pad_token_id"):
        if not hasattr(PretrainedConfig, name):
            setattr(PretrainedConfig, name, None)
    if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
        PreTrainedTokenizerBase.additional_special_tokens = property(
            lambda tokenizer: tokenizer.special_tokens_map.get("additional_special_tokens", [])
        )


class Florence2Backend:
    """Run the official pinned Florence-2 code without any network access."""

    def __init__(self, model_dir):
        self.model_dir = validate_model_directory(model_dir, "Florence-2")
        self.last_metrics = {}
        revision_file = self.model_dir / "kbl_model_revision.json"
        self.revision = json.loads(revision_file.read_text(encoding="utf-8"))["revision"]

    def inventory(self, image, tasks, max_regions=64):
        from transformers import AutoModelForCausalLM, AutoProcessor
        _transformers_5_compatibility()
        configure_local_caches(self.model_dir.parent)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16 if device.type == "cuda" else torch.float32)
        model = None
        timings = {}
        started_all = time.perf_counter()
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            processor = AutoProcessor.from_pretrained(
                self.model_dir, trust_remote_code=True, local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_dir, trust_remote_code=True, local_files_only=True,
                dtype=dtype, attn_implementation="eager",
            )
            # Transformers 5 no longer resolves the old BART tied-weight map.
            shared = model.language_model.model.shared
            model.language_model.model.encoder.embed_tokens.weight = shared.weight
            model.language_model.model.decoder.embed_tokens.weight = shared.weight
            model.language_model.lm_head.weight = shared.weight
            model.to(device).eval()
            pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            candidates = []
            caption = ""
            for task in tasks:
                started = time.perf_counter()
                prompt = processor._construct_prompts([task])
                tokens = processor.tokenizer(prompt, return_tensors="pt", padding=False,
                                             return_token_type_ids=False)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=tokens["input_ids"].to(device), pixel_values=pixel_values,
                        max_new_tokens=1024, num_beams=3, do_sample=False, use_cache=False,
                    )
                text = processor.batch_decode(generated, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(text, task=task, image_size=image.size)
                candidates.extend(parse_florence_result(parsed, task, image.width, image.height))
                timings[TASK_SOURCES[task]] = round(time.perf_counter() - started, 4)
                if len(candidates) >= int(max_regions):
                    break
            # Dense captioning can miss a small object's box while the global
            # caption still names it.  Let Florence itself decide the missing
            # concepts, then ask Florence's open-vocabulary task for boxes.
            if "<DENSE_REGION_CAPTION>" in tasks and len(candidates) < int(max_regions):
                from ..utils.scene_utils import CANONICAL_TERMS, normalize_scene_label
                started = time.perf_counter()
                caption_task = "<MORE_DETAILED_CAPTION>"
                caption_tokens = processor.tokenizer(
                    processor._construct_prompts([caption_task]), return_tensors="pt",
                    padding=False, return_token_type_ids=False,
                )
                with torch.inference_mode():
                    caption_ids = model.generate(
                        input_ids=caption_tokens["input_ids"].to(device), pixel_values=pixel_values,
                        max_new_tokens=256, num_beams=3, do_sample=False, use_cache=False,
                    )
                caption_text = processor.batch_decode(caption_ids, skip_special_tokens=False)[0]
                caption_result = processor.post_process_generation(
                    caption_text, task=caption_task, image_size=image.size
                )
                caption = str(caption_result.get(caption_task, "")).lower()
                existing = {
                    normalize_scene_label(item.get("raw_label", ""), "")
                    for item in candidates if item.get("semantic")
                }
                discovered = []
                for canonical, aliases in CANONICAL_TERMS.items():
                    # Person routing must come from an actual localized
                    # Florence detection, never merely from caption text.
                    if canonical == "person":
                        continue
                    if canonical in existing:
                        continue
                    if any(re.search(rf"\b{re.escape(alias)}\b", caption) for alias in aliases):
                        discovered.append(canonical)
                timings["florence_detailed_caption"] = round(time.perf_counter() - started, 4)
                for concept in discovered[:8]:
                    started = time.perf_counter()
                    prompt = f"<OPEN_VOCABULARY_DETECTION>{concept}"
                    tokens = processor.tokenizer(
                        processor._construct_prompts([prompt]), return_tensors="pt",
                        padding=False, return_token_type_ids=False,
                    )
                    with torch.inference_mode():
                        generated = model.generate(
                            input_ids=tokens["input_ids"].to(device), pixel_values=pixel_values,
                            max_new_tokens=256, num_beams=3, do_sample=False, use_cache=False,
                        )
                    text = processor.batch_decode(generated, skip_special_tokens=False)[0]
                    parsed = processor.post_process_generation(
                        text, task="<OPEN_VOCABULARY_DETECTION>", image_size=image.size
                    )
                    grounded = parse_florence_result(
                        parsed, "<OPEN_VOCABULARY_DETECTION>", image.width, image.height
                    )
                    for item in grounded:
                        item["raw_label"] = concept
                        item["label"] = concept
                        item["semantic"] = True
                    candidates.extend(grounded)
                    timings[f"florence_ground_{concept}"] = round(time.perf_counter() - started, 4)
                    if len(candidates) >= int(max_regions):
                        break
            self.last_metrics = {
                "device": str(device), "dtype": str(dtype), "revision": self.revision,
                "tasks": timings, "elapsed_seconds": round(time.perf_counter() - started_all, 4),
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "caption": caption,
            }
            return candidates[: int(max_regions)]
        finally:
            cleanup_model(model)
            gc.collect()
