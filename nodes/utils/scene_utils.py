"""Scene candidate naming, filtering, and merge helpers."""

import re
import numpy as np

CANONICAL_TERMS = {
    "person": ("person", "woman", "man", "human", "people", "girl", "boy"),
    "hair": ("hair",), "face": ("face",), "hand": ("hand",), "foot": ("foot", "feet"),
    "clothing": ("dress", "jacket", "shirt", "coat", "trousers", "pants", "skirt", "miniskirt", "clothing", "clothes"),
    "shoe": ("shoe", "shoes", "footwear", "boot", "boots", "sneaker", "sneakers", "heel", "heels"),
    "chair": ("chair", "chairs", "stool", "stools", "bench", "benches"), "table": ("table", "tables", "desk", "desks"),
    "flower": ("flower", "flowers", "bouquet", "bouquets"), "glass": ("glass", "glasses", "cup", "cups", "goblet", "goblets"),
    "cloth": ("cloth", "fabric", "curtain", "curtains", "drape", "drapes"), "mirror": ("mirror", "mirrors"),
    "bag": ("bag", "bags", "handbag", "handbags", "purse", "purses", "backpack", "backpacks"),
    "jewelry": ("jewelry", "necklace", "necklaces", "bracelet", "bracelets", "earring", "earrings"),
    "plant": ("plant", "plants", "tree", "trees", "foliage"), "rope": ("rope", "ropes"),
    "box": ("box", "boxes", "gift", "gifts", "present", "presents"),
    "ball": ("ball", "balls", "ornament", "ornaments"),
    "telephone": ("telephone", "telephones", "phone", "phones"),
    "card": ("card", "cards"), "candle": ("candle", "candles"),
    "cabinet": ("cabinet", "cabinets", "cupboard", "cupboards"),
    "blanket": ("blanket", "blankets"),
}


def normalize_scene_label(raw_label, fallback="object_001"):
    words = set(re.findall(r"[a-z]+", str(raw_label).lower()))
    for canonical, aliases in CANONICAL_TERMS.items():
        if words.intersection(aliases):
            return canonical
    return fallback


def safe_asset_name(value, fallback="object", max_length=96):
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(value))
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", cleaned).strip(" ._")
    cleaned = re.sub(r"_+", "_", cleaned)
    return (cleaned or fallback)[: int(max_length)].rstrip(" ._") or fallback


def bbox_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / union if union else 0.0


def merge_scene_candidates(candidates, max_regions):
    kept = []
    for item in sorted(candidates, key=lambda x: (bool(x.get("semantic")), x.get("confidence") or 0.0), reverse=True):
        if any(bbox_iou(item["bbox"], old["bbox"]) >= 0.92 and item.get("label") == old.get("label") for old in kept):
            continue
        kept.append(dict(item))
        if len(kept) >= int(max_regions):
            break
    object_counter = region_counter = 0
    label_counts = {}
    for item in kept:
        if item.get("semantic"):
            object_counter += 1
            canonical = normalize_scene_label(item.get("raw_label", item.get("label", "")), f"object_{object_counter:03d}")
        else:
            region_counter += 1
            canonical = f"region_{region_counter:03d}"
        item["canonical_label"] = canonical
        item["label"] = canonical
        label_counts[canonical] = label_counts.get(canonical, 0) + 1
        item["id"] = f"{canonical}_{label_counts[canonical]:02d}"
    return kept


def background_candidate_filter(elements, image_size, max_area_ratio=0.90):
    width, height = image_size; image_area = max(1, width * height)
    semantic_masks = [np.asarray(x["mask"], bool) for x in elements if x.get("semantic")]
    kept, rejected = [], []
    for item in elements:
        mask = np.asarray(item["mask"], bool); ratio = float(mask.sum()) / image_area
        edges = sum((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any())) if mask.any() else 0
        ys, xs = np.nonzero(mask)
        bbox_ratio = 0.0 if not len(xs) else float((xs.max()-xs.min()+1) * (ys.max()-ys.min()+1)) / image_area
        contains = sum(float(np.logical_and(mask, semantic).sum()) / max(1, semantic.sum()) >= 0.8 for semantic in semantic_masks)
        background = not item.get("semantic") and (
            ratio > float(max_area_ratio) or (ratio > 0.65 and edges >= 3)
            or (bbox_ratio > 0.65 and edges >= 3) or (ratio > 0.55 and contains >= 2)
        )
        (rejected if background else kept).append(item)
    return kept, rejected
