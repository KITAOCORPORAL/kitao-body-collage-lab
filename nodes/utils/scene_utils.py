"""Scene candidate naming, filtering, and merge helpers."""

import re
import numpy as np

CANONICAL_TERMS = {
    "person": ("person", "woman", "man", "human", "people", "girl", "boy"),
    "hair": ("hair",), "head": ("head",), "face": ("face",),
    "arm": ("arm", "arms"), "hand": ("hand", "hands"),
    "leg": ("leg", "legs"), "foot": ("foot", "feet"),
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

PERSON_CHILD_LABELS = {
    "hair", "head", "face", "arm", "hand", "leg", "foot", "clothing", "shoe",
    "torso", "body-region", "body_region",
}

SEMANTIC_INVENTORY_SOURCES = {
    "florence_od", "florence_caption_grounding", "grounding_dino_extra", "guided",
}


def normalize_scene_label(raw_label, fallback="object_001"):
    words = set(re.findall(r"[a-z]+", str(raw_label).lower()))
    # A localized anatomy noun is more specific than a generic modifier such
    # as "human" (for example Florence OD's "human face").
    for canonical in ("hair", "head", "face", "arm", "hand", "leg", "foot", "shoe"):
        if words.intersection(CANONICAL_TERMS[canonical]):
            return canonical
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


def scene_inventory_tasks(strategy, include_region_proposals=False):
    """Resolve v0.1.2 inventory strategies while accepting v0.1.1 aliases."""
    aliases = {
        "semantic": "semantic_plus_dense",
        "complete": "semantic_plus_regions",
    }
    resolved = aliases.get(str(strategy), str(strategy))
    if resolved == "semantic_only":
        return ["<OD>"]
    if resolved == "semantic_plus_dense":
        return ["<OD>", "<DENSE_REGION_CAPTION>"]
    if resolved == "semantic_plus_regions":
        tasks = ["<OD>", "<DENSE_REGION_CAPTION>"]
        if include_region_proposals:
            tasks.append("<REGION_PROPOSAL>")
        return tasks
    raise ValueError(f"未知 Scene Inventory strategy: {strategy}")


def _is_generic_label(label):
    return bool(re.fullmatch(r"(?:object|region)_\d{3}", str(label).lower()))


def _stable_object_semantic(item):
    """OD/grounded labels count as stable even when they are outside our small taxonomy."""
    if not item.get("semantic"):
        return False
    label = str(item.get("label", "")).lower()
    if label in CANONICAL_TERMS:
        return True
    source = str(item.get("discovery_source", item.get("source", ""))).lower()
    raw = str(item.get("raw_label", "")).strip().lower()
    vague = not raw or raw in {"object", "item", "thing", "scene", "background", "region"}
    return source in SEMANTIC_INVENTORY_SOURCES and not vague


def _mask_geometry(mask, image_area):
    mask = np.asarray(mask, dtype=bool)
    area = int(mask.sum())
    if area == 0:
        return {
            "area_ratio": 0.0, "bbox_ratio": 0.0, "bbox_fill": 0.0,
            "aspect_ratio": float("inf"), "edge_count": 0, "fragment_score": 1.0,
        }
    ys, xs = np.nonzero(mask)
    box_width = int(xs.max() - xs.min() + 1)
    box_height = int(ys.max() - ys.min() + 1)
    bbox_area = max(1, box_width * box_height)
    bbox_fill = float(area / bbox_area)
    aspect = float(max(box_width, box_height) / max(1, min(box_width, box_height)))
    mask_edge_count = int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())
    margin_x = max(1, int(mask.shape[1] * 0.01))
    margin_y = max(1, int(mask.shape[0] * 0.01))
    bbox_edge_count = (
        int(xs.min() <= margin_x) + int(xs.max() >= mask.shape[1] - 1 - margin_x)
        + int(ys.min() <= margin_y) + int(ys.max() >= mask.shape[0] - 1 - margin_y)
    )
    edge_count = max(mask_edge_count, bbox_edge_count)
    thin_penalty = min(1.0, max(0.0, (aspect - 4.0) / 8.0))
    sparse_penalty = min(1.0, max(0.0, (0.45 - bbox_fill) / 0.45))
    small_penalty = min(1.0, max(0.0, (0.003 - area / image_area) / 0.003))
    fragment_score = 0.45 * sparse_penalty + 0.35 * thin_penalty + 0.20 * small_penalty
    return {
        "area_ratio": float(area / image_area),
        "bbox_ratio": float(bbox_area / image_area),
        "bbox_fill": bbox_fill,
        "aspect_ratio": aspect,
        "edge_count": edge_count,
        "fragment_score": float(np.clip(fragment_score, 0.0, 1.0)),
    }


def _inside_fraction(inner, outer):
    inner_mask = np.asarray(inner, dtype=bool)
    area = int(inner_mask.sum())
    if area == 0:
        return 0.0
    return float(np.logical_and(inner_mask, np.asarray(outer, dtype=bool)).sum() / area)


def filter_clean_objects(elements, image_size, allow_person_subelements=False, prefer_whole_objects=True):
    """Apply the v0.1.2 complete-object policy to already-segmented candidates.

    This deliberately favours false negatives over exporting a background-shaped false object.
    It mutates no masks; accepted items receive diagnostic scores only.
    """
    width, height = image_size
    image_area = max(1, int(width) * int(height))
    prepared = []
    for source_item in elements:
        item = dict(source_item)
        geometry = _mask_geometry(item["mask"], image_area)
        item.update({
            "fragment_score": round(geometry["fragment_score"], 6),
            "whole_object_score": round(float(np.clip(
                (0.48 if _stable_object_semantic(item) else 0.0)
                + (0.14 if item.get("semantic") else 0.0)
                + 0.12 * float(item.get("sam_score", 0.0) or 0.0)
                + 0.10 * float(item.get("confidence", 0.0) or 0.0)
                + 0.16 * min(1.0, geometry["bbox_fill"] / 0.65)
                - 0.28 * geometry["fragment_score"]
                - 0.025 * geometry["edge_count"],
                0.0, 1.0,
            )), 6),
            "_clean_geometry": geometry,
            "_stable_semantic": _stable_object_semantic(item),
        })
        prepared.append(item)

    people = [item for item in prepared if str(item.get("label", "")).lower() == "person"]
    stable = [item for item in prepared if item["_stable_semantic"]]
    rejected_background = []
    rejected_fragments = []
    suppressed_children = []
    remaining = []

    for item in prepared:
        label = str(item.get("label", "")).lower()
        mask = item["mask"]
        geometry = item["_clean_geometry"]
        generic = _is_generic_label(label) or not item["_stable_semantic"]

        if people and not allow_person_subelements and label != "person":
            inside_person = max((_inside_fraction(mask, person["mask"]) for person in people), default=0.0)
            is_person_child = label in PERSON_CHILD_LABELS or label.startswith("body_") or label.startswith("region_")
            if is_person_child and inside_person >= 0.72:
                suppressed_children.append(item)
                continue

        contained_semantics = sum(
            other is not item
            and _inside_fraction(other["mask"], mask) >= 0.72
            and int(other.get("area", 0)) < int(item.get("area", 0))
            for other in stable
        )
        background_like = generic and (
            geometry["area_ratio"] > 0.46
            or (geometry["area_ratio"] > 0.20 and geometry["edge_count"] >= 3)
            or (geometry["bbox_ratio"] > 0.58 and geometry["edge_count"] >= 3)
            or (geometry["bbox_ratio"] > 0.60 and geometry["area_ratio"] > 0.12)
            or (geometry["area_ratio"] > 0.16 and contained_semantics >= 2)
            or geometry["edge_count"] == 4
        )
        if background_like:
            rejected_background.append(item)
            continue

        overlaps_stable = any(
            other is not item
            and (_inside_fraction(mask, other["mask"]) >= 0.82 or bbox_iou(item["bbox"], other["bbox"]) >= 0.60)
            for other in stable
        )
        fragment_like = generic and (
            (geometry["aspect_ratio"] >= 10.0 and geometry["bbox_fill"] < 0.55)
            or geometry["bbox_fill"] < 0.07
            or geometry["fragment_score"] >= 0.76
            or (geometry["area_ratio"] < 0.003 and overlaps_stable)
        )
        if fragment_like:
            rejected_fragments.append(item)
            continue
        if generic and overlaps_stable:
            rejected_fragments.append(item)
            continue
        remaining.append(item)

    if prefer_whole_objects:
        ordered = sorted(
            remaining,
            key=lambda item: (
                str(item.get("label", "")).lower() == "person",
                int(item.get("area", 0)) if str(item.get("label", "")).lower() == "person" else 0,
                float(item["whole_object_score"]),
                int(item.get("area", 0)),
            ),
            reverse=True,
        )
        whole = []
        for item in ordered:
            same_label_overlap = any(
                item.get("label") == old.get("label")
                and (
                    _inside_fraction(item["mask"], old["mask"]) >= 0.82
                    or _inside_fraction(old["mask"], item["mask"]) >= 0.82
                    or bbox_iou(item["bbox"], old["bbox"]) >= 0.58
                )
                for old in whole
            )
            if same_label_overlap:
                rejected_fragments.append(item)
            else:
                whole.append(item)
        remaining = whole

    for item in remaining:
        item.pop("_clean_geometry", None)
        item.pop("_stable_semantic", None)
    diagnostics = {
        "background_like_rejected_count": len(rejected_background),
        "fragment_rejected_count": len(rejected_fragments),
        "parent_child_suppressed_count": len(suppressed_children),
        "background_like_rejected_labels": [item.get("label") for item in rejected_background],
        "fragment_rejected_labels": [item.get("label") for item in rejected_fragments],
        "parent_child_suppressed_labels": [item.get("label") for item in suppressed_children],
    }
    return remaining, diagnostics


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
