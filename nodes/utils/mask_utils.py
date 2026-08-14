"""Independent-mask filtering and deduplication."""

import numpy as np
import torch

REQUIRED_ELEMENT_FIELDS = {"id", "label", "mask", "bbox", "confidence", "area", "source"}


def mask_iou(mask_a, mask_b):
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def mask_containment(mask_a, mask_b):
    """Return the fraction of the smaller mask covered by the other mask."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    smaller = min(int(a.sum()), int(b.sum()))
    if smaller == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / smaller)


def deduplicate_elements(elements, iou_threshold, min_mask_area, max_candidates):
    eligible = [item for item in elements if int(item["area"]) >= int(min_mask_area)]
    eligible.sort(
        key=lambda item: (
            item.get("source") == "guided",
            float(item.get("confidence", 0.0)),
            int(item.get("area", 0)),
        ),
        reverse=True,
    )

    kept = []
    for candidate in eligible:
        duplicate = False
        for existing in kept:
            iou = mask_iou(candidate["mask"], existing["mask"])
            same_label = candidate["label"] == existing["label"]
            contained = same_label and mask_containment(candidate["mask"], existing["mask"]) >= 0.95
            if iou >= float(iou_threshold) or contained:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= int(max_candidates):
            break
    return kept


def assign_element_ids(elements):
    counters = {}
    for item in elements:
        label = item["label"]
        counters[label] = counters.get(label, 0) + 1
        item["id"] = f"{label}_{counters[label]:02d}"
    return elements


def masks_to_batch(elements, height, width):
    if not elements:
        return torch.zeros((1, height, width), dtype=torch.float32)
    return torch.stack(
        [torch.from_numpy(np.asarray(item["mask"], dtype=np.float32)) for item in elements],
        dim=0,
    )


def validate_elements(elements, height, width):
    """Freeze the KBL_ELEMENTS v0.1 contract at the production boundary."""
    for index, item in enumerate(elements):
        missing = REQUIRED_ELEMENT_FIELDS.difference(item)
        if missing:
            raise ValueError(f"KBL_ELEMENTS[{index}] 缺少字段: {', '.join(sorted(missing))}")
        mask = np.asarray(item["mask"])
        if mask.shape != (height, width):
            raise ValueError(
                f"KBL_ELEMENTS[{index}] mask 尺寸 {mask.shape} 与原图 {(height, width)} 不一致"
            )
        if len(item["bbox"]) != 4:
            raise ValueError(f"KBL_ELEMENTS[{index}] bbox 必须包含 4 个坐标")
        if int(item["area"]) != int(np.count_nonzero(mask)):
            raise ValueError(f"KBL_ELEMENTS[{index}] area 与 mask 非零面积不一致")
    return elements
