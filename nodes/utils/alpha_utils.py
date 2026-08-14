"""Non-generative binary-mask cleanup and straight-alpha helpers."""

import cv2
import numpy as np
from PIL import Image, ImageFilter


def mask_bbox(mask):
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if not len(xs):
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _remove_small_components(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    kept = np.zeros_like(mask, dtype=bool)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area):
            kept |= labels == index
    return kept


def _fill_small_holes(mask, max_area):
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    filled = mask.copy()
    height, width = mask.shape
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        touches_edge = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_edge and int(area) <= int(max_area):
            filled[labels == index] = True
    return filled


def refine_binary_mask(
    raw_mask,
    expand_px=1,
    erode_px=0,
    min_component_area=16,
    fill_small_holes=True,
    remove_small_islands=True,
):
    """Conservative topology cleanup. The input array is never modified."""
    mask = np.asarray(raw_mask, dtype=bool).copy()
    if remove_small_islands and min_component_area > 1:
        mask = _remove_small_components(mask, min_component_area)
    if fill_small_holes and min_component_area > 0:
        mask = _fill_small_holes(mask, min_component_area)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    if int(expand_px) > 0:
        radius = int(expand_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
    if int(erode_px) > 0:
        radius = int(erode_px)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return mask


def soft_alpha_from_binary(binary_mask, feather_px):
    """Feather only the boundary band while preserving a fully opaque interior."""
    binary = np.asarray(binary_mask, dtype=bool)
    if feather_px <= 0:
        return binary.astype(np.float32)
    image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(float(feather_px))), dtype=np.float32) / 255.0
    interior_distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 5)
    blurred[interior_distance >= max(1.0, float(feather_px) * 2.0)] = 1.0
    blurred[blurred < (1.0 / 255.0)] = 0.0
    return np.clip(blurred, 0.0, 1.0).astype(np.float32)


def rgba_from_source(source_rgb, alpha_mask):
    """Return straight-alpha RGBA. RGB bytes are copied directly from the source."""
    rgb = np.asarray(source_rgb.convert("RGB"), dtype=np.uint8)
    alpha = np.rint(np.clip(np.asarray(alpha_mask), 0.0, 1.0) * 255.0).astype(np.uint8)
    if alpha.shape != rgb.shape[:2]:
        raise ValueError(f"alpha mask 尺寸 {alpha.shape} 与图像 {rgb.shape[:2]} 不一致")
    return Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
