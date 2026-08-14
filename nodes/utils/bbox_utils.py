"""Bounding-box normalization helpers."""


def clamp_box(box, width, height):
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = max(0.0, min(x1, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    x2 = max(x1, min(x2, float(width)))
    y2 = max(y1, min(y2, float(height)))
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def mask_bbox(mask):
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]

