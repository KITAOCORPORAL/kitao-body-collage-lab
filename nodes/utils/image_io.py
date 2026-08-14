"""PIL/NumPy/ComfyUI image conversions."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def load_image_file(image_path):
    """Load a local image with EXIF orientation applied and return RGB pixels."""
    path = Path(image_path).expanduser().resolve()
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(f"KBL 不支持该图片格式: {path.suffix}。支持: {supported}")
    if not path.is_file():
        raise FileNotFoundError(f"KBL 找不到输入图片: {path}")

    with Image.open(path) as opened:
        oriented = ImageOps.exif_transpose(opened)
        rgb = oriented.convert("RGB")
        image = rgb.copy()

    width, height = image.size
    meta = {
        "filename": path.name,
        "source_path": str(path),
        "width": width,
        "height": height,
    }
    return image, meta


def pil_to_comfy_image(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def comfy_image_to_pil(image):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError("KBL 需要形状为 [B,H,W,C] 的 ComfyUI IMAGE")
    array = image[0, :, :, :3].detach().cpu().clamp(0, 1).numpy()
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")


def mask_array_to_tensor(mask):
    return torch.from_numpy(np.asarray(mask, dtype=np.float32))

