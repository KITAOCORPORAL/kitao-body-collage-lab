"""High-resolution local image input node."""

import json

from .utils.image_io import load_image_file, pil_to_comfy_image


class KBLLoadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_path": ("STRING", {"default": "N:\\Comfy-Desktop\\ComfyUI-Shared\\input\\photo.jpg"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "image_meta", "source_path", "width", "height", "filename")
    FUNCTION = "load"
    CATEGORY = "Kitao Body Collage/输入"
    DESCRIPTION = "读取 JPG/JPEG/PNG/WEBP，应用 EXIF 方向并保持原始分辨率。"

    def load(self, image_path):
        image, meta = load_image_file(image_path)
        return (
            pil_to_comfy_image(image),
            json.dumps(meta, ensure_ascii=False),
            meta["source_path"],
            meta["width"],
            meta["height"],
            meta["filename"],
        )

NODE_CLASS_MAPPINGS = {"KBL_Load_Image": KBLLoadImage}
NODE_DISPLAY_NAME_MAPPINGS = {"KBL_Load_Image": "KBL 加载图片"}
