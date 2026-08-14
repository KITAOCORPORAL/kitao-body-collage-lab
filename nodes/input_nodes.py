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


class KBLLoadImagePicker:
    """Use the exact upload/input selection protocol of the installed ComfyUI."""
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        files = [name for name in folder_paths.recursive_search(input_dir)[0]]
        return {"required": {"image": (sorted(files), {"image_upload": True})}}

    RETURN_TYPES = KBLLoadImage.RETURN_TYPES
    RETURN_NAMES = KBLLoadImage.RETURN_NAMES
    FUNCTION = "load"
    CATEGORY = "Kitao Body Collage/输入"
    DESCRIPTION = "从 ComfyUI Shared Input 选择或上传图片。"

    def load(self, image):
        import folder_paths
        return KBLLoadImage().load(folder_paths.get_annotated_filepath(image))

    @classmethod
    def IS_CHANGED(cls, image):
        import folder_paths
        return folder_paths.get_annotated_filepath(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        import folder_paths
        return True if folder_paths.exists_annotated_filepath(image) else f"Invalid image file: {image}"

NODE_CLASS_MAPPINGS = {"KBL_Load_Image": KBLLoadImage, "KBL_Load_Image_Picker": KBLLoadImagePicker}
NODE_DISPLAY_NAME_MAPPINGS = {"KBL_Load_Image": "KBL 加载图片", "KBL_Load_Image_Picker": "KBL 选择图片"}
