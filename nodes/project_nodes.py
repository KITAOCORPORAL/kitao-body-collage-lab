"""Installation diagnostics used before the inference nodes are connected."""

import json

from .utils.path_config import get_path_config


class KBLProjectInfo:
    """Expose resolved storage paths so an installation can be checked in ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("version", "model_root", "export_root")
    FUNCTION = "inspect"
    CATEGORY = "Kitao Body Collage/系统"
    DESCRIPTION = "显示 KBL 版本与当前解析到的 N 盘模型、导出目录。"

    def inspect(self):
        config = get_path_config()
        print("[KBL] " + json.dumps(config, ensure_ascii=False))
        return config["version"], config["model_root"], config["export_root"]

