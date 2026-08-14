"""KBL local model repository integrity checks."""

from pathlib import Path

MODEL_REQUIREMENTS = {
    "GroundingDINO": {
        "directory": "grounding_dino",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.txt",
            "model.safetensors",
        ],
    },
    "SAM2": {
        "directory": "sam2",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "model.safetensors",
        ],
    },
    "DWPose": {
        "directory": "dwpose",
        "files": ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"],
    },
    "Florence-2": {
        "directory": "florence2",
        "files": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "model.safetensors",
            "configuration_florence2.py",
            "modeling_florence2.py",
            "processing_florence2.py",
            "kbl_model_revision.json",
        ],
    },
}


def check_model(model_root, model_name):
    definition = MODEL_REQUIREMENTS[model_name]
    directory = Path(model_root) / definition["directory"]
    missing = [name for name in definition["files"] if not (directory / name).is_file()]
    return directory, missing


def require_models(model_root):
    failures = []
    for model_name in MODEL_REQUIREMENTS:
        directory, missing = check_model(model_root, model_name)
        if missing:
            failures.append(
                "\n".join(
                    [
                        "[缺少模型]",
                        f"模型：{model_name}",
                        f"目录：{directory}",
                        "缺少：",
                        *[f"- {name}" for name in missing],
                    ]
                )
            )
    if failures:
        raise FileNotFoundError("\n\n".join(failures))


def print_model_report(model_root):
    """Print exact missing files using the same format as the installer."""
    root = Path(model_root)
    for model_name in MODEL_REQUIREMENTS:
        directory, missing = check_model(root, model_name)
        if missing:
            print("[缺少 DWPose 模型]" if model_name == "DWPose" else "[缺少模型]")
            print(f"模型：{model_name}")
            print(f"期待目录：{directory}" if model_name == "DWPose" else f"目录：{directory}")
            print("缺少：")
            for name in missing:
                print(f"- {name}")
        else:
            print(f"[模型完整] {model_name}: {directory}")


def repository_size(directory):
    return sum(path.stat().st_size for path in Path(directory).rglob("*") if path.is_file())


if __name__ == "__main__":
    print_model_report(Path("N:/Comfy-Desktop/ComfyUI-Shared/models/Kitao_Body_Collage_Lab"))
