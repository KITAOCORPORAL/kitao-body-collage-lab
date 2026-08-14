"""Download complete KBL Transformers repositories to N drive."""

import os
import sys
from pathlib import Path

CACHE_ROOT = Path("N:/Comfy-Desktop/HF_Cache")
MODEL_ROOT = Path("N:/Comfy-Desktop/ComfyUI-Shared/models/Kitao_Body_Collage_Lab")

os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_ROOT / "transformers")

from huggingface_hub import snapshot_download

from model_integrity import MODEL_REQUIREMENTS, check_model, repository_size, require_models

REPOSITORIES = {
    "GroundingDINO": "IDEA-Research/grounding-dino-tiny",
    "SAM2": "facebook/sam2.1-hiera-small",
    "DWPose": "yzd-v/DWPose",
}


def install_model(model_name, repo_id):
    definition = MODEL_REQUIREMENTS[model_name]
    target = MODEL_ROOT / definition["directory"]
    target.mkdir(parents=True, exist_ok=True)
    _, missing = check_model(MODEL_ROOT, model_name)
    if not missing:
        print(f"[已完整] {model_name}: {target}")
        return
    print(f"[下载] {repo_id} -> {target}")
    download_args = {
        "repo_id": repo_id,
        "local_dir": target,
        "cache_dir": CACHE_ROOT / "hub",
        "max_workers": 4,
    }
    if model_name == "DWPose":
        download_args["allow_patterns"] = ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"]
    else:
        download_args["ignore_patterns"] = ["pytorch_model.bin", "*.pt"]
    snapshot_download(**download_args)


def main():
    print(f"Python: {sys.executable}")
    print(f"HF_HOME: {os.environ['HF_HOME']}")
    print(f"HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")
    print(f"TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for model_name, repo_id in REPOSITORIES.items():
        install_model(model_name, repo_id)
    require_models(MODEL_ROOT)
    print("\n模型完整性检查通过：")
    for model_name, definition in MODEL_REQUIREMENTS.items():
        directory = MODEL_ROOT / definition["directory"]
        print(f"- {model_name}: {directory}")
        print(f"  文件数: {sum(1 for path in directory.rglob('*') if path.is_file())}")
        print(f"  总大小: {repository_size(directory) / (1024 ** 2):.2f} MiB")


if __name__ == "__main__":
    main()
