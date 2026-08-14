"""Unified release-readiness checks for KBL v0.1.0."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from Kitao_Body_Collage_Lab.nodes.utils.path_config import get_path_config
from Kitao_Body_Collage_Lab.version import KBL_VERSION

try:
    from .model_integrity import MODEL_REQUIREMENTS, check_model
except ImportError:  # Direct execution: python scripts/kbl_doctor.py
    from model_integrity import MODEL_REQUIREMENTS, check_model


DEFAULT_DEPLOYMENT = Path("N:/comfyui/ComfyUI-Installs/ComfyUI/ComfyUI/custom_nodes/Kitao_Body_Collage_Lab")


def _model_status(model_root, name):
    directory, missing = check_model(model_root, name)
    return {"status": "PASS" if not missing else "FAILED", "detail": str(directory), "missing": missing}


def run_doctor():
    config = get_path_config()
    model_root = Path(config["model_root"]) / "Kitao_Body_Collage_Lab"
    export_root = Path(config["export_root"])
    checks = {
        "Python": {"status": "PASS", "detail": sys.version.split()[0]},
    }
    try:
        import torch
        checks["Torch"] = {"status": "PASS", "detail": str(torch.__version__)}
        checks["CUDA"] = {
            "status": "PASS" if torch.cuda.is_available() else "FAILED",
            "detail": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE",
        }
    except Exception as exc:
        checks["Torch"] = {"status": "FAILED", "detail": str(exc)}
        checks["CUDA"] = {"status": "FAILED", "detail": "Torch unavailable"}
    for name in MODEL_REQUIREMENTS:
        checks[name] = _model_status(model_root, name)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
        if checks["DWPose"]["status"] == "PASS":
            checks["DWPose"]["detail"] += f" — {provider}"
    except Exception as exc:
        checks["DWPose"] = {"status": "FAILED", "detail": f"onnxruntime unavailable: {exc}", "missing": []}
    birefnet = model_root / "birefnet"
    checks["BiRefNet"] = {
        "status": "OPTIONAL / INSTALLED" if birefnet.is_dir() and any(birefnet.iterdir()) else "OPTIONAL / NOT INSTALLED",
        "detail": str(birefnet),
    }
    try:
        export_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="kbl_doctor_", suffix=".tmp", dir=export_root, delete=True) as probe:
            probe.write(b"KBL")
            probe.flush()
        checks["Output directory"] = {"status": "PASS", "detail": str(export_root)}
    except Exception as exc:
        checks["Output directory"] = {"status": "FAILED", "detail": str(exc)}
    deployment = Path(os.environ.get("KBL_CUSTOM_NODE_PATH", str(DEFAULT_DEPLOYMENT))).resolve()
    deployed_version = deployment / "version.py"
    deployment_ok = (deployment / "__init__.py").is_file() and deployed_version.is_file() and f'KBL_VERSION = "{KBL_VERSION}"' in deployed_version.read_text(encoding="utf-8")
    checks["ComfyUI custom node deployment"] = {"status": "PASS" if deployment_ok else "FAILED", "detail": str(deployment)}
    mandatory = ["Python", "Torch", "CUDA", "GroundingDINO", "SAM2", "DWPose", "Output directory", "ComfyUI custom node deployment"]
    overall = "READY" if all(checks[name]["status"] == "PASS" for name in mandatory) else "NOT READY"
    return {"kbl_version": KBL_VERSION, "checks": checks, "overall": overall}


def _print_report(report):
    print("Kitao Body Collage Lab Doctor\n")
    for name, result in report["checks"].items():
        print(f"{name}:")
        print(result["status"] + (f" — {result['detail']}" if result.get("detail") else ""))
        if result.get("missing"):
            for item in result["missing"]:
                print(f"  missing: {item}")
        print()
    print("Overall:")
    print(report["overall"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = run_doctor()
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else "", end="" if args.json else "")
    if not args.json:
        _print_report(report)
    return 0 if report["overall"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
