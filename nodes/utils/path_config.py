"""Resolve KBL storage paths without writing outside the selected roots."""

import os
from pathlib import Path

from ...version import KBL_VERSION

DEFAULT_N_ROOT = Path("N:/ComfyUI")
DESKTOP_SHARED_MODELS = Path("N:/Comfy-Desktop/ComfyUI-Shared/models")


def _resolve_path(env_name, candidates):
    configured = os.environ.get(env_name)
    if configured:
        return Path(configured).expanduser()
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_path_config():
    """Return deterministic model and export paths, preferring existing N-drive roots."""
    model_root = _resolve_path(
        "KBL_MODEL_ROOT",
        [DESKTOP_SHARED_MODELS, DEFAULT_N_ROOT / "models"],
    )
    export_root = _resolve_path(
        "KBL_EXPORT_ROOT",
        [DEFAULT_N_ROOT / "output" / "Kitao_Body_Collage_Lab"],
    )
    return {
        "version": KBL_VERSION,
        "model_root": str(model_root),
        "export_root": str(export_root),
    }
