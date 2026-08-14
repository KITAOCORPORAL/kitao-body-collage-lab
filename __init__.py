"""ComfyUI node registration for Kitao Body Collage Lab."""

from .version import KBL_VERSION
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = KBL_VERSION

__all__ = ["KBL_VERSION", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
