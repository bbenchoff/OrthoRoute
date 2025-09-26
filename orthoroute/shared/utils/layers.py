"""
Layer utility functions for consistent layer representation throughout OrthoRoute.

Provides canonical KiCad-style layer name handling to eliminate int/string conversion issues.
"""
import re
from typing import Union

LAYER_NAME_RE = re.compile(r"^(F\.Cu|B\.Cu|In\d+\.Cu)$")

def norm_layer(layer: Union[int, str]) -> str:
    """
    Return a canonical KiCad-style layer name string.

    Args:
        layer: Either an int index or string name

    Returns:
        Canonical layer name (e.g., "F.Cu", "In1.Cu", "B.Cu")

    Raises:
        ValueError: If layer cannot be normalized
    """
    if isinstance(layer, str):
        s = layer.strip()
        if LAYER_NAME_RE.match(s):
            return s
        raise ValueError(f"Unknown layer string: {layer}")

    # Map integer layer indices to KiCad names
    IDX2NAME = {
        0: "F.Cu",
        1: "In1.Cu",
        2: "In2.Cu",
        3: "In3.Cu",
        4: "In4.Cu",
        5: "B.Cu",
    }

    if layer in IDX2NAME:
        return IDX2NAME[layer]
    raise ValueError(f"Unknown layer index: {layer}")

def get_layer_stackup():
    """Get the standard layer stackup for visibility filtering."""
    return {"F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"}