"""
Bulk-site mask for excluding open-boundary effects from real-space
observables. 
"""

from __future__ import annotations

import numpy as np
import alphashape
from shapely.geometry import Point


def bulk_mask(
    sites: np.ndarray,
    buffer_dist: float = 1.8,
    alpha: float = 0.9,
) -> np.ndarray:
    """
    Create mask for 'bulk' sites by eroding the tiling boundary inward.

    Computes the concave hull (alpha-shape) of the site cloud, then
    shrinks it inward by buffer_dist. Sites inside the eroded polygon
    are bulk and the sites outside are boundary.

    Args:
        sites: (N, 2) site coordinates
        buffer_dist: inward erosion distance (lattice units)
        alpha: alpha-shape parameter controlling hull concavity

    Returns:
        (N,) boolean mask: True for bulk sites.
    """
    pts = np.array(sites)
    shape = alphashape.alphashape(pts, alpha=alpha)
    if hasattr(shape, "geom_type") and shape.geom_type != "Polygon":
        shape = max(shape.geoms, key=lambda g: g.area)
    inner_poly = shape.buffer(-buffer_dist)
    if inner_poly.is_empty:
        return np.zeros(len(sites), dtype=bool)
    mask = np.array([inner_poly.contains(Point(xy)) for xy in pts])
    return mask
