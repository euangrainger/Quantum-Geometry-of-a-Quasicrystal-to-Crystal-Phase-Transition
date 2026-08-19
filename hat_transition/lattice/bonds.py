"""
Bond classification utilities.

Adds an inner-outer binary classificaion to the dual, tetrille tags for further transitions. 
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


def classify_inner_outer(
    n_sites: int,
    bonds: np.ndarray,
    inner_degree: int = 6,
) -> Dict[Tuple[int, int], str]:
    """
    Classify bonds as 'inner' or 'outer' based on vertex degree.

    A bond is 'inner' if either endpoint has exactly `inner_degree`
    connections (i.e. it is a hexagon-center vertex in the tetrille).
    """
    degree = np.zeros(n_sites, dtype=int)
    for a, b in bonds:
        degree[a] += 1
        degree[b] += 1

    is_inner_site = degree == inner_degree
    io_labels = {}
    for a, b in bonds:
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if is_inner_site[a] or is_inner_site[b]:
            io_labels[key] = "inner"
        else:
            io_labels[key] = "outer"
    return io_labels


def build_bond_labels(
    bond_labels_ab: Dict[Tuple[int, int], str],
    bond_labels_io: Dict[Tuple[int, int], str],
) -> Dict[Tuple[int, int], Dict[str, str]]:
    """
    Merge ab and io classifications into a single dict.

    Returns:
        {(i, j): {'ab': 'dual'|'tetrille', 'io': 'inner'|'outer'}}
    """
    combined = {}
    for key, ab in bond_labels_ab.items():
        io = bond_labels_io.get(key, "outer")
        combined[key] = {"ab": ab, "io": io}
    return combined
