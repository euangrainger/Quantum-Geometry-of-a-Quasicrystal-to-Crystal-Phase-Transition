"""
Programmatic lattice generation from kite-partitioned hat tiling.

Tiling generation with the unit hat modified to include additional vertices 
and bonds for the kite decomposition. Extracts sites, bonds, and classifications 
directly.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from hat_transition._tiling import (
    HatTile, MetaTile, hat_outline, mul, trans_pt, generate_hat_tiling,
)

KITE_BONDS = [
    [0, 1, 2, 13],
    [0, 13, 3, 4],
    [0, 4, 14, 16],
    [0, 16, 11, 12],
    [9, 10, 11, 16],
    [9, 16, 14, 8],
    [5, 6, 7, 8],
    [5, 8, 14, 4],
]

_HAT_EDGE_SET = set()
for _i in range(13):
    _j = (_i + 1) % 13
    _HAT_EDGE_SET.add((min(_i, _j), max(_i, _j)))

_POLYGON_ORDER = [0, 1, 2, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_POLYGON_EDGE_SET = set()
for _k in range(len(_POLYGON_ORDER)):
    _a = _POLYGON_ORDER[_k]
    _b = _POLYGON_ORDER[(_k + 1) % len(_POLYGON_ORDER)]
    _POLYGON_EDGE_SET.add((min(_a, _b), max(_a, _b)))

ROUND_PRECISION = 3

@dataclass
class KiteData:
    """A single kite quadrilateral in world space."""
    vertices: np.ndarray          
    center: Tuple[float, float]   
    bond_type: str               
    label: str                    
    generated: bool = False       
    hat_indices: Tuple[int, ...] = ()  
    tile_id: int = -1            


@dataclass
class LatticeData:
    """Complete lattice data for simulation input."""
    sites: np.ndarray
    bonds: np.ndarray
    bond_labels: Dict[Tuple[int, int], str]
    site_labels: Dict[int, str]
    sublattice: np.ndarray
    hex_centres: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int))
    kites: List[KiteData] = field(default_factory=list)
    n_original_kites: int = 0
    n_generated_kites: int = 0

def _classify_kite_bond(bond: List[int]) -> str:
    """Classify a kite's bond type from hat edge membership."""
    hat_count = sum(
        1 for k in range(4)
        if (min(bond[k], bond[(k + 1) % 4]), max(bond[k], bond[(k + 1) % 4]))
        in _HAT_EDGE_SET
    )
    if hat_count == 4:
        return "hat"
    elif hat_count == 0:
        return "tetrille"
    return "dual"

def _collect_kites(tile, transform, kites_out: List[KiteData],
                   _tile_counter: List[int] | None = None):
    """Recursively extract all kites from the tile hierarchy."""
    if _tile_counter is None:
        _tile_counter = [0]
    if isinstance(tile, HatTile):
        tid = _tile_counter[0]
        _tile_counter[0] += 1
        for bond in KITE_BONDS:
            verts = np.array([trans_pt(transform, hat_outline[i]) for i in bond])
            kites_out.append(KiteData(
                vertices=verts,
                center=(verts[0, 0], verts[0, 1]),
                bond_type=_classify_kite_bond(bond),
                label=tile.label,
                generated=False,
                hat_indices=tuple(bond),
                tile_id=tid,
            ))
        return
    for childT, childGeom in tile.children:
        _collect_kites(childGeom, mul(transform, childT), kites_out,
                       _tile_counter)

def _round_pt(p, prec=ROUND_PRECISION):
    return (round(p[0], prec), round(p[1], prec))


def _kite_sector(kite: KiteData) -> int:
    """60-degree sector (0..5) from angle to opposite vertex."""
    cx, cy = kite.center
    ox, oy = kite.vertices[2]
    angle_deg = math.degrees(math.atan2(oy - cy, ox - cx)) % 360
    return round(angle_deg / 60) % 6


def _rotate_kite(kite: KiteData, angle_rad: float) -> KiteData:
    """Rotate kite by angle around its center."""
    cx, cy = kite.center
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    new_verts = np.empty_like(kite.vertices)
    for i, (x, y) in enumerate(kite.vertices):
        dx, dy = x - cx, y - cy
        new_verts[i] = (cos_a * dx - sin_a * dy + cx,
                        sin_a * dx + cos_a * dy + cy)
    return KiteData(
        vertices=new_verts,
        center=(float(new_verts[0, 0]), float(new_verts[0, 1])),
        bond_type="boundary",
        label="boundary",
        generated=True,
    )


def _complete_hexagons(kites: List[KiteData], min_kites: int = 1):
    """Complete boundary hexagons via 60-degree rotation.

    Finite tiling patches have incomplete hexagons at the boundary
    (fewer than 6 kites). This fills them by rotating an existing
    kite into the missing sectors, labelled as 'boundary'."""
    groups: Dict[Tuple, List[KiteData]] = {}
    for k in kites:
        key = _round_pt(k.center)
        groups.setdefault(key, []).append(k)

    completed = []
    n_generated = 0
    for center, group in groups.items():
        if len(group) >= 6:
            completed.extend(group)
            continue
        if len(group) < min_kites:
            completed.extend(group)
            continue
        occupied = {}
        for k in group:
            occupied[_kite_sector(k)] = k
        ref_sector, ref_kite = next(iter(occupied.items()))
        for s in range(6):
            if s in occupied:
                completed.append(occupied[s])
            else:
                rotation = (s - ref_sector) * math.pi / 3
                completed.append(_rotate_kite(ref_kite, rotation))
                n_generated += 1
    return completed, n_generated

def _extract_sites_and_bonds(kites: List[KiteData], atol: float = 1e-3):
    """
    Extract unique sites and bonds from kite list.

    Returns:
        sites: (N, 2) array
        bonds: (M, 2) int array of site index pairs
        bond_labels: dict mapping (i, j) sorted -> 'dual' | 'tetrille'
        site_labels: dict mapping index -> 'dual' | 'tetrille'
        hex_centres: sorted indices of the hexagon-centre sites
    """
    site_map: Dict[Tuple[int, int], int] = {}
    site_coords: List[np.ndarray] = []
    site_on_hat = set()
    hex_centres = set()

    def _site_key(x, y):
        return (round(x / atol), round(y / atol))

    def _get_or_add(x, y):
        k = _site_key(x, y)
        if k not in site_map:
            site_map[k] = len(site_coords)
            site_coords.append(np.array([x, y]))
        return site_map[k]

    edge_set: Dict[Tuple[int, int], str] = {}

    for kite in kites:
        indices = []
        for v in kite.vertices:
            indices.append(_get_or_add(v[0], v[1]))
        hex_centres.add(indices[0])

        if kite.bond_type in ("hat", "dual"):
            for idx in indices:
                site_on_hat.add(idx)

        hat_idx = kite.hat_indices  
        for i in range(4):
            j = (i + 1) % 4
            a, b = sorted([indices[i], indices[j]])
            key = (a, b)

            if hat_idx and not kite.generated:
                hi, hj = hat_idx[i], hat_idx[j]
                edge_on_polygon = (min(hi, hj), max(hi, hj)) in _POLYGON_EDGE_SET
            else:
                edge_on_polygon = False

            if key not in edge_set:
                edge_set[key] = "dual" if edge_on_polygon else "tetrille"
            elif edge_on_polygon:
                edge_set[key] = "dual"

    sites = np.array(site_coords)
    bonds = np.array(sorted(edge_set.keys()), dtype=int)
    bond_labels = {k: v for k, v in edge_set.items()}
    site_labels = {
        i: ("dual" if i in site_on_hat else "tetrille")
        for i in range(len(sites))
    }

    return sites, bonds, bond_labels, site_labels, np.array(
        sorted(hex_centres), dtype=int)


def generate_lattice(
    level: int = 2,
    tile: str = "H",
    min_kites: int = 1,
) -> LatticeData:
    """
    Generate a complete lattice from the kite-partitioned hat tiling.

    Args:
        level: substitution level (1 = base, 2 = one substitution, ...)
        to note, level 1 is the base metalite (so H1 has 4 hats etc) as
        a choice of labelling. 

    Returns:
        LatticeData with sites, bonds, labels, and sublattice assignment
    """
    from .bipartite import sublattice_assignment

    tiles = generate_hat_tiling(level)
    tile_idx = {"H": 0, "T": 1, "P": 2, "F": 3}[tile]
    meta_tile = tiles[tile_idx]

    kites: List[KiteData] = []
    _collect_kites(meta_tile, [1, 0, 0, 0, 1, 0], kites)
    n_original = len(kites)

    kites, n_generated = _complete_hexagons(kites, min_kites=min_kites)
    sites, bonds, bond_labels, site_labels, hex_centres = \
        _extract_sites_and_bonds(kites)
    sublattice, _ = sublattice_assignment(
        len(sites), bonds, reference_sites=hex_centres)

    return LatticeData(
        sites=sites,
        bonds=bonds,
        bond_labels=bond_labels,
        site_labels=site_labels,
        sublattice=sublattice,
        hex_centres=hex_centres,
        kites=kites,
        n_original_kites=n_original,
        n_generated_kites=n_generated,
    )
