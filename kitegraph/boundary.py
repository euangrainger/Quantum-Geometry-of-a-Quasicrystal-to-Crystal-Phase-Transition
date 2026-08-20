"""
Boundary (excess) construction for both parent configurations.

Both open-boundary configurations of the transition:

  tetrille excess (periodic parent)
      1. completion: every cell a hat touches is filled to a full 6-kite
         tetrille cell — the minimal boundary, identical to the old
         pipeline's full simulation region;
      2. halo: rings of full tetrille cells grown outward. Ring 1 gives
         every boundary cell at least one whole parent cell outwards
         (where the hat outline was flush with a cell wall there was
         none), ring 2 covers the cells ring 1 itself added; hence the
         default of two rings.

  hat excess (aperiodic parent)
      The outermost hats are left as bare hat tiles, outline (dual)
      edges only with no kite decoration while the interior hats are
      decorated and tetrille-completed inside that rim. The parent hat
      lattice then extends beyond the embedded kite-decorated region.
      Small metatiles cannot support this (every hat is on the rim, no
      interior remains) so construction raises rather than emitting a
      wrong lattice.

The imbalance search adds further whole tetrille cells in groups spread
symmetrically about the patch centroid until the chiral imbalance
N_A - N_B hits the requested target. Tt is deterministic (integer-key
ordering) and meant to be run offline a few times.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .tiling import HAT_VERTICES, HAT_KITES, iter_hats, metatile, trans_pt
from .registry import (CellRegistry, Key, LatticeFrame, NEIGHBOUR_OFFSETS)
from .graph import KiteGraph


# --------------------------------------------------------------------------
# tetrille excess (periodic-parent open patch)
# --------------------------------------------------------------------------

def build_tetrille_parent(level: int = 2, tile: str = "H",
                          halo_rings: int = 2) -> KiteGraph:
    """Decorated hats + completion + halo_rings of tetrille excess."""
    reg = _register_all_hats(level, tile)
    reg.complete_partial_cells()
    for ring in range(1, halo_rings + 1):
        reg.add_halo_ring(ring)
    return KiteGraph.from_registry(
        reg, parent_mode="tetrille",
        meta={"level": level, "tile": tile, "halo_rings": halo_rings})


def _register_all_hats(level: int, tile: str) -> CellRegistry:
    hats = list(iter_hats(metatile(level, tile)))
    frame = LatticeFrame.from_first_hat(hats[0][0])
    reg = CellRegistry(frame)
    for transform, _label in hats:
        reg.register_hat(transform)
    return reg


# --------------------------------------------------------------------------
# hat excess (aperiodic-parent open patch)
# --------------------------------------------------------------------------

def build_hat_parent(level: int = 2, tile: str = "H",
                     rim_layers: int = 1) -> KiteGraph:
    """Interior hats decorated + completed; the outermost rim_layers of
    hats registered outline-only (the parent-hat excess)."""
    hats = list(iter_hats(metatile(level, tile)))
    frame = LatticeFrame.from_first_hat(hats[0][0])
    interior, rim = _split_interior_rim(hats, frame, rim_layers)
    if not interior:
        raise ValueError(
            f"hat-excess boundary needs interior hats, but every hat of "
            f"{tile}{level} sits on the rim (patch too small for "
            f"rim_layers={rim_layers}) — use a larger level")

    reg = CellRegistry(frame)
    for i in interior:
        reg.register_hat(hats[i][0])
    for i in rim:
        reg.register_hat_outline(hats[i][0])
    reg.complete_partial_cells()
    return KiteGraph.from_registry(
        reg, parent_mode="hat",
        meta={"level": level, "tile": tile, "rim_layers": rim_layers,
              "n_interior_hats": len(interior), "n_rim_hats": len(rim)})


# --------------------------------------------------------------------------
# imbalance search
# --------------------------------------------------------------------------

def imbalance_search(level: int = 2, tile: str = "H",
                     target: int = 1, halo_rings: int = 2,
                     group_size: int = 6, max_steps: int = 200
                     ) -> Tuple[KiteGraph, List[Tuple[int, int]]]:
    """Grow symmetric groups of tetrille cells until the imbalance hits
    the target.
    """
    reg = _register_all_hats(level, tile)
    reg.complete_partial_cells()

    added = 0
    trajectory: List[Tuple[int, int]] = []
    imb = _structural_imbalance(reg)
    trajectory.append((added, imb))

    while abs(imb) != target and len(trajectory) <= max_steps:
        candidates = _boundary_candidates(reg)
        if not candidates:
            break
        for key in _symmetric_group(reg, candidates, group_size):
            reg.add_cell(key, role="completion")
            added += 1
        imb = _structural_imbalance(reg)
        trajectory.append((added, imb))

    for ring in range(1, halo_rings + 1):
        reg.add_halo_ring(ring)

    graph = KiteGraph.from_registry(
        reg, parent_mode="tetrille",
        meta={"level": level, "tile": tile, "halo_rings": halo_rings,
              "imbalance_target": target, "cells_added": added,
              "imbalance_final": imb})
    return graph, trajectory


def hat_parent_imbalance_search(level: int = 2, tile: str = "H",
                                target: int = 1, rim_layers: int = 1,
                                group_size: int = 3
                                ) -> Tuple[KiteGraph, List[Tuple[int, int]]]:
    """Hat-parent variant of the search, admit rim hats one symmetric
    group at a time until the chiral imbalance matches the target.
    """
    hats = list(iter_hats(metatile(level, tile)))
    frame = LatticeFrame.from_first_hat(hats[0][0])
    interior, rim = _split_interior_rim(hats, frame, rim_layers)
    if not interior:
        raise ValueError(
            f"hat-excess boundary needs interior hats, but every hat of "
            f"{tile}{level} sits on the rim — use a larger level")

    # rim hats ranked by angle about the interior centroid
    interior_centres = [trans_pt(hats[i][0], HAT_VERTICES[0])
                        for i in interior]
    centroid = np.mean(interior_centres, axis=0)

    def angle(i: int) -> float:
        p = trans_pt(hats[i][0], HAT_VERTICES[0])
        return math.atan2(p[1] - centroid[1], p[0] - centroid[0]) \
            % (2 * math.pi)

    ranked = sorted(rim, key=angle)

    def build(admitted: List[int]) -> KiteGraph:
        reg = CellRegistry(frame)
        for i in interior:
            reg.register_hat(hats[i][0])
        for i in admitted:
            reg.register_hat_outline(hats[i][0])
        reg.complete_partial_cells()
        return KiteGraph.from_registry(
            reg, parent_mode="hat",
            meta={"level": level, "tile": tile, "rim_layers": rim_layers,
                  "n_interior_hats": len(interior),
                  "n_rim_hats": len(admitted)})

    def spread(n: int) -> List[int]:
        """n rim hats spread evenly in angle (all of them for n = full)."""
        if n >= len(ranked):
            return list(ranked)
        stride = len(ranked) / n if n else 1.0
        return [ranked[int(k * stride)] for k in range(n)]

    trajectory: List[Tuple[int, int]] = []
    for n in range(len(ranked), -1, -max(group_size, 1)):
        g = build(spread(n))
        imb = g.imbalance()[0]
        trajectory.append((n, imb))
        if abs(imb) == target:
            return g, trajectory
    return build(spread(len(ranked))), trajectory


def _split_interior_rim(hats, frame: LatticeFrame, rim_layers: int
                        ) -> Tuple[List[int], List[int]]:
    """Partition hat indices into (interior, rim) by cell coverage."""
    hat_cells: List[Set[Key]] = []
    kite_count: Dict[Key, int] = {}
    for transform, _ in hats:
        keys = [frame.key(trans_pt(transform, p)) for p in HAT_VERTICES]
        cells = set()
        for (c, _m1, _v, _m2) in HAT_KITES:
            cells.add(keys[c])
            kite_count[keys[c]] = kite_count.get(keys[c], 0) + 1
        hat_cells.append(cells)

    def rim_cell(k: Key) -> bool:
        if kite_count[k] < 6:
            return True
        return any((k[0] + dx, k[1] + dy) not in kite_count
                   for dx, dy in NEIGHBOUR_OFFSETS)

    rim_set = {i for i, cells in enumerate(hat_cells)
               if any(rim_cell(k) for k in cells)}
    for _ in range(rim_layers - 1):
        rim_cells = {k for i in rim_set for k in hat_cells[i]}
        rim_set |= {i for i, cells in enumerate(hat_cells)
                    if cells & rim_cells}
    interior = [i for i in range(len(hats)) if i not in rim_set]
    return interior, sorted(rim_set)


def _structural_imbalance(reg: CellRegistry) -> int:
    """N_A - N_B without building the full graph: A = centres + corners,
    B = midpoints, counted over the union of all kite vertices."""
    from .registry import kite_vertex_keys, vertex_class, CLASS_MIDPOINT
    seen: Set[Key] = set()
    n_a = n_b = 0
    for key, cell in reg.cells.items():
        for sector in cell.kites:
            for vk in kite_vertex_keys(key, sector):
                if vk in seen:
                    continue
                seen.add(vk)
                if vertex_class(vk) == CLASS_MIDPOINT:
                    n_b += 1
                else:
                    n_a += 1
    return n_a - n_b


def _boundary_candidates(reg: CellRegistry) -> List[Key]:
    out: Set[Key] = set()
    for key in reg.cells:
        for dx, dy in NEIGHBOUR_OFFSETS:
            nb = (key[0] + dx, key[1] + dy)
            if nb not in reg.cells:
                out.add(nb)
    return sorted(out)


def _symmetric_group(reg: CellRegistry, candidates: List[Key],
                     group_size: int) -> List[Key]:
    """Pick group_size candidates spread evenly in angle about the
    centroid, so growth stays balanced about the interior."""
    centroid = np.mean([reg.frame.pos(k) for k in reg.cells], axis=0)

    def angle(k: Key) -> float:
        p = reg.frame.pos(k) - centroid
        return math.atan2(p[1], p[0]) % (2 * math.pi)

    ranked = sorted(candidates, key=angle)
    if len(ranked) <= group_size:
        return ranked
    stride = len(ranked) / group_size
    return [ranked[int(i * stride)] for i in range(group_size)]
