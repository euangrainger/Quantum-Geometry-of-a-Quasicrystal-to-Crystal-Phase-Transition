"""
Integer lattice frame and the hexagon-cell registry.

Every vertex of the kite lattice sits on a refinement of the triangular
lattice of hexagon centres.  Writing positions as alpha*A1 + beta*A2 in
the basis (A1, A2) of neighbouring-centre vectors (60 degrees apart) and
scaling by 6, each vertex has an exact integer key whose residue mod 6
identifies its structural class:

    centre    (0, 0)
    midpoint  (3, 0), (0, 3), (3, 3)
    corner    (2, 2), (4, 4)


The registry itself is a map from hexagon cells (centre keys) to the
kites they contain.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .tiling import HAT_VERTICES, HAT_KITES, HAT_CENTRES, outline_edges, \
    trans_pt

Key = Tuple[int, int]

CORNER_OFFSETS: List[Key] = [(2, 2), (-2, 4), (-4, 2), (-2, -2), (2, -4), (4, -2)]
MIDPOINT_OFFSETS: List[Key] = [(3, 0), (0, 3), (-3, 3), (-3, 0), (0, -3), (3, -3)]
NEIGHBOUR_OFFSETS: List[Key] = [(6, 0), (0, 6), (-6, 6), (-6, 0), (0, -6), (6, -6)]

CLASS_CENTRE, CLASS_CORNER, CLASS_MIDPOINT = 0, 1, 2

_RESIDUE_CLASS = {
    (0, 0): CLASS_CENTRE,
    (2, 2): CLASS_CORNER, (4, 4): CLASS_CORNER,
    (3, 0): CLASS_MIDPOINT, (0, 3): CLASS_MIDPOINT, (3, 3): CLASS_MIDPOINT,
}

def vertex_class(key: Key) -> int:
    """Structural class of a vertex from its key residue mod 6."""
    cls = _RESIDUE_CLASS.get((key[0] % 6, key[1] % 6))
    if cls is None:
        raise ValueError(f"key {key} is not on the kite lattice")
    return cls


class LatticeFrame:
    """Affine frame mapping world points to exact integer keys.

    The origin is an arbitrary placed hexagon centre and A1 an arbitrary placed
    centre-to-neighbour vector, with A2 = A1 rotated by +60 degrees.
    """

    def __init__(self, origin: Tuple[float, float],
                 a1: Tuple[float, float], snap_tol: float = 1e-6):
        c, s = math.cos(math.pi / 3), math.sin(math.pi / 3)
        a2 = (c * a1[0] - s * a1[1], s * a1[0] + c * a1[1])
        self.origin = np.asarray(origin, dtype=float)
        self.basis = np.column_stack([a1, a2])          # world <- (alpha, beta)
        self.basis_inv = np.linalg.inv(self.basis)
        self.snap_tol = snap_tol

    @classmethod
    def from_first_hat(cls, transform) -> "LatticeFrame":
        """Calibrate from one placed hat."""
        c0 = trans_pt(transform, HAT_VERTICES[0])
        m13 = trans_pt(transform, HAT_VERTICES[13])
        a1 = (2 * (m13[0] - c0[0]), 2 * (m13[1] - c0[1]))
        return cls(c0, a1)

    def key(self, p: Tuple[float, float]) -> Key:
        """Snap a world point to its integer key; error if off-lattice."""
        ab = 6.0 * (self.basis_inv @ (np.asarray(p, dtype=float) - self.origin))
        k = (int(round(ab[0])), int(round(ab[1])))
        err = float(np.hypot(ab[0] - k[0], ab[1] - k[1]))
        if err > self.snap_tol * 6.0:
            raise ValueError(f"point {p} is {err:.2e} (6x units) off-lattice")
        vertex_class(k)  # validates the residue as a side effect
        return k

    def pos(self, key: Key) -> np.ndarray:
        """Exact world position of a key (float only at the last step)."""
        return self.origin + self.basis @ (np.asarray(key, dtype=float) / 6.0)


@dataclass
class KiteRecord:
    """One kite inside a cell, identified by its 60-degree sector."""
    sector: int
    source: str          # 'hat' | 'completion' | 'halo'
    tile_id: int = -1    # id of the placing hat (-1 for generated kites)
    slot: int = -1       # kite index within HAT_KITES (-1 for generated)


@dataclass
class Cell:
    """One hexagon of the parent (tetrille) lattice."""
    key: Key                                   # centre vertex key
    kites: Dict[int, KiteRecord] = field(default_factory=dict)
    role: str = "interior"                     # interior|completion|halo
    ring: int = 0                              # halo ring index (halo only)

    @property
    def complete(self) -> bool:
        return len(self.kites) == 6

    @property
    def hat_covered(self) -> bool:
        return all(r.source == "hat" for r in self.kites.values()) \
            and self.complete


def kite_vertex_keys(centre: Key, sector: int) -> Tuple[Key, Key, Key, Key]:
    """Role-ordered vertex keys (centre, midpoint, corner, midpoint) of
    the kite in the given sector of the given cell."""
    cx, cy = centre
    m1 = MIDPOINT_OFFSETS[sector]
    v = CORNER_OFFSETS[sector]
    m2 = MIDPOINT_OFFSETS[(sector + 1) % 6]
    return (centre, (cx + m1[0], cy + m1[1]),
            (cx + v[0], cy + v[1]), (cx + m2[0], cy + m2[1]))


def kite_edge_keys(centre: Key, sector: int) -> List[Tuple[Key, Key]]:
    """The four edges of a kite."""
    c, m1, v, m2 = kite_vertex_keys(centre, sector)
    return [(c, m1), (m1, v), (v, m2), (m2, c)]


_SECTOR_BY_OFFSET = {off: s for s, off in enumerate(CORNER_OFFSETS)}


class CellRegistry:
    """Hexagon cells of one tiling patch, with kite provenance.

    Build order: register hats (register_hat / register_hat_outline),
    then boundary passes from boundary.py (complete_partial_cells,
    add_halo_ring), then hand to KiteGraph.from_registry.
    """

    def __init__(self, frame: LatticeFrame):
        self.frame = frame
        self.cells: Dict[Key, Cell] = {}
        # Edges of placed decorated hat outlines, the dual bonds.
        self.dual_edges: Set[frozenset] = set()
        # Hats registered outline-only (hat-excess boundary mode).
        self.outline_only_hats: List[int] = []
        self._n_hats = 0

    # -- hat registration --------------------------------------------------

    def register_hat(self, transform) -> int:
        """Register a fully decorated hat: its 8 kites into their cells,
        and its outline edges as dual bonds.  Returns the hat id."""
        tile_id = self._n_hats
        self._n_hats += 1

        keys = [self.frame.key(trans_pt(transform, p)) for p in HAT_VERTICES]
        for slot, (c, m1, v, m2) in enumerate(HAT_KITES):
            centre = keys[c]
            sector = self._sector(centre, keys[v])
            cell = self.cells.setdefault(centre, Cell(centre))
            if sector in cell.kites:
                raise ValueError(
                    f"cell {centre} sector {sector}: kite already present "
                    f"(hat {cell.kites[sector].tile_id}) — overlapping hats")
            cell.kites[sector] = KiteRecord(sector, "hat", tile_id, slot)
        for a, b in outline_edges():
            self.dual_edges.add(frozenset((keys[a], keys[b])))
        return tile_id

    def register_hat_outline(self, transform) -> int:
        """Register a hat as a bare tile. The outline (dual) edges only, no
        kite decoration.  This is the hat-excess boundary: the parent hat
        lattice extends past the embedded kite-decorated region."""
        tile_id = self._n_hats
        self._n_hats += 1
        keys = [self.frame.key(trans_pt(transform, p)) for p in HAT_VERTICES]
        for a, b in outline_edges():
            self.dual_edges.add(frozenset((keys[a], keys[b])))
        self.outline_only_hats.append(tile_id)
        return tile_id

    def _sector(self, centre: Key, corner: Key) -> int:
        off = (corner[0] - centre[0], corner[1] - centre[1])
        try:
            return _SECTOR_BY_OFFSET[off]
        except KeyError:
            raise ValueError(
                f"corner {corner} is not adjacent to centre {centre}")

    # -- boundary passes (driven by boundary.py) ---------------------------

    def complete_partial_cells(self) -> int:
        """Fill every partially covered cell to a full 6-kite tetrille
        cell.  These 'completion' kites define the minimal boundary."""
        n_added = 0
        for cell in self.cells.values():
            if cell.complete:
                continue
            cell.role = "completion"
            for s in range(6):
                if s not in cell.kites:
                    cell.kites[s] = KiteRecord(s, "completion")
                    n_added += 1
        return n_added

    def add_halo_ring(self, ring: int) -> int:
        """Add one ring of full tetrille cells adjacent to the current
        registry (role 'halo').  Called repeatedly for deeper excess."""
        new_keys: Set[Key] = set()
        for key in self.cells:
            for dx, dy in NEIGHBOUR_OFFSETS:
                nb = (key[0] + dx, key[1] + dy)
                if nb not in self.cells:
                    new_keys.add(nb)
        for key in sorted(new_keys):
            cell = Cell(key, role="halo", ring=ring)
            for s in range(6):
                cell.kites[s] = KiteRecord(s, "halo")
            self.cells[key] = cell
        return len(new_keys)

    def add_cell(self, key: Key, role: str = "halo", ring: int = 0) -> Cell:
        """Add one full tetrille cell (used by the imbalance search)."""
        if key in self.cells:
            raise ValueError(f"cell {key} already registered")
        cell = Cell(key, role=role, ring=ring)
        for s in range(6):
            cell.kites[s] = KiteRecord(s, role)
        self.cells[key] = cell
        return cell

    # -- derived cell structure -------------------------------------------

    def neighbours(self, key: Key) -> Iterable[Key]:
        for dx, dy in NEIGHBOUR_OFFSETS:
            nb = (key[0] + dx, key[1] + dy)
            if nb in self.cells:
                yield nb

    def mask_depths(self) -> Dict[Key, int]:
        """Whole-cell depth inward from the minimal (completion) boundary.
        """
        from collections import deque

        depths: Dict[Key, int] = {}
        q: deque = deque()
        for key, cell in self.cells.items():
            if cell.role == "halo":
                continue
            on_edge = cell.role == "completion" or any(
                (key[0] + dx, key[1] + dy) not in self.cells
                or self.cells[(key[0] + dx, key[1] + dy)].role == "halo"
                for dx, dy in NEIGHBOUR_OFFSETS)
            if on_edge:
                depths[key] = 0
                q.append(key)
        while q:
            k = q.popleft()
            for nb in self.neighbours(k):
                if self.cells[nb].role != "halo" and nb not in depths:
                    depths[nb] = depths[k] + 1
                    q.append(nb)
        return depths
