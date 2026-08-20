"""
KiteGraph is the fundamental data object. 

Everything downstream takes a KiteGraph as its argument.  
The object is the kite lattice structure of one hat-tiling patch. 

    vertex_class  centre / corner / midpoint      (structural, from keys, in terms of the kite graph)
    edge_io       inner (spoke) / outer (rim)     (structural, from classes, in terms of the Tetrille unit cell)
    edge_ab       dual (on a placed hat outline) / tetrille (provenance, assigned in tiling for the hat / tetrille classifcation)
    cells         parent tetrille cells with role interior/completion/halo (for boundary conditions)
"""

from __future__ import annotations
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import (CellRegistry, CLASS_CENTRE, CLASS_CORNER,
                       CLASS_MIDPOINT, Key, kite_edge_keys,
                       kite_vertex_keys, vertex_class)

# integer codes used in the arrays for labels. 
AB_DUAL, AB_TETRILLE = 0, 1
IO_INNER, IO_OUTER = 0, 1
ROLE_CODES = {"interior": 0, "completion": 1, "halo": 2}
ROLE_NAMES = {v: k for k, v in ROLE_CODES.items()}


@dataclass
class KiteGraph:
    """One lattice patch as arrays, plus the parent-cell table.

    parent_mode records which sublattice plays the parent: 'tetrille'
    (periodic-parent open patch: tetrille completion + halo excess boundary complete) or
    'hat' (aperiodic-parent: outline-only hat excess boundary complete). swap_parent() relabels the mode.
    """

    # vertices
    vertices: np.ndarray        # (N, 2) float world positions
    vertex_keys: np.ndarray     # (N, 2) int exact identity
    vertex_class: np.ndarray    # (N,)   0 centre / 1 corner / 2 midpoint

    # edges
    edges: np.ndarray           # (M, 2) int vertex-index pairs, i < j
    edge_ab: np.ndarray         # (M,)   0 dual / 1 tetrille
    edge_io: np.ndarray         # (M,)   0 inner (spoke) / 1 outer (rim)

    # parent cells
    cell_centre: np.ndarray     # (C,) vertex index of each cell centre
    cell_role: np.ndarray       # (C,) 0 interior / 1 completion / 2 halo
    cell_ring: np.ndarray       # (C,) halo ring index (0 for non-halo)
    cell_depth: np.ndarray      # (C,) whole-cell depth inward from the
                                #      minimal (completion) boundary;
                                #      -1 for halo cells
    vertex_cells: List[Tuple[int, ...]] = field(default_factory=list)
                                # per vertex: indices of incident cells

    parent_mode: str = "tetrille"
    meta: Dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_registry(cls, reg: CellRegistry,
                      parent_mode: str = "tetrille",
                      meta: Optional[Dict] = None) -> "KiteGraph":
        """Derive the graph from a finished cell registry.

        Vertices and edges come from the kites cell by cell (plus any
        outline-only hats), deduplicated by integer key. labels are assigned from structure
        and provenance. 
        """
        vindex: Dict[Key, int] = {}
        vkeys: List[Key] = []

        def vid(key: Key) -> int:
            i = vindex.get(key)
            if i is None:
                i = len(vkeys)
                vindex[key] = i
                vkeys.append(key)
            return i

        edge_set: Dict[Tuple[int, int], None] = {}
        cell_items = sorted(reg.cells.items())

        for key, cell in cell_items:
            for sector in sorted(cell.kites):
                for a, b in kite_edge_keys(key, sector):
                    ia, ib = vid(a), vid(b)
                    edge_set[(min(ia, ib), max(ia, ib))] = None

        # outline-only hats contribute edges that may run outside every
        # registered cell (the bare parent-hat rim)
        for pair in sorted(reg.dual_edges,
                           key=lambda fs: tuple(sorted(fs))):
            a, b = sorted(pair)
            ia, ib = vid(a), vid(b)
            edge_set[(min(ia, ib), max(ia, ib))] = None

        n = len(vkeys)
        vertex_keys = np.array(vkeys, dtype=np.int64)
        vclass = np.array([vertex_class(k) for k in vkeys], dtype=np.int8)
        vertices = np.array([reg.frame.pos(k) for k in vkeys])

        edges = np.array(sorted(edge_set), dtype=np.int64)
        # ab is dual, if the edge lies on a placed hat outline
        ab = np.full(len(edges), AB_TETRILLE, dtype=np.int8)
        for m, (i, j) in enumerate(edges):
            if frozenset((vkeys[i], vkeys[j])) in reg.dual_edges:
                ab[m] = AB_DUAL
        # io (inner/outer), it is inner if the edge is a spoke (touches a centre)
        io = np.where((vclass[edges[:, 0]] == CLASS_CENTRE)
                      | (vclass[edges[:, 1]] == CLASS_CENTRE),
                      IO_INNER, IO_OUTER).astype(np.int8)

        depths = reg.mask_depths()
        cell_centre = np.array([vindex[k] for k, _ in cell_items],
                               dtype=np.int64)
        cell_role = np.array([ROLE_CODES[c.role] for _, c in cell_items],
                             dtype=np.int8)
        cell_ring = np.array([c.ring for _, c in cell_items], dtype=np.int16)
        cell_depth = np.array([depths.get(k, -1) for k, _ in cell_items],
                              dtype=np.int32)

        # incident cells per vertex (centres 1 cell, midpoints 2, corners 3
        # when all neighbours exist)
        incident: List[List[int]] = [[] for _ in range(n)]
        for ci, (key, cell) in enumerate(cell_items):
            seen = set()
            for sector in cell.kites:
                for vk in kite_vertex_keys(key, sector):
                    i = vindex[vk]
                    if i not in seen:
                        seen.add(i)
                        incident[i].append(ci)

        return cls(vertices=vertices, vertex_keys=vertex_keys,
                   vertex_class=vclass, edges=edges, edge_ab=ab,
                   edge_io=io, cell_centre=cell_centre,
                   cell_role=cell_role, cell_ring=cell_ring,
                   cell_depth=cell_depth,
                   vertex_cells=[tuple(c) for c in incident],
                   parent_mode=parent_mode, meta=dict(meta or {}))

    # -- basic views -------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def sublattice(self) -> np.ndarray:
        """Structural sublattice: A (0) = centres and corners, B (1) =
        midpoints.  This is the anchored two-colouring conventional choice"""
        return (self.vertex_class == CLASS_MIDPOINT).astype(np.int8)

    def imbalance(self) -> Tuple[int, float]:
        """Chiral imbalance N_A - N_B (signed, and per site).  Its
        modulus lower-bounds the number of E = 0 modes protected by the
        bipartite (chiral) symmetry."""
        s = self.sublattice()
        signed = int(np.sum(s == 0) - np.sum(s == 1))
        return signed, signed / max(len(s), 1)

    # -- verification ------------------------------------------------------

    def two_colouring(self, edge_mask: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """BFS two-colouring over the selected edges, anchored so that
        every centre gets colour 0.

        Returns (colours, conflicts). Conflicts are same-colour edges
        (odd cycles).  On the full edge set this must reproduce
        sublattice() with no conflicts; a failure means the construction
        broke the lattice (this should only happen w dev).
        """
        sel = self.edges if edge_mask is None else self.edges[edge_mask]
        adj: List[List[int]] = [[] for _ in range(self.n_vertices)]
        for i, j in sel:
            adj[i].append(j)
            adj[j].append(i)

        colours = np.full(self.n_vertices, -1, dtype=np.int8)
        conflicts: List[Tuple[int, int]] = []
        for s in range(self.n_vertices):
            if colours[s] != -1 or not adj[s]:
                continue
            # anchor each component at its lowest-index centre if it has
            # one, so colour 0 always means "centre-side"
            comp = [s]
            seen = {s}
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        comp.append(v)
                        q.append(v)
            centres = [v for v in comp
                       if self.vertex_class[v] == CLASS_CENTRE]
            root = min(centres) if centres else min(comp)
            colours[root] = 0
            q = deque([root])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if colours[v] == -1:
                        colours[v] = 1 - colours[u]
                        q.append(v)
                    elif colours[v] == colours[u]:
                        conflicts.append((min(u, v), max(u, v)))
        return colours, sorted(set(conflicts))

    # -- bulk mask ---------------------------------------------------------

    def bulk_mask(self, depth: int = 0) -> np.ndarray:
        """Boolean vertex mask eroded inward in whole parent cells.

        A vertex is bulk at the given depth when it belongs to at least
        one non-halo cell of cell_depth >= depth and to no shallower
        non-halo cell. Depth 0 is the mask ceiling. Vertices that
        exist only on bare hat outlines (hat-excess rim) are never bulk.
        """
        mask = np.zeros(self.n_vertices, dtype=bool)
        for i, cells in enumerate(self.vertex_cells):
            non_halo = [self.cell_depth[c] for c in cells
                        if self.cell_role[c] != ROLE_CODES["halo"]]
            if non_halo and min(non_halo) >= depth:
                mask[i] = True
        return mask

    def max_mask_depth(self) -> int:
        non_halo = self.cell_depth[self.cell_role != ROLE_CODES["halo"]]
        return int(non_halo.max()) if len(non_halo) else -1

    # -- parent swap -------------------------------------------------------

    def swap_parent(self) -> "KiteGraph":
        """Flip which sublattice is regarded as the parent.  Geometry and
        labels are shared (views), only the mode tag changes."""
        return KiteGraph(
            vertices=self.vertices, vertex_keys=self.vertex_keys,
            vertex_class=self.vertex_class, edges=self.edges,
            edge_ab=self.edge_ab, edge_io=self.edge_io,
            cell_centre=self.cell_centre, cell_role=self.cell_role,
            cell_ring=self.cell_ring, cell_depth=self.cell_depth,
            vertex_cells=self.vertex_cells,
            parent_mode="hat" if self.parent_mode == "tetrille"
            else "tetrille",
            meta=dict(self.meta))

    # -- serialisation -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Flat .npz (arrays) + .json sidecar (mode, meta, cell lists)."""
        path = Path(path)
        np.savez_compressed(
            path.with_suffix(".npz"),
            vertices=self.vertices, vertex_keys=self.vertex_keys,
            vertex_class=self.vertex_class, edges=self.edges,
            edge_ab=self.edge_ab, edge_io=self.edge_io,
            cell_centre=self.cell_centre, cell_role=self.cell_role,
            cell_ring=self.cell_ring, cell_depth=self.cell_depth)
        sidecar = {
            "parent_mode": self.parent_mode,
            "meta": self.meta,
            "vertex_cells": [list(c) for c in self.vertex_cells],
        }
        path.with_suffix(".json").write_text(json.dumps(sidecar))

    @classmethod
    def load(cls, path: str | Path) -> "KiteGraph":
        path = Path(path)
        arrays = np.load(path.with_suffix(".npz"))
        sidecar = json.loads(path.with_suffix(".json").read_text())
        return cls(
            vertices=arrays["vertices"], vertex_keys=arrays["vertex_keys"],
            vertex_class=arrays["vertex_class"], edges=arrays["edges"],
            edge_ab=arrays["edge_ab"], edge_io=arrays["edge_io"],
            cell_centre=arrays["cell_centre"], cell_role=arrays["cell_role"],
            cell_ring=arrays["cell_ring"], cell_depth=arrays["cell_depth"],
            vertex_cells=[tuple(c) for c in sidecar["vertex_cells"]],
            parent_mode=sidecar["parent_mode"], meta=sidecar["meta"])
