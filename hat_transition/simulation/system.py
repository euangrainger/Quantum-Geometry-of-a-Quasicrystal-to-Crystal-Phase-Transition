"""
Tight-binding system built on Kwant.
"""

from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import kwant
from .hopping import HoppingParams, hopping_value

class _Amorphous(kwant.builder.SiteFamily):
    def __init__(self, coords: np.ndarray):
        self.coords = np.asarray(coords, dtype=float)
        super().__init__("amorphous", "", 1)

    def normalize_tag(self, tag):
        return int(tag[0]) if isinstance(tag, tuple) else int(tag)

    def pos(self, tag):
        return self.coords[int(tag)]

def _onsite(site,
            coupling_tetrille, coupling_dual, inner_scale,
            inner_dual_lerp, outer_tetrille_scale, inner_dual_target,
            magnetic_field, bond_label_dict, bond_label2_dict):
    """Legacy onsite ennergy modification"""
    return 0.0

def _hopping(site1, site2,
             coupling_tetrille, coupling_dual, inner_scale,
             inner_dual_lerp, outer_tetrille_scale, inner_dual_target,
             magnetic_field, bond_label_dict, bond_label2_dict):
    """Kwant parametric hopping: looks up bond type (dual/tetrille,
    inner/outer) from the label dicts, computes the magnitude via
    hopping_value, and applies a Peierls phase if B != 0."""
    params = HoppingParams(
        coupling_tetrille=float(coupling_tetrille),
        coupling_dual=float(coupling_dual),
        inner_scale=float(inner_scale),
        inner_dual_lerp=float(inner_dual_lerp),
        outer_tetrille_scale=float(outer_tetrille_scale),
        inner_dual_target=float(inner_dual_target),
        magnetic_field=float(magnetic_field),
    )
    i1 = int(site1.tag[0]) if isinstance(site1.tag, tuple) else int(site1.tag)
    i2 = int(site2.tag[0]) if isinstance(site2.tag, tuple) else int(site2.tag)
    key = (min(i1, i2), max(i1, i2))

    ab = bond_label_dict.get(key, "tetrille")
    io_dict = bond_label2_dict.get(key, {"io": "outer"})
    io = io_dict["io"] if isinstance(io_dict, dict) else "outer"

    mag = hopping_value(params, ab, io)
    B = params.magnetic_field
    if B != 0.0:
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        phase = np.exp(-0.5j * B * (x1 - x2) * (y1 + y2))
        return -mag * phase
    return -mag

@dataclass
class System:
    """Tight-binding system backed by a finalised Kwant system."""
    fsyst: object
    sites: np.ndarray
    bonds: np.ndarray
    n_sites: int
    old_to_new: Optional[np.ndarray] = None

    def _params_dict(self, params, bond_label_dict, bond_label2_dict):
        return {
            "coupling_tetrille": params.coupling_tetrille,
            "coupling_dual": params.coupling_dual,
            "inner_scale": params.inner_scale,
            "inner_dual_lerp": params.inner_dual_lerp,
            "outer_tetrille_scale": params.outer_tetrille_scale,
            "inner_dual_target": params.inner_dual_target,
            "magnetic_field": params.magnetic_field,
            "bond_label_dict": bond_label_dict,
            "bond_label2_dict": bond_label2_dict,
        }

    def hamiltonian(
        self,
        params: HoppingParams,
        bond_label_dict: Dict[Tuple[int, int], str],
        bond_label2_dict: Dict[Tuple[int, int], Dict[str, str]],
    ) -> np.ndarray:
        """
        Full NxN dense Hamiltonian via Kwant.

        Sites that are dynamically disconnected (zero hopping at the
        current parameters) will have empty rows/columns. 
        """
        p = self._params_dict(params, bond_label_dict, bond_label2_dict)
        return self.fsyst.hamiltonian_submatrix(params=p, sparse=False)

    def hamiltonian_reduced(
        self,
        params: HoppingParams,
        bond_label_dict: Dict[Tuple[int, int], str],
        bond_label2_dict: Dict[Tuple[int, int], Dict[str, str]],
        hopping_threshold: float = 1e-12,
    ):
        """
        Reduced Hamiltonian excluding dynamically disconnected sites.

        At intermediate parameter values that sit on phase edges, 
        many configurations contain disconnected sites.  
        This method removes them, returning only the physically active subspace.
        """
        H = self.hamiltonian(params, bond_label_dict, bond_label2_dict)
        active_mask = np.any(np.abs(H) > hopping_threshold, axis=1)
        n_eff = int(active_mask.sum())
        if n_eff == 0:
            return np.zeros((0, 0)), active_mask, np.array([], dtype=int)
        active_to_full = np.where(active_mask)[0]
        H_red = H[np.ix_(active_to_full, active_to_full)]
        return H_red, active_mask, active_to_full

def remap_bond_labels(
    label_dict: Dict[Tuple[int, int], any],
    old_to_new: np.ndarray,
) -> Dict[Tuple[int, int], any]:
    """
    Translate bond label dict keys from old site indices to new indices
    after dangling removal.
    """
    remapped = {}
    for (i, j), v in label_dict.items():
        ni, nj = int(old_to_new[i]), int(old_to_new[j])
        if ni >= 0 and nj >= 0:
            key = (min(ni, nj), max(ni, nj))
            remapped[key] = v
    return remapped

def build_system(sites: np.ndarray, bonds: np.ndarray) -> System:
    """
    Build a System from site coordinates and bond pairs.
    """
    n_original = len(sites)
    lat = _Amorphous(sites)
    builder = kwant.Builder()

    for i in range(n_original):
        builder[lat(i)] = _onsite

    for a, b in bonds:
        a, b = int(a), int(b)
        if a != b:
            builder[lat(a), lat(b)] = _hopping

    builder.eradicate_dangling()
    fsyst = builder.finalized()
    n_final = fsyst.graph.num_nodes
    n_removed = n_original - n_final
    if n_removed > 0:
        warnings.warn(
            f"build_system: eradicate_dangling removed {n_removed} of "
            f"{n_original} sites (degree <= 1 in the geometric graph). "
            "Interior kite vertices have degree >= 2 by construction, so "
            "this points at boundary-completion remnants from "
            "_complete_hexagons; check the input lattice if unexpected.",
            stacklevel=2,
        )
    new_sites = np.empty((n_final, 2))
    old_to_new = np.full(n_original, -1, dtype=int)

    for new_idx in range(n_final):
        site = fsyst.sites[new_idx]
        old_idx = int(site.tag[0]) if isinstance(site.tag, tuple) else int(site.tag)
        old_to_new[old_idx] = new_idx
        new_sites[new_idx] = site.pos

    new_bonds = []
    for i in range(n_final):
        for j in fsyst.graph.out_neighbors(i):
            if j > i:
                new_bonds.append([i, j])
    new_bonds = np.array(new_bonds, dtype=int) if new_bonds else np.empty((0, 2), dtype=int)

    if np.all(old_to_new >= 0):
        old_to_new = None 

    return System(
        fsyst=fsyst,
        sites=new_sites,
        bonds=new_bonds,
        n_sites=n_final,
        old_to_new=old_to_new,
    )
