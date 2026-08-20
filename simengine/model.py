"""
Nearest-neighbour tight-binding model on a KiteGraph.

The model interacts with kitegraph currently in terms of hopping magnitude, 
onsite energy or data type (in the case of complex values for kernel)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace

import numpy as np

from kitegraph.graph import KiteGraph, AB_DUAL, IO_INNER
from kitegraph.registry import CLASS_CENTRE, CLASS_CORNER, CLASS_MIDPOINT


@dataclass
class ModelParams:
    """All parameters of the NN tight-binding Hamiltonian."""
    # hopping (4-phase schedule)
    coupling_dual: float = 1.0
    coupling_tetrille: float = 0.0
    inner_scale: float = 1.0
    inner_dual_lerp: float = 0.0
    outer_tetrille_scale: float = 1.0
    inner_dual_target: float = 1.0
    magnetic_field: float = 0.0  # (Peierls phase)
    eps_centre: float = 0.0
    eps_corner: float = 0.0
    eps_midpoint: float = 0.0

    def copy(self, **changes) -> "ModelParams":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelParams":
        return cls(**d)


class NNTightBinding:
    """The one implemented model: NN hoppings driven by the edge labels.
    """

    norb = 1 ## Orbitals

    def edge_values(self, graph: KiteGraph, params: ModelParams
                    ) -> np.ndarray:
        """ hopping magnitudes t_ij  
        """
        dual = graph.edge_ab == AB_DUAL
        inner = graph.edge_io == IO_INNER

        eff = np.where(dual, params.coupling_dual,
                       params.coupling_tetrille).astype(float)
        eff[inner] *= params.inner_scale
        w = params.inner_dual_lerp
        m = inner & dual
        eff[m] = (1.0 - w) * eff[m] + w * params.inner_dual_target
        m = ~inner & ~dual
        eff[m] *= params.outer_tetrille_scale
        return eff

    def vertex_values(self, graph: KiteGraph, params: ModelParams
                      ) -> np.ndarray:
        """onsite energies from the structural vertex classes."""
        eps = np.empty(graph.n_vertices, dtype=float)
        eps[graph.vertex_class == CLASS_CENTRE] = params.eps_centre
        eps[graph.vertex_class == CLASS_CORNER] = params.eps_corner
        eps[graph.vertex_class == CLASS_MIDPOINT] = params.eps_midpoint
        return eps

    def needs_complex(self, params: ModelParams) -> bool:
        """A Peierls phase is the only source of complex entries so far"""
        return params.magnetic_field != 0.0


def four_phase_path(s: float) -> ModelParams:
    """The canonical transition path as one global coordinate."""
    if not 0.0 <= s <= 4.0:
        raise ValueError(f"path coordinate {s} outside [0, 4]")
    p = ModelParams()
    p.coupling_tetrille = min(s, 1.0)
    p.inner_scale = 1.0 - min(max(s - 1.0, 0.0), 1.0)
    p.inner_dual_lerp = min(max(s - 2.0, 0.0), 1.0)
    p.outer_tetrille_scale = 1.0 - min(max(s - 3.0, 0.0), 1.0)
    return p


def phase1_lambda(lam: float) -> ModelParams:
    """Phase 1 only."""
    return ModelParams(coupling_tetrille=lam)
