"""
Observable hooks for on-the-fly (per lambda) computation interface.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence

import numpy as np

from kitegraph.graph import KiteGraph
from .model import ModelParams


class SpectrumHook:
    """Base class for on-the-fly observables.
    """

    name: str = "hook"
    needs_vectors: bool = False

    def begin(self, graph: KiteGraph, n_points: int) -> None:
        pass

    def accumulate(self, point: int, params: ModelParams,
                   energies: np.ndarray,
                   vectors: Optional[np.ndarray],
                   active_to_full: Optional[np.ndarray]
                   ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def finalize(self) -> Dict[str, np.ndarray]:
        return {}


class HookSet:
    """The registered hooks of one run which is driven by the runner."""

    def __init__(self, hooks: Sequence[SpectrumHook] = ()):
        self.hooks: List[SpectrumHook] = list(hooks)
        names = [h.name for h in self.hooks]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate hook names: {names}")

    @property
    def needs_vectors(self) -> bool:
        """True if any hook needs eigenvectors."""
        return any(h.needs_vectors for h in self.hooks)

    def begin(self, graph: KiteGraph, n_points: int) -> None:
        for h in self.hooks:
            h.begin(graph, n_points)

    def accumulate(self, point: int, params: ModelParams,
                   energies: np.ndarray,
                   vectors: Optional[np.ndarray],
                   active_to_full: Optional[np.ndarray]
                   ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for h in self.hooks:
            per_point = h.accumulate(point, params, energies,
                                     vectors if h.needs_vectors else None,
                                     active_to_full) or {}
            for key, val in per_point.items():
                out[f"{h.name}_{key}"] = np.asarray(val)
        return out

    def finalize(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for h in self.hooks:
            for key, val in h.finalize().items():
                out[f"{h.name}_{key}"] = np.asarray(val)
        return out
