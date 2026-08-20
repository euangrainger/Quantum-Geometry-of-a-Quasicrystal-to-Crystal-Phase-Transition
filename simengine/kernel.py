"""
Dense spectrum kernel, utilities for getting things to run. 
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import linalg as sla

from kitegraph.graph import KiteGraph
from .model import ModelParams, NNTightBinding

GIB = 1024 ** 3

@dataclass
class MemoryReport:
    """Predicted peak memory of one diagonalisation"""
    n: int
    dtype: str
    need_vectors: bool
    driver: str
    h_bytes: int
    vec_bytes: int
    work_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.h_bytes + self.vec_bytes + self.work_bytes

    def __str__(self) -> str:
        return (f"N = {self.n}: H buffer {self.h_bytes / GIB:.2f} GiB "
                f"({self.dtype}), eigenvectors "
                f"{self.vec_bytes / GIB:.2f} GiB, workspace est. "
                f"{self.work_bytes / GIB:.2f} GiB ({self.driver}) — "
                f"peak ~ {self.total_bytes / GIB:.2f} GiB")


def memory_report(n: int, complex_dtype: bool, need_vectors: bool,
                  driver: str = "evr") -> MemoryReport:
    item = 16 if complex_dtype else 8
    h = n * n * item
    vec = n * n * item if need_vectors else 0
    work = int((1.5 if driver == "evd" else 0.1) * n * n * item)
    return MemoryReport(n=n, dtype="complex128" if complex_dtype
                        else "float64", need_vectors=need_vectors,
                        driver=driver, h_bytes=h, vec_bytes=vec,
                        work_bytes=work)


@dataclass
class SpectrumResult:
    energies: np.ndarray
    vectors: Optional[np.ndarray]
    active_to_full: Optional[np.ndarray]


class SpectrumKernel:
    """Forward pass + diagonalisation for one graph and one model."""

    def __init__(self, graph: KiteGraph, model: Optional[NNTightBinding]
                 = None):
        self.graph = graph
        self.model = model or NNTightBinding()
        self.n = graph.n_vertices
        self._rows = graph.edges[:, 0]
        self._cols = graph.edges[:, 1]
        x = graph.vertices[:, 0]
        y = graph.vertices[:, 1]
        self._peierls_factor = ((x[self._rows] - x[self._cols])
                                * (y[self._rows] + y[self._cols]))
        self._buf: Optional[np.ndarray] = None

    # -- forward pass ------------------------------------------------------

    def assemble(self, params: ModelParams,
                 dtype: Optional[type] = None) -> np.ndarray:
        if dtype is None:
            dtype = (np.complex128 if self.model.needs_complex(params)
                     else np.float64)
        if self._buf is None or self._buf.dtype != dtype:
            self._buf = np.zeros((self.n, self.n), dtype=dtype)
        else:
            self._buf[:] = 0

        t = self.model.edge_values(self.graph, params)
        if dtype == np.complex128 and params.magnetic_field != 0.0:
            phase = np.exp(-0.5j * params.magnetic_field
                           * self._peierls_factor)
            upper = -t * phase
        else:
            upper = -t
        self._buf[self._rows, self._cols] = upper
        self._buf[self._cols, self._rows] = np.conj(upper)
        eps = self.model.vertex_values(self.graph, params)
        self._buf[np.arange(self.n), np.arange(self.n)] = eps
        return self._buf

    # -- diagonalisation ---------------------------------------------------

    def spectrum(self, params: ModelParams, need_vectors: bool = False,
                 driver: str = "evr", reduce: bool = False,
                 hopping_threshold: float = 1e-12) -> SpectrumResult:
        """Diagonalise H(params)
        """
        H = self.assemble(params)
        active_to_full = None
        if reduce:
            active = np.any(np.abs(H) > hopping_threshold, axis=1)
            if not active.all():
                active_to_full = np.where(active)[0]
                H = np.ascontiguousarray(
                    H[np.ix_(active_to_full, active_to_full)])
                if len(active_to_full) == 0:
                    return SpectrumResult(np.empty(0), None,
                                          active_to_full)

        if need_vectors:
            energies, vectors = sla.eigh(
                H, driver=driver, overwrite_a=True, check_finite=False)
        else:
            energies = sla.eigh(
                H, driver=driver, overwrite_a=True, check_finite=False,
                eigvals_only=True)
            vectors = None
        return SpectrumResult(energies, vectors, active_to_full)

    def report(self, params: ModelParams, need_vectors: bool,
               driver: str = "evr") -> MemoryReport:
        return memory_report(self.n, self.model.needs_complex(params),
                             need_vectors, driver)
