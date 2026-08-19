"""
Bipartite lattice utilities.

Used to verify bipartite correctness of lattices in simulation, as well as identify minimum number of zero modes protected
by chiral symmetry. 
"""

from __future__ import annotations
from collections import deque
from typing import Iterable, List, Tuple
import numpy as np


def sublattice_assignment(
    n_sites: int,
    bonds: np.ndarray,
    reference_sites: Iterable[int] | None = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Assign sublattice labels (0 or 1) via BFS 2-coloring.

    A 2-coloring is only defined up to a global swap per connected component,
    so which physical site type ends up on A is otherwise an artefact of site
    ordering. Passing reference_sites pins that choice.

    Args:
        reference_sites: sites that must be labelled 0 (sublattice A). Every
            component containing at least one is flipped so its reference
            sites are 0; components containing none keep the lowest-index-
            first convention. Reference sites disagreeing within a component
            raise ValueError, since they cannot all be sublattice A.

    Returns:
        colors: (N,) int array of sublattice labels
        conflicts: list of (i, j) same-sublattice edges
    """
    adj = [[] for _ in range(n_sites)]
    for a, b in bonds:
        a, b = int(a), int(b)
        if a != b:
            adj[a].append(b)
            adj[b].append(a)

    colors = np.full(n_sites, -1, dtype=int)
    component = np.full(n_sites, -1, dtype=int)
    conflicts = []
    n_components = 0

    for s in range(n_sites):
        if colors[s] != -1:
            continue
        colors[s] = 0
        component[s] = n_components
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if colors[v] == -1:
                    colors[v] = 1 - colors[u]
                    component[v] = n_components
                    q.append(v)
                elif colors[v] == colors[u]:
                    conflicts.append((min(u, v), max(u, v)))
        n_components += 1

    if reference_sites is not None:
        _anchor(colors, component, n_components, reference_sites, bool(conflicts))

    return colors, sorted(set(conflicts))


def _anchor(
    colors: np.ndarray,
    component: np.ndarray,
    n_components: int,
    reference_sites: Iterable[int],
    has_conflicts: bool,
) -> None:
    """Flip whole components in place so reference sites carry label 0."""
    flip = np.zeros(n_components, dtype=bool)
    decided = np.zeros(n_components, dtype=bool)

    for r in sorted({int(r) for r in reference_sites}):
        if not 0 <= r < len(colors):
            raise ValueError(f"reference site {r} out of range for {len(colors)} sites")
        c = component[r]
        wants_flip = colors[r] == 1
        if not decided[c]:
            decided[c] = True
            flip[c] = wants_flip
        elif wants_flip != flip[c]:
            detail = (
                " (lattice is not bipartite, so the coloring is ill-defined)"
                if has_conflicts else ""
            )
            raise ValueError(
                f"reference sites are not all on one sublattice within component "
                f"{int(c)}: site {r} disagrees{detail}"
            )

    if flip.any():
        mask = flip[component]
        colors[mask] = 1 - colors[mask]


def verify_bipartite(
    n_sites: int,
    bonds: np.ndarray,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Check whether the lattice is bipartite.

    Returns:
        is_bipartite: True if no conflicts
        conflicts: list of same-sublattice edges
    """
    _, conflicts = sublattice_assignment(n_sites, bonds)
    return len(conflicts) == 0, conflicts


def sublattice_imbalance(sublattice: np.ndarray) -> Tuple[int, float]:
    """
    Compute sublattice imbalance.

    Signed, so it is only comparable across runs when A was pinned to a fixed
    site type (see reference_sites in sublattice_assignment); |signed| is the
    minimum bound of zero modes either way.

    Returns:
        signed: N_A - N_B (positive when the anchored sublattice is in excess)
        normalised: (N_A - N_B) / N (intensive)
    """
    n = len(sublattice)
    if n == 0:
        return 0, 0.0
    n_a = int(np.sum(sublattice == 0))
    n_b = int(np.sum(sublattice == 1))
    signed = n_a - n_b
    return signed, signed / n

