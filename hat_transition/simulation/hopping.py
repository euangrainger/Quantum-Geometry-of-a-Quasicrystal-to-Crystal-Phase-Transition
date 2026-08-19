"""
Parametric hopping functions.

Two orthogonal binary bond classifications drive the effective hopping:
  - ab: 'dual' vs 'tetrille' -> base coupling
  - io: 'inner' vs 'outer' -> scaling / lerp modifiers
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class HoppingParams:
    """
    All parameters controlling the tight-binding hopping.

    The 4-phase transition sweeps these one at a time:
      Phase 1: coupling_tetrille  0->1  (turn on tetrille bonds)
      Phase 2: inner_scale        1->0  (remove intra-hexagon bonds)
      Phase 3: inner_dual_lerp    0->1  (restore inner dual bonds)
      Phase 4: outer_tetrille_scale 1->0 (remove outer tetrille bonds)
    """
    coupling_tetrille: float = 0.0
    coupling_dual: float = 1.0
    inner_scale: float = 1.0
    inner_dual_lerp: float = 0.0
    outer_tetrille_scale: float = 1.0
    inner_dual_target: float = 1.0
    magnetic_field: float = 0.0

    def copy(self) -> "HoppingParams":
        return HoppingParams(**self.__dict__)


def hopping_value(
    params: HoppingParams,
    ab: str,
    io: str,
) -> float:
    """
    Compute effective hopping magnitude for a bond.

    Args:
        params: current hopping parameters
        ab: 'dual' or 'tetrille'
        io: 'inner' or 'outer'

    Returns:
        effective hopping strength (positive = attractive)
    """
    eff = params.coupling_dual if ab == "dual" else params.coupling_tetrille

    if io == "inner":
        eff *= params.inner_scale

    if io == "inner" and ab == "dual":
        w = params.inner_dual_lerp
        eff = (1.0 - w) * eff + w * params.inner_dual_target

    if io == "outer" and ab == "tetrille":
        eff *= params.outer_tetrille_scale

    return eff
