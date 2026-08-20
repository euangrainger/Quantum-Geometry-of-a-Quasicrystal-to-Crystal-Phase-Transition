"""
Linear walkthrough of the kitegraph construction.

  STEP 0  tiling            hats placed by the substitution
  STEP 1  registry          integer-frame snapping, cells and kites
  STEP 2  boundary          completion + halo (or hat-excess rim)
  STEP 3  graph             vertices, edges, dual/tetrille, inner/outer
  STEP 4  bipartite         BFS verification, chiral imbalance
  STEP 5  mask              whole-cell erosion, depth by depth

Run from the repo root:
   kitegraph.main   # H3 (default)
   kitegraph.main --level 3
   kitegraph.main --mode hat
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from .tiling import iter_hats, metatile
from .registry import CellRegistry, LatticeFrame, CLASS_CENTRE, \
    CLASS_CORNER, CLASS_MIDPOINT
from .graph import KiteGraph, ROLE_CODES
from .boundary import _split_interior_rim


def banner(s: str) -> None:
    print("\n" + "=" * 72 + f"\n{s}\n" + "=" * 72)


def main() -> KiteGraph:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--level", type=int, default=3)
    p.add_argument("--tile", choices="HTPF", default="H")
    p.add_argument("--mode", choices=["tetrille", "hat"], default="tetrille",
                   help="which lattice is the parent that extends past "
                        "the embedded region")
    p.add_argument("--halo", type=int, default=2,
                   help="tetrille mode: halo rings of excess parent cells")
    p.add_argument("--rim-layers", type=int, default=1,
                   help="hat mode: layers of outline-only rim hats")
    p.add_argument("--viz", action="store_true",
                   help="render the construction figures (SVG) (not currently configured)")
    p.add_argument("--viz-dir", default="kitegraph/figures")
    p.add_argument("--save", default=None, metavar="PATH",
                   help="save the finished KiteGraph (npz + json)")
    args = p.parse_args()

    # ------------------------------------------------------------- STEP 0
    banner(f"STEP 0  - tiling {args.tile}{args.level} substitution")
    hats = list(iter_hats(metatile(args.level, args.tile)))
    print(f"{len(hats)} hats placed "
          f"(labels: { {lab for _, lab in hats} })")

    # ------------------------------------------------------------- STEP 1
    banner("STEP 1  - registry snap onto the integer kite lattice")
    frame = LatticeFrame.from_first_hat(hats[0][0])
    # note: the substitution places hats at half the scale of the
    # HAT_VERTICES table, so the world lattice constant is sqrt(3), not
    # 2*sqrt(3) — the calibrated frame absorbs this automatically
    print(f"frame calibrated from hat 0: origin {frame.origin.round(4)}, "
          f"lattice constant |A1| = {np.linalg.norm(frame.basis[:, 0]):.6f}")

    reg = CellRegistry(frame)
    if args.mode == "tetrille":
        for transform, _ in hats:
            reg.register_hat(transform)
        n_rim = 0
    else:
        interior, rim = _split_interior_rim(hats, frame, args.rim_layers)
        if not interior:
            raise SystemExit(
                f"every hat of {args.tile}{args.level} sits on the rim. "
                "The hat-excess boundary needs a larger level")
        for i in interior:
            reg.register_hat(hats[i][0])
        for i in rim:
            reg.register_hat_outline(hats[i][0])
        n_rim = len(rim)
        print(f"hat-excess split: {len(interior)} interior (decorated), "
              f"{n_rim} rim (outline only)")
    n_hat_kites = sum(len(c.kites) for c in reg.cells.values())
    print(f"{len(reg.cells)} cells touched, {n_hat_kites} hat kites, "
          f"{len(reg.dual_edges)} outline (dual) edges recorded")

    # ------------------------------------------------------------- STEP 2
    banner("STEP 2  - boundary completion, then excess")
    n_completed = reg.complete_partial_cells()
    n_partial = sum(1 for c in reg.cells.values()
                    if c.role == "completion")
    print(f"completion: {n_completed} kites added in {n_partial} partial "
          "cells (the minimal boundary)")
    if args.mode == "tetrille":
        for ring in range(1, args.halo + 1):
            n_new = reg.add_halo_ring(ring)
            print(f"halo ring {ring}: {n_new} full parent cells")

    # ------------------------------------------------------------- STEP 3
    banner("STEP 3  - graph derive vertices, edges and labels")
    g = KiteGraph.from_registry(
        reg, parent_mode=args.mode,
        meta={"level": args.level, "tile": args.tile,
              "halo_rings": args.halo if args.mode == "tetrille" else 0,
              "n_rim_hats": n_rim})
    n_c = int(np.sum(g.vertex_class == CLASS_CENTRE))
    n_v = int(np.sum(g.vertex_class == CLASS_CORNER))
    n_m = int(np.sum(g.vertex_class == CLASS_MIDPOINT))
    print(f"N = {g.n_vertices} vertices ({n_c} centres, {n_v} corners, "
          f"{n_m} midpoints), M = {g.n_edges} edges")
    print(f"labels: {int(np.sum(g.edge_ab == 0))} dual / "
          f"{int(np.sum(g.edge_ab == 1))} tetrille;  "
          f"{int(np.sum(g.edge_io == 0))} inner / "
          f"{int(np.sum(g.edge_io == 1))} outer")

    # ------------------------------------------------------------- STEP 4
    banner("STEP 4  - bipartite verification and chiral imbalance")
    colours, conflicts = g.two_colouring()
    active = colours >= 0
    ok = bool(np.all(colours[active] == g.sublattice()[active]))
    signed, per_site = g.imbalance()
    print(f"BFS two-colouring: {len(conflicts)} conflicts; matches the "
          f"structural sublattice (A = centres + corners): {ok}")
    print(f"imbalance N_A - N_B = {signed:+d} ({per_site:+.5f} per site) ")

    # ------------------------------------------------------------- STEP 5
    banner("STEP 5  - mask whole-cell erosion from the minimal boundary")
    for d in range(g.max_mask_depth() + 1):
        mask = g.bulk_mask(d)
        note = "  <- ceiling: minimal completion region" if d == 0 else ""
        print(f"depth {d}: {int(mask.sum()):5d} bulk sites{note}")

    return g


if __name__ == "__main__":
    main()
