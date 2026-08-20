"""
Hat-tiling substitution and the decorated unit hat.

The substitution machinery is a pythonic version of https://github.com/isohedral/hatviz:
Tt produces, for a requested substitution level, a hierarchy of metatiles. 

However, we modify the vertices of the hat to be declarative in terms of the kite object.
Each of its 16 vertices is named by its structural class on the kite lattice (hexagon centre / hexagon corner / edge midpoint), 
the 8 kites of the hat are written role-ordered and grouped by the hexagon they belong to, 
and the interior decoration (vertices 13, 14, 15) is a single named block explaining why it exists.
"""

from __future__ import annotations
import math
from typing import Iterator, List, Tuple

# --------------------------------------------------------------------------
# affine-transform funcs (2x3 row-major [a, b, tx, c, d, ty])
# --------------------------------------------------------------------------

def pt(x: float, y: float) -> Tuple[float, float]:
    return (x, y)


def hex_pt(x: float, y: float) -> Tuple[float, float]:
    """Hex coordinates -> Cartesian, used for the hat vertex table."""
    return (x + 0.5 * y, (math.sqrt(3) / 2) * y)


def padd(p, q):
    return (p[0] + q[0], p[1] + q[1])


def psub(p, q):
    return (p[0] - q[0], p[1] - q[1])


def trot(ang: float) -> List[float]:
    c, s = math.cos(ang), math.sin(ang)
    return [c, -s, 0, s, c, 0]


def ttrans(tx: float, ty: float) -> List[float]:
    return [1, 0, tx, 0, 1, ty]


def mul(A: List[float], B: List[float]) -> List[float]:
    aA, bA, txA, cA, dA, tyA = A
    aB, bB, txB, cB, dB, tyB = B
    return [
        aA * aB + bA * cB, aA * bB + bA * dB, aA * txB + bA * tyB + txA,
        cA * aB + dA * cB, cA * bB + dA * dB, cA * txB + dA * tyB + tyA,
    ]


def inv(T: List[float]) -> List[float]:
    a, b, tx, c, d, ty = T
    det = a * d - b * c
    if abs(det) < 1e-14:
        raise ValueError("degenerate transform")
    return [d / det, -b / det, (b * ty - tx * d) / det,
            -c / det, a / det, (tx * c - a * ty) / det]


def trans_pt(M: List[float], P: Tuple[float, float]) -> Tuple[float, float]:
    x, y = P
    a, b, tx, c, d, ty = M
    return (a * x + b * y + tx, c * x + d * y + ty)


def rot_about(p, ang):
    return mul(ttrans(p[0], p[1]), mul(trot(ang), ttrans(-p[0], -p[1])))


def match_seg(p, q):
    return [q[0] - p[0], p[1] - q[1], p[0],
            q[1] - p[1], q[0] - p[0], p[1]]


def match_two(p1, q1, p2, q2):
    return mul(match_seg(p2, q2), inv(match_seg(p1, q1)))


def intersect(p1, q1, p2, q2):
    denom = (q2[1] - p2[1]) * (q1[0] - p1[0]) - (q2[0] - p2[0]) * (q1[1] - p1[1])
    if abs(denom) < 1e-14:
        raise ValueError("parallel lines")
    ua = ((q2[0] - p2[0]) * (p1[1] - p2[1])
          - (q2[1] - p2[1]) * (p1[0] - p2[0])) / denom
    return (p1[0] + ua * (q1[0] - p1[0]), p1[1] + ua * (q1[1] - p1[1]))

## Updates to the hat unit cell based on the kite decomposition. 

_R3 = math.sqrt(3)
HAT_VERTICES: List[Tuple[float, float]] = [
    hex_pt(0, 0),    #  0 centre   (hexagon A)
    hex_pt(-1, -1),  #  1 midpoint
    hex_pt(0, -2),   #  2 corner
    hex_pt(2, -2),   #  3 corner
    hex_pt(2, -1),   #  4 midpoint
    hex_pt(4, -2),   #  5 centre   (hexagon C)
    hex_pt(5, -1),   #  6 midpoint
    hex_pt(4, 0),    #  7 corner
    hex_pt(3, 0),    #  8 midpoint
    hex_pt(2, 2),    #  9 centre   (hexagon B)
    hex_pt(0, 3),    # 10 midpoint
    hex_pt(0, 2),    # 11 corner
    hex_pt(-1, 2),   # 12 midpoint
    (0.0, -_R3),     # 13 midpoint (decoration: bisects long edge 2--3)
    (2.0, 0.0),      # 14 corner   (decoration: triple point of A, B, C)
    (1.5, _R3 / 2),  # 15 midpoint (decoration: between centres 0 and 9)
]

# The index of the hex centres in in HAT_VERTICES.
HAT_CENTRES = (0, 9, 5)

HAT_KITES: List[Tuple[int, int, int, int]] = [
    # hexagon A (centre 0): four kites from midpoint 1 to 12
    (0, 1, 2, 13),
    (0, 13, 3, 4),
    (0, 4, 14, 15),
    (0, 15, 11, 12),
    # hexagon B (centre 9): two kites
    (9, 10, 11, 15),
    (9, 15, 14, 8),
    # hexagon C (centre 5): two kites
    (5, 6, 7, 8),
    (5, 8, 14, 4),
]

HAT_OUTLINE: List[int] = [0, 1, 2, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def outline_edges() -> List[Tuple[int, int]]:
    """The 14 local vertex-index pairs of the decorated hat outline."""
    n = len(HAT_OUTLINE)
    return [(HAT_OUTLINE[k], HAT_OUTLINE[(k + 1) % n]) for k in range(n)]


# --------------------------------------------------------------------------
# metatile hierarchy and substitution rules 
# --------------------------------------------------------------------------

class HatTile:
    """A single hat; the leaf of the metatile hierarchy."""

    def __init__(self, label: str):
        self.label = label
        self.shape = HAT_VERTICES


class MetaTile:
    def __init__(self, shape, width: float):
        self.shape = shape
        self.width = width
        self.children: List[tuple] = []  # (transform, HatTile | MetaTile)

    def add_child(self, T, geom):
        self.children.append((T, geom))

    def eval_child(self, n: int, i: int):
        T, g = self.children[n]
        return trans_pt(T, g.shape[i])

    def recentre(self):
        if not self.shape:
            return
        cx = sum(p[0] for p in self.shape) / len(self.shape)
        cy = sum(p[1] for p in self.shape) / len(self.shape)
        self.shape = [(x - cx, y - cy) for (x, y) in self.shape]
        shift = ttrans(-cx, -cy)
        self.children = [(mul(shift, T), g) for (T, g) in self.children]


_hr3 = math.sqrt(3) / 2

hat_H1 = HatTile("H1")
hat_H = HatTile("H")
hat_T = HatTile("T")
hat_P = HatTile("P")
hat_F = HatTile("F")


def _construct_base_H() -> MetaTile:
    H_outline = [pt(0, 0), pt(4, 0), pt(4.5, _hr3),
                 pt(2.5, 5 * _hr3), pt(1.5, 5 * _hr3), pt(-0.5, _hr3)]
    meta = MetaTile(H_outline, 2.0)
    meta.add_child(match_two(HAT_VERTICES[5], HAT_VERTICES[7],
                             H_outline[5], H_outline[0]), hat_H)
    meta.add_child(match_two(HAT_VERTICES[9], HAT_VERTICES[11],
                             H_outline[1], H_outline[2]), hat_H)
    meta.add_child(match_two(HAT_VERTICES[5], HAT_VERTICES[7],
                             H_outline[3], H_outline[4]), hat_H)
    T4a = [-0.5, -_hr3, 0, _hr3, -0.5, 0]
    T4b = [0.5, 0, 0, 0, -0.5, 0]
    meta.add_child(mul(ttrans(2.5, _hr3), mul(T4a, T4b)), hat_H1)
    return meta


def _construct_base_T() -> MetaTile:
    T_outline = [pt(0, 0), pt(3, 0), pt(1.5, 3 * _hr3)]
    meta = MetaTile(T_outline, 2.0)
    meta.add_child([0.5, 0, 0.5, 0, 0.5, _hr3], hat_T)
    return meta


def _construct_base_P() -> MetaTile:
    P_outline = [pt(0, 0), pt(4, 0), pt(3, 2 * _hr3), pt(-1, 2 * _hr3)]
    meta = MetaTile(P_outline, 2.0)
    meta.add_child([0.5, 0, 1.5, 0, 0.5, _hr3], hat_P)
    meta.add_child(mul(ttrans(0, 2 * _hr3),
                       mul([0.5, _hr3, 0, -_hr3, 0.5, 0],
                           [0.5, 0, 0, 0, 0.5, 0])), hat_P)
    return meta


def _construct_base_F() -> MetaTile:
    F_outline = [pt(0, 0), pt(3, 0), pt(3.5, _hr3),
                 pt(3, 2 * _hr3), pt(-1, 2 * _hr3)]
    meta = MetaTile(F_outline, 2.0)
    meta.add_child([0.5, 0, 1.5, 0, 0.5, _hr3], hat_F)
    meta.add_child(mul(ttrans(0, 2 * _hr3),
                       mul([0.5, _hr3, 0, -_hr3, 0.5, 0],
                           [0.5, 0, 0, 0, 0.5, 0])), hat_F)
    return meta


def _construct_patch(H, T, P, F) -> MetaTile:
    """Assemble the level-(n+1) patch from level-n H, T, P, F."""
    ret = MetaTile([], H.width)
    shapes = {"H": H, "T": T, "P": P, "F": F}
    rules = [
        ["H"],
        [0, 0, "P", 2], [1, 0, "H", 2], [2, 0, "P", 2], [3, 0, "H", 2],
        [4, 4, "P", 2], [0, 4, "F", 3], [2, 4, "F", 3],
        [4, 1, 3, 2, "F", 0],
        [8, 3, "H", 0], [9, 2, "P", 0], [10, 2, "H", 0], [11, 4, "P", 2],
        [12, 0, "H", 2], [13, 0, "F", 3], [14, 2, "F", 1], [15, 3, "H", 4],
        [8, 2, "F", 1], [17, 3, "H", 0], [18, 2, "P", 0], [19, 2, "H", 2],
        [20, 4, "F", 3], [20, 0, "P", 2], [22, 0, "H", 2], [23, 4, "F", 3],
        [23, 0, "F", 3], [16, 0, "P", 2],
        [9, 4, 0, 2, "T", 2],
        [4, 0, "F", 3],
    ]
    for r in rules:
        if len(r) == 1:
            ret.add_child([1, 0, 0, 0, 1, 0], shapes[r[0]])
        elif len(r) == 4:
            idx_ch, seg_i, shape_label, seg_j = r
            T_ch, geom_ch = ret.children[idx_ch]
            poly = geom_ch.shape
            P1 = trans_pt(T_ch, poly[(seg_i + 1) % len(poly)])
            Q1 = trans_pt(T_ch, poly[seg_i])
            new_poly = shapes[shape_label].shape
            ret.add_child(match_two(new_poly[seg_j],
                                    new_poly[(seg_j + 1) % len(new_poly)],
                                    P1, Q1), shapes[shape_label])
        else:
            cP = ret.children[r[0]]
            cQ = ret.children[r[2]]
            P1 = trans_pt(cQ[0], cQ[1].shape[r[3]])
            Q1 = trans_pt(cP[0], cP[1].shape[r[1]])
            new_poly = shapes[r[4]].shape
            ret.add_child(match_two(new_poly[r[5]],
                                    new_poly[(r[5] + 1) % len(new_poly)],
                                    P1, Q1), shapes[r[4]])
    return ret


def _construct_metatiles(patch: MetaTile):
    """Slice the assembled patch back into H, T, P, F supertiles."""
    bps1 = patch.eval_child(8, 2)
    bps2 = patch.eval_child(21, 2)
    rbps = trans_pt(rot_about(bps1, -2 * math.pi / 3), bps2)
    p72 = patch.eval_child(7, 2)
    p252 = patch.eval_child(25, 2)
    llc = intersect(bps1, rbps, patch.eval_child(6, 2), p72)
    w = psub(patch.eval_child(6, 2), llc)

    new_H_outline = [llc, bps1]
    w2 = trans_pt(trot(-math.pi / 3), w)
    new_H_outline.append(padd(new_H_outline[1], w2))
    new_H_outline.append(patch.eval_child(14, 2))
    w3 = trans_pt(trot(-math.pi / 3), w2)
    new_H_outline.append(psub(new_H_outline[3], w3))
    new_H_outline.append(patch.eval_child(6, 2))
    new_H = MetaTile(new_H_outline, patch.width * 2)
    for ch in [0, 9, 16, 27, 26, 6, 1, 8, 10, 15]:
        new_H.add_child(*patch.children[ch])

    new_P = MetaTile([p72, padd(p72, psub(bps1, llc)), bps1, llc],
                     patch.width * 2)
    for ch in [7, 2, 3, 4, 28]:
        new_P.add_child(*patch.children[ch])

    new_F = MetaTile([bps2, patch.eval_child(24, 2), patch.eval_child(25, 0),
                      p252, padd(p252, psub(llc, bps1))], patch.width * 2)
    for ch in [21, 20, 22, 23, 24, 25]:
        new_F.add_child(*patch.children[ch])

    AAA = new_H_outline[2]
    BBB = padd(new_H_outline[1], psub(new_H_outline[4], new_H_outline[5]))
    CCC = trans_pt(rot_about(BBB, -math.pi / 3), AAA)
    new_T = MetaTile([BBB, CCC, AAA], patch.width * 2)
    new_T.add_child(*patch.children[11])

    for m in (new_H, new_P, new_F, new_T):
        m.recentre()
    return (new_H, new_T, new_P, new_F)


def generate_hat_tiling(level: int):
    """(H, T, P, F) metatiles at the requested substitution level.

    level 1 is the base metatile (H1 holds 4 hats), each further level
    applies one substitution.
    """
    tiles = [_construct_base_H(), _construct_base_T(),
             _construct_base_P(), _construct_base_F()]
    for _ in range(level - 1):
        tiles = _construct_metatiles(_construct_patch(*tiles))
    return tiles


def metatile(level: int, tile: str = "H") -> MetaTile:
    """The single requested metatile at the requested level."""
    return generate_hat_tiling(level)[{"H": 0, "T": 1, "P": 2, "F": 3}[tile]]


def iter_hats(tile) -> Iterator[Tuple[List[float], str]]:
    """Yield (world transform, hat label) for every hat in the hierarchy,
    in deterministic depth-first order (so hat ids are reproducible)."""

    def _walk(node, T):
        if isinstance(node, HatTile):
            yield (T, node.label)
            return
        for childT, child in node.children:
            yield from _walk(child, mul(T, childT))

    yield from _walk(tile, [1, 0, 0, 0, 1, 0])
