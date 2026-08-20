"""
Pythonised script based on https://github.com/isohedral/hatviz with a modified hat unit cell to 
generate a bipartite hat metatile and contain neccessary internal verticies. 
"""

import math
from typing import List, Tuple

def pt(x: float, y: float) -> Tuple[float, float]:
    """Construct a point (x,y)."""
    return (x, y)

def hex_pt(x: float, y: float) -> Tuple[float, float]:
    """Return the hex-coordinates turned into standard 2D points,
       used for the 'hat' shape definitions."""
    hr3 = math.sqrt(3) / 2
    return (x + 0.5*y, hr3*y)

def padd(p: Tuple[float, float], q: Tuple[float, float]) -> Tuple[float, float]:
    return (p[0] + q[0], p[1] + q[1])

def psub(p: Tuple[float, float], q: Tuple[float, float]) -> Tuple[float, float]:
    return (p[0] - q[0], p[1] - q[1])

def mag(x: float, y: float) -> float:
    return math.hypot(x, y)

def trot(ang: float) -> List[float]:
    """Return a 2D rotation matrix (in 2x3 'affine' form)."""
    c = math.cos(ang)
    s = math.sin(ang)
    return [c, -s, 0, s, c, 0]

def ttrans(tx: float, ty: float) -> List[float]:
    return [1, 0, tx, 0, 1, ty]

def mul(A: List[float], B: List[float]) -> List[float]:
    """Multiply two affine transformations A*B (each 2x3)."""
    aA, bA, txA, cA, dA, tyA = A
    aB, bB, txB, cB, dB, tyB = B
    return [
        aA*aB + bA*cB,
        aA*bB + bA*dB,
        aA*txB + bA*tyB + txA,
        cA*aB + dA*cB,
        cA*bB + dA*dB,
        cA*txB + dA*tyB + tyA
    ]

def inv(T: List[float]) -> List[float]:
    """Affine matrix inverse for 2D transforms."""
    a, b, tx, c, d, ty = T
    det = a*d - b*c
    if abs(det) < 1e-14:
        raise ValueError("Degenerate or nearly degenerate transform.")
    return [
        d/det, -b/det, (b*ty - tx*d)/det,
        -c/det, a/det, (tx*c - a*ty)/det
    ]

def trans_pt(M: List[float], P: Tuple[float, float]) -> Tuple[float, float]:
    """Apply affine transform M to point P."""
    x, y = P
    a, b, tx, c, d, ty = M
    return (a*x + b*y + tx, c*x + d*y + ty)

def rot_about(p: Tuple[float, float], ang: float) -> List[float]:
    return mul(ttrans(p[0], p[1]),
               mul(trot(ang),
                   ttrans(-p[0], -p[1])))

def match_seg(p: Tuple[float, float], q: Tuple[float, float]) -> List[float]:
    return [
        q[0] - p[0],  # a
        p[1] - q[1],  # b
        p[0],         # tx
        q[1] - p[1],  # c
        q[0] - p[0],  # d
        p[1]          # ty
    ]

def match_two(p1: Tuple[float, float],
              q1: Tuple[float, float],
              p2: Tuple[float, float],
              q2: Tuple[float, float]) -> List[float]:
    return mul(match_seg(p2, q2), inv(match_seg(p1, q1)))

def intersect(p1, q1, p2, q2):
    denom = (q2[1] - p2[1])*(q1[0] - p1[0]) - (q2[0] - p2[0])*(q1[1] - p1[1])
    if abs(denom) < 1e-14:
        raise ValueError("Lines are parallel or nearly parallel.")
    ua = ((q2[0] - p2[0])*(p1[1] - p2[1]) - (q2[1] - p2[1])*(p1[0] - p2[0])) / denom
    ix = p1[0] + ua*(q1[0] - p1[0])
    iy = p1[1] + ua*(q1[1] - p1[1])
    return (ix, iy)

class HatTile:
    def __init__(self, label: str):
        self.label = label
        self.shape = hat_outline

class MetaTile:
    def __init__(self, shape: List[Tuple[float, float]], width: float):
        self.shape = shape
        self.width = width
        self.children = []

    def add_child(self, T: List[float], geom):
        self.children.append((T, geom))

    def eval_child(self, n: int, i: int) -> Tuple[float, float]:
        T, g = self.children[n]
        return trans_pt(T, g.shape[i])

    def recentre(self):
        if len(self.shape) == 0:
            return
        cx = sum(p[0] for p in self.shape) / len(self.shape)
        cy = sum(p[1] for p in self.shape) / len(self.shape)
        new_shape = [(x - cx, y - cy) for (x, y) in self.shape]
        self.shape = new_shape
        shift = ttrans(-cx, -cy)
        new_children = []
        for (Tm, geom) in self.children:
            new_children.append((mul(shift, Tm), geom))
        self.children = new_children

# define hat outline with additional verticies added for tetrille embedding. 
r3  = math.sqrt(3)
hr3 = math.sqrt(3) / 2

hat_outline = [
    hex_pt(0,  0),  hex_pt(-1, -1), hex_pt(0, -2),  hex_pt(2, -2),
    hex_pt(2, -1), hex_pt(4,  -2), hex_pt(5, -1),  hex_pt(4,  0),
    hex_pt(3,  0), hex_pt(2,   2), hex_pt(0,  3),  hex_pt(0,  2),
    hex_pt(-1,  2), (0, -r3), (2, 0), (1.5, hr3)
]

hat_H1 = HatTile('H1')
hat_H  = HatTile('H')
hat_T  = HatTile('T')
hat_P  = HatTile('P')
hat_F  = HatTile('F')

def construct_base_H():
    H_outline = [
        pt(0, 0), pt(4, 0), pt(4.5, hr3),
        pt(2.5, 5 * hr3), pt(1.5, 5 * hr3), pt(-0.5, hr3)
    ]
    meta = MetaTile(H_outline, 2.0)
    M1 = match_two(hat_outline[5], hat_outline[7], H_outline[5], H_outline[0])
    meta.add_child(M1, hat_H)
    M2 = match_two(hat_outline[9], hat_outline[11], H_outline[1], H_outline[2])
    meta.add_child(M2, hat_H)
    M3 = match_two(hat_outline[5], hat_outline[7], H_outline[3], H_outline[4])
    meta.add_child(M3, hat_H)
    T4a = [-0.5, -hr3, 0, hr3, -0.5, 0] 
    T4b = [0.5, 0, 0, 0, -0.5, 0]
    T4 = mul(ttrans(2.5, hr3), mul(T4a, T4b))
    meta.add_child(T4, hat_H1)
    return meta

def construct_base_T():
    T_outline = [pt(0, 0), pt(3, 0), pt(1.5, 3 * hr3)]
    meta = MetaTile(T_outline, 2.0)
    M = [0.5, 0, 0.5, 0, 0.5, hr3]
    meta.add_child(M, hat_T)
    return meta

def construct_base_P():
    P_outline = [pt(0, 0), pt(4, 0), pt(3, 2 * hr3), pt(-1, 2 * hr3)]
    meta = MetaTile(P_outline, 2.0)
    M1 = [0.5, 0, 1.5, 0, 0.5, hr3]
    meta.add_child(M1, hat_P)
    M2 = mul(ttrans(0, 2 * hr3),
             mul([0.5, hr3, 0, -hr3, 0.5, 0],
                 [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]))
    meta.add_child(M2, hat_P)
    return meta

def construct_base_F():
    F_outline = [pt(0, 0), pt(3, 0), pt(3.5, hr3), pt(3, 2 * hr3), pt(-1, 2 * hr3)]
    meta = MetaTile(F_outline, 2.0)
    M1 = [0.5, 0, 1.5, 0, 0.5, hr3]
    meta.add_child(M1, hat_F)
    M2 = mul(ttrans(0, 2 * hr3),
             mul([0.5, hr3, 0, -hr3, 0.5, 0],
                 [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]))
    meta.add_child(M2, hat_F)
    return meta

base_H = construct_base_H()
base_T = construct_base_T()
base_P = construct_base_P()
base_F = construct_base_F()


def construct_patch(H: MetaTile, T: MetaTile, P: MetaTile, F: MetaTile) -> MetaTile:
    """Assembles a big patch from H, T, P, F using the substituion rules."""
    ret = MetaTile([], H.width)
    shapes = {'H': H, 'T': T, 'P': P, 'F': F}
    rules = [
        ['H'],
        [0, 0, 'P', 2],
        [1, 0, 'H', 2],
        [2, 0, 'P', 2],
        [3, 0, 'H', 2],
        [4, 4, 'P', 2],
        [0, 4, 'F', 3],
        [2, 4, 'F', 3],
        [4, 1, 3, 2, 'F', 0],
        [8, 3, 'H', 0],
        [9, 2, 'P', 0],
        [10, 2, 'H', 0],
        [11, 4, 'P', 2],
        [12, 0, 'H', 2],
        [13, 0, 'F', 3],
        [14, 2, 'F', 1],
        [15, 3, 'H', 4],
        [8, 2, 'F', 1],
        [17, 3, 'H', 0],
        [18, 2, 'P', 0],
        [19, 2, 'H', 2],
        [20, 4, 'F', 3],
        [20, 0, 'P', 2],
        [22, 0, 'H', 2],
        [23, 4, 'F', 3],
        [23, 0, 'F', 3],
        [16, 0, 'P', 2],
        [9, 4, 0, 2, 'T', 2],
        [4, 0, 'F', 3]
    ]
    for i, r in enumerate(rules):
        if len(r) == 1:
            ret.add_child([1, 0, 0, 0, 1, 0], shapes[r[0]])
        elif len(r) == 4:
            idx_ch, seg_i, shape_label, seg_j = r
            T_ch, geom_ch = ret.children[idx_ch]
            poly = geom_ch.shape
            P1 = trans_pt(T_ch, poly[(seg_i + 1) % len(poly)])
            Q1 = trans_pt(T_ch, poly[seg_i])
            new_geom = shapes[shape_label]
            new_poly = new_geom.shape
            transformM = match_two(new_poly[seg_j], new_poly[(seg_j + 1) % len(new_poly)],
                                   P1, Q1)
            ret.add_child(transformM, new_geom)
        else:
            cP = ret.children[r[0]]
            cQ = ret.children[r[2]]
            P1 = trans_pt(cQ[0], cQ[1].shape[r[3]])
            Q1 = trans_pt(cP[0], cP[1].shape[r[1]])
            shape_label = r[4]
            seg_j = r[5]
            new_geom = shapes[shape_label]
            new_poly = new_geom.shape
            transformM = match_two(new_poly[seg_j], new_poly[(seg_j + 1) % len(new_poly)],
                                   P1, Q1)
            ret.add_child(transformM, new_geom)
    return ret

def construct_metatiles(patch: MetaTile):
    """Slices the patch into bigger versions of H, T, P, F."""
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
        new_H.add_child(patch.children[ch][0], patch.children[ch][1])
    new_P_outline = [p72, padd(p72, psub(bps1, llc)), bps1, llc]
    new_P = MetaTile(new_P_outline, patch.width * 2)
    for ch in [7, 2, 3, 4, 28]:
        new_P.add_child(patch.children[ch][0], patch.children[ch][1])
    new_F_outline = [bps2,
                     patch.eval_child(24, 2),
                     patch.eval_child(25, 0),
                     p252,
                     padd(p252, psub(llc, bps1))]
    new_F = MetaTile(new_F_outline, patch.width * 2)
    for ch in [21, 20, 22, 23, 24, 25]:
        new_F.add_child(patch.children[ch][0], patch.children[ch][1])
    AAA = new_H_outline[2]
    BBB = padd(new_H_outline[1], psub(new_H_outline[4], new_H_outline[5]))
    CCC = trans_pt(rot_about(BBB, -math.pi / 3), AAA)
    new_T_outline = [BBB, CCC, AAA]
    new_T = MetaTile(new_T_outline, patch.width * 2)
    new_T.add_child(patch.children[11][0], patch.children[11][1])
    new_H.recentre()
    new_P.recentre()
    new_F.recentre()
    new_T.recentre()
    return (new_H, new_T, new_P, new_F)

def generate_hat_tiling(level: int):
    """
    Return a tuple (H, T, P, F) for the tiling at the requested substitution level.
    level = 1 -> base metatile shapes only
    level = 2 -> one substitution, etc.
    """
    tiles = [base_H, base_T, base_P, base_F]
    current_level = 1
    while current_level < level:
        patch = construct_patch(*tiles)
        tiles = construct_metatiles(patch)
        current_level += 1
    return tiles
