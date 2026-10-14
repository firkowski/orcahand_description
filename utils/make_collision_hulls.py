# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================
"""Generate the simplified convex collision hulls of the v2 RIGHT hand.

Each entry of HULLS is the convex hull of one or more source STLs, optionally
clipped to the intersection of half-spaces, reduced to at most MAX_VERTS
vertices by greedy farthest-point insertion: start from the axis extremes and
repeatedly add the source hull vertex lying farthest outside the current hull.
The result is written in mm (part frame, same frame and scale as this repo's
part meshes) to v2/models/assets/right/collision/<name>.stl.

Finger, thumb and palm hulls are built from the printed parts in
orcahand/orcahand_hardware (orca_v2/base), which share this repo's part frames;
this repo's palm mesh is an older revision whose surface lies up to 4 mm
outside the printed palm. The tower keeps this repo's filled envelope meshes.

A single hull bridges concavities that other links move through, and URDF
simulators with self-collision only filter parent-child pairs, so:
  - the palm is cut with planes along its real features: each knuckle housing
    is a clevis whose slot (y ~ 0) holds the abduction link and the proximal
    phalanx knuckle, the knuckles are separated at the valleys between them,
    and the thumb pocket is split along its floor, rim, wall and heel;
  - each finger's proximal phalanx bone is split at its neck (z = 9 mm).
With these, the hulls stay clear of the palm over the full abduction x MCP and
thumb CMC x abduction ranges wherever the printed parts do.

The left hulls are mirrored from these by utils/mirror_left_from_right.py.

Usage:
    python utils/make_collision_hulls.py [--hardware ../orcahand_hardware]
"""

import argparse
import functools
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = REPO_ROOT / "v2" / "models" / "assets" / "right"
OUT_DIR = MESH_DIR / "collision"
MAX_VERTS = 64


def z(lo=None, hi=None):
    """Half-spaces lo <= z <= hi (mm); None leaves that side open."""
    return ([((0, 0, 1), lo)] if lo is not None else []) + ([((0, 0, -1), -hi)] if hi is not None else [])


def x(lo=None, hi=None):
    return ([((1, 0, 0), lo)] if lo is not None else []) + ([((-1, 0, 0), -hi)] if hi is not None else [])


def y(lo=None, hi=None):
    return ([((0, 1, 0), lo)] if lo is not None else []) + ([((0, -1, 0), -hi)] if hi is not None else [])


# Printed palm: carpals, cast skin, core and teeth (orcahand_hardware, orca_v2/base).
PALM = ["02_Carpals/R-Carpals.stl", "09_Skin/Carpals_R-Carpals_Skin.stl",
        "02_Carpals/CORE-R.stl", "02_Carpals/CarpalsTeeth.stl"]

# Output name -> (source STLs, half-spaces). Sources are paths in
# orcahand_hardware's orca_v2/base, or "repo:<stem>" for this repo's meshes.
# Palm frame: z towards the fingers, +y palmar, -x thumb side.
HULLS = {
    # Tower: filled hulls, internals ignored.
    "ForeArmStructure-Model": (["repo:ForeArmStructure-Model"], []),
    "TopTower-Model": (["repo:TopTower-Model"], []),
    # Palm, wrist to knuckles.
    "R-Carpals_wrist": (PALM, z(None, -40)),
    "R-Carpals_heel": (PALM, z(-40, -30) + x(None, -8)),
    "R-Carpals_center_lo": (PALM, z(-40, -21) + x(-15.5, None)),
    "R-Carpals_center_hi": (PALM, z(-21, -5) + x(-15.5, None)),
    "R-Carpals_thumb_floor": (PALM, z(-30, -9) + x(None, -15.5)),
    "R-Carpals_thumb_top_out": (PALM, z(-9, 1) + x(None, -26)),
    "R-Carpals_thumb_top_in": (PALM, z(-9, 1) + x(-26, -15.5)),
    "R-Carpals_plate": (PALM, z(-5, 15) + x(-15.5, None)),
    "R-Carpals_thenar_out": (PALM, z(1, 15) + x(None, -26)),
    "R-Carpals_thenar_in": (PALM, z(1, 15) + x(-26, -15.5)),
    # Knuckle housings: dorsal (y < 0) and palmar (y > 0) walls of each slot.
    "R-Carpals_index_dorsal": (PALM, z(15) + x(None, -15) + y(None, 0)),
    "R-Carpals_index_palmar": (PALM, z(15) + x(None, -15) + y(0)),
    "R-Carpals_middle_base": (PALM, z(15, 25) + x(-15, 11.5)),
    "R-Carpals_middle_dorsal": (PALM, z(25) + x(-12, 9.5) + y(None, 0)),
    "R-Carpals_middle_palmar_lo": (PALM, z(25, 32.5) + x(-12, 9.5) + y(0)),
    "R-Carpals_middle_palmar_hi": (PALM, z(32.5) + x(-12, 9.5) + y(0)),
    "R-Carpals_ring_dorsal": (PALM, z(15) + x(11.5, 28.5) + y(None, 0)),
    "R-Carpals_ring_palmar": (PALM, z(15) + x(11.5, 28.5) + y(0)),
    "R-Carpals_ring_pinky_step": (PALM, z(15) + x(28.5, 32)),
    "R-Carpals_pinky_dorsal": (PALM, z(15) + x(32) + y(None, 0)),
    "R-Carpals_pinky_palmar": (PALM, z(15) + x(32) + y(0)),
    # Fingers: abduction link, then bone + skin per phalanx; the proximal
    # phalanx bone is split at its neck. Middle and ring share the M-* hulls.
    **{name: spec for c in ("I", "M", "P") for name, spec in {
        f"{c}-AP" if c != "I" else "I-AP-R": ([f"01_Fingers/{c}-AP{'-R' if c == 'I' else ''}.stl"], []),
        f"{c}-PP_PP_base": ([f"01_Fingers/{c}-PP.stl"], z(None, 9)),
        f"{c}-PP_PP_shaft": ([f"01_Fingers/{c}-PP.stl"], z(9)),
        f"{c}-PP_Skin": ([f"09_Skin/PP-Skin_{c}-PP-Skin_Skin.stl"], []),
        f"{c}-IP_IP": ([f"01_Fingers/{c}-IP.stl"], []),
        f"{c}-DP-Skin": ([f"09_Skin/DP-Skin_{c}-DP-Skin_Skin.stl"], []),
    }.items()},
    # Thumb: two base links, then bone + skin per phalanx.
    "T-TP-R": (["01_Fingers/T-TP-R.stl"], []),
    "R-T-AP": (["01_Fingers/R-T-AP.stl"], []),
    "T-PP_PP": (["01_Fingers/T-PP.stl"], []),
    "T-PP_Skin": (["09_Skin/PP-Skin_T-PP-Skin_Skin.stl"], []),
    "T-DP_T-DP": (["01_Fingers/T-DP.stl"], []),
    "T-DP_Skin": (["09_Skin/T-DP_Skin.stl"], []),
}


def clip(meshes, planes):
    """Surface points of the meshes inside all half-spaces n . p >= c (mm).

    Vertices inside plus the points where mesh edges cross a bounding plane,
    so neighbouring pieces share their cut faces exactly.
    """
    pts = []
    for mesh in meshes:
        v = mesh.vertices
        a, b = v[mesh.edges_unique[:, 0]], v[mesh.edges_unique[:, 1]]
        pts.append(v)
        for n, c in planes:
            da, db = a @ n - c, b @ n - c
            cross = da * db < 0
            t = da[cross] / (da[cross] - db[cross])
            pts.append(a[cross] + t[:, None] * (b[cross] - a[cross]))
    pts = np.vstack(pts)
    for n, c in planes:
        pts = pts[pts @ np.asarray(n, float) >= c - 1e-9]
    return pts


def reduced_hull(points, max_verts=MAX_VERTS):
    """Greedy inner approximation of the convex hull with <= max_verts vertices."""
    pts = points[ConvexHull(points).vertices]
    if len(pts) > max_verts:
        sel = sorted({int(f(pts[:, a])) for a in range(3) for f in (np.argmin, np.argmax)})
        while len(sel) < max_verts:
            eq = ConvexHull(pts[sel]).equations
            sel.append(int((pts @ eq[:, :3].T + eq[:, 3]).max(axis=1).argmax()))
        pts = pts[sel]
    return trimesh.Trimesh(pts).convex_hull


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hardware", type=Path, default=REPO_ROOT.parent / "orcahand_hardware",
                        help="path to a checkout of orcahand/orcahand_hardware")
    args = parser.parse_args()
    base = args.hardware / "orca_v2" / "base"

    @functools.cache
    def load(src):
        f = MESH_DIR / f"{src[5:]}.stl" if src.startswith("repo:") else base / src
        return trimesh.load_mesh(f, force="mesh")

    OUT_DIR.mkdir(exist_ok=True)
    for stale in set(p.stem for p in OUT_DIR.glob("*.stl")) - set(HULLS):
        (OUT_DIR / f"{stale}.stl").unlink()
        print(f"removed stale {stale}.stl")
    for name, (sources, planes) in HULLS.items():
        points = clip([load(s) for s in sources], planes)
        full = trimesh.Trimesh(points).convex_hull
        hull = reduced_hull(points)
        hull.export(OUT_DIR / f"{name}.stl")
        print(f"{name:28s} {len(full.vertices):5d} -> {len(hull.vertices):2d} verts, "
              f"volume {hull.volume / full.volume:6.1%} of full hull")


if __name__ == "__main__":
    main()
