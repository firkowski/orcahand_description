# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================
"""Generate the simplified convex collision hulls of the v2 RIGHT hand.

Each entry of HULLS is the convex hull of one visual STL (optionally clipped
to a slab along one mesh axis), reduced to at most MAX_VERTS vertices by
greedy farthest-point insertion: start from the axis extremes and repeatedly
add the source hull vertex lying farthest outside the current hull. The
result is written in mm (same frame and scale as the source STL) to
v2/models/assets/right/collision/<name>.stl, so a collision geom with the
visual geom's pose overlays the visual mesh exactly.

The left hulls are mirrored from these by utils/mirror_left_from_right.py.

Usage:
    python utils/make_collision_hulls.py
"""

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = REPO_ROOT / "v2" / "models" / "assets" / "right"
OUT_DIR = MESH_DIR / "collision"
MAX_VERTS = 64

# Output name -> (source STL stem, clip slab (axis, lo_mm, hi_mm) or None).
HULLS = {
    # Tower: filled hulls, internals ignored.
    "ForeArmStructure-Model": ("ForeArmStructure-Model", None),
    "TopTower-Model": ("TopTower-Model", None),
    # Palm (body frame: z towards the fingers, +y palmar): distal and proximal
    # palm plus the wrist bracket, so the hull doesn't bridge the concavities.
    "R-Carpals_distal": ("R-Carpals", (2, -5.0, None)),
    "R-Carpals_proximal": ("R-Carpals", (2, -40.0, -5.0)),
    "R-Carpals_wrist": ("R-Carpals", (2, None, -40.0)),
    # Fingers: abduction link, then one bone + one skin hull per phalanx.
    # Middle and ring share the M-* hulls.
    "I-AP-R": ("I-AP-R", None),
    "I-PP_PP": ("I-PP_PP", None),
    "I-PP_Skin": ("I-PP_Skin", None),
    "I-IP_IP": ("I-IP_IP", None),
    "I-DP-Skin": ("I-FingerTipAssembly_I-DP-Skin", None),
    "M-AP": ("M-AP", None),
    "M-PP_PP": ("M-PP_PP", None),
    "M-PP_Skin": ("M-PP_Skin", None),
    "M-IP_IP": ("M-IP_IP", None),
    "M-DP-Skin": ("M-FingerTipAssembly_M-DP-Skin", None),
    "P-AP": ("P-AP", None),
    "P-PP_PP": ("P-PP_PP", None),
    "P-PP_Skin": ("P-PP_Skin", None),
    "P-IP_IP": ("P-IP_IP", None),
    "P-DP-Skin": ("P-FingerTipAssembly_P-DP-Skin", None),
    # Thumb: two base links, then bone + skin per phalanx.
    "T-TP-R": ("T-TP-R", None),
    "R-T-AP": ("R-T-AP", None),
    "T-PP_PP": ("T-PP_PP", None),
    "T-PP_Skin": ("T-PP_Skin", None),
    "T-DP_T-DP": ("T-DP_T-DP", None),
    "T-DP_Skin": ("T-DP_Skin", None),
}


def clip(mesh, axis, lo, hi):
    """Points of the mesh surface between lo and hi (mm) along axis.

    Vertices inside the slab plus the points where mesh edges cross its
    bounding planes, so neighbouring slabs share their cut faces exactly.
    """
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi
    v = mesh.vertices
    pts = [v[(v[:, axis] >= lo) & (v[:, axis] <= hi)]]
    a, b = v[mesh.edges_unique[:, 0]], v[mesh.edges_unique[:, 1]]
    for c in (lo, hi):
        if np.isfinite(c):
            cross = (a[:, axis] - c) * (b[:, axis] - c) < 0
            t = (c - a[cross, axis]) / (b[cross, axis] - a[cross, axis])
            pts.append(a[cross] + t[:, None] * (b[cross] - a[cross]))
    return np.vstack(pts)


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
    OUT_DIR.mkdir(exist_ok=True)
    for name, (src, slab) in HULLS.items():
        mesh = trimesh.load_mesh(MESH_DIR / f"{src}.stl", force="mesh")
        points = mesh.vertices if slab is None else clip(mesh, *slab)
        full = trimesh.Trimesh(points).convex_hull
        hull = reduced_hull(points)
        hull.export(OUT_DIR / f"{name}.stl")
        print(f"{name:28s} {len(full.vertices):5d} -> {len(hull.vertices):2d} verts, "
              f"volume {hull.volume / full.volume:6.1%} of full hull")


if __name__ == "__main__":
    main()
