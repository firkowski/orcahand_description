# ==============================================================================
# Copyright (c) 2025 ORCA
#
# This file is part of ORCA and is licensed under the MIT License.
# You may use, copy, modify, and distribute this file under the terms of the MIT License.
# See the LICENSE file at the root of this repository for full license information.
# ==============================================================================
"""Write the inertial properties of the v2 RIGHT hand into its URDF and MJCF.

Each link is the sum of its parts, each part a uniform solid of known mass:

  - Printed PLA parts: the mass Bambu Studio reports when slicing
    orcahand_hardware's orca_v2/base/Prints-1100.3mf with its own settings
    (eSUN PLA+ 1.25 g/cm3, 15% gyroid, 2 walls, 6/4 top/bottom layers),
    support excluded. Geometry: the printed part's STL from orcahand_hardware,
    which is in the same frame as this repo's part meshes.
  - Cast skins: the skin STL volume at SKIN_DENSITY (assumed silicone).
  - The forearm uses this repo's ForeArmStructure envelope as geometry, since
    the printed structure is a different shape.
  - Servos (Feetech, models from orca_core's MOTOR_MODELS; masses from
    retailer specs, no official datasheet found): 16 HLS3915 finger servos in
    the forearm, one HLS3930 wrist servo in the tower. Their exact pockets are
    not modelled: each mass is spread uniformly over the link's envelope mesh.
    Spools, hulls, fans, electronics, cables, bearings and screws are NOT
    included.

The centre of mass becomes the <inertial> origin (URDF, link frame) / the
<inertial pos> (MJCF, body frame) and the tensor is taken about it. The MJCF
<inertial> overrides the geom-derived inertia. Run
utils/mirror_left_from_right.py afterwards to regenerate the left hand.

Usage:
    python utils/compute_inertials.py [--hardware ../orcahand_hardware]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF = REPO_ROOT / "v2" / "models" / "urdf" / "orcahand_right.urdf"
BODY = REPO_ROOT / "v2" / "models" / "mjcf" / "orcahand_right_body.xml"
MESH_DIR = REPO_ROOT / "v2" / "models" / "assets" / "right"
SKIN_DENSITY = 1100.0  # kg/m3
HLS3915_G = 35.8  # finger servo (HL-3915-C001), 35.8 +- 2 g
HLS3930_G = 70.5  # wrist servo (HL-3930-C001), 70.5 +- 1 g

# Link -> parts: (STL path relative to orca_v2/base of orcahand_hardware, or
# "repo:<file>" for this repo's right mesh dir; printed mass in g, or None for
# a cast skin at SKIN_DENSITY).
LINKS = {
    "right_forearm": [("repo:ForeArmStructure-Model.stl", 86.6),
                      ("repo:ForeArmStructure-Model.stl", 16 * HLS3915_G)],
    "right_tower": [("03_Wrist/WristStructure_Dynamixel.stl", 61.6),
                    ("repo:TopTower-Model.stl", HLS3930_G)],
    "right_carpals": [
        ("02_Carpals/R-Carpals.stl", 54.21),
        ("02_Carpals/CORE-R.stl", 3.23),
        ("02_Carpals/CarpalsTeeth.stl", 1.39),
        ("09_Skin/Carpals_R-Carpals_Skin.stl", None),
    ],
    "right_index_ap": [("01_Fingers/I-AP-R.stl", 3.49)],
    "right_index_pp": [("01_Fingers/I-PP.stl", 5.47), ("09_Skin/PP-Skin_I-PP-Skin_Skin.stl", None)],
    "right_index_fingertip_assembly": [("01_Fingers/I-IP.stl", 2.82), ("09_Skin/DP-Skin_I-DP-Skin_Skin.stl", None)],
    "right_middle_ap": [("01_Fingers/M-AP.stl", 3.71)],
    "right_middle_pp": [("01_Fingers/M-PP.stl", 5.875), ("09_Skin/PP-Skin_M-PP-Skin_Skin.stl", None)],
    "right_middle_fingertip_assembly": [("01_Fingers/M-IP.stl", 2.81), ("09_Skin/DP-Skin_M-DP-Skin_Skin.stl", None)],
    "right_ring_ap": [("01_Fingers/M-AP.stl", 3.71)],
    "right_ring_pp": [("01_Fingers/M-PP.stl", 5.875), ("09_Skin/PP-Skin_M-PP-Skin_Skin.stl", None)],
    "right_ring_fingertip_assembly": [("01_Fingers/M-IP.stl", 2.81), ("09_Skin/DP-Skin_M-DP-Skin_Skin.stl", None)],
    "right_pinky_ap": [("01_Fingers/P-AP.stl", 3.07)],
    "right_pinky_pp": [("01_Fingers/P-PP.stl", 4.19), ("09_Skin/PP-Skin_P-PP-Skin_Skin.stl", None)],
    "right_pinky_fingertip_assembly": [("01_Fingers/P-IP.stl", 2.04), ("09_Skin/DP-Skin_P-DP-Skin_Skin.stl", None)],
    "right_thumb_tp": [("01_Fingers/T-TP-R.stl", 5.54)],
    "right_thumb_ap": [("01_Fingers/R-T-AP.stl", 5.07)],
    "right_thumb_pp": [("01_Fingers/T-PP.stl", 5.26), ("09_Skin/PP-Skin_T-PP-Skin_Skin.stl", None)],
    "right_thumb_dp": [("01_Fingers/T-DP.stl", 2.24), ("09_Skin/T-DP_Skin.stl", None)],
}


def link_inertia(parts, hardware):
    """Mass (kg), COM (m) and inertia about the COM (kg m2) in the part frame."""
    masses, coms, inertias = [], [], []
    for path, grams in parts:
        f = MESH_DIR / path[5:] if path.startswith("repo:") else hardware / "orca_v2" / "base" / path
        mesh = trimesh.load_mesh(f, force="mesh")
        assert mesh.is_watertight, f"{f} is not watertight"
        mesh.apply_scale(1e-3)
        mass = SKIN_DENSITY * mesh.volume if grams is None else grams * 1e-3
        mesh.density = mass / mesh.volume
        masses.append(mass)
        coms.append(mesh.center_mass)
        inertias.append(mesh.moment_inertia)
    m = sum(masses)
    com = sum(mi * ci for mi, ci in zip(masses, coms)) / m
    inertia = sum(I + mi * ((d @ d) * np.eye(3) - np.outer(d, d))
                  for mi, ci, I in zip(masses, coms, inertias) for d in [ci - com])
    return m, com, inertia


def fmt(v):
    return f"{v:.6g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hardware", type=Path, default=REPO_ROOT.parent / "orcahand_hardware",
                        help="path to a checkout of orcahand/orcahand_hardware")
    args = parser.parse_args()

    urdf, body = URDF.read_text(), BODY.read_text()
    for link, parts in LINKS.items():
        m, com, I = link_inertia(parts, args.hardware)
        tensor = (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])

        # URDF: link frame = part frame shifted by the (unrotated) visual origin.
        block = re.search(rf'<link name="{link}">.*?</link>', urdf, re.S).group(0)
        vxyz, vrpy = re.search(r'<visual>\s*<origin xyz="([^"]+)" rpy="([^"]+)"', block).groups()
        assert all(float(a) == 0 for a in vrpy.split()), f"{link}: rotated visual origin"
        origin = np.array([float(a) for a in vxyz.split()]) + com
        new = re.sub(
            r'<inertial>.*?</inertial>',
            lambda _: (
                f'<inertial>\n\t\t\t<origin xyz="{" ".join(fmt(a) for a in origin)}" rpy="0.0 0.0 0.0"/>\n'
                f'\t\t\t<mass value="{fmt(m)}"/>\n'
                '\t\t\t<inertia ' + " ".join(f'{k}="{fmt(v)}"' for k, v in
                                             zip(("ixx", "iyy", "izz", "ixy", "ixz", "iyz"), tensor)) + '/>\n'
                '\t\t</inertial>'),
            block, count=1, flags=re.S)
        urdf = urdf.replace(block, new)

        # MJCF: <inertial> as the body's first child, replacing any previous one.
        line = re.search(rf'^(\t*)<body name="{link}"[^\n]*\n', body, re.M)
        inertial = (f'{line.group(1)}\t<inertial pos="{" ".join(fmt(a) for a in com)}" mass="{fmt(m)}" '
                    f'fullinertia="{" ".join(fmt(v) for v in tensor)}" />\n')
        rest = body[line.end():]
        if rest.lstrip("\t").startswith("<inertial "):
            rest = rest[rest.index("\n") + 1:]
        body = body[:line.end()] + inertial + rest
        print(f"{link:32s} {m * 1e3:7.2f} g  COM {np.round(com * 1e3, 1)} mm")

    URDF.write_text(urdf)
    BODY.write_text(body)


if __name__ == "__main__":
    main()
