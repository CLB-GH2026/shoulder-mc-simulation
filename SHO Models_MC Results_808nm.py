# SHO002 — Awaiting STL Files

Place the following STL files in this directory to run the simulation:

- `humerus_raw.stl`
- `scapula_raw.stl`
- `clavicle_raw.stl`
- `humeral_cartilage_raw.stl`
- `glenoid_cartilage_raw.stl`
- `labrum_raw.stl`

## Sourcing

See the repository CLAUDE.md for recommended sources (BodyParts3D, TotalSegmentator, SpineWeb, SimTK).

## Coordinate Convention

All meshes must share a common coordinate system:
- **+Z = superior** (cranial)
- **+Y = anterior** (ventral)
- **+X = lateral** (right side of body)

The pipeline auto-corrects Z-axis inversion via `AUTO_ORIENT = True`.
Meshes from BodyParts3D are already in this convention.
