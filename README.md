[README.md](https://github.com/user-attachments/files/28965975/README.md)
# PBM Monte Carlo Simulation — Multi-Joint

Photobiomodulation (PBM) photon transport simulations for Kineon Move+ therapeutic
light device placements across four musculoskeletal targets.  All simulations use
GPU-accelerated Monte Carlo (pmcx) with anatomically-derived tissue geometry.

## Repository Structure

```
knee-mc-simulation/         ← Existing knee pipeline (OpenKnee STL source)
shoulder/
├── CLAUDE.md
└── Scripts/
    ├── SHO Models_MC Results_808nm.py
    ├── SHO Models_MC Results_650nm.py
    └── Raw_Mesh_Files_SHO###/   ← STL files go here (see PLACEHOLDER.md)
elbow/
├── CLAUDE.md
└── Scripts/
    ├── ELB Models_MC Results_808nm.py
    ├── ELB Models_MC Results_650nm.py
    └── Raw_Mesh_Files_ELB###/
lower_back/
├── CLAUDE.md
└── Scripts/
    ├── LBK Models_MC Results_808nm.py
    ├── LBK Models_MC Results_650nm.py
    └── Raw_Mesh_Files_LBK###/
```

## Joint Comparison

| Target | Tissue Depth (posterior) | Primary Clinical Indication | MUSCLE_THICK_MM |
|---|---|---|---|
| Knee | ~3.5 cm to joint space | OA, chondromalacia | 8 mm |
| Shoulder | ~4.5 cm to GH joint | Rotator cuff, OA | 20 mm |
| Elbow | ~2.0 cm to radiocapitellar | Lateral epicondylitis | 10 mm |
| Lower Back | ~6.5 cm to L4/L5 disc | Disc pain, LBP | 35 mm |

## Pipeline Architecture

All four joint simulations share the same pipeline functions — only the
tissue table, grid dimensions, wrapping thicknesses, orientation check,
and source positions differ per joint.  See each joint's `CLAUDE.md` for
the specific values.

## STL File Sources

See individual `CLAUDE.md` files.  Recommended sources:

- **Bone geometry**: [BodyParts3D GitHub mirror](https://github.com/Kevin-Mattheus-Moerman/BodyParts3D) (CC-BY-SA)
- **Lumbar discs**: [SpineWeb](http://spineweb.digitalimaginggroup.ca/)
- **Auto-segmentation from CT**: TotalSegmentator (`pip install TotalSegmentator`)
- **Visible Human DICOM**: [University of Denver](https://digitalcommons.du.edu/visiblehuman/) (CC-BY 4.0)

## Dependencies

```bash
pip install numpy trimesh pmcx plotly scipy
```

Requires a CUDA-capable GPU for pmcx (`gpuid=1`).
