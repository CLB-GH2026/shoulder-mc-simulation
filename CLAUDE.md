# CLAUDE.md — Shoulder PBM Monte Carlo Simulation

This file provides guidance to Claude Code when working with code in this repository.
Mirrors the structure of the knee `knee-mc-simulation` repo.

## Active Scripts

All simulation scripts live in `Scripts/`:

| File | Purpose |
|---|---|
| `SHO Models_MC Results_808nm.py` | Batch pipeline — all subjects at 808 nm |
| `SHO Models_MC Results_650nm.py` | Batch pipeline — all subjects at 650 nm |

Run from the `Scripts/` directory:
```bash
python "SHO Models_MC Results_808nm.py"
```

## Project Overview

Monte Carlo photon transport simulation of PBM laser delivery to the glenohumeral
(shoulder) joint.  Models how 808 nm / 650 nm NIR/red light propagates through
skin → adipose → deltoid/rotator cuff muscle → joint capsule to reach the
glenohumeral cartilage and synovial fluid.

## Pipeline (per subject / per run)

Identical to the knee pipeline:

1. **STL → voxel label volume** (`build_label_volume`): loads one STL mesh per tissue,
   ray-casts along Z to fill an integer label array.  Auto-detects Z-axis inversion by
   comparing humerus vs scapula centroid Z (`AUTO_ORIENT = True`).
2. **Synovial fluid fill** (`add_synovial_fluid`): dilates cartilage + labrum mask to
   fill joint space with label 14.
3. **Soft-tissue wrapping** (`add_wrapping_layers`): binary dilation adds concentric
   muscle → adipose → skin shells.
4. **Epidermis labelling** (`add_epidermis_layer`): outermost skin voxel ring relabelled
   as epidermis (label 15).
5. **Joint-line Z detection** (`find_joint_line_z`): finds Z slice with peak
   cartilage + labrum + synovial density.
6. **Source placement** (`find_surface_source_positions`): 3 sources (1 posterior over
   infraspinatus, 2 anterior flanking the acromion) snapped to nearest epidermis voxel.
7. **pmcx simulation** (`run_pmcx`): GPU-accelerated MC.
8. **Analysis** (`analyze_fluence_absorption`): per-tissue mean fluence, absorbed power.
9. **Output**: depth histogram HTML + CSV results.

## Key Configuration Constants

```python
VOXEL_SIZE       = 1.0      # mm/voxel
GRID_DIMS_MM     = (180, 180, 200)   # wider/shallower than knee
AUTO_ORIENT      = True     # auto-correct Z-axis inversion (humerus vs scapula)
MUSCLE_THICK_MM  = 20       # deltoid/rotator cuff — reduce to 15 for anterior access
ADIPOSE_THICK_MM = 5
SKIN_THICK_MM    = 2
SOURCE_POWER_MW  = 50       # 808nm  (120 for 650nm)
SOURCE_DUTY_CYCLE = 0.75
SOURCE_OPT_EFF   = 0.85
```

## Coordinate System

- **+Z = superior** (humeral head above glenoid)
- **+Y = anterior**, −Y = posterior
- **+X = lateral** (away from body midline)
- Source `world_pos` is relative to `mesh_center` (bounding box midpoint)

## Tissue Labels

| Label | Tissue |
|---|---|
| 1 | Humerus bone |
| 2 | Scapula bone |
| 3 | Clavicle bone |
| 5 | Labrum (fibrocartilage) |
| 7 | Humeral head cartilage |
| 8 | Glenoid cartilage |
| 11 | Muscle (synthesised wrapping) |
| 12 | Adipose (synthesised wrapping) |
| 13 | Skin (synthesised wrapping) |
| 14 | Synovial fluid (synthesised) |
| 15 | Epidermis (synthesised) |

## Optical Properties (808 nm / 650 nm)

Properties follow `opt(µa, µs', g, n)` — same convention as knee pipeline.
Tissue values are identical to knee for shared tissue types (bone, cartilage,
muscle, adipose, skin, epidermis). Labrum uses the same values as knee meniscus
(fibrocartilage class).

## Required STL Files Per Subject

Place files in `Scripts/Raw_Mesh_Files_SHO###/`:

```
humerus_raw.stl
scapula_raw.stl
clavicle_raw.stl
humeral_cartilage_raw.stl
glenoid_cartilage_raw.stl
labrum_raw.stl
```

## Recommended STL Sources

| Tissue | Source | Notes |
|---|---|---|
| Humerus, scapula, clavicle | [BodyParts3D GitHub mirror](https://github.com/Kevin-Mattheus-Moerman/BodyParts3D) | CC-BY-SA 2.1 JP; already STL |
| Cartilage, labrum | 3D Slicer segmentation of Visible Human Male MRI | Segment glenohumeral joint from NLM DICOM |
| All tissues (auto) | TotalSegmentator on any shoulder CT | `pip install TotalSegmentator` |
| SimTK biomechanics models | [simtk.org/projects/wu2016shoulder](https://simtk.org/projects/wu2016shoulder) | FEM-registered geometry |

## Adding a New Subject

1. Create `Scripts/Raw_Mesh_Files_SHO###/` and place the 6 STL files inside.
2. Add `"SHO###"` to `SUBJECT_IDS` list in both scripts.
3. Run the 808 nm script; inspect the depth histogram HTML before running 650 nm.

## Key Differences from Knee Pipeline

- `GRID_DIMS_MM` is wider (180×180) and shallower (200 mm) than knee (150×140×285)
- `MUSCLE_THICK_MM = 20` vs knee's 8 mm — deltoid is substantially thicker
- Orientation check uses `humerus-bone` vs `scapula-bone` Z centroids (not femur/tibia)
- Joint line detection includes `labrum` labels (not meniscus)
- No patella geometry
