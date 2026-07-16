"""
Shoulder STL Mesh → 3D Voxel Volume + pmcx Fluence Overlay  (808 nm)
----------------------------------------------------------------------
Pipeline (shared logic lives in pbm_mc_core; see that package's README for
the full stage list and the tissue-label convention this script's `tissues`
dict follows):

Tissue hierarchy (highest label wins when meshes overlap):
  1  humerus-bone         Cortical/cancellous humerus
  2  scapula-bone         Scapula + glenoid fossa
  3  clavicle-bone        Clavicle (optional geometry anchor)
  7  humeral-cart         Humeral head articular cartilage
  8  glenoid-cart         Glenoid articular cartilage
  5  labrum               Fibrocartilaginous glenoid labrum
  11 muscle               Synthesised — concentric dilation (deltoid, rotator cuff)
  12 adipose              Synthesised — concentric dilation
  13 skin                 Synthesised — concentric dilation
  14 synovial             Synthesised — dilation of cartilage/labrum gap
  15 epidermis            Synthesised — outermost 1-voxel skin ring

Wrapping note:
  MUSCLE_THICK_MM is set conservatively at 20 mm.  In reality the deltoid is
  25–30 mm thick posteriorly and ~18 mm anteriorly over the subscapularis.
  Reduce to 15 mm to model the anterior/superior access window used by the
  Move+ shoulder attachment.

Source positions (default):
  +Y = anterior,  −Y = posterior,  +Z = superior
  Posterior source aims at mid-joint line; two anterior sources flank the
  acromion.  All three Z values are auto-set to the joint-line height.

Dependencies:
    pip install numpy trimesh pmcx plotly scipy
    pip install git+https://github.com/CLB-GH2026/pbm-mc-core.git@v0.1.1
"""

import numpy as np
import time
from pathlib import Path
from datetime import datetime

from pbm_mc_core import (
    opt, EPIDERMIS_LABEL, build_melanin_conditions,
    build_label_volume,
    add_synovial_fluid, add_wrapping_layers, add_epidermis_layer,
    find_joint_line_z, find_surface_source_positions,
    optimize_source_positions_reciprocity,
    run_pmcx,
    analyze_fluence_absorption, analyze_penetration_depth, plot_depth_histogram,
    target_depth_zone,
    results_to_csv, melanin_comparison_to_csv,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()

WAVELENGTH_M  = 808e-9
WAVELENGTH_NM = 808

# Target-tissue predicate shared by joint-line detection and the reciprocity
# source optimiser: shoulder's joint-relevant tissues are cartilage, labrum
# (fibrocartilage), and synovial fluid — the core library's default predicate
# only covers 'cart'/'synovial', so shoulder must supply its own to include
# the labrum.
TARGET_MATCH_FN = lambda name: ('cart' in name) or ('labrum' in name) or ('synovial' in name)

# Epidermal optical properties by melanin condition at 808 nm.
# True (unscaled) values; build_melanin_conditions() applies the epidermis
# thickness-correction scale (0.2 mm physical / 1 mm voxel).
_MELANIN_RAW_808NM = {
    #        µa      µs'    g     n
    'fair':  (0.008, 1.50, 0.80, 1.40),  # Fitzpatrick I-II
    'olive': (0.025, 1.60, 0.80, 1.40),  # Fitzpatrick III-IV
    'dark':  (0.075, 1.70, 0.80, 1.40),  # Fitzpatrick V-VI
}

# ── Source optimiser ──────────────────────────────────────────────────────────
OPTIMIZE_SOURCES = False   # True → per-subject reciprocity scan before main run
OPT_N_SOURCES    = 3
OPT_MIN_SEP_MM   = 25.0
OPT_NPHOTON      = 1e6

# ── Grid / voxel ─────────────────────────────────────────────────────────────
VOXEL_SIZE    = 1.0               # mm per voxel
GRID_DIMS_MM  = (180, 180, 200)   # x, y, z — shoulder is wider & shallower than knee
VOXEL_RES     = tuple(int(round(d / VOXEL_SIZE)) for d in GRID_DIMS_MM)
AUTO_ORIENT   = True              # auto-correct Z-axis inversion (humerus above glenoid)
FLUENCE_OUTPUT = None             # None = run pmcx; path string = load saved .npy

# ── Soft-tissue wrapping (mm) ─────────────────────────────────────────────────
# Deltoid posterior:  ~25 mm    anterior:  ~18 mm
# Rotator cuff:       ~10 mm (infraspinatus / supraspinatus)
# Subcutaneous fat:    ~5 mm typical
# Skin dermis:         ~2 mm
MUSCLE_THICK_MM  = 15   # anterior/superior access window (was 20; uniform wrap overestimates anterior depth)
ADIPOSE_THICK_MM =  5
SKIN_THICK_MM    =  2

# ── Source power ──────────────────────────────────────────────────────────────
SOURCE_POWER_MW   = 50
SOURCE_DUTY_CYCLE = 0.75
SOURCE_OPT_EFF    = 0.85
CONE_ANGLE_DEG    = 20     # source cone full angle

MELANIN_CONDITIONS = build_melanin_conditions(_MELANIN_RAW_808NM, voxel_size_mm=VOXEL_SIZE)

# ─────────────────────────────────────────────────────────────────────────────
# TISSUE GROUPS (shoulder anatomy) — passed into analyze_fluence_absorption /
# results_to_csv / melanin_comparison_to_csv, which are anatomy-agnostic.
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = {
    'Bone':      lambda n: 'bone'     in n,
    'Cartilage': lambda n: 'cart'     in n,
    'Labrum':    lambda n: 'labrum'   in n,
    'Synovial':  lambda n: 'synovial' in n,
    'Muscle':    lambda n: 'muscle'   in n,
    'Adipose':   lambda n: 'adipose'  in n,
    'Skin+Epidermis': lambda n: ('skin' in n) or ('epidermis' in n),
}
# Cross-subject dose summary (results_to_csv Section 4) — joint + muscle
# targets only, mirroring the knee pipeline's narrower DOSE_GROUPS.
DOSE_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'Labrum':         lambda n: 'labrum'   in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
}
COMP_GROUPS = {
    'Cartilage':      lambda n: 'cart'     in n,
    'Labrum':         lambda n: 'labrum'   in n,
    'Synovial Fluid': lambda n: 'synovial' in n,
    'Muscle':         lambda n: 'muscle'   in n,
    'Bone':           lambda n: 'bone'     in n,
    'Skin+Epidermis': lambda n: 'skin' in n or 'epidermis' in n,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. PER-SUBJECT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_subject(subject_id, mesh_dir_base, output_dir, melanin_condition='fair'):
    """Run the full pipeline for a single shoulder subject."""

    mesh_dir = Path(mesh_dir_base) / f"Raw_Mesh_Files_{subject_id}"
    if not mesh_dir.exists():
        print(f"  Skipping {subject_id} — directory not found: {mesh_dir}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  Processing {subject_id}  [{melanin_condition}]")
    print(f"{'=' * 60}")

    # ── Tissue table ─────────────────────────────────────────────────────────
    # Optical properties at 808 nm (µa, µs', g, n).
    # Cartilage, labrum, bone values are the same as the knee pipeline.
    # Labrum is fibrocartilage — same optical class as knee meniscus.
    tissues = {
        "synovial":      (None,                                             14, opt(0.0005, 0.01,  0.90, 1.36)),
        "skin":          (None,                                             13, opt(0.003,  1.22,  0.79, 1.40)),
        "adipose":       (None,                                             12, opt(0.0013, 1.00,  0.90, 1.44)),
        "muscle":        (None,                                             11, opt(0.0180, 0.55,  0.93, 1.37)),
        "labrum":        (mesh_dir / "labrum_raw.stl",                       5, opt(0.006,  1.80,  0.90, 1.37)),  # fibrocartilage
        "glenoid-cart":  (mesh_dir / "glenoid_cartilage_raw.stl",            8, opt(0.015,  1.00,  0.90, 1.37)),  # hyaline
        "humeral-cart":  (mesh_dir / "humeral_cartilage_raw.stl",            7, opt(0.015,  1.00,  0.90, 1.37)),  # hyaline
        "clavicle-bone": (mesh_dir / "clavicle_raw.stl",                     3, opt(0.040,  2.50,  0.92, 1.37)),
        "scapula-bone":  (mesh_dir / "scapula_raw.stl",                      2, opt(0.040,  2.50,  0.92, 1.37)),
        "humerus-bone":  (mesh_dir / "humerus_raw.stl",                      1, opt(0.040,  2.50,  0.92, 1.37)),
    }
    tissues["epidermis"] = (None, EPIDERMIS_LABEL, MELANIN_CONDITIONS[melanin_condition])

    try:
        vol, origin, mesh_center = build_label_volume(
            tissues, VOXEL_RES, VOXEL_SIZE,
            auto_orient=AUTO_ORIENT,
            orient_ref_a='scapula-bone', orient_ref_b='humerus-bone',
        )

        bone_labels      = [t[1] for name, t in tissues.items() if "bone" in name]
        cartilage_labels = [t[1] for name, t in tissues.items() if "cart" in name]
        labrum_labels    = [t[1] for name, t in tissues.items() if "labrum" in name]

        vol = add_synovial_fluid(
            vol,
            cartilage_labels=cartilage_labels + labrum_labels,
            bone_labels=bone_labels,
            fluid_label=tissues["synovial"][1],
            dilation_vox=3
        )

        layer_configs_vox = [
            (tissues["muscle"][1],  int(round(MUSCLE_THICK_MM  / VOXEL_SIZE))),
            (tissues["adipose"][1], int(round(ADIPOSE_THICK_MM / VOXEL_SIZE))),
            (tissues["skin"][1],    int(round(SKIN_THICK_MM    / VOXEL_SIZE))),
        ]
        vol = add_wrapping_layers(vol, layer_configs_vox)
        vol = add_epidermis_layer(vol, skin_label=tissues["skin"][1],
                                   epidermis_label=EPIDERMIS_LABEL)

        jl_z = find_joint_line_z(vol, tissues, origin, VOXEL_SIZE, mesh_center,
                                  target_match_fn=TARGET_MATCH_FN)

        _colors = ['red', 'green', 'blue', 'orange', 'purple']
        if OPTIMIZE_SOURCES:
            print("\n--- Reciprocity source position optimisation ---")
            opt_positions = optimize_source_positions_reciprocity(
                vol, tissues, origin, mesh_center, VOXEL_SIZE,
                OPT_N_SOURCES, OPT_MIN_SEP_MM, OPT_NPHOTON,
                epidermis_label=EPIDERMIS_LABEL,
                target_match_fn=TARGET_MATCH_FN,
            )
            if opt_positions:
                src_configs = [
                    {'name': f'Opt-{i+1}', 'world_pos': pos, 'color': _colors[i % len(_colors)]}
                    for i, pos in enumerate(opt_positions)
                ]
            else:
                print("  [OPT] Falling back to default positions")
                src_configs = _default_src_configs(jl_z)
        else:
            src_configs = _default_src_configs(jl_z)

        for cfg in src_configs:
            d = np.array([0, 0, jl_z]) - np.array(cfg['world_pos'])
            cfg['srcdir'] = (d / np.linalg.norm(d)).tolist()

        pmcx_source_plus = find_surface_source_positions(
            vol, origin, VOXEL_SIZE, mesh_center, src_configs
        )
        pmcx_source = [{'srcpos': s['srcpos'], 'srcdir': s['srcdir']}
                       for s in pmcx_source_plus]

        fluence_combined, fluence_list = run_pmcx(
            vol, tissues, pmcx_source,
            wavelength_m=WAVELENGTH_M,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
            cone_angle_deg=CONE_ANGLE_DEG,
            voxel_size_mm=VOXEL_SIZE,
        )

        results = analyze_fluence_absorption(
            fluence_combined, vol, tissues, VOXEL_SIZE,
            pmcx_source=pmcx_source,
            groups=GROUPS,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
        )

        subj_dir = Path(output_dir) / melanin_condition / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        cart_names  = [n for n in results if 'cart'     in n]
        cart_vox    = sum(results[n]['n_voxels'] for n in cart_names)
        cart_flu_mw = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in cart_names) / cart_vox) if cart_vox > 0 else 0.0

        labrum_names   = [n for n in results if 'labrum' in n]
        labrum_vox     = sum(results[n]['n_voxels'] for n in labrum_names)
        labrum_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                              for n in labrum_names) / labrum_vox) if labrum_vox > 0 else 0.0

        syn_names   = [n for n in results if 'synovial' in n]
        syn_vox     = sum(results[n]['n_voxels'] for n in syn_names)
        syn_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in syn_names) / syn_vox) if syn_vox > 0 else 0.0

        print("\n=== Penetration depth analysis ===")
        bin_centers, mean_flu, max_depth = analyze_penetration_depth(
            fluence_combined, vol, VOXEL_SIZE, mesh_center, origin
        )
        # Shoulder anatomy depth references (approximate, posterior access) —
        # NOT knee's zone; the glenohumeral joint sits much deeper than the
        # knee's joint line, so this must be passed explicitly (see
        # pbm_mc_core.analysis.plot_depth_histogram docstring).
        # Data-driven dose-integration zone: the actual depth band (from the
        # skin surface) spanned by the target tissues in THIS model, rather than
        # a hardcoded anatomical guess (the glenohumeral targets sit ~1.5-2.2 cm
        # deep here, not the 4.5 cm a generic GH-joint estimate would assume).
        z_lo, z_hi, z_med = target_depth_zone(vol, tissues, VOXEL_SIZE, TARGET_MATCH_FN)
        if z_lo is None:
            z_lo, z_hi, z_med = 2.0, 3.5, 2.5   # fallback if no target voxels
        print(f"  Target depth zone: {z_lo:.2f}-{z_hi:.2f} cm (median {z_med:.2f} cm)")

        fig_depth = plot_depth_histogram(
            bin_centers, mean_flu, subject_id, WAVELENGTH_NM,
            depth_refs=[(z_med, 'Cartilage/labrum/synovial (targets)')],
            zone_lo=z_lo, zone_hi=z_hi,
            group_flu_mw={
                'Cartilage': cart_flu_mw,
                'Labrum': labrum_flu_mw,
                'Synovial Fluid': syn_flu_mw,
            },
        )
        depth_html = str(subj_dir / f"depth_histogram_{subject_id}_{melanin_condition}.html")
        fig_depth.write_html(depth_html)
        print(f"  Saved: {depth_html}")

        np.save(subj_dir / "label_volume.npy", vol)
        np.save(subj_dir / "fluence_combined.npy", fluence_combined)
        for i, flu in enumerate(fluence_list):
            np.save(subj_dir / f"fluence_src{i + 1}.npy", flu)

        return subject_id, results

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _default_src_configs(jl_z):
    """
    Default source positions for the shoulder at 808 nm.

    Coordinate convention (same as knee):
      +Y = anterior,  −Y = posterior,  +X = lateral,  +Z = superior

    Shoulder anatomy at the glenohumeral joint line:
      Posterior source:  placed posteriorly over the infraspinatus (~Y = −70 mm).
                         This mirrors the Move+ posterior pad placement.
      Anterior (Sup):    superior-anterior, targeting the supraspinatus window
                         above the acromion (Y ≈ +55, X ≈ 0).
      Anterior (Inf):    inferior-anterior, below the coracoid process
                         (Y ≈ +50, X ≈ +25, slightly lateral).

    All Z values are auto-set to jl_z (glenohumeral joint line height).
    """
    return [
        {'name': 'Posterior',       'world_pos': [  0, -70, jl_z], 'color': 'red'  },
        {'name': 'Anterior (Sup)',  'world_pos': [  0,  55, jl_z], 'color': 'green'},
        {'name': 'Anterior (Inf)',  'world_pos': [ 25,  50, jl_z], 'color': 'blue' },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Subject list ─────────────────────────────────────────────────────────
    # Populate once STL files are available.
    # Expected directory name format: Raw_Mesh_Files_SHO001, SHO002, …
    # Required STL files per subject (see TISSUE TABLE above):
    #   humerus_raw.stl, scapula_raw.stl, clavicle_raw.stl,
    #   humeral_cartilage_raw.stl, glenoid_cartilage_raw.stl, labrum_raw.stl
    SUBJECT_IDS = ["SHO001"]   # e.g. ["SHO001", "SHO002"]

    BASE_DIR   = Path(".")
    RUN_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path(f"results_shoulder_808nm_{RUN_ID}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Shoulder MC Simulation — 808 nm")
    print(f"Subjects: {SUBJECT_IDS if SUBJECT_IDS else '(none configured — add to SUBJECT_IDS)'}")
    print(f"Output:   {OUTPUT_DIR}")

    if not SUBJECT_IDS:
        print("\n⚠  No subjects configured.  Add subject IDs to SUBJECT_IDS and place "
              "STL files in Raw_Mesh_Files_SHO### directories.")
        raise SystemExit(0)

    all_condition_results = {}
    for condition in MELANIN_CONDITIONS:
        print(f"\n{'=' * 60}\n  Melanin: {condition.upper()}\n{'=' * 60}")
        (OUTPUT_DIR / condition).mkdir(exist_ok=True)
        cond_results = []
        for subject_id in SUBJECT_IDS:
            result = run_subject(subject_id, BASE_DIR, OUTPUT_DIR,
                                 melanin_condition=condition)
            if result is not None:
                cond_results.append(result)
        all_condition_results[condition] = cond_results
        if cond_results:
            results_to_csv(
                cond_results,
                groups=GROUPS,
                dose_groups=DOSE_GROUPS,
                source_power_mw=SOURCE_POWER_MW,
                duty_cycle=SOURCE_DUTY_CYCLE,
                opt_eff=SOURCE_OPT_EFF,
                n_sources=3,
                output_path=str(OUTPUT_DIR / f"MC_Shoulder_808nm_{condition}.csv"),
            )

    melanin_comparison_to_csv(
        all_condition_results,
        groups=COMP_GROUPS,
        output_path=str(OUTPUT_DIR / "MC_Shoulder_Melanin_Comparison_808nm.csv"),
        wavelength_nm=WAVELENGTH_NM,
    )
    print(f"\nDone.")
