"""
Shoulder STL Mesh → 3D Voxel Volume + pmcx Fluence Overlay  (808 nm)
----------------------------------------------------------------------
Pipeline mirrors the knee OKS batch pipeline exactly.

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
"""

import numpy as np
import trimesh
import time
import pmcx
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, distance_transform_edt
from pathlib import Path
import webbrowser
import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

start_time = time.perf_counter()


def opt(mua, mus_prime, g, n):
    """Convert reduced scattering coefficient to transport scattering for pmcx."""
    return [mua, mus_prime / (1 - g), g, n]


# Epidermal optical properties by melanin condition at 808 nm
# Thickness correction: epidermis is 0.2 mm physically but occupies 1 voxel
# (1 mm) in the simulation.  Both µa and µs' are scaled by
#   EPI_SCALE = EPI_THICKNESS_MM / VOXEL_SIZE_MM = 0.2 / 1.0 = 0.2
_EPI_THICKNESS_MM = 0.2
_EPI_SCALE        = _EPI_THICKNESS_MM / 1.0

MELANIN_CONDITIONS = {
    #                   µa (true × scale)        µs' (true × scale)   g     n
    'fair':  opt(0.008 * _EPI_SCALE, 1.50 * _EPI_SCALE, 0.80, 1.40),  # Fitzpatrick I-II
    'olive': opt(0.025 * _EPI_SCALE, 1.60 * _EPI_SCALE, 0.80, 1.40),  # Fitzpatrick III-IV
    'dark':  opt(0.075 * _EPI_SCALE, 1.70 * _EPI_SCALE, 0.80, 1.40),  # Fitzpatrick V-VI
}
EPIDERMIS_LABEL = 15

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
MUSCLE_THICK_MM  = 20
ADIPOSE_THICK_MM =  5
SKIN_THICK_MM    =  2

# ── Source power ──────────────────────────────────────────────────────────────
SOURCE_POWER_MW   = 50
SOURCE_DUTY_CYCLE = 0.75
SOURCE_OPT_EFF    = 0.85


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
        vol, origin, mesh_center = build_label_volume(tissues, VOXEL_RES, VOXEL_SIZE)

        bone_labels      = [t[1] for name, t in tissues.items() if "bone"    in name]
        cartilage_labels = [t[1] for name, t in tissues.items() if "cart"    in name]
        labrum_labels    = [t[1] for name, t in tissues.items() if "labrum"  in name]

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

        jl_z = find_joint_line_z(vol, tissues, origin, VOXEL_SIZE, mesh_center)

        _colors = ['red', 'green', 'blue', 'orange', 'purple']
        if OPTIMIZE_SOURCES:
            print("\n--- Reciprocity source position optimisation ---")
            opt_positions = optimize_source_positions_reciprocity(
                vol, tissues, origin, mesh_center, VOXEL_SIZE,
                OPT_N_SOURCES, OPT_MIN_SEP_MM, OPT_NPHOTON, wavelength_m=808e-9
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
            pmcx_source_list=pmcx_source,
            source_power_mw=SOURCE_POWER_MW,
            duty_cycle=SOURCE_DUTY_CYCLE,
            opt_eff=SOURCE_OPT_EFF,
        )

        results = analyze_fluence_absorption(
            fluence_combined, vol, tissues, VOXEL_SIZE,
            pmcx_source=pmcx_source
        )

        subj_dir = Path(output_dir) / melanin_condition / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        cart_names  = [n for n in results if 'cart'     in n]
        cart_vox    = sum(results[n]['n_voxels'] for n in cart_names)
        cart_flu_mw = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in cart_names) / cart_vox) if cart_vox > 0 else 0.0

        syn_names   = [n for n in results if 'synovial' in n]
        syn_vox     = sum(results[n]['n_voxels'] for n in syn_names)
        syn_flu_mw  = (sum(results[n]['mean_flu'] * results[n]['n_voxels']
                           for n in syn_names) / syn_vox) if syn_vox > 0 else 0.0

        print("\n=== Penetration depth analysis ===")
        bin_centers, mean_flu, max_depth = analyze_penetration_depth(
            fluence_combined, vol, VOXEL_SIZE, mesh_center, origin
        )
        fig_depth = plot_depth_histogram(
            bin_centers, mean_flu, subject_id,
            cartilage_flu_mw=cart_flu_mw,
            synovial_flu_mw=syn_flu_mw,
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
# ALL FUNCTIONS BELOW ARE IDENTICAL TO THE KNEE PIPELINE
# (stl_to_voxels, build_label_volume, add_wrapping_layers,
#  add_epidermis_layer, add_synovial_fluid, find_surface_source_positions,
#  find_joint_line_z, optimize_source_positions_reciprocity, run_pmcx,
#  analyze_fluence_absorption, analyze_penetration_depth,
#  plot_depth_histogram, results_to_csv, melanin_comparison_to_csv,
#  write_interactive_html, plot_results, make_coord_arrays)
# ─────────────────────────────────────────────────────────────────────────────
# ── Orientation: humerus (label 1) should sit SUPERIOR to scapula (label 2)
#    in a standard anatomical right-shoulder view (+Z = superior).
#    AUTO_ORIENT swaps Z sign if humerus_z_mean < scapula_z_mean.
# ─────────────────────────────────────────────────────────────────────────────

def stl_to_voxels(mesh_path, label, origin, spacing, shape, z_flip=False):
    mesh = trimesh.load(mesh_path, force="mesh")
    if not mesh.is_watertight:
        print(f"  ⚠  {mesh_path} is not watertight — attempting repair")
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)
    if z_flip:
        mesh.vertices[:, 2] *= -1
    nx, ny, nz = shape
    vol = np.zeros(shape, dtype=np.uint8)
    xs = origin[0] + (np.arange(nx) + 0.5) * spacing
    ys = origin[1] + (np.arange(ny) + 0.5) * spacing
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            ray_origin = np.array([[x, y, origin[2] - spacing]])
            ray_dir    = np.array([[0.0, 0.0, 1.0]])
            locs, _, _ = mesh.ray.intersects_location(
                ray_origins=ray_origin, ray_directions=ray_dir)
            if len(locs) == 0:
                continue
            hit_zs = np.sort(locs[:, 2])
            for k in range(0, len(hit_zs) - 1, 2):
                z0, z1 = hit_zs[k], hit_zs[k + 1]
                iz0 = max(0, int(np.floor((z0 - origin[2]) / spacing)))
                iz1 = min(nz - 1, int(np.ceil((z1 - origin[2]) / spacing)))
                vol[ix, iy, iz0:iz1 + 1] = label
    return vol


def build_label_volume(tissues, res, spacing):
    """Merge all tissue STLs into one integer label volume.

    Orientation check: humerus (label 1) should be SUPERIOR to scapula (label 2).
    If humerus_z_mean < scapula_z_mean the Z axis is flipped.
    """
    all_verts      = []
    humerus_z_mean = None
    scapula_z_mean = None

    for name, (path, label, _) in tissues.items():
        if path is not None:
            m = trimesh.load(path, force="mesh")
            all_verts.append(m.vertices)
            if 'humerus-bone' in name:
                humerus_z_mean = m.vertices[:, 2].mean()
            elif 'scapula-bone' in name:
                scapula_z_mean = m.vertices[:, 2].mean()

    z_flip = False
    if AUTO_ORIENT and humerus_z_mean is not None and scapula_z_mean is not None:
        # Humeral head centroid should sit above the glenoid fossa (+Z)
        if humerus_z_mean < scapula_z_mean:
            z_flip = True
            print(f"  [ORIENT] Z-axis inversion detected: "
                  f"humerus_z={humerus_z_mean:.1f} < scapula_z={scapula_z_mean:.1f} "
                  f"— applying Z correction")
            all_verts = [v * np.array([1.0, 1.0, -1.0]) for v in all_verts]

    verts       = np.vstack(all_verts)
    mn          = verts.min(axis=0)
    mx          = verts.max(axis=0)
    mesh_center = (mn + mx) / 2.0
    grid_half   = np.array(res) * spacing / 2.0
    origin      = mesh_center - grid_half

    vol = np.zeros(res, dtype=np.uint8)
    for name, (path, label, _) in tissues.items():
        if path is not None:
            print(f"  Voxelizing {name} (label={label})...")
            layer = stl_to_voxels(path, label, origin, spacing, res, z_flip=z_flip)
            vol[layer > 0] = layer[layer > 0]

    return vol, origin, mesh_center


def add_wrapping_layers(vol, layer_configs):
    result      = vol.copy()
    outer_shell = result > 0
    for label, thickness_vox in layer_configs:
        dilated   = binary_dilation(outer_shell, iterations=thickness_vox)
        new_layer = dilated & ~outer_shell
        result[new_layer & (result == 0)] = label
        outer_shell = dilated
    return result


def add_epidermis_layer(vol, skin_label, epidermis_label):
    skin_mask  = vol == skin_label
    inner_skin = binary_erosion(skin_mask, iterations=1)
    epi_mask   = skin_mask & ~inner_skin
    result     = vol.copy()
    result[epi_mask] = epidermis_label
    return result


def add_synovial_fluid(vol, cartilage_labels, bone_labels, fluid_label, dilation_vox):
    cartilage_mask = np.isin(vol, cartilage_labels)
    bone_mask      = np.isin(vol, bone_labels)
    if cartilage_mask.sum() == 0:
        print("  Warning: no cartilage/labrum voxels found")
        return vol
    dilated_cart     = binary_dilation(cartilage_mask, iterations=dilation_vox)
    INNER_FILL_LABELS = set(cartilage_labels) | set(bone_labels)
    fluid_mask = (
        dilated_cart
        & ~cartilage_mask
        & ~bone_mask
        & ~np.isin(vol, list(INNER_FILL_LABELS))
    )
    result = vol.copy()
    result[fluid_mask] = fluid_label
    return result


def find_joint_line_z(vol, tissues, origin, spacing, mesh_center):
    cart_labels = [t[1] for name, t in tissues.items() if 'cart'    in name]
    lab_labels  = [t[1] for name, t in tissues.items() if 'labrum'  in name]
    syn_labels  = [t[1] for name, t in tissues.items() if 'synovial' in name]
    target_mask = np.isin(vol, cart_labels + lab_labels + syn_labels)
    if target_mask.sum() == 0:
        print("  [JLINE] No cartilage/labrum/synovial voxels — using Z=0")
        return 0.0
    counts_per_z = target_mask.sum(axis=(0, 1))
    iz_peak      = int(np.argmax(counts_per_z))
    world_z      = origin[2] + (iz_peak + 0.5) * spacing
    z_offset     = world_z - mesh_center[2]
    print(f"  [JLINE] Joint-line Z slice: {iz_peak}  "
          f"world_z={world_z:.1f} mm  offset={z_offset:+.1f} mm  "
          f"({int(counts_per_z[iz_peak])} target voxels)")
    return z_offset


def find_surface_source_positions(vol, origin, spacing, mesh_center, src_configs):
    sources       = []
    tissue_coords = np.argwhere(vol > 0)
    for cfg in src_configs:
        intended_world = np.array(cfg['world_pos'])
        intended_vox   = (intended_world + mesh_center - origin) / spacing
        distances      = np.linalg.norm(tissue_coords - intended_vox, axis=1)
        nearest        = tissue_coords[distances.argmin()]
        print(f"  '{cfg['name']}': intended_vox={intended_vox.round(1)}, "
              f"nearest={nearest}, distance={distances.min():.1f} vox, "
              f"label={vol[nearest[0], nearest[1], nearest[2]]}")
        srcdir = np.array(cfg['srcdir'], dtype=float)
        srcdir = srcdir / np.linalg.norm(srcdir)
        srcpos = nearest.astype(float).copy()
        for step in range(1, 21):
            sp         = [int(round(x)) for x in srcpos]
            sp_clipped = [np.clip(sp[i], 0, vol.shape[i] - 1) for i in range(3)]
            if vol[sp_clipped[0], sp_clipped[1], sp_clipped[2]] > 0:
                break
            srcpos = nearest.astype(float) + srcdir * step
        sources.append({'srcpos': srcpos.tolist(), 'srcdir': srcdir.tolist(),
                        'color': cfg['color'], 'name': cfg['name']})
    return sources


def optimize_source_positions_reciprocity(vol, tissues, origin, mesh_center,
                                           spacing, n_sources, min_sep_mm,
                                           n_photon, wavelength_m=808e-9):
    cart_labels = [t[1] for name, t in tissues.items() if 'cart'    in name]
    lab_labels  = [t[1] for name, t in tissues.items() if 'labrum'  in name]
    syn_labels  = [t[1] for name, t in tissues.items() if 'synovial' in name]
    target_mask = np.isin(vol, cart_labels + lab_labels + syn_labels)
    if target_mask.sum() == 0:
        return None
    centroid_vox = np.argwhere(target_mask).mean(axis=0)
    max_label    = max(t[1] for t in tissues.values())
    prop_table   = [[0, 0, 1, 1]] * (max_label + 1)
    for name, (path, label, props) in tissues.items():
        prop_table[label] = props
    cfg_opt = {
        "nphoton": n_photon, "srctype": 'isotropic',
        "srcpos": centroid_vox.tolist(), "srcdir": [0.0, 0.0, 1.0],
        "vol": vol.astype(np.uint8), "prop": prop_table,
        "tstart": 0, "tend": 1e-9, "tstep": 1e-9,
        "unitinmm": spacing, "autopilot": 1, "gpuid": 1,
        "issavedet": 0, "outputtype": "fluence", "normalize": 1,
    }
    res      = pmcx.run(cfg_opt)
    flu_map  = res['flux'].squeeze()
    epi_coords = np.argwhere(vol == EPIDERMIS_LABEL)
    epi_flu    = flu_map[epi_coords[:, 0], epi_coords[:, 1], epi_coords[:, 2]]
    sort_idx   = np.argsort(epi_flu)[::-1]
    epi_coords = epi_coords[sort_idx]
    epi_flu    = epi_flu[sort_idx]
    peak_flu   = epi_flu[0]
    min_sep_vox    = min_sep_mm / spacing
    selected_world = []
    active         = np.ones(len(epi_flu), dtype=bool)
    for i in range(n_sources):
        live = np.where(active)[0]
        if len(live) == 0:
            break
        best_vox   = epi_coords[live[0]]
        dists      = np.linalg.norm(epi_coords - best_vox, axis=1)
        active[dists < min_sep_vox] = False
        world_abs  = origin + (best_vox.astype(float) + 0.5) * spacing
        world_cen  = (world_abs - mesh_center).tolist()
        selected_world.append(world_cen)
        flu_val = flu_map[best_vox[0], best_vox[1], best_vox[2]]
        print(f"    Src {i+1}: world={[f'{v:+.1f}' for v in world_cen]} mm  "
              f"(rel_flu={flu_val/peak_flu:.3f})")
    return selected_world


def run_pmcx(vol, tissues, src_cfg, pmcx_source_list,
             source_power_mw=50, wavelength_m=808e-9,
             duty_cycle=0.75, opt_eff=0.85, modulation_hz=40):
    h           = 6.626e-34
    c           = 3e8
    E_photon    = h * c / wavelength_m
    power_avg_W = (source_power_mw * 1e-3) * duty_cycle * opt_eff
    Q_avg_per_s = power_avg_W / E_photon
    scale       = Q_avg_per_s * E_photon * 100.0 * 1e3
    print(f"  Average power:  {power_avg_W*1e3:.2f} mW")
    max_label   = max(t[1] for t in tissues.values())
    prop_table  = [[0, 0, 1, 1]] * (max_label + 1)
    for name, (path, label, opts) in tissues.items():
        prop_table[label] = opts
    half_angle_rad = np.deg2rad(20 / 2)
    cfg = {
        "nphoton": 1e7, "srctype": 'cone',
        "srcparam1": [half_angle_rad, 0, 0, 0],
        "vol": vol.astype(np.uint8), "prop": prop_table,
        "tstart": 0, "tend": 1e-9, "tstep": 1e-9,
        "unitinmm": VOXEL_SIZE, "autopilot": 1, "gpuid": 1,
        "issavedet": 0, "outputtype": "fluence", "normalize": 1,
    }
    cfg.update(src_cfg)
    individual_fluences = []
    combined_flux       = None
    for i, src in enumerate(pmcx_source_list):
        cfg['srcpos'] = src['srcpos']
        cfg['srcdir'] = src['srcdir']
        res           = pmcx.run(cfg)
        flux_mwcm2    = res['flux'].squeeze() * scale
        individual_fluences.append(flux_mwcm2)
        np.save(f"fluence_src{i+1}.npy", flux_mwcm2)
        combined_flux = flux_mwcm2.copy() if combined_flux is None \
                        else combined_flux + flux_mwcm2
    np.save("fluence_combined.npy", combined_flux)
    return combined_flux, individual_fluences


def analyze_fluence_absorption(fluence, vol, tissues, voxel_size_mm, pmcx_source):
    voxel_vol_cm3 = (voxel_size_mm * 0.1) ** 3
    print("\n=== Fluence & Absorption Analysis ===")
    results           = {}
    total_absorbed_mw = 0.0
    sorted_tissues    = sorted(tissues.items(), key=lambda kv: kv[1][1])
    for name, (path, label, opts) in sorted_tissues:
        mua      = opts[0]
        mua_cm   = mua * 10.0
        mask     = vol == label
        n_voxels = mask.sum()
        if n_voxels == 0:
            continue
        flu_vals    = fluence[mask]
        mean_flu    = flu_vals.mean()
        max_flu     = flu_vals.max()
        vol_cm3     = n_voxels * voxel_vol_cm3
        absorbed_mw = (mua_cm * flu_vals * voxel_vol_cm3).sum()
        results[name] = {
            'label': label, 'n_voxels': n_voxels, 'vol_cm3': vol_cm3,
            'mua_mm': mua, 'mean_flu': mean_flu, 'max_flu': max_flu,
            'absorbed_mw': absorbed_mw,
        }
        total_absorbed_mw += absorbed_mw
    for name, r in results.items():
        pct = 100 * r['absorbed_mw'] / total_absorbed_mw if total_absorbed_mw > 0 else 0
        print(f"  {name:<16} label={r['label']:>2}  "
              f"mean_flu={r['mean_flu']:.3e}  absorbed={r['absorbed_mw']:.4f} mW  "
              f"({pct:.2f}%)")
    n_sources      = len(pmcx_source)
    power_per_src  = SOURCE_POWER_MW * SOURCE_DUTY_CYCLE * SOURCE_OPT_EFF
    total_power_mw = n_sources * power_per_src
    print(f"\n  Total input: {total_power_mw:.1f} mW  |  "
          f"Total absorbed: {total_absorbed_mw:.4f} mW  |  "
          f"Ratio: {100*total_absorbed_mw/total_power_mw:.2f}%")
    return results


def analyze_penetration_depth(fluence, vol, voxel_size_mm, mesh_center, origin,
                               bin_width_cm=0.25):
    tissue_mask = vol > 0
    depth_vox   = distance_transform_edt(tissue_mask)
    depth_cm    = depth_vox * voxel_size_mm / 10.0
    center_vox  = (mesh_center - origin) / voxel_size_mm
    ci = tuple(int(np.clip(round(center_vox[i]), 0, vol.shape[i] - 1)) for i in range(3))
    max_depth_cm = float(depth_cm[ci])
    if max_depth_cm <= 0:
        max_depth_cm = float(depth_cm.max())
    n_bins      = max(1, int(np.ceil(max_depth_cm / bin_width_cm)))
    bin_edges   = np.linspace(0.0, n_bins * bin_width_cm, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    valid       = tissue_mask & (fluence > 0)
    depths      = depth_cm[valid]
    fluences    = fluence[valid]
    mean_flu    = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (depths >= bin_edges[i]) & (depths < bin_edges[i + 1])
        if in_bin.sum() > 0:
            mean_flu[i] = fluences[in_bin].mean()
    print(f"  Depth range: 0 – {max_depth_cm:.2f} cm  |  "
          f"{n_bins} bins @ {bin_width_cm} cm each")
    return bin_centers, mean_flu, max_depth_cm


def plot_depth_histogram(bin_centers, mean_flu, subject_id, bin_width_cm=0.25,
                          treatment_times_s=(300, 600, 900),
                          cartilage_flu_mw=0.0, synovial_flu_mw=0.0):
    # Shoulder anatomy depth references (approximate, posterior access):
    #   ~1.0 cm  skin + adipose
    #   ~2.5 cm  deltoid / infraspinatus muscle bulk
    #   ~4.5 cm  glenohumeral joint space
    DEPTH_REFS  = [(1.0, 'Skin/Adipose'), (2.5, 'Muscle'), (4.5, 'GH Joint')]
    ZONE_LO, ZONE_HI = 2.5, 4.5

    bin_centers = np.asarray(bin_centers)
    mean_flu    = np.asarray(mean_flu)
    zone_mask   = (bin_centers >= ZONE_LO) & (bin_centers <= ZONE_HI)
    zone_width  = ZONE_HI - ZONE_LO
    n_zone      = zone_mask.sum()
    if n_zone >= 2:
        zone_integral = float(np.trapz(mean_flu[zone_mask], bin_centers[zone_mask]))
    elif n_zone == 1:
        zone_integral = float(mean_flu[zone_mask][0] * bin_width_cm)
    else:
        zone_integral = 0.0
    zone_norm_mw  = zone_integral / zone_width
    dose_lines    = [f"  {t // 60:.0f} min:  {zone_norm_mw * 1e-3 * t:.4f} J/cm²"
                     for t in treatment_times_s]
    cart_doses    = [f"  {t // 60:.0f} min:  {cartilage_flu_mw * 1e-3 * t:.4f} J/cm²"
                     for t in treatment_times_s]
    syn_doses     = [f"  {t // 60:.0f} min:  {synovial_flu_mw * 1e-3 * t:.4f} J/cm²"
                     for t in treatment_times_s]
    annot_text = (
        f"<b>Zone {ZONE_LO}–{ZONE_HI} cm  ∫F·dz / Δz</b><br>"
        f"Norm. fluence rate: {zone_norm_mw:.4f} mW/cm²<br>"
        + "<br>".join(dose_lines)
        + f"<br><br><b>Cartilage (vol-weighted)</b><br>"
        + f"Fluence rate: {cartilage_flu_mw:.4f} mW/cm²<br>"
        + "<br>".join(cart_doses)
        + f"<br><br><b>Synovial Fluid</b><br>"
        + f"Fluence rate: {synovial_flu_mw:.4f} mW/cm²<br>"
        + "<br>".join(syn_doses)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers, y=mean_flu, width=bin_width_cm * 0.85,
        marker=dict(color=mean_flu, colorscale='Hot', reversescale=True,
                    showscale=True,
                    colorbar=dict(title=dict(text='mW/cm²', side='right'),
                                  thickness=15, len=0.6)),
        name='Mean Fluence Rate',
    ))
    max_depth = float(bin_centers[-1]) + bin_width_cm / 2 if len(bin_centers) else 6.0
    for depth, label in DEPTH_REFS:
        if depth <= max_depth:
            fig.add_shape(type='line', x0=depth, x1=depth, y0=0, y1=1,
                          xref='x', yref='paper',
                          line=dict(color='rgba(100,200,255,0.55)', width=1, dash='dash'))
            fig.add_annotation(x=depth, y=1, xref='x', yref='paper',
                                text=label, showarrow=False,
                                font=dict(size=9, color='#8b949e'),
                                xanchor='left', yanchor='bottom', xshift=3)
    fig.add_annotation(x=0.98, y=0.98, xref='paper', yref='paper',
                        text=annot_text, showarrow=False, align='left',
                        xanchor='right', yanchor='top',
                        font=dict(size=10, color='#e6edf3'),
                        bgcolor='rgba(22,27,34,0.85)', bordercolor='#30363d',
                        borderwidth=1, borderpad=6)
    fig.update_layout(
        title=dict(text=f'Fluence Rate vs Penetration Depth — {subject_id} (808 nm) Shoulder',
                   font=dict(size=14)),
        xaxis=dict(title='Penetration Depth from Skin Surface (cm)',
                   gridcolor='#30363d', zeroline=False, dtick=0.25),
        yaxis=dict(title='Mean Fluence Rate (mW/cm²)', type='log',
                   gridcolor='#30363d', zeroline=False),
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font_color='#e6edf3',
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1),
        margin=dict(l=70, r=20, t=55, b=55), bargap=0.05,
    )
    return fig


def results_to_csv(all_results, output_path="MC_Analysis_Shoulder_808nm.csv",
                   treatment_times_s=(300, 600, 900)):
    import csv
    GROUPS = {
        'Bone':      lambda n: 'bone'     in n,
        'Cartilage': lambda n: 'cart'     in n,
        'Labrum':    lambda n: 'labrum'   in n,
        'Synovial':  lambda n: 'synovial' in n,
        'Muscle':    lambda n: 'muscle'   in n,
        'Adipose':   lambda n: 'adipose'  in n,
        'Skin':      lambda n: 'skin'     in n,
    }
    n_sources      = 3
    power_per_src  = SOURCE_POWER_MW * SOURCE_DUTY_CYCLE * SOURCE_OPT_EFF
    total_power_mw = n_sources * power_per_src
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for subj_id, results in all_results:
            total_absorbed_mw = sum(r['absorbed_mw'] for r in results.values())
            writer.writerow([])
            writer.writerow([f'Subject: {subj_id}'])
            writer.writerow([])
            writer.writerow(['Section', 'Subject', 'Tissue', 'Label',
                             'Voxels', 'Vol (cm³)', 'mua (mm^-1)',
                             'Mean Fluence Rate (mW/cm²)', 'Max Fluence Rate (mW/cm²)',
                             'Absorbed Power (mW)', '% of Total Absorbed'])
            for name, r in sorted(results.items(), key=lambda kv: kv[1]['label']):
                pct = 100 * r['absorbed_mw'] / total_absorbed_mw if total_absorbed_mw > 0 else 0
                writer.writerow(['Layer', subj_id, name, r['label'], r['n_voxels'],
                                 f"{r['vol_cm3']:.6f}", f"{r['mua_mm']:.4f}",
                                 f"{r['mean_flu']:.6e}", f"{r['max_flu']:.6e}",
                                 f"{r['absorbed_mw']:.6f}", f"{pct:.4f}"])
            writer.writerow([])
            writer.writerow(['Section', 'Subject', 'Group',
                             'Total Voxels', 'Total Vol (cm³)',
                             'Mean Fluence Rate (mW/cm²)', 'Max Fluence Rate (mW/cm²)',
                             'Absorbed Power (mW)', '% of Total Absorbed'])
            for group, match_fn in GROUPS.items():
                group_names  = [n for n in results if match_fn(n)]
                if not group_names:
                    continue
                grp_voxels   = sum(results[n]['n_voxels']   for n in group_names)
                grp_vol      = sum(results[n]['vol_cm3']     for n in group_names)
                grp_absorbed = sum(results[n]['absorbed_mw'] for n in group_names)
                grp_pct      = 100 * grp_absorbed / total_absorbed_mw if total_absorbed_mw > 0 else 0
                grp_flu_sum  = sum(results[n]['mean_flu'] * results[n]['n_voxels'] for n in group_names)
                grp_mean_flu = grp_flu_sum / grp_voxels if grp_voxels > 0 else 0
                grp_max_flu  = max(results[n]['max_flu'] for n in group_names)
                writer.writerow(['Group', subj_id, group, grp_voxels,
                                 f"{grp_vol:.6f}", f"{grp_mean_flu:.6e}",
                                 f"{grp_max_flu:.6e}", f"{grp_absorbed:.6f}", f"{grp_pct:.4f}"])
            writer.writerow(['Power', subj_id,
                             f'Sources: {n_sources}',
                             f'Power per source: {power_per_src:.1f} mW',
                             f'Total input: {total_power_mw:.1f} mW',
                             f'Total absorbed: {total_absorbed_mw:.4f} mW',
                             f'Ratio: {100*total_absorbed_mw/total_power_mw:.2f}%'])
            writer.writerow([])
            writer.writerow(['-' * 40])
    print(f"\nCSV written: {output_path}  ({len(all_results)} subjects)")


def melanin_comparison_to_csv(all_condition_results, output_path, wavelength_nm,
                               treatment_times_s=(300, 600, 900)):
    import csv
    COMP_GROUPS = {
        'Cartilage':      lambda n: 'cart'     in n,
        'Labrum':         lambda n: 'labrum'   in n,
        'Synovial Fluid': lambda n: 'synovial' in n,
        'Muscle':         lambda n: 'muscle'   in n,
        'Bone':           lambda n: 'bone'     in n,
        'Skin+Epidermis': lambda n: 'skin' in n or 'epidermis' in n,
    }
    conditions = list(all_condition_results.keys())

    def vol_weighted_mean(results_dict, match_fn):
        names     = [n for n in results_dict if match_fn(n)]
        total_vox = sum(results_dict[n]['n_voxels'] for n in names)
        if total_vox == 0:
            return 0.0
        return sum(results_dict[n]['mean_flu'] * results_dict[n]['n_voxels']
                   for n in names) / total_vox

    seen, all_subj = set(), []
    for cond_list in all_condition_results.values():
        for subj_id, _ in cond_list:
            if subj_id not in seen:
                seen.add(subj_id); all_subj.append(subj_id)
    lookup = {cond: dict(pairs) for cond, pairs in all_condition_results.items()}

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'MELANIN CONDITION COMPARISON — Shoulder {wavelength_nm} nm'])
        writer.writerow([])
        for group_name, match_fn in COMP_GROUPS.items():
            writer.writerow([f'=== {group_name} ==='])
            writer.writerow(['Fluence Rate (mW/cm²)'] + [c.capitalize() for c in conditions])
            group_vals = {c: [] for c in conditions}
            for subj_id in all_subj:
                row = [subj_id]
                for cond in conditions:
                    if subj_id in lookup[cond]:
                        v = vol_weighted_mean(lookup[cond][subj_id], match_fn)
                        row.append(f'{v:.4f}')
                        if v > 0:
                            group_vals[cond].append(v)
                    else:
                        row.append('N/A')
                writer.writerow(row)
            for label, fn in [('Mean', np.mean), ('StDev', np.std)]:
                writer.writerow([label] + [
                    f'{fn(group_vals[c]):.4f}' if group_vals[c] else 'N/A'
                    for c in conditions])
            writer.writerow([])
    print(f"\nMelanin comparison CSV written: {output_path}")


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
    SUBJECT_IDS = []   # e.g. ["SHO001", "SHO002"]

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
            results_to_csv(cond_results,
                           output_path=str(OUTPUT_DIR / f"MC_Shoulder_808nm_{condition}.csv"))

    melanin_comparison_to_csv(
        all_condition_results,
        output_path=str(OUTPUT_DIR / "MC_Shoulder_Melanin_Comparison_808nm.csv"),
        wavelength_nm=808,
    )
    print(f"\nDone.")
