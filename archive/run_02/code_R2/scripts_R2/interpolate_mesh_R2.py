#!/usr/bin/env python3
"""
scripts/interpolate_mesh.py - Non-Matching Mesh Field Interpolator & Restart Generator.

This script reads a high-resolution simulation snapshot (e.g. at t ~ 0.50) from an HDF5
velocity field file, generates a refined target mesh (e.g. lc_pole = 0.0005), interpolates
the axisymmetric 3D velocity vector (u_r, u_z, u_theta) onto the fine mesh quadratic P2^3
function space with boundary condition enforcement, and outputs restart checkpoint files:
  - checkpoint_un.npy (P2^3 velocity state)
  - checkpoint_pn.npy (P1 pressure state)
  - checkpoint_meta.json (simulation metadata)
"""

import os
import sys
import json
import time
import argparse
import datetime
import re
import numpy as np
import h5py
import matplotlib.tri as mtri
from scipy.interpolate import NearestNDInterpolator

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.mesh import generate_mesh
import gmsh


def key_to_time(key_str):
    """
    Convert an HDF5 dataset key string to a floating point timestamp.
    Examples:
      '0' -> 0.0
      '0_5' -> 0.5
      '0_499998586184208' -> 0.499998586184208
      '1_9832873250839785e-06' -> 1.9832873250839785e-06
    """
    if "_" in key_str:
        # Replace only the first underscore with a decimal point
        parts = key_str.split("_", 1)
        normalized = parts[0] + "." + parts[1]
    else:
        normalized = key_str
    try:
        return float(normalized)
    except ValueError:
        return None


def find_nearest_snapshot(source_h5, target_time, source_xdmf=None, verbose=True):
    """
    Find the HDF5 dataset key closest to target_time in source_h5.

    Returns:
    --------
    tuple: (best_key, best_time, time_diff)
    """
    if not os.path.exists(source_h5):
        raise FileNotFoundError(f"Source HDF5 file not found: {source_h5}")

    with h5py.File(source_h5, "r") as f:
        if "Function/Velocity" not in f:
            raise KeyError(f"Dataset group '/Function/Velocity' not found in {source_h5}")
        
        velocity_grp = f["Function/Velocity"]
        keys = list(velocity_grp.keys())
        if not keys:
            raise ValueError(f"No velocity datasets found in {source_h5}:/Function/Velocity")

    valid_snapshots = []
    for k in keys:
        t_val = key_to_time(k)
        if t_val is not None:
            valid_snapshots.append((k, t_val))

    if not valid_snapshots:
        raise ValueError("Could not parse any timestamps from HDF5 velocity dataset keys.")

    # Find nearest
    best_key, best_time = min(valid_snapshots, key=lambda item: abs(item[1] - target_time))
    time_diff = abs(best_time - target_time)

    if verbose:
        print(f"[Snapshot Discovery] Target time: {target_time:.6f}s")
        print(f"[Snapshot Discovery] Found {len(valid_snapshots)} snapshots in {source_h5}")
        print(f"[Snapshot Discovery] Selected snapshot '{best_key}' at t = {best_time:.15f}s (delta = {time_diff:.2e}s)")

    return best_key, best_time, time_diff


def load_source_snapshot(source_h5, snapshot_key, verbose=True):
    """
    Selectively read only the required snapshot geometry, topology, and velocity dataset
    from source_h5 without loading the entire multi-gigabyte file into memory.
    """
    t0 = time.time()
    with h5py.File(source_h5, "r") as f:
        pts_src = f["Mesh/mesh/geometry"][:]       # (N_v, 2)
        topo_src = f["Mesh/mesh/topology"][:]     # (N_c, 3)
        u_src = f[f"Function/Velocity/{snapshot_key}"][:] # (N_v, 3)

    load_duration = time.time() - t0

    if np.isnan(u_src).any() or np.isinf(u_src).any():
        raise ValueError(f"Source velocity dataset '{snapshot_key}' contains NaN or Inf values.")

    if verbose:
        print(f"[Source Load] Loaded snapshot '{snapshot_key}' in {load_duration:.3f}s")
        print(f"[Source Load] Source mesh: {pts_src.shape[0]} vertices, {topo_src.shape[0]} cells")
        print(f"[Source Load] Velocity range: ur=[{np.min(u_src[:,0]):.2e}, {np.max(u_src[:,0]):.2e}], "
              f"uz=[{np.min(u_src[:,1]):.2e}, {np.max(u_src[:,1]):.2e}], "
              f"ut=[{np.min(u_src[:,2]):.2e}, {np.max(u_src[:,2]):.2e}]")

    return pts_src, topo_src, u_src


def extract_mesh_p2_dofs(mesh_path, verbose=True):
    """
    Load a .msh file with Gmsh, extract linear vertices and triangle cell topology,
    compute all unique edge midpoints, and return full P2 DOF coordinates.
    """
    if not gmsh.isInitialized():
        gmsh.initialize()
    else:
        gmsh.clear()

    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(mesh_path)

    # 1. Linear vertex nodes
    node_tags_v, node_coords_v, _ = gmsh.model.mesh.getNodes()
    coords_v = node_coords_v.reshape(-1, 3)[:, :2]
    tag_to_vidx = {tag: i for i, tag in enumerate(node_tags_v)}

    # 2. 2D triangle elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
    if not elem_node_tags or len(elem_node_tags) == 0:
        raise ValueError(f"No 2D triangle elements found in {mesh_path}")

    triangles = np.array(elem_node_tags[0]).reshape(-1, 3)

    # 3. Collect unique edges
    edges_set = set()
    for tri in triangles:
        for i in range(3):
            e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            edges_set.add(e)
    edges_list = list(edges_set)

    # 4. Compute edge midpoints
    edge_midpoints = np.array([
        0.5 * (coords_v[tag_to_vidx[e[0]]] + coords_v[tag_to_vidx[e[1]]])
        for e in edges_list
    ])

    # 5. P2 DOF coordinates: vertices followed by edge midpoints
    p2_coords = np.vstack([coords_v, edge_midpoints])

    gmsh.finalize()

    if verbose:
        print(f"[Mesh DOFs] Target mesh '{mesh_path}': {len(coords_v)} vertices (P1), "
              f"{len(triangles)} triangles, {len(edges_list)} edges, {len(p2_coords)} P2 nodes")

    return coords_v, p2_coords, triangles


def interpolate_field(
    pts_src,
    topo_src,
    u_src,
    p2_coords,
    R0=1.0,
    H=2.0,
    k=0.5,
    verbose=True
):
    """
    Interpolate source 3D velocity field (u_r, u_z, u_theta) onto destination P2 coordinates
    using exact source triangulation, nearest extrapolation fallback for boundary fringes,
    and strict physical boundary condition projection (no-slip on outer wall, symmetry on axis).
    """
    t0 = time.time()
    
    # 1. Exact Delaunay / Triangulation from source topology
    triang = mtri.Triangulation(pts_src[:, 0], pts_src[:, 1], triangles=topo_src)
    interp_ur = mtri.LinearTriInterpolator(triang, u_src[:, 0])
    interp_uz = mtri.LinearTriInterpolator(triang, u_src[:, 1])
    interp_ut = mtri.LinearTriInterpolator(triang, u_src[:, 2])

    near_uz = NearestNDInterpolator(pts_src, u_src[:, 1])
    near_ur = NearestNDInterpolator(pts_src, u_src[:, 0])
    near_ut = NearestNDInterpolator(pts_src, u_src[:, 2])

    r_dst = p2_coords[:, 0]
    z_dst = p2_coords[:, 1]

    # Evaluate linear interpolator
    val_ur = interp_ur(r_dst, z_dst)
    val_uz = interp_uz(r_dst, z_dst)
    val_ut = interp_ut(r_dst, z_dst)

    mask_r = np.ma.getmaskarray(val_ur)
    mask_z = np.ma.getmaskarray(val_uz)
    mask_t = np.ma.getmaskarray(val_ut)

    ur_clean = np.array(val_ur, dtype=np.float64)
    uz_clean = np.array(val_uz, dtype=np.float64)
    ut_clean = np.array(val_ut, dtype=np.float64)

    # Fill masked fringe points using nearest neighbor
    masked_count = 0
    if mask_r.any():
        masked_count = max(masked_count, int(np.sum(mask_r)))
        ur_clean[mask_r] = near_ur(p2_coords[mask_r])
    if mask_z.any():
        masked_count = max(masked_count, int(np.sum(mask_z)))
        uz_clean[mask_z] = near_uz(p2_coords[mask_z])
    if mask_t.any():
        masked_count = max(masked_count, int(np.sum(mask_t)))
        ut_clean[mask_t] = near_ut(p2_coords[mask_t])

    # 2. Strict Boundary Conditions Enforcement
    # Wall boundary r = f(z) = R0 * cos(pi * z / 2H) * exp(-k * z^2)
    def f_wall(z):
        return R0 * np.cos(np.pi * z / (2.0 * H)) * np.exp(-k * z**2)

    wall_r = f_wall(z_dst)
    # No-slip condition on outer wall: u_r = u_z = u_theta = 0
    is_wall = np.isclose(r_dst, wall_r, atol=1e-4) | (r_dst >= wall_r - 1e-5)
    ur_clean[is_wall] = 0.0
    uz_clean[is_wall] = 0.0
    ut_clean[is_wall] = 0.0

    # Symmetry condition on axis r = 0: u_r = 0, u_theta = 0
    is_axis = np.isclose(r_dst, 0.0, atol=1e-6)
    ur_clean[is_axis] = 0.0
    ut_clean[is_axis] = 0.0

    # 3. Interleave into flat P2^3 array matching FEniCSx Taylor-Hood vector layout
    num_p2 = len(p2_coords)
    u_interleaved = np.empty(num_p2 * 3, dtype=np.float64)
    u_interleaved[0::3] = ur_clean
    u_interleaved[1::3] = uz_clean
    u_interleaved[2::3] = ut_clean

    # 4. Check for NaNs/Infs
    if np.isnan(u_interleaved).any() or np.isinf(u_interleaved).any():
        raise ValueError("Interpolation produced NaN or Inf values in the velocity field.")

    duration = time.time() - t0
    max_vel = np.max(np.sqrt(ur_clean**2 + uz_clean**2 + ut_clean**2))

    if verbose:
        print(f"[Interpolation] Complete in {duration:.3f}s (masked boundary fringes resolved: {masked_count})")
        print(f"[Interpolation] Resulting P2 DOFs: {len(u_interleaved)} ({num_p2} nodes x 3 components)")
        print(f"[Interpolation] Peak velocity magnitude: {max_vel:.6e}")

    stats = {
        "num_p2_nodes": num_p2,
        "masked_fringes_resolved": masked_count,
        "max_velocity": float(max_vel),
        "wall_nodes_clamped": int(np.sum(is_wall)),
        "axis_nodes_clamped": int(np.sum(is_axis)),
        "duration_sec": duration,
    }

    return u_interleaved, stats


def run_interpolation_pipeline(
    source_h5="runs/run_02_standard_aws/velocity.h5",
    source_xdmf="runs/run_02_standard_aws/velocity.xdmf",
    target_time=0.50,
    lc_pole=0.0005,
    lc_boundary=0.015,
    out_dir="runs/run_03_fine_interpolated",
    target_mesh=None,
    dry_run=False,
    R0=1.0,
    H=2.0,
    k=0.5,
    num_points=400,
    verbose=True,
):
    """
    Execute the end-to-end mesh generation, field interpolation, and checkpoint setup.
    """
    if target_mesh is None:
        target_mesh = os.path.join(out_dir, "apple_domain.msh")

    if verbose:
        print("=" * 70)
        print(" Navier-Stokes Non-Matching Mesh Interpolation & Restart Setup")
        print("=" * 70)
        print(f" Source HDF5:    {source_h5}")
        print(f" Target Time:    {target_time}s")
        print(f" Target lc_pole: {lc_pole}")
        print(f" Target Mesh:    {target_mesh}")
        print(f" Output Dir:     {out_dir}")
        print(f" Dry Run Mode:   {dry_run}")
        print("-" * 70)

    # 1. Discover target snapshot
    snapshot_key, actual_time, time_diff = find_nearest_snapshot(
        source_h5, target_time, source_xdmf=source_xdmf, verbose=verbose
    )

    # 2. Selectively load source snapshot
    pts_src, topo_src, u_src = load_source_snapshot(source_h5, snapshot_key, verbose=verbose)

    # 3. Generate refined target mesh
    if verbose:
        print(f"[Mesh Generation] Generating target mesh with lc_pole={lc_pole}, lc_boundary={lc_boundary}...")
    
    mesh_out_path = target_mesh if not dry_run else os.path.join(out_dir, "temp_dryrun_mesh.msh")
    mesh_info = generate_mesh(
        output_file=mesh_out_path,
        lc_pole=lc_pole,
        lc_boundary=lc_boundary,
        R0=R0,
        H=H,
        k=k,
        num_points=num_points,
        verbosity=0,
        gui=False,
    )

    # 4. Extract target P2 DOFs
    coords_v, p2_coords, triangles = extract_mesh_p2_dofs(mesh_out_path, verbose=verbose)

    # 5. Interpolate field onto P2 space
    u_interleaved, stats = interpolate_field(
        pts_src,
        topo_src,
        u_src,
        p2_coords,
        R0=R0,
        H=H,
        k=k,
        verbose=verbose,
    )

    # 6. Pressure field (P1)
    p_p1 = np.zeros(len(coords_v), dtype=np.float64)

    # 7. Checkpoint Output or Dry-Run Reporting
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        un_path = os.path.join(out_dir, "checkpoint_un.npy")
        pn_path = os.path.join(out_dir, "checkpoint_pn.npy")
        meta_path = os.path.join(out_dir, "checkpoint_meta.json")

        np.save(un_path, u_interleaved)
        np.save(pn_path, p_p1)

        # Approximate simulation step for t ~ 0.50 (estimated from blowup tracking ~ 340,590)
        est_step = int(round(actual_time * 681180)) # Scaled step index

        meta_data = {
            "step": est_step,
            "t": float(actual_time),
            "target_time": float(target_time),
            "time_diff": float(time_diff),
            "lc_pole": float(lc_pole),
            "lc_boundary": float(lc_boundary),
            "source_h5": source_h5,
            "source_snapshot": snapshot_key,
            "num_p2_nodes": len(p2_coords),
            "num_p1_nodes": len(coords_v),
            "total_velocity_dofs": len(u_interleaved),
            "max_velocity": stats["max_velocity"],
            "created_at": datetime.datetime.now().isoformat(),
        }

        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)

        if verbose:
            print("-" * 70)
            print(" Checkpoint Files Successfully Written:")
            print(f"  - {un_path} ({os.path.getsize(un_path):,} bytes, {len(u_interleaved)} float64)")
            print(f"  - {pn_path} ({os.path.getsize(pn_path):,} bytes, {len(p_p1)} float64)")
            print(f"  - {meta_path} ({os.path.getsize(meta_path):,} bytes)")
            print(f"  - {mesh_out_path} ({os.path.getsize(mesh_out_path):,} bytes)")
            print("=" * 70)
    else:
        # Clean up temporary dry-run mesh if created
        if os.path.exists(mesh_out_path):
            try:
                os.remove(mesh_out_path)
            except OSError:
                pass

        if verbose:
            print("-" * 70)
            print(" [DRY RUN SUCCESS] Mathematical validation passed without writing files:")
            print(f"  - Snapshot at t={actual_time:.6f}s (target: {target_time:.6f}s)")
            print(f"  - Mesh P1 nodes: {len(coords_v)}, P2 nodes: {len(p2_coords)}")
            print(f"  - Interpolated DOFs: {len(u_interleaved)} (all finite, 0 NaNs/Infs)")
            print(f"  - Peak velocity: {stats['max_velocity']:.4e}")
            print("=" * 70)

    return {
        "status": "success",
        "snapshot_key": snapshot_key,
        "actual_time": actual_time,
        "target_time": target_time,
        "num_p2_nodes": len(p2_coords),
        "num_p1_nodes": len(coords_v),
        "u_interleaved": u_interleaved,
        "p_p1": p_p1,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Interpolate Navier-Stokes velocity snapshot from coarse/AWS mesh onto refined fine mesh."
    )
    parser.add_argument(
        "--source-h5",
        default="runs/run_02_standard_aws/velocity.h5",
        help="Path to source velocity.h5 file (default: runs/run_02_standard_aws/velocity.h5)",
    )
    parser.add_argument(
        "--source-xdmf",
        default="runs/run_02_standard_aws/velocity.xdmf",
        help="Path to source velocity.xdmf file (default: runs/run_02_standard_aws/velocity.xdmf)",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=0.50,
        help="Target simulation timestamp to extract and interpolate (default: 0.50)",
    )
    parser.add_argument(
        "--lc-pole",
        type=float,
        default=0.0005,
        help="Mesh characteristic size at cusp poles z = +/- H (default: 0.0005)",
    )
    parser.add_argument(
        "--lc-boundary",
        type=float,
        default=0.015,
        help="Mesh characteristic size at equatorial boundary (default: 0.015)",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/run_03_fine_interpolated",
        help="Output directory for fine mesh restart checkpoints (default: runs/run_03_fine_interpolated)",
    )
    parser.add_argument(
        "--target-mesh",
        default=None,
        help="Path for target mesh file (default: <out-dir>/apple_domain.msh)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute mathematical verification without writing permanent checkpoint files",
    )
    parser.add_argument("--R0", type=float, default=1.0, help="Equatorial radius parameter (default: 1.0)")
    parser.add_argument("--H", type=float, default=2.0, help="Domain half-height parameter (default: 2.0)")
    parser.add_argument("--k", type=float, default=0.5, help="Cusp exponential parameter (default: 0.5)")
    parser.add_argument(
        "--num-points",
        type=int,
        default=400,
        help="Number of spline discretization points for boundary (default: 400)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed verbose progress logging")

    args = parser.parse_args()

    try:
        run_interpolation_pipeline(
            source_h5=args.source_h5,
            source_xdmf=args.source_xdmf,
            target_time=args.target_time,
            lc_pole=args.lc_pole,
            lc_boundary=args.lc_boundary,
            out_dir=args.out_dir,
            target_mesh=args.target_mesh,
            dry_run=args.dry_run,
            R0=args.R0,
            H=args.H,
            k=args.k,
            num_points=args.num_points,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"\n[ERROR] Interpolation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
