"""
Unit and integration tests for scripts/interpolate_mesh.py.
"""
import os
import json
import numpy as np
import pytest
from scripts.interpolate_mesh import (
    key_to_time,
    find_nearest_snapshot,
    load_source_snapshot,
    extract_mesh_p2_dofs,
    run_interpolation_pipeline,
)


def test_key_to_time():
    assert key_to_time("0") == 0.0
    assert key_to_time("0_5") == 0.5
    assert key_to_time("0_499998586184208") == 0.499998586184208
    assert key_to_time("1_9832873250839785e-06") == 1.9832873250839785e-06
    assert key_to_time("invalid_string_abc") is None


def test_find_nearest_snapshot():
    source_h5 = "runs/run_02_standard_aws/velocity.h5"
    if not os.path.exists(source_h5):
        pytest.skip(f"{source_h5} not found")
    
    best_key, best_time, time_diff = find_nearest_snapshot(source_h5, target_time=0.50, verbose=False)
    assert best_key == "0_499998586184208"
    assert np.isclose(best_time, 0.499998586184208, atol=1e-8)
    assert time_diff < 1e-4


def test_load_source_snapshot():
    source_h5 = "runs/run_02_standard_aws/velocity.h5"
    if not os.path.exists(source_h5):
        pytest.skip(f"{source_h5} not found")
    
    pts_src, topo_src, u_src = load_source_snapshot(source_h5, "0_499998586184208", verbose=False)
    assert pts_src.ndim == 2 and pts_src.shape[1] == 2
    assert topo_src.ndim == 2 and topo_src.shape[1] == 3
    assert u_src.ndim == 2 and u_src.shape[1] == 3
    assert pts_src.shape[0] == 10029
    assert topo_src.shape[0] == 19485
    assert u_src.shape[0] == 10029
    assert not np.isnan(u_src).any()
    assert not np.isinf(u_src).any()


def test_extract_mesh_p2_dofs(tmp_path):
    from src.mesh import generate_mesh
    mesh_path = str(tmp_path / "test_p2.msh")
    generate_mesh(output_file=mesh_path, lc_pole=0.002, lc_boundary=0.03, num_points=100, verbosity=0)
    
    coords_v, p2_coords, triangles = extract_mesh_p2_dofs(mesh_path, verbose=False)
    assert len(coords_v) > 50
    assert len(p2_coords) > len(coords_v) # P2 contains vertices + edge midpoints
    assert len(triangles) > 50
    assert p2_coords.shape[1] == 2


def test_dry_run_interpolation(tmp_path):
    source_h5 = "runs/run_02_standard_aws/velocity.h5"
    if not os.path.exists(source_h5):
        pytest.skip(f"{source_h5} not found")
    
    out_dir = str(tmp_path / "dry_run_dir")
    res = run_interpolation_pipeline(
        source_h5=source_h5,
        target_time=0.50,
        lc_pole=0.001,
        lc_boundary=0.02,
        out_dir=out_dir,
        dry_run=True,
        verbose=False,
    )
    
    assert res["status"] == "success"
    assert not np.isnan(res["u_interleaved"]).any()
    assert not np.isinf(res["u_interleaved"]).any()
    assert not np.isnan(res["p_p1"]).any()
    assert not np.isinf(res["p_p1"]).any()
    # In dry-run, no files should be saved in out_dir
    assert not os.path.exists(os.path.join(out_dir, "checkpoint_un.npy"))
    assert not os.path.exists(os.path.join(out_dir, "checkpoint_pn.npy"))


def test_full_interpolation_pipeline(tmp_path):
    source_h5 = "runs/run_02_standard_aws/velocity.h5"
    if not os.path.exists(source_h5):
        pytest.skip(f"{source_h5} not found")
    
    out_dir = str(tmp_path / "prod_run_dir")
    res = run_interpolation_pipeline(
        source_h5=source_h5,
        target_time=0.50,
        lc_pole=0.0005,
        lc_boundary=0.015,
        out_dir=out_dir,
        dry_run=False,
        verbose=False,
    )
    
    assert res["status"] == "success"
    
    un_file = os.path.join(out_dir, "checkpoint_un.npy")
    pn_file = os.path.join(out_dir, "checkpoint_pn.npy")
    meta_file = os.path.join(out_dir, "checkpoint_meta.json")
    mesh_file = os.path.join(out_dir, "apple_domain.msh")
    
    assert os.path.exists(un_file)
    assert os.path.exists(pn_file)
    assert os.path.exists(meta_file)
    assert os.path.exists(mesh_file)
    
    un = np.load(un_file)
    pn = np.load(pn_file)
    with open(meta_file, "r") as f:
        meta = json.load(f)
    
    assert len(un) == res["num_p2_nodes"] * 3
    assert len(pn) == res["num_p1_nodes"]
    assert not np.isnan(un).any()
    assert not np.isinf(un).any()
    assert not np.isnan(pn).any()
    assert not np.isinf(pn).any()
    
    assert meta["t"] == res["actual_time"]
    assert meta["num_p2_nodes"] == res["num_p2_nodes"]
    assert meta["num_p1_nodes"] == res["num_p1_nodes"]
    assert meta["lc_pole"] == 0.0005
    assert meta["max_velocity"] > 1e10
