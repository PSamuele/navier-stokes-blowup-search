"""
Unit and integration tests for src/mesh.py mesh generator.
"""
import os
import pytest
import gmsh
from src.mesh import generate_mesh


def test_generate_mesh_default(tmp_path):
    out_mesh = str(tmp_path / "default_apple_domain.msh")
    info = generate_mesh(
        output_file=out_mesh,
        lc_pole=0.002,
        lc_boundary=0.03,
        num_points=100,
        verbosity=0,
    )
    
    assert os.path.exists(out_mesh), f"Mesh file {out_mesh} was not created"
    assert os.path.getsize(out_mesh) > 1000, "Mesh file is too small"
    assert info["output_file"] == out_mesh
    assert info["num_nodes"] > 50, f"Expected >50 nodes, got {info['num_nodes']}"
    assert info["num_elements_2d"] > 50, f"Expected >50 elements, got {info['num_elements_2d']}"


def test_generate_mesh_custom_parameters(tmp_path):
    out_mesh = str(tmp_path / "fine_test.msh")
    info = generate_mesh(
        output_file=out_mesh,
        lc_pole=0.001,
        lc_boundary=0.02,
        R0=1.2,
        H=1.8,
        k=0.6,
        num_points=150,
        verbosity=0,
    )
    
    assert os.path.exists(out_mesh)
    assert info["lc_pole"] == 0.001
    assert info["lc_boundary"] == 0.02
    assert info["R0"] == 1.2
    assert info["H"] == 1.8
    assert info["k"] == 0.6
    assert info["num_nodes"] > 100


def test_mesh_physical_groups(tmp_path):
    out_mesh = str(tmp_path / "pg_test.msh")
    generate_mesh(
        output_file=out_mesh,
        lc_pole=0.003,
        lc_boundary=0.04,
        num_points=50,
        verbosity=0,
    )
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(out_mesh)
    
    physical_groups = gmsh.model.getPhysicalGroups()
    # Expect 1D SymmetryAxis, 1D AppleWall, 2D FluidDomain
    dim_tags = [pg[0] for pg in physical_groups]
    assert 1 in dim_tags, "Missing 1D physical groups"
    assert 2 in dim_tags, "Missing 2D physical group"
    
    # Check group names
    names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in physical_groups]
    assert "SymmetryAxis" in names
    assert "AppleWall" in names
    assert "FluidDomain" in names
    
    gmsh.finalize()
