"""
Regression tests for the Run 3 mesh generator.

The headline test is :func:`test_polar_refinement_is_actually_delivered`, which is
the check that Run 2 lacked.  R2 asked gmsh for h = 1e-4 at the poles and received
0.015 -- a factor of 150 -- because its MathEval size field referenced the gmsh
coordinate ``z`` while the geometry lives in the *xy* plane, making the field a
constant.  Nothing in the project ever compared the request with the result, so both
published runs used a uniform mesh while the README described exponential polar
refinement.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import mesh  # noqa: E402


# ---------------------------------------------------------------- geometry

def test_boundary_profile_matches_the_study_definition():
    z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    r = mesh.boundary_radius(z, R0=1.0, H=2.0, k=0.5)
    assert r[2] == pytest.approx(1.0)            # equator
    assert r[0] == pytest.approx(0.0, abs=1e-15)  # south pole
    assert r[4] == pytest.approx(0.0, abs=1e-15)  # north pole
    assert np.all(r >= 0.0)


def test_boundary_derivative_agrees_with_finite_differences():
    z = np.linspace(-1.9, 1.9, 41)
    eps = 1e-6
    fd = (mesh.boundary_radius(z + eps) - mesh.boundary_radius(z - eps)) / (2 * eps)
    assert np.allclose(mesh.boundary_radius_prime(z), fd, rtol=1e-5, atol=1e-7)


def test_domain_is_a_cone_not_a_cusp_at_the_poles():
    """Documents finding D1 so the discrepancy cannot be forgotten again.

    f(z) vanishes linearly at z = H, so the "cusp" is a cone of half-angle ~6 deg.
    The README claims the boundary "narrows exponentially near z = H" and that
    "k >> 1 dictates the severity of the polar cusp"; neither is true of this
    profile, and the study runs with k = 0.5.
    """
    H = 2.0
    slope = mesh.boundary_radius_prime(np.array([H]), R0=1.0, H=H, k=0.5)[0]
    assert slope == pytest.approx(-0.10633, rel=1e-3)

    # A cusp would have f/(H-z) -> 0.  Here the ratio tends to |f'(H)|, a constant.
    d = np.array([1e-3, 1e-4, 1e-5])
    ratio = mesh.boundary_radius(H - d, R0=1.0, H=H, k=0.5) / d
    assert np.allclose(ratio, abs(slope), rtol=1e-2)


# ---------------------------------------------------------------- meshing

@pytest.fixture(scope="module")
def graded_mesh(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("mesh") / "graded.msh")
    info = mesh.generate_mesh(
        output_file=out, lc_pole=0.004, lc_boundary=0.030, num_points=None, verbosity=0
    )
    return info


def test_polar_refinement_is_actually_delivered(graded_mesh):
    """The regression guard for finding A1.

    Run against the R2 recipe this fails by a factor of 150.
    """
    info = graded_mesh
    achieved = info["h_pole_actual"]
    requested = info["lc_pole"]
    assert achieved == pytest.approx(requested, rel=0.5), (
        f"polar element size {achieved:.3e} does not match the requested "
        f"{requested:.3e} (ratio {achieved / requested:.1f}); the size field is not "
        "reaching the poles"
    )


def test_equatorial_size_is_delivered(graded_mesh):
    info = graded_mesh
    assert info["h_equator_actual"] == pytest.approx(info["lc_boundary"], rel=0.5)


def test_pole_region_is_resolved(graded_mesh):
    """R2 had 12 cells in |z| > 0.95H across the entire domain."""
    assert graded_mesh["n_cells_pole_region"] > 100


def test_mesh_stays_in_the_half_plane(graded_mesh):
    """Finding A4: a spline through sparse points can overshoot to r < 0."""
    assert graded_mesh["r_min"] >= 0.0
    assert graded_mesh["z_min"] == pytest.approx(-2.0)
    assert graded_mesh["z_max"] == pytest.approx(2.0)


def test_boundary_sampling_scales_with_the_requested_size():
    """Finding A3: R2 used 400 uniform points regardless of the target size."""
    coarse, _, _ = mesh._graded_boundary_points(0.02, 0.08, 1.0, 2.0, 0.5, 2.0, None)
    fine, _, _ = mesh._graded_boundary_points(0.002, 0.008, 1.0, 2.0, 0.5, 2.0, None)
    assert len(fine) > 3 * len(coarse)

    # Points must cluster toward the poles, not spread uniformly in z.
    near_pole = np.sum(np.abs(fine) > 1.9)
    uniform_expectation = len(fine) * (0.2 / 4.0)
    assert near_pole > uniform_expectation


def test_refinement_is_monotone_in_the_requested_size(tmp_path):
    """Halving lc_pole must actually produce smaller cells and more of them."""
    prev_cells, prev_h = 0, np.inf
    for lc in (0.016, 0.008, 0.004):
        info = mesh.generate_mesh(
            output_file=str(tmp_path / f"m_{lc}.msh"),
            lc_pole=lc, lc_boundary=lc * 4, verbosity=0,
        )
        assert info["num_elements_2d"] > prev_cells
        assert info["h_pole_actual"] < prev_h
        prev_cells, prev_h = info["num_elements_2d"], info["h_pole_actual"]


def test_info_dict_reports_the_requested_api(tmp_path):
    """The API tests/test_mesh.py and interpolate_mesh_R2.py already assumed."""
    out = str(tmp_path / "api.msh")
    info = mesh.generate_mesh(
        output_file=out, lc_pole=0.01, lc_boundary=0.04,
        R0=1.2, H=1.8, k=0.6, num_points=300, verbosity=0,
    )
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000
    assert info["output_file"] == out
    assert info["lc_pole"] == 0.01
    assert info["lc_boundary"] == 0.04
    assert info["R0"] == 1.2 and info["H"] == 1.8 and info["k"] == 0.6
    assert info["num_nodes"] > 50
    assert info["num_elements_2d"] > 50


def test_physical_groups_are_present(tmp_path):
    import gmsh

    out = str(tmp_path / "pg.msh")
    mesh.generate_mesh(output_file=out, lc_pole=0.02, lc_boundary=0.06, verbosity=0)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(out)
        groups = gmsh.model.getPhysicalGroups()
        names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups]
        assert "SymmetryAxis" in names
        assert "AppleWall" in names
        assert "FluidDomain" in names
    finally:
        gmsh.finalize()


def test_invalid_sizes_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        mesh.generate_mesh(output_file=str(tmp_path / "a.msh"), lc_pole=-1.0)
    with pytest.raises(ValueError, match="must not exceed"):
        mesh.generate_mesh(
            output_file=str(tmp_path / "b.msh"), lc_pole=0.1, lc_boundary=0.01
        )


def test_verification_rejects_a_mesh_that_missed_its_target():
    """The guard itself must fire, otherwise it is decoration."""
    info = {
        "output_file": "synthetic.msh",
        "lc_pole": 1e-4,
        "lc_boundary": 0.015,
        # Exactly what the R2 recipe produced.
        "h_pole_actual": 0.015,
        "h_equator_actual": 0.015,
        "r_min": 0.0,
    }
    with pytest.raises(RuntimeError, match="polar element size is 150"):
        mesh._verify(info, tol=3.0)


def test_cli_emits_json(tmp_path):
    out = str(tmp_path / "cli.msh")
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "mesh.py"),
         "-o", out, "--lc_pole", "0.02", "--lc_boundary", "0.06", "--verbosity", "0"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout)
    assert info["output_file"] == out
    assert os.path.exists(out)
