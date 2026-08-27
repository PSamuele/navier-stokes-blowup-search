"""
CLI and restart-guard tests for the Run 3 solver.

R2 had no command line at all: ``run_solver()`` took no arguments and hard-coded
``out_dir = "results"``, so the command published in the project README --
``python main_R2.py --out_dir ... --mesh ...`` -- silently ignored both flags
(finding B9).  The flag names below deliberately match the ones the repository's
existing ``tests/test_main_cli.py`` already expects, so the intended API is the one
that now exists.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("dolfinx", reason="requires the FEniCSx environment")

from src import solver  # noqa: E402


EXPECTED_FLAGS = [
    "--mesh", "--out_dir", "--checkpoint_dir", "--restart", "-T", "--t_final",
    "--num_steps", "--adaptive_dt", "--no_adaptive_dt", "--cfl", "--dx_min",
    "--nu", "--log_interval", "--scheme", "--dt_min", "--max_velocity",
    "--sample_dt", "--xdmf_dt", "--label",
]


def test_help_lists_every_documented_flag():
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "src", "solver.py"), "--help"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    for flag in EXPECTED_FLAGS:
        assert flag in proc.stdout, f"missing flag {flag}"


def test_parser_defaults():
    args = solver.build_parser().parse_args([])
    assert args.mesh == "assets/apple_domain.msh"
    assert args.out_dir == "results"
    assert args.checkpoint_dir is None
    assert args.restart is False
    assert np.isclose(args.t_final, 0.55)
    assert args.num_steps == 1100
    assert args.adaptive_dt is True
    assert np.isclose(args.cfl, 0.5)
    assert args.dx_min is None          # measured from the mesh, not hard-coded
    assert np.isclose(args.nu, 1e-3)
    assert args.log_interval == 10
    assert args.scheme == "ipcs"


def test_parser_custom_arguments():
    args = solver.build_parser().parse_args([
        "--mesh", "assets/custom.msh",
        "--out_dir", "runs/custom_out",
        "--checkpoint_dir", "runs/custom_cp",
        "--restart",
        "-T", "0.85",
        "--num_steps", "2500",
        "--no_adaptive_dt",
        "--cfl", "0.35",
        "--dx_min", "0.0005",
        "--nu", "0.005",
        "--log_interval", "50",
        "--scheme", "chorin",
        "--label", "experiment",
    ])
    assert args.mesh == "assets/custom.msh"
    assert args.out_dir == "runs/custom_out"
    assert args.checkpoint_dir == "runs/custom_cp"
    assert args.restart is True
    assert np.isclose(args.t_final, 0.85)
    assert args.num_steps == 2500
    assert args.adaptive_dt is False
    assert np.isclose(args.cfl, 0.35)
    assert np.isclose(args.dx_min, 0.0005)
    assert np.isclose(args.nu, 0.005)
    assert args.log_interval == 50
    assert args.scheme == "chorin"
    assert args.label == "experiment"


def test_adaptive_dt_toggle():
    parser = solver.build_parser()
    assert parser.parse_args([]).adaptive_dt is True
    assert parser.parse_args(["--no_adaptive_dt"]).adaptive_dt is False
    assert parser.parse_args(["--adaptive_dt"]).adaptive_dt is True


def test_scheme_choices_are_constrained():
    with pytest.raises(SystemExit):
        solver.build_parser().parse_args(["--scheme", "nonsense"])


def test_missing_mesh_raises():
    with pytest.raises(FileNotFoundError, match="Mesh file.*not found"):
        solver.run_solver(mesh_file="nonexistent_mesh_xyz123.msh")


def test_restart_without_checkpoint_dir_raises(tmp_path, mesh_file):
    with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
        solver.run_solver(
            mesh_file=mesh_file,
            out_dir=str(tmp_path / "o"),
            restart=True,
            checkpoint_dir=str(tmp_path / "nonexistent_cp_9999"),
        )


def test_restart_without_metadata_raises(tmp_path, mesh_file):
    cp = tmp_path / "empty_cp"
    cp.mkdir()
    with pytest.raises(FileNotFoundError, match="Checkpoint metadata file.*not found"):
        solver.run_solver(
            mesh_file=mesh_file,
            out_dir=str(tmp_path / "o"),
            restart=True,
            checkpoint_dir=str(cp),
        )


def test_restart_rejects_a_different_rank_count(tmp_path, mesh_file):
    """R2 stored per-rank arrays with no record of how many ranks wrote them."""
    cp = tmp_path / "cp"
    cp.mkdir()
    (cp / "checkpoint_meta.json").write_text(
        json.dumps({"step": 10, "t": 0.1, "mpi_size": 99})
    )
    with pytest.raises(ValueError, match="written with 99 ranks"):
        solver.run_solver(
            mesh_file=mesh_file,
            out_dir=str(tmp_path / "o"),
            restart=True,
            checkpoint_dir=str(cp),
        )


def test_measure_mesh_spacing_reflects_the_actual_mesh(mesh_file):
    """Finding B4: dx_min must come from the mesh, not from a literal."""
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    domain = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2).mesh
    h_min, h_max, dx_eff, h_cells = solver.measure_mesh_spacing(domain, degree=2)
    assert 0.0 < h_min <= h_max
    assert dx_eff == pytest.approx(h_min / 2.0)
    # The fixture mesh asks for lc_pole = 0.02, so h_min must be near that scale --
    # nothing like the 1e-4 R2 assumed for a mesh whose true minimum was 1.6e-3.
    assert 1e-3 < h_min < 1e-1

    # Per-cell diameters back the cell-local CFL condition.
    n_local = domain.topology.index_map(domain.topology.dim).size_local
    assert h_cells.shape == (n_local,)
    assert np.isclose(h_cells.min(), h_min) and np.isclose(h_cells.max(), h_max)


def test_cell_local_cfl_beats_the_global_pairing(mesh_file):
    """The global min(h)/max|u| pairing is needlessly conservative.

    It couples the smallest cell in the mesh to the fastest fluid anywhere in the
    mesh. On these graded meshes that costs a large factor in step size for no
    stability benefit.
    """
    from mpi4py import MPI
    from dolfinx import fem
    from dolfinx.io import gmsh as gmshio

    from src import ic
    from src.diagnostics import Diagnostics

    domain = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, 0, gdim=2).mesh
    h_min, _, dx_eff, h_cells = solver.measure_mesh_spacing(domain, degree=2)
    V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    u = fem.Function(V)
    u.interpolate(ic.velocity_callable())
    u.x.scatter_forward()

    diag = Diagnostics(domain, u, quadrature_degree=6)
    dt_cell, vmax = diag.cellwise_cfl_dt(h_cells, cfl=0.5, degree=2)
    dt_global = 0.5 * dx_eff / vmax

    assert np.isfinite(dt_cell) and dt_cell > 0
    assert vmax == pytest.approx(25.21, rel=0.02)
    assert dt_cell > dt_global, "the cell-local step must never be the smaller one"


def test_check_ksp_raises_on_a_diverged_solve():
    """The guard for the silent-failure class of bug.

    PETSc signals a preconditioner breakdown by setting a negative converged
    reason and leaving Inf in the solution vector -- it does not raise.  On the
    production mesh this happened for real: the r-weighted mass matrix has
    diagonal entries near 7e-15 at the axis, ILU inside block Jacobi failed, and
    the projection returned reason -11 with an Inf velocity field.
    """
    class FakeKSP:
        def __init__(self, reason):
            self._reason = reason

        def getConvergedReason(self):
            return self._reason

        def getIterationNumber(self):
            return 0

    with pytest.raises(solver.SolverStop) as excinfo:
        solver.check_ksp(FakeKSP(-11), "projection", step=42, t=0.123)
    assert excinfo.value.reason == "linear_solver_diverged"
    assert "DIVERGED_PC_FAILED" in excinfo.value.detail
    assert "step 42" in excinfo.value.detail

    # A converged solve returns the iteration count and does not raise.
    assert solver.check_ksp(FakeKSP(2), "momentum", step=1, t=0.0) == 0


def test_projection_preconditioner_defaults_to_sor():
    """bjacobi/ilu breaks down on the near-axis mass matrix; sor does not."""
    args = solver.build_parser().parse_args([])
    assert args.pc_projection == "sor"
    assert args.pc_pressure == "hypre"
