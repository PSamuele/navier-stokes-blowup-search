"""
End-to-end smoke tests: a few dozen real time steps on a tiny mesh.

These are the tests that would have caught the Run 2 failures at runtime rather
than in a post-mortem:

* the boundary conditions still hold after the projection step (finding B2);
* the solver refuses to integrate below its CFL floor instead of clamping and
  carrying on (finding B5);
* the reported vorticity stays physical instead of exploding on the axis (B1).
"""

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("dolfinx", reason="requires the FEniCSx environment")

from src import solver                     # noqa: E402
from src import ic           # noqa: E402


def read_csv(path):
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in fh if ln.strip()]
    data = np.array([[float(v) if v else np.nan for v in row] for row in rows])
    return {name: data[:, i] for i, name in enumerate(header)}


@pytest.fixture(scope="module")
def short_run(tmp_path_factory, mesh_file):
    out = str(tmp_path_factory.mktemp("smoke"))
    status = solver.run_solver(
        mesh_file=mesh_file,
        out_dir=out,
        T=0.004,
        cfl=0.5,
        log_interval=1000,
        sample_dt=5e-4,
        xdmf_dt=1e9,
        checkpoint_dt=1e9,
    )
    return status, out


def test_run_completes(short_run):
    status, _ = short_run
    assert status["terminated_reason"] == "completed"
    assert status["reached_T"]
    assert status["final_step"] > 5


def test_boundary_conditions_hold_after_the_projection(short_run):
    """Finding B2: R2 solved step 3 with no BCs, so this residual grew unchecked."""
    _, out = short_run
    csv = read_csv(os.path.join(out, "blowup_data.csv"))
    assert np.nanmax(csv["bc_residual"]) < 1e-10


def test_vorticity_stays_physical(short_run):
    """Finding B1: R2 reported 4.8e5 at t ~ 0 against an analytic 351.6."""
    _, out = short_run
    csv = read_csv(os.path.join(out, "blowup_data.csv"))
    reference = ic.reference_extrema()["max_vorticity"]
    assert np.nanmax(csv["max_vorticity"]) < 10 * reference
    assert csv["max_vorticity"][0] == pytest.approx(reference, rel=0.30)


def test_energy_does_not_grow_in_a_viscous_flow(short_run):
    """No forcing and no inflow, so kinetic energy must not increase."""
    _, out = short_run
    csv = read_csv(os.path.join(out, "blowup_data.csv"))
    e = csv["kinetic_energy"]
    assert np.all(np.isfinite(e))
    assert e[-1] <= e[0] * 1.02, f"kinetic energy grew from {e[0]:.4f} to {e[-1]:.4f}"


def test_cfl_target_is_respected(short_run):
    _, out = short_run
    csv = read_csv(os.path.join(out, "blowup_data.csv"))
    assert np.nanmax(csv["cfl"]) <= 0.5 + 1e-6


def test_divergence_stays_small(short_run):
    _, out = short_run
    csv = read_csv(os.path.join(out, "blowup_data.csv"))
    assert np.nanmax(csv["div_u_rel"]) < 0.5


def test_metadata_records_provenance(short_run):
    status, out = short_run
    with open(os.path.join(out, "run_meta.json")) as fh:
        meta = json.load(fh)
    for key in ("mesh_fingerprint", "n_cells", "h_min", "dx_min_cfl",
                "dolfinx_version", "petsc_version", "scheme", "mpi_size"):
        assert key in meta
    # Finding B4: the CFL length scale is derived from the mesh.
    assert meta["dx_min_cfl"] == pytest.approx(meta["h_min"] / 2.0)


def test_solver_stops_instead_of_violating_cfl(tmp_path, mesh_file):
    """Finding B5, the mechanism behind the Run 2 'blow-up'.

    With dt_min set above the step the CFL condition demands, the solver must stop
    and say so.  R2 clamped at 1e-6 and kept integrating to |u| = 1e22.
    """
    status = solver.run_solver(
        mesh_file=mesh_file,
        out_dir=str(tmp_path / "cfl_guard"),
        T=0.01,
        dt_min=1.0,          # absurdly large: the CFL step is ~1e-4
        log_interval=1000,
        sample_dt=1e9,
        xdmf_dt=1e9,
        checkpoint_dt=1e9,
    )
    assert status["terminated_reason"] == "cfl_below_dt_min"
    assert "Refusing to integrate" in status["terminated_detail"]
    assert not status["reached_T"]


def test_velocity_guard_aborts(tmp_path, mesh_file):
    """A divergent run must be caught, not written to disk for 39,000 rows."""
    status = solver.run_solver(
        mesh_file=mesh_file,
        out_dir=str(tmp_path / "vel_guard"),
        T=0.01,
        max_velocity=1.0,    # the initial condition already peaks at 25
        log_interval=1000,
        sample_dt=1e9,
        xdmf_dt=1e9,
        checkpoint_dt=1e9,
    )
    assert status["terminated_reason"] == "velocity_limit_exceeded"


def test_chorin_scheme_still_runs(tmp_path, mesh_file):
    """--scheme chorin reproduces the R2 splitting, for comparison studies."""
    status = solver.run_solver(
        mesh_file=mesh_file,
        out_dir=str(tmp_path / "chorin"),
        T=0.002,
        scheme="chorin",
        log_interval=1000,
        sample_dt=1e-3,
        xdmf_dt=1e9,
        checkpoint_dt=1e9,
    )
    assert status["terminated_reason"] == "completed"


def test_restart_resumes_from_checkpoint(tmp_path, mesh_file):
    out = str(tmp_path / "restart")
    first = solver.run_solver(
        mesh_file=mesh_file, out_dir=out, T=0.002,
        log_interval=1000, sample_dt=1e-3, xdmf_dt=1e9, checkpoint_dt=1e-3,
    )
    assert first["terminated_reason"] == "completed"

    second = solver.run_solver(
        mesh_file=mesh_file, out_dir=out, T=0.004, restart=True,
        log_interval=1000, sample_dt=1e-3, xdmf_dt=1e9, checkpoint_dt=1e-3,
    )
    assert second["terminated_reason"] == "completed"
    assert second["final_t"] > first["final_t"]
    # Finding B8: the XDMF series is appended to, not truncated.
    assert os.path.exists(os.path.join(out, "velocity.xdmf"))


def test_energy_guard_stops_an_unphysical_run(tmp_path, mesh_file):
    """Finding: the coarse grid loses resolution long before it violates CFL.

    In a closed domain with no-slip walls and no body force the exact solution
    obeys dE/dt = -2 nu int|D(u)|^2 <= 0, so kinetic energy growth is numerical by
    definition. Observed on a coarse grid: energy fell monotonically to t = 0.1223
    and then rose 0.3%, 1.4%, 4.2%, 15%, 48% while max|u| went 25.5 -> 480, all with
    the CFL number sitting exactly on its 0.5 target. A velocity threshold needs an
    arbitrary constant and a CFL guard never fires; this one is exact.
    """
    status = solver.run_solver(
        mesh_file=mesh_file,
        out_dir=str(tmp_path / "energy_guard"),
        T=0.01,
        max_energy_growth=-1.0,   # any decrease at all counts as "growth"
        log_interval=1000,
        sample_dt=1e-3,
        xdmf_dt=1e9,
        checkpoint_dt=1e9,
    )
    assert status["terminated_reason"] == "energy_growth"
    assert "Energy cannot increase" in status["terminated_detail"]
    assert status["t_at_energy_min"] >= 0.0


def test_energy_guard_allows_a_healthy_run(short_run):
    """A well-resolved run must not trip the guard."""
    status, out = short_run
    assert status["terminated_reason"] == "completed"
    csv = read_csv(os.path.join(out, "blowup_data.csv"))
    ke = csv["kinetic_energy"]
    # Monotone decay, to within the splitting error of the projection scheme.
    assert np.all(np.diff(ke) < 1e-6 * ke[0])
    assert status["kinetic_energy_min"] == pytest.approx(np.min(ke), rel=1e-9)


def test_tag_suffixes_every_output_file(tmp_path, mesh_file):
    """A convergence level must produce self-identifying filenames.

    Three levels write into sibling directories; without a tag their files are
    named identically and stop being distinguishable once moved or downloaded.
    """
    out = str(tmp_path / "tagged")
    solver.run_solver(mesh_file=mesh_file, out_dir=out, T=0.002, cfl=0.5,
                      log_interval=1000, sample_dt=5e-4, xdmf_dt=1e-3,
                      checkpoint_dt=1e9, tag="medium")

    for name in ("blowup_data_medium.csv", "status_medium.json",
                 "run_meta_medium.json", "velocity_medium.xdmf"):
        assert os.path.exists(os.path.join(out, name)), f"missing {name}"

    # The untagged names must not also appear, or both would be picked up.
    for name in ("blowup_data.csv", "status.json", "run_meta.json"):
        assert not os.path.exists(os.path.join(out, name)), f"unexpected {name}"

    # XDMF references its HDF5 companion by name; the tag has to reach both.
    with open(os.path.join(out, "velocity_medium.xdmf")) as fh:
        assert "velocity_medium.h5" in fh.read()
