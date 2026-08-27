"""
Regression tests for the Run 3 diagnostics.

The headline test is :func:`test_vorticity_matches_the_analytic_initial_condition`.
Run 2 logged ``max_vorticity = 4.826e5`` on its very first sample where the exact
initial condition gives ``351.6`` -- a factor of 1373 before any physics had
happened -- because it evaluated ``u_theta / (r + 1e-14)`` at DG1 interpolation
points, which are the cell vertices, and every cell touching the symmetry axis has
vertices at exactly r = 0.  Every conclusion Run 2 drew from its vorticity curve
rests on that number.
"""

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("dolfinx", reason="requires the FEniCSx environment")

import ufl                                             # noqa: E402
from mpi4py import MPI                                 # noqa: E402
from dolfinx import fem                                # noqa: E402
from dolfinx.io import gmsh as gmshio                  # noqa: E402

from src import ic, mesh                      # noqa: E402
from src.diagnostics import Diagnostics          # noqa: E402


REFERENCE = ic.reference_extrema()


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("diag") / "diag.msh")
    info = mesh.generate_mesh(
        output_file=path, lc_pole=0.006, lc_boundary=0.035, verbosity=0
    )
    domain = gmshio.read_from_msh(path, MPI.COMM_WORLD, 0, gdim=2).mesh
    V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    u = fem.Function(V)
    u.interpolate(ic.velocity_callable())
    u.x.scatter_forward()
    return domain, u, info


# ------------------------------------------------------------------

def test_analytic_reference_is_what_we_think_it_is():
    assert REFERENCE["max_velocity"] == pytest.approx(25.2117, rel=1e-4)
    assert REFERENCE["max_vorticity"] == pytest.approx(351.64, rel=1e-3)


def test_initial_condition_is_regular_on_the_axis():
    """R2 guarded with Max(r, 1e-12); the fields are genuinely regular instead."""
    import sympy as sp

    u_r, u_z, u_theta, _, _, (r, z) = ic.symbolic_fields()
    for expr in (u_r, u_z, u_theta):
        value = expr.subs({r: 0, z: 0.3})
        assert value.is_finite, f"{expr} is singular on the axis"
        assert np.isfinite(float(sp.N(value)))


def test_quadrature_points_never_touch_the_axis(state):
    """The invariant that removes the need for any epsilon at all."""
    domain, u, _ = state
    diag = Diagnostics(domain, u, quadrature_degree=6)
    assert diag.audit() > 0.0


def test_vorticity_matches_the_analytic_initial_condition(state):
    """The regression guard for finding B1.

    The R2 diagnostic returns ~4.8e5 here.  Anything above a few hundred means the
    axis singularity has come back.
    """
    domain, u, _ = state
    diag = Diagnostics(domain, u, quadrature_degree=6)
    res = diag.compute()

    assert res["max_vorticity"] == pytest.approx(REFERENCE["max_vorticity"], rel=0.05)
    assert res["max_velocity"] == pytest.approx(REFERENCE["max_velocity"], rel=0.02)

    # Explicit statement of the failure mode, so a regression names itself.
    assert res["max_vorticity"] < 1e4, (
        f"max|omega| = {res['max_vorticity']:.3e} is far above the analytic 351.6; "
        "the axis 1/r singularity is back (finding B1)"
    )


def test_vorticity_maximum_sits_on_the_vortex_ring(state):
    """The ring is centred at (r, z) = (Rv, 0) = (0.5, 0)."""
    domain, u, _ = state
    res = Diagnostics(domain, u, quadrature_degree=6).compute()
    assert res["r_at_max_vorticity"] == pytest.approx(0.5, abs=0.12)
    assert res["z_at_max_vorticity"] == pytest.approx(0.0, abs=0.12)


def test_the_r2_formula_reproduces_the_bug(state):
    """Direct demonstration that the old recipe is what produced 4.8e5.

    Interpolating the same field with R2's DG1 + 1e-14 recipe must give a number
    orders of magnitude above the analytic value; if it did not, the diagnosis of
    finding B1 would be wrong.
    """
    domain, u, _ = state
    x = ufl.SpatialCoordinate(domain)
    r_safe = x[0] + 1e-14  # R2, verbatim

    # Perturb the axis DOFs by solver-tolerance noise, which is what R2 had after
    # its boundary-condition-free projection step (finding B2).
    u_noisy = fem.Function(u.function_space)
    u_noisy.x.array[:] = u.x.array
    coords = u.function_space.tabulate_dof_coordinates()
    bs = u.function_space.dofmap.index_map_bs
    on_axis = np.where(np.isclose(coords[:, 0], 0.0))[0]
    rng = np.random.default_rng(0)
    u_noisy.x.array[on_axis * bs + 2] = rng.normal(0.0, 1e-9, on_axis.size)
    u_noisy.x.scatter_forward()

    omega_z = u_noisy[2].dx(0) + u_noisy[2] / r_safe
    W = fem.functionspace(domain, ("DG", 1))
    f = fem.Function(W)
    f.interpolate(fem.Expression(omega_z**2, W.element.interpolation_points))
    r2_value = float(np.sqrt(np.max(f.x.array)))

    assert r2_value > 1e4, (
        "the R2 recipe should blow up on axis noise; if it does not, the "
        "explanation of finding B1 needs revisiting"
    )


def test_conserved_quantities_are_positive_and_finite(state):
    domain, u, _ = state
    res = Diagnostics(domain, u, quadrature_degree=6).compute()
    for key in ("kinetic_energy", "enstrophy", "max_circulation"):
        assert np.isfinite(res[key]) and res[key] > 0.0
    # Volume of revolution, 2 pi * int r dr dz = pi * int f(z)^2 dz = 4.285398.
    # This validates the r-weighted quadrature the whole module is built on.
    assert Diagnostics(domain, u).volume == pytest.approx(4.285398, rel=2e-3)


def test_bkm_integral_accumulates_monotonically(state):
    domain, u, _ = state
    diag = Diagnostics(domain, u, quadrature_degree=4)
    values = [diag.accumulate_bkm(t, 100.0) for t in (0.0, 0.1, 0.2, 0.3)]
    assert values == sorted(values)
    assert values[-1] == pytest.approx(30.0, rel=1e-6)


def test_divergence_is_reported_both_strongly_and_weakly(state):
    domain, u, _ = state
    res = Diagnostics(domain, u, quadrature_degree=6).compute()
    for key in ("div_u_l2", "div_u_rel", "div_u_weak"):
        assert key in res and np.isfinite(res[key])
    # The raw P2 interpolant of the streamfunction is not discretely solenoidal,
    # but it should not be wildly divergent either.
    assert res["div_u_rel"] < 0.5
