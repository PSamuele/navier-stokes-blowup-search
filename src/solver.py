#!/usr/bin/env python3
"""
solver.py - Axisymmetric incompressible Navier-Stokes solver (Run 3).

Fixed-point of comparison: ``runs/run_02/code_R2/main_R2.py``.  Every numbered
finding below is documented with evidence in ``docs/findings.md``.

Corrections carried by this file
--------------------------------
B1   Vorticity is no longer evaluated as ``u_theta/(r + 1e-14)`` at DG1 vertices,
     which sit exactly on the symmetry axis.  See ``src/diagnostics.py``.

B2   The projection step now applies the boundary conditions.  R2 solved step 3
     with a bare mass matrix and no ``apply_lifting``/``set_bc``, so the corrected
     velocity satisfied neither no-slip on the wall nor u_r = u_theta = 0 on the
     axis, which is what fed B1.

B3   The scheme is genuine incremental pressure correction (IPCS): the predictor
     carries ``-grad(p^n)``, step 2 solves for the increment ``phi``, and
     ``p^{n+1} = p^n + phi``.  R2 advertised IPCS in the README but implemented
     first-order Chorin.  ``--scheme chorin`` reproduces the old behaviour.

B4   ``dx_min`` is measured from the mesh (``dolfinx.cpp.mesh.h``, global MPI min)
     and divided by the polynomial degree for the P2 node spacing.  R2 hard-coded
     1e-4 while its mesh had a true minimum edge of 1.63e-3.

B5   The CFL time step is no longer silently clamped.  If the stability limit falls
     below ``--dt_min`` the run stops and records why.  R2 clamped at 1e-6 and kept
     integrating to |u| = 1e22.

B6   The pressure is fixed by a constant null space, not by a Dirichlet point
     constraint at an interior node (not H^1-admissible in 2-D, and it degrades AMG).

B7   The HYPRE/GAMG fallback actually works: availability is probed with
     ``PCSetUp``, since ``PCSetType`` does not fail on a missing package.

B8   Restart appends to the XDMF series instead of truncating it, validates the
     rank count and mesh fingerprint, and is reachable from the command line.

B9   Full CLI (``build_parser``/``main``/``run_solver``).  R2 took no arguments at
     all, so the documented ``--out_dir``/``--mesh`` flags were silently ignored.

B10  Logging, field output and checkpointing have independent cadences and are
     sampled on *physical time*, not step count -- required for runs at different
     resolutions to be comparable on a common grid.

B12  The initial condition is projected onto a discretely divergence-free field and
     the no-slip residual is measured rather than ignored.

B13  ``A1`` is reassembled every step.  R2 reassembled it only when ``dt`` changed,
     but ``a1`` carries ``u_n`` as a coefficient through the linearised advection;
     with ``dt`` pinned at its floor for 62.8% of the run, the momentum operator was
     frozen at a stale velocity field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

try:
    import ufl
    import dolfinx
    import dolfinx.cpp as _dcpp
    from dolfinx import fem, mesh as dmesh
    from dolfinx.io import XDMFFile
    from dolfinx.io import gmsh as gmshio
    from dolfinx.fem.petsc import (
        assemble_matrix,
        assemble_vector,
        apply_lifting,
        set_bc,
        create_vector,
    )
except ImportError as exc:  # pragma: no cover
    print(f"Import error: {exc}", file=sys.stderr)
    print("Activate the FEniCSx environment first (conda activate fenicsx-env).", file=sys.stderr)
    raise

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import ic                                          # noqa: E402
from src.diagnostics import Diagnostics, boundary_condition_residual  # noqa: E402


def tagged(name, tag):
    """Insert a run tag before the extension: blowup_data.csv -> blowup_data_fine.csv.

    A convergence study writes three levels whose per-level files would otherwise
    be named identically, distinguished only by their directory.  The tag keeps
    each file self-identifying once it is downloaded, quoted or archived away
    from that directory.  Without a tag the plain names are used.
    """
    if not tag:
        return name
    root, ext = os.path.splitext(name)
    return f"{root}_{tag}{ext}"


CSV_COLUMNS = [
    "step", "t", "dt", "cfl",
    "max_velocity", "max_vorticity", "max_circulation",
    "kinetic_energy", "enstrophy", "div_u_l2", "div_u_rel", "div_u_weak", "bkm_integral",
    "r_at_max_vorticity", "z_at_max_vorticity",
    "bc_residual", "iters_momentum", "iters_pressure", "iters_projection",
    "wall_time",
]


# =====================================================================
# helpers
# =====================================================================

def _log(comm, msg):
    """Print from rank 0 only (finding B10: R2 printed from all 16 ranks)."""
    if comm.rank == 0:
        print(msg, flush=True)


def _file_fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit(cwd):
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def measure_mesh_spacing(domain, degree):
    """Global minimum/maximum cell diameter, and the effective node spacing.

    Finding B4: R2 hard-coded ``dx_min = 0.0001``, unrelated to the mesh it was
    actually running on (true minimum edge 1.63e-3).  For a degree-``p`` Lagrange
    space the node spacing is roughly ``h/p``, which is what the CFL condition
    should use.
    """
    tdim = domain.topology.dim
    n_local = domain.topology.index_map(tdim).size_local
    cells = np.arange(n_local, dtype=np.int32)
    h = (_dcpp.mesh.h(domain._cpp_object, tdim, cells) if n_local
         else np.zeros(0, dtype=np.float64))

    comm = domain.comm
    h_min = float(comm.allreduce(float(np.min(h)) if n_local else np.inf, op=MPI.MIN))
    h_max = float(comm.allreduce(float(np.max(h)) if n_local else -np.inf, op=MPI.MAX))
    return h_min, h_max, h_min / max(degree, 1), h


def _make_ksp(comm, A, ksp_type, pc_type, rtol, atol, fallback_pc=None, name=""):
    """Create a KSP, verifying the preconditioner really is available.

    Finding B7: R2 wrapped ``PCSetType(HYPRE)`` in a bare ``try/except``.  Setting a
    type never raises for a missing package -- the failure surfaces later inside
    ``PCSetUp`` -- so its GAMG fallback could never trigger.
    """
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    pc.setType(pc_type)
    chosen = pc_type
    if fallback_pc is not None:
        try:
            pc.setUp()
        except Exception as exc:
            _log(comm, f"  [{name}] preconditioner '{pc_type}' unavailable ({exc}); "
                       f"falling back to '{fallback_pc}'")
            pc.setType(fallback_pc)
            chosen = fallback_pc
    ksp.setTolerances(rtol=rtol, atol=atol)
    ksp.setFromOptions()
    return ksp, chosen


def check_ksp(ksp, name, step, t):
    """Abort on a KSP that did not converge.

    PETSc does **not** raise when a preconditioner breaks down: it sets a negative
    converged reason and leaves the solution vector filled with Inf.  On the
    production mesh the r-weighted mass matrix has diagonal entries down to 7e-15
    near the axis, ILU inside block Jacobi fails on it, and the projection step
    returned reason -11 (DIVERGED_PC_FAILED) with an Inf velocity field -- which an
    unchecked solve would have propagated into every subsequent step.  Silent
    numerical failure is the whole reason Run 2 produced 39,000 rows of nonsense,
    so every solve is checked.
    """
    reason = ksp.getConvergedReason()
    if reason < 0:
        raise SolverStop(
            "linear_solver_diverged",
            f"{name} solve failed at step {step} (t = {t:.6f}): PETSc converged "
            f"reason {reason} after {ksp.getIterationNumber()} iterations. "
            f"Reason -11 is DIVERGED_PC_FAILED (preconditioner breakdown); try a "
            f"different preconditioner for this stage.",
        )
    return ksp.getIterationNumber()


class SolverStop(RuntimeError):
    """Raised when the run must stop for a physical or numerical reason."""

    def __init__(self, reason, detail):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


# =====================================================================
# solver
# =====================================================================

def run_solver(
    mesh_file="assets/apple_domain.msh",
    out_dir="results",
    checkpoint_dir=None,
    restart=False,
    T=0.55,
    num_steps=1100,
    adaptive_dt=True,
    cfl=0.5,
    dx_min=None,
    dt_min=1e-9,
    dt_max=5e-3,
    nu=1e-3,
    scheme="ipcs",
    log_interval=10,
    sample_dt=None,
    xdmf_dt=None,
    checkpoint_dt=None,
    max_velocity=1e6,
    max_energy_growth=0.01,
    quadrature_degree=6,
    pc_momentum="bjacobi",
    pc_pressure="hypre",
    pc_projection="sor",
    degree_u=2,
    degree_p=1,
    taper_ic=False,
    project_ic=True,
    label=None,
    tag=None,
    comm=None,
):
    """Integrate the axisymmetric Navier-Stokes equations and write diagnostics.

    Returns a status dict, also written to ``<out_dir>/status.json``.
    """
    comm = comm or MPI.COMM_WORLD
    t_wall0 = time.time()

    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file '{mesh_file}' not found")

    checkpoint_dir = checkpoint_dir or os.path.join(out_dir, "checkpoints")
    meta_file = os.path.join(checkpoint_dir, tagged("checkpoint_meta.json", tag))
    csv_file = os.path.join(out_dir, tagged("blowup_data.csv", tag))
    xdmf_path = os.path.join(out_dir, tagged("velocity.xdmf", tag))
    status_file = os.path.join(out_dir, tagged("status.json", tag))

    # Validate a restart request *before* creating anything, otherwise the
    # directory check below can never fail: makedirs would have just created it.
    if restart:
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Checkpoint metadata file not found: {meta_file}")

    if comm.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    comm.Barrier()

    _log(comm, f"--- Navier-Stokes cusp solver, Run 3 (dolfinx {dolfinx.__version__}) ---")
    _log(comm, f"    mesh    : {mesh_file}")
    _log(comm, f"    out_dir : {out_dir}")
    _log(comm, f"    ranks   : {comm.size}   scheme: {scheme}")

    # ---------------- mesh & spaces ----------------
    domain = gmshio.read_from_msh(mesh_file, comm, 0, gdim=2).mesh

    V = fem.functionspace(domain, ("Lagrange", degree_u, (3,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))

    n_u = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    n_p = Q.dofmap.index_map.size_global
    h_min, h_max, dx_eff, h_cells = measure_mesh_spacing(domain, degree_u)
    n_cells = domain.topology.index_map(domain.topology.dim).size_global

    dx_min_override = dx_min
    if dx_min is None:
        dx_min = dx_eff
    _log(comm, f"    cells   : {n_cells}   velocity DOFs: {n_u}   pressure DOFs: {n_p}")
    _log(comm, f"    h_min={h_min:.4e}  h_max={h_max:.4e}  dx_min(CFL)={dx_min:.4e}")

    u_n = fem.Function(V, name="Velocity")
    u_s = fem.Function(V, name="TentativeVelocity")
    p_n = fem.Function(Q, name="Pressure")
    phi = fem.Function(Q, name="PressureIncrement")

    # ---------------- boundary conditions ----------------
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = dmesh.exterior_facet_indices(domain.topology)

    def is_axis(x):
        return np.isclose(x[0], 0.0)

    axis_facets = dmesh.locate_entities_boundary(domain, 1, is_axis)
    wall_facets = np.setdiff1d(boundary_facets, axis_facets)

    n_axis = comm.allreduce(len(axis_facets), op=MPI.SUM)
    n_wall = comm.allreduce(len(wall_facets), op=MPI.SUM)
    _log(comm, f"    boundary: {n_axis} axis facets, {n_wall} wall facets")
    if n_axis == 0 or n_wall == 0:
        raise RuntimeError(
            f"Boundary classification failed (axis={n_axis}, wall={n_wall}); "
            "the mesh does not look like the expected half-plane domain."
        )

    wall_dofs_V = fem.locate_dofs_topological(V, 1, wall_facets)
    u_bc_wall = fem.Function(V)
    bc_wall = fem.dirichletbc(u_bc_wall, wall_dofs_V)

    V_r, _ = V.sub(0).collapse()
    V_t, _ = V.sub(2).collapse()
    axis_dofs_r = fem.locate_dofs_topological((V.sub(0), V_r), 1, axis_facets)
    axis_dofs_t = fem.locate_dofs_topological((V.sub(2), V_t), 1, axis_facets)
    zero_r = fem.Function(V_r)
    zero_t = fem.Function(V_t)
    bc_axis_r = fem.dirichletbc(zero_r, axis_dofs_r, V.sub(0))
    bc_axis_t = fem.dirichletbc(zero_t, axis_dofs_t, V.sub(2))
    bcs_u = [bc_wall, bc_axis_r, bc_axis_t]

    # Unrolled indices into u.x.array for the BC residual check.  DOLFINx returns
    # *block* indices from locate_dofs_topological(V, ...) on a blocked space but
    # *already unrolled* indices from the (V.sub(i), Vi) form; mixing the two
    # conventions silently reads the wrong entries.
    _bs = V.dofmap.index_map_bs
    bc_dof_sets = [
        np.concatenate([wall_dofs_V * _bs + c for c in range(_bs)]) if len(wall_dofs_V) else
        np.empty(0, dtype=np.int64),
        axis_dofs_r[0],
        axis_dofs_t[0],
    ]

    # Finding B6: no Dirichlet constraint on the pressure.  The increment problem is
    # pure Neumann and its constant null space is handed to PETSc explicitly.
    bcs_p = []

    # ---------------- variational forms ----------------
    dt_val = T / max(num_steps, 1)
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    nu_c = fem.Constant(domain, PETSc.ScalarType(nu))

    x = ufl.SpatialCoordinate(domain)
    r = x[0]
    md = {"quadrature_degree": quadrature_degree}
    dx = ufl.Measure("dx", domain=domain, metadata=md)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    ur, uz, ut = u[0], u[1], u[2]
    vr, vz, vt = v[0], v[1], v[2]
    un_r, un_z, un_t = u_n[0], u_n[1], u_n[2]

    # -- step 1: tentative velocity ------------------------------------
    # (u* - u^n)/dt + (u^n . grad) u* + swirl(u^n, u*) = -grad p^n + nu lap u*
    F1 = (ufl.dot(u - u_n, v) / dt) * r * dx
    adv_r = un_r * ur.dx(0) + un_z * ur.dx(1)
    adv_z = un_r * uz.dx(0) + un_z * uz.dx(1)
    adv_t = un_r * ut.dx(0) + un_z * ut.dx(1)
    F1 += (adv_r * vr + adv_z * vz + adv_t * vt) * r * dx
    # Centrifugal (-u_theta^2/r) and Coriolis (+u_r u_theta/r) terms, linearised
    # semi-implicitly.  The r of the measure cancels the 1/r analytically, so these
    # are written without any division at all.
    F1 += (-un_t * ut * vr + un_r * ut * vt) * dx
    # Vector Laplacian in cylindrical coordinates: the extra u_r/r^2 and
    # u_theta/r^2 terms integrate against r dx to give a single 1/r weight.
    F1 += nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * r * dx
    F1 += nu_c * (ur * vr + ut * vt) / r * dx
    if scheme == "ipcs":
        # Finding B3: the incremental pressure term R2 omitted.
        F1 += (p_n.dx(0) * vr + p_n.dx(1) * vz) * r * dx
    a1, L1 = ufl.lhs(F1), ufl.rhs(F1)

    # -- step 2: pressure increment ------------------------------------
    # laplace(phi) = div(u*)/dt.  Written with the 1/r of the divergence already
    # cancelled against the r of the measure, so no division appears.
    div_u_s_weighted = u_s[0].dx(0) * r + u_s[0] + u_s[1].dx(1) * r
    a2 = ufl.inner(ufl.grad(p), ufl.grad(q)) * r * dx
    L2 = -(1.0 / dt) * div_u_s_weighted * q * dx

    # -- step 3: projection --------------------------------------------
    a3 = ufl.dot(u, v) * r * dx
    L3 = ufl.dot(u_s, v) * r * dx - dt * (phi.dx(0) * vr + phi.dx(1) * vz) * r * dx

    form_a1, form_L1 = fem.form(a1), fem.form(L1)
    form_a2, form_L2 = fem.form(a2), fem.form(L2)
    form_a3, form_L3 = fem.form(a3), fem.form(L3)

    _log(comm, "    assembling operators...")
    A1 = assemble_matrix(form_a1, bcs=bcs_u)
    A1.assemble()
    A2 = assemble_matrix(form_a2, bcs=bcs_p)
    A2.assemble()
    A3 = assemble_matrix(form_a3, bcs=bcs_u)
    A3.assemble()

    # Finding B6: constant null space for the pure-Neumann pressure problem.
    nullspace = PETSc.NullSpace().create(constant=True, comm=comm)
    A2.setNullSpace(nullspace)
    A2.setNearNullSpace(nullspace)

    solver1, pc1 = _make_ksp(comm, A1, PETSc.KSP.Type.BCGS, pc_momentum,
                             1e-8, 1e-12, name="momentum")
    solver2, pc2 = _make_ksp(comm, A2, PETSc.KSP.Type.CG, pc_pressure,
                             1e-8, 1e-12, fallback_pc=PETSc.PC.Type.GAMG, name="pressure")
    # SOR, not block Jacobi.  The r-weighted mass matrix has diagonal entries down
    # to 7e-15 in the cells that touch the axis on the production grid; the ILU
    # inside BJACOBI breaks down there and PETSc returns DIVERGED_PC_FAILED with an
    # Inf solution.  SOR cannot break down and converges in fewer iterations than
    # point Jacobi (16 against 22 on the fine mesh).
    solver3, pc3 = _make_ksp(comm, A3, PETSc.KSP.Type.CG, pc_projection,
                             1e-9, 1e-14, name="projection")
    _log(comm, f"    preconditioners: momentum={pc1}, pressure={pc2}, projection={pc3}")

    b1 = create_vector(V)
    b2 = create_vector(Q)
    b3 = create_vector(V)

    diag = Diagnostics(domain, u_n, quadrature_degree=quadrature_degree)
    r_sampled_min = diag.audit()
    _log(comm, f"    min sampled radius = {r_sampled_min:.3e} (must be > 0; finding B1)")

    # ---------------- initial state ----------------
    t = 0.0
    step = 0
    if restart:
        with open(meta_file) as fh:
            meta = json.load(fh)
        if meta.get("mpi_size") != comm.size:
            raise ValueError(
                f"Checkpoint was written with {meta.get('mpi_size')} ranks but this run "
                f"uses {comm.size}. Per-rank checkpoints are partition dependent; "
                "restart with the same rank count."
            )
        if meta.get("mesh_fingerprint") not in (None, _file_fingerprint(mesh_file)):
            raise ValueError("Checkpoint was written for a different mesh file.")
        un_path = os.path.join(checkpoint_dir, tagged(f"checkpoint_un_rank{comm.rank}.npy", tag))
        pn_path = os.path.join(checkpoint_dir, tagged(f"checkpoint_pn_rank{comm.rank}.npy", tag))
        if not (os.path.exists(un_path) and os.path.exists(pn_path)):
            raise FileNotFoundError(f"Checkpoint state arrays not found in {checkpoint_dir}")
        arr_u = np.load(un_path)
        arr_p = np.load(pn_path)
        if arr_u.shape != u_n.x.array.shape:
            raise ValueError(
                f"Checkpoint DOF mismatch for velocity field u_n: loaded array has shape "
                f"{arr_u.shape}, but function space V has {u_n.x.array.shape[0]} DOFs on "
                f"rank {comm.rank}."
            )
        if arr_p.shape != p_n.x.array.shape:
            raise ValueError(
                f"Checkpoint DOF mismatch for pressure field p_n: loaded array has shape "
                f"{arr_p.shape}, but function space Q has {p_n.x.array.shape[0]} DOFs on "
                f"rank {comm.rank}."
            )
        u_n.x.array[:] = arr_u
        p_n.x.array[:] = arr_p
        u_n.x.scatter_forward()
        p_n.x.scatter_forward()
        t = float(meta["t"])
        step = int(meta["step"])
        _log(comm, f"    restarted from step {step} (t = {t:.6f})")
    else:
        _log(comm, "    building initial condition...")
        u_n.interpolate(ic.velocity_callable(taper=taper_ic))
        u_n.x.scatter_forward()
        set_bc(u_n.x.petsc_vec, bcs_u)
        u_n.x.scatter_forward()

    bc_res = boundary_condition_residual(u_n, bc_dof_sets, comm)
    _log(comm, f"    initial boundary-condition residual: {bc_res:.3e}")

    if not restart and project_ic:
        # Finding B12: remove the divergence the P2 interpolant of the analytic
        # streamfunction carries, so the first pressure solve is not a shock.
        before = diag.compute(include_location=False)
        u_s.x.array[:] = u_n.x.array
        u_s.x.scatter_forward()
        old_dt = dt.value
        dt.value = 1.0
        _solve_pressure(b2, form_L2, form_a2, bcs_p, nullspace, solver2, phi)
        _solve_projection(b3, form_L3, form_a3, bcs_u, solver3, u_n)
        dt.value = old_dt
        after = diag.compute(include_location=False)
        _log(comm,
             "    divergence-free projection of IC: "
             f"weak residual {before['div_u_weak']:.3e} -> {after['div_u_weak']:.3e}, "
             f"||div u||/||grad u|| {before['div_u_rel']:.3e} -> {after['div_u_rel']:.3e}")

    # ---------------- output setup ----------------
    sample_dt = sample_dt if sample_dt else T / 500.0
    xdmf_dt = xdmf_dt if xdmf_dt else T / 50.0
    checkpoint_dt = checkpoint_dt if checkpoint_dt else T / 20.0

    V_out = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u_out = fem.Function(V_out, name="Velocity")
    # Finding B8: append instead of truncating a restarted series.
    xdmf_mode = "a" if (restart and os.path.exists(xdmf_path)) else "w"
    xdmf = XDMFFile(comm, xdmf_path, xdmf_mode)
    if xdmf_mode == "w":
        xdmf.write_mesh(domain)

    if comm.rank == 0 and (not restart or not os.path.exists(csv_file)):
        with open(csv_file, "w") as fh:
            fh.write(",".join(CSV_COLUMNS) + "\n")

    run_meta = {
        "label": label or os.path.basename(os.path.normpath(out_dir)),
        "mesh_file": mesh_file,
        "mesh_fingerprint": _file_fingerprint(mesh_file),
        "n_cells": int(n_cells),
        "n_velocity_dofs": int(n_u),
        "n_pressure_dofs": int(n_p),
        "h_min": h_min, "h_max": h_max, "dx_min_cfl": float(dx_min),
        "min_sampled_radius": float(r_sampled_min),
        "scheme": scheme, "nu": nu, "T": T, "cfl": cfl,
        "adaptive_dt": bool(adaptive_dt), "dt_min": dt_min, "dt_max": dt_max,
        "max_energy_growth": max_energy_growth,
        "degree_u": degree_u, "degree_p": degree_p,
        "quadrature_degree": quadrature_degree,
        "sample_dt": sample_dt, "xdmf_dt": xdmf_dt, "checkpoint_dt": checkpoint_dt,
        "mpi_size": comm.size,
        "dolfinx_version": dolfinx.__version__,
        "petsc_version": ".".join(str(PETSc.Sys.getVersionInfo()[k])
                                  for k in ("major", "minor", "subminor")),
        "preconditioners": {"momentum": pc1, "pressure": pc2, "projection": pc3},
        "python": platform.python_version(),
        "git_commit": _git_commit(os.path.dirname(os.path.abspath(__file__))),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if comm.rank == 0:
        with open(os.path.join(out_dir, tagged("run_meta.json", tag)), "w") as fh:
            json.dump(run_meta, fh, indent=2)

    # ---------------- time loop ----------------
    _log(comm, f"--- integrating to T = {T} (adaptive_dt={adaptive_dt}) ---")
    next_sample = t
    next_xdmf = t
    next_ckpt = t + checkpoint_dt
    stop_reason, stop_detail = "completed", ""
    n_dt_limited = 0
    last = {}
    energy_min = np.inf      # running minimum of the kinetic energy
    t_energy_min = 0.0

    try:
        while t < T - 1e-15:
            # Cell-local CFL: min over cells of h_cell / (degree * max_cell|u|).
            # One pass gives both the stable step and the global peak velocity.
            dt_cfl_cell, max_vel = diag.cellwise_cfl_dt(h_cells, cfl, degree_u)

            if not np.isfinite(max_vel):
                raise SolverStop("non_finite_state", "velocity field contains NaN or Inf")
            if max_vel > max_velocity:
                raise SolverStop(
                    "velocity_limit_exceeded",
                    f"max|u| = {max_vel:.4e} exceeds --max_velocity = {max_velocity:.4e}",
                )

            # -------- time step selection (findings B4, B5) --------
            if adaptive_dt:
                if dx_min_override is not None:
                    # Explicit --dx_min: fall back to the global pairing.
                    dt_cfl = (cfl * dx_min / max_vel) if max_vel > 1e-12 else dt_max
                else:
                    dt_cfl = dt_cfl_cell if np.isfinite(dt_cfl_cell) else dt_max
                if dt_cfl < dt_min:
                    # R2 clamped here and kept going.  That silent clamp is the
                    # mechanism behind its reported "blow-up".
                    raise SolverStop(
                        "cfl_below_dt_min",
                        f"CFL requires dt = {dt_cfl:.3e} < --dt_min = {dt_min:.3e} "
                        f"at t = {t:.6f} with max|u| = {max_vel:.4e}. Refusing to "
                        "integrate outside the stability limit.",
                    )
                dt_new = min(dt_cfl, dt_max)
                if dt_new >= dt_max:
                    n_dt_limited += 1
            else:
                dt_new = dt_val

            if t + dt_new > T:
                dt_new = T - t
            dt.value = dt_new
            # CFL actually attained, measured against the cell-local stability limit.
            if np.isfinite(dt_cfl_cell) and dt_cfl_cell > 0.0:
                cfl_actual = cfl * dt_new / dt_cfl_cell
            else:
                cfl_actual = max_vel * dt_new / dx_min

            step_t0 = time.time()

            # -------- step 1: tentative velocity --------
            # Finding B13: reassembled every step; a1 carries u_n as a coefficient.
            A1.zeroEntries()
            assemble_matrix(A1, form_a1, bcs=bcs_u)
            A1.assemble()

            with b1.localForm() as loc:
                loc.set(0)
            assemble_vector(b1, form_L1)
            apply_lifting(b1, [form_a1], [bcs_u])
            b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            set_bc(b1, bcs_u)
            solver1.solve(b1, u_s.x.petsc_vec)
            it1 = check_ksp(solver1, "momentum", step, t)
            u_s.x.scatter_forward()

            # -------- step 2: pressure increment --------
            it2 = _solve_pressure(b2, form_L2, form_a2, bcs_p, nullspace, solver2,
                                  phi, step, t)

            # -------- step 3: projection (finding B2: with BCs) --------
            it3 = _solve_projection(b3, form_L3, form_a3, bcs_u, solver3, u_n, step, t)

            # -------- pressure update (finding B3) --------
            if scheme == "ipcs":
                p_n.x.array[:] += phi.x.array
            else:
                p_n.x.array[:] = phi.x.array
            p_n.x.scatter_forward()

            t += dt.value
            step += 1
            elapsed = time.time() - step_t0

            # -------- output on a physical-time cadence (finding B10) --------
            if t >= next_sample - 1e-15 or t >= T - 1e-15:
                res = diag.compute(t=t)
                res["bc_residual"] = boundary_condition_residual(u_n, bc_dof_sets, comm)
                row = {
                    "step": step, "t": t, "dt": float(dt.value), "cfl": cfl_actual,
                    "iters_momentum": it1, "iters_pressure": it2, "iters_projection": it3,
                    "wall_time": time.time() - t_wall0,
                    **res,
                }
                last = row

                # -------- energy guard --------
                # In a closed domain with no-slip walls and no body force the exact
                # solution obeys dE/dt = -2 nu * int |D(u)|^2 <= 0, strictly.  Any
                # growth in kinetic energy is therefore numerical, full stop -- and
                # unlike a velocity threshold it needs no arbitrary constant to
                # compare against.  It is also the guard that actually fires first:
                # the breakdown here comes from under-resolution, not from a CFL
                # violation, so the time step remains perfectly well behaved while
                # the solution stops meaning anything.
                # Evaluate the check before updating the running minimum: writing
                # it as an `elif` on the update would skip the test entirely while
                # the energy is still falling, so a threshold could never fire on a
                # decreasing series even when asked to.
                ke = res["kinetic_energy"]
                if (max_energy_growth is not None and np.isfinite(energy_min)
                        and energy_min > 0):
                    growth = ke / energy_min - 1.0
                    if growth > max_energy_growth:
                        raise SolverStop(
                            "energy_growth",
                            f"kinetic energy rose {growth * 100:.2f}% above its running "
                            f"minimum ({ke:.6g} against {energy_min:.6g} at "
                            f"t = {t_energy_min:.6f}) by t = {t:.6f}. Energy cannot "
                            f"increase in this configuration, so the solution has "
                            f"stopped being physical -- almost always because the grid "
                            f"can no longer resolve the flow. Refine, or treat "
                            f"t = {t_energy_min:.6f} as the reliable horizon.",
                        )
                if ke < energy_min:
                    energy_min, t_energy_min = ke, t
                if comm.rank == 0:
                    with open(csv_file, "a") as fh:
                        fh.write(",".join(f"{row.get(c, '')}" for c in CSV_COLUMNS) + "\n")
                while next_sample <= t:
                    next_sample += sample_dt

            if step % log_interval == 0 or step == 1:
                _log(
                    comm,
                    f"step {step:>7d} | t={t:.6f} | dt={dt.value:.3e} | CFL={cfl_actual:.3f} "
                    f"| max|u|={max_vel:9.3f} | its({it1},{it2},{it3}) | {elapsed:.3f}s/step",
                )

            if t >= next_xdmf - 1e-15 or t >= T - 1e-15:
                u_out.interpolate(u_n)
                xdmf.write_function(u_out, t)
                while next_xdmf <= t:
                    next_xdmf += xdmf_dt

            if t >= next_ckpt - 1e-15:
                _write_checkpoint(comm, checkpoint_dir, meta_file, u_n, p_n, step, t, run_meta, tag=tag)
                while next_ckpt <= t:
                    next_ckpt += checkpoint_dt

    except SolverStop as stop:
        stop_reason, stop_detail = stop.reason, stop.detail
        _log(comm, f"!! stopping: {stop_detail}")
    except FloatingPointError as exc:
        stop_reason, stop_detail = "non_finite_state", str(exc)
        _log(comm, f"!! stopping: {exc}")
    finally:
        xdmf.close()
        _write_checkpoint(comm, checkpoint_dir, meta_file, u_n, p_n, step, t, run_meta, tag=tag)

    status = {
        **run_meta,
        "terminated_reason": stop_reason,
        "terminated_detail": stop_detail,
        "final_t": t,
        "final_step": step,
        "reached_T": bool(t >= T - 1e-12),
        "steps_at_dt_max": n_dt_limited,
        "kinetic_energy_min": None if not np.isfinite(energy_min) else energy_min,
        "t_at_energy_min": t_energy_min,
        "wall_time_sec": time.time() - t_wall0,
        "final_diagnostics": {k: v for k, v in last.items() if k in CSV_COLUMNS},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if comm.rank == 0:
        with open(status_file, "w") as fh:
            json.dump(status, fh, indent=2)

    _log(comm, f"--- {stop_reason} at t = {t:.6f} after {step} steps "
               f"({status['wall_time_sec']:.1f} s) ---")
    return status


def _solve_pressure(b2, form_L2, form_a2, bcs_p, nullspace, solver2, phi,
                    step=0, t=0.0):
    with b2.localForm() as loc:
        loc.set(0)
    assemble_vector(b2, form_L2)
    if bcs_p:
        apply_lifting(b2, [form_a2], [bcs_p])
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    if bcs_p:
        set_bc(b2, bcs_p)
    # Pure Neumann: make the right-hand side compatible with the constant mode.
    nullspace.remove(b2)
    solver2.solve(b2, phi.x.petsc_vec)
    iters = check_ksp(solver2, "pressure", step, t)
    phi.x.scatter_forward()
    return iters


def _solve_projection(b3, form_L3, form_a3, bcs_u, solver3, u_n, step=0, t=0.0):
    """Finding B2: R2 solved this without boundary conditions."""
    with b3.localForm() as loc:
        loc.set(0)
    assemble_vector(b3, form_L3)
    apply_lifting(b3, [form_a3], [bcs_u])
    b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b3, bcs_u)
    solver3.solve(b3, u_n.x.petsc_vec)
    iters = check_ksp(solver3, "projection", step, t)
    u_n.x.scatter_forward()
    return iters


def _write_checkpoint(comm, checkpoint_dir, meta_file, u_n, p_n, step, t, run_meta,
                      tag=None):
    np.save(os.path.join(checkpoint_dir, tagged(f"checkpoint_un_rank{comm.rank}.npy", tag)), u_n.x.array)
    np.save(os.path.join(checkpoint_dir, tagged(f"checkpoint_pn_rank{comm.rank}.npy", tag)), p_n.x.array)
    if comm.rank == 0:
        with open(meta_file, "w") as fh:
            json.dump(
                {
                    "step": step,
                    "t": t,
                    "mpi_size": comm.size,
                    "mesh_fingerprint": run_meta["mesh_fingerprint"],
                    "n_velocity_dofs": run_meta["n_velocity_dofs"],
                },
                fh,
                indent=2,
            )


# =====================================================================
# CLI  (finding B9)
# =====================================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog="solver.py",
        description="Axisymmetric Navier-Stokes cusp solver (Run 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mesh", dest="mesh", default="assets/apple_domain.msh",
                   help="Gmsh .msh file describing the half-plane domain")
    p.add_argument("--out_dir", default="results", help="Directory for CSV, XDMF and status")
    p.add_argument("--checkpoint_dir", default=None,
                   help="Checkpoint directory (default: <out_dir>/checkpoints)")
    p.add_argument("--restart", action="store_true", help="Resume from a checkpoint")
    p.add_argument("-T", "--t_final", type=float, default=0.55, help="Final time")
    p.add_argument("--num_steps", type=int, default=1100,
                   help="Step count used for the fixed time step, and for the initial dt")
    p.add_argument("--adaptive_dt", dest="adaptive_dt", action="store_true", default=True,
                   help="Choose dt from the CFL condition")
    p.add_argument("--no_adaptive_dt", dest="adaptive_dt", action="store_false",
                   help="Use the fixed step T/num_steps")
    p.add_argument("--cfl", type=float, default=0.5, help="Target CFL number")
    p.add_argument("--dx_min", type=float, default=None,
                   help="Override the CFL length scale (default: measured from the mesh)")
    p.add_argument("--dt_min", type=float, default=1e-9,
                   help="Stop rather than integrate below this step (see finding B5)")
    p.add_argument("--dt_max", type=float, default=5e-3, help="Upper bound on dt")
    p.add_argument("--nu", type=float, default=1e-3, help="Kinematic viscosity")
    p.add_argument("--scheme", choices=["ipcs", "chorin"], default="ipcs",
                   help="ipcs = incremental pressure correction; chorin reproduces Run 2")
    p.add_argument("--log_interval", type=int, default=10, help="Steps between console lines")
    p.add_argument("--sample_dt", type=float, default=None,
                   help="Physical time between CSV rows (default: T/500)")
    p.add_argument("--xdmf_dt", type=float, default=None,
                   help="Physical time between field snapshots (default: T/50)")
    p.add_argument("--checkpoint_dt", type=float, default=None,
                   help="Physical time between checkpoints (default: T/20)")
    p.add_argument("--max_velocity", type=float, default=1e6,
                   help="Abort if max|u| exceeds this")
    p.add_argument("--max_energy_growth", type=float, default=0.01,
                   help="Abort when kinetic energy rises this fraction above its "
                        "running minimum. Energy cannot grow in a closed, unforced "
                        "domain, so this detects loss of resolution with no arbitrary "
                        "scale. Set to a large number to disable")
    p.add_argument("--quadrature_degree", type=int, default=6)
    p.add_argument("--pc_momentum", default="bjacobi",
                   help="PETSc preconditioner for the momentum solve")
    p.add_argument("--pc_pressure", default="hypre",
                   help="PETSc preconditioner for the pressure Poisson (falls back to gamg)")
    p.add_argument("--pc_projection", default="sor",
                   help="PETSc preconditioner for the projection solve; bjacobi/ilu "
                        "breaks down on the near-axis mass matrix")
    p.add_argument("--taper_ic", action="store_true",
                   help="Taper the vortex ring so the IC satisfies no-slip exactly")
    p.add_argument("--no_project_ic", dest="project_ic", action="store_false", default=True,
                   help="Skip the initial divergence-free projection")
    p.add_argument("--label", default=None, help="Name recorded in run_meta.json")
    p.add_argument("--tag", default=None,
                   help="Suffix every output filename, e.g. --tag fine writes "
                        "blowup_data_fine.csv. Convergence levels set this automatically.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_solver(
        mesh_file=args.mesh,
        out_dir=args.out_dir,
        checkpoint_dir=args.checkpoint_dir,
        restart=args.restart,
        T=args.t_final,
        num_steps=args.num_steps,
        adaptive_dt=args.adaptive_dt,
        cfl=args.cfl,
        dx_min=args.dx_min,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        nu=args.nu,
        scheme=args.scheme,
        log_interval=args.log_interval,
        sample_dt=args.sample_dt,
        xdmf_dt=args.xdmf_dt,
        checkpoint_dt=args.checkpoint_dt,
        max_velocity=args.max_velocity,
        max_energy_growth=args.max_energy_growth,
        quadrature_degree=args.quadrature_degree,
        pc_momentum=args.pc_momentum,
        pc_pressure=args.pc_pressure,
        pc_projection=args.pc_projection,
        taper_ic=args.taper_ic,
        project_ic=args.project_ic,
        label=args.label,
        tag=args.tag,
    )


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status["terminated_reason"] == "completed" else 1)
