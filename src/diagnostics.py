"""
diagnostics.py - Axis-safe flow diagnostics for the axisymmetric cusp study.

Why this module exists
----------------------
R2 measured vorticity as::

    r_safe  = r + 1e-14
    omega_z = u[2].dx(0) + u[2] / r_safe
    ...
    W_DG      = functionspace(domain, ("DG", 1))
    vort_expr = Expression(vorticity_sq, W_DG.element.interpolation_points)
    max_vort  = sqrt(max(vort_func.x.array))

The interpolation points of a DG1 element are the cell *vertices*, and every cell
touching the symmetry axis has vertices at exactly ``r = 0``.  There ``u_theta``
should be zero but is only zero to solver tolerance -- R2 additionally never
re-applied the boundary conditions after the projection step (finding B2) -- so the
quotient divides a round-off-level number by 1e-14 and amplifies it by 1e14.

The consequences are not subtle.  The very first row of ``blowup_data_R2.csv``
reports ``max_vorticity = 4.826e5`` where the exact analytic initial condition gives
``351.6``: a factor of 1373 before a single physical event.  At the published
"reliability threshold" t = 0.2095 s the velocity is still 20.3 (initial 25.2) while
the logged vorticity reads 3.3e8, on a mesh whose largest representable vorticity is
about u/h = 1.4e3.  The entire BKM diagnostic of Run 2 measured axis noise.

How this module avoids it
-------------------------
Everything is evaluated at **interior quadrature points**, where ``r > 0`` strictly
(a Gauss point of a triangle has all barycentric coordinates positive, so it can sit
on the axis only if all three vertices do, which is impossible for a non-degenerate
cell).  No epsilon is needed anywhere and none is used.  :func:`Diagnostics.audit`
asserts the property on the actual mesh rather than assuming it.

Integral quantities use the axisymmetric measure ``r dx``, against which the 1/r
terms cancel analytically.

Finding B11 is fixed at the same time: sampling at quadrature points of a
sufficiently high degree tracks the true cell maximum of a field derived from P2
velocities far better than nodal DG1 values, which systematically under-report the
peak the study is about.

Finding D5 is fixed by also reporting the quantities the theory section of the
project README talks about but the solver never computed: enstrophy, circulation
Gamma = r u_theta, kinetic energy, the divergence residual and the running BKM
integral.
"""

from __future__ import annotations

import numpy as np
import ufl
from mpi4py import MPI

import basix.ufl
from dolfinx import fem, la


# 2 pi from integrating the ignorable azimuthal coordinate: the half-plane integral
# with weight r times 2 pi is the true 3-D volume integral.
TWO_PI = 2.0 * np.pi


def vorticity_ufl(u, r):
    """Cylindrical vorticity components of an axisymmetric field ``u = (u_r, u_z, u_t)``.

        omega_r     = -d(u_t)/dz
        omega_z     =  d(u_t)/dr + u_t / r
        omega_theta =  d(u_r)/dz - d(u_z)/dr

    ``r`` must be the spatial coordinate itself, not a regularised ``r + eps``.
    Callers are responsible for only evaluating the result where ``r > 0``, which
    is what quadrature points guarantee.
    """
    omega_r = -u[2].dx(1)
    omega_z = u[2].dx(0) + u[2] / r
    omega_t = u[0].dx(1) - u[1].dx(0)
    return omega_r, omega_z, omega_t


def divergence_ufl(u, r):
    """Axisymmetric divergence  d(u_r)/dr + u_r/r + d(u_z)/dz."""
    return u[0].dx(0) + u[0] / r + u[1].dx(1)


class Diagnostics:
    """Compiled, reusable diagnostics for one velocity function.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    u : dolfinx.fem.Function
        Velocity in a vector Lagrange space, components ``(u_r, u_z, u_theta)``.
    quadrature_degree : int
        Degree of the quadrature element used for the L-infinity estimates.  The
        integrands are non-polynomial (they contain 1/r), so this controls sampling
        density rather than exactness; 4 gives 6 interior points per triangle.
    """

    def __init__(self, domain, u, quadrature_degree=4):
        self.domain = domain
        self.comm = domain.comm
        self.u = u
        self.quadrature_degree = quadrature_degree

        x = ufl.SpatialCoordinate(domain)
        r = x[0]
        self.dx = ufl.Measure("dx", domain=domain)

        w_r, w_z, w_t = vorticity_ufl(u, r)
        vort_sq = w_r**2 + w_z**2 + w_t**2
        vel_sq = u[0] ** 2 + u[1] ** 2 + u[2] ** 2
        gamma = r * u[2]
        div_u = divergence_ufl(u, r)

        # --- point-sampled quantities (L-infinity estimates) ----------------
        # A quadrature element interpolates *at the quadrature points*, which are
        # strictly interior to each cell.  This is the whole point: no vertex, so
        # no r = 0, so no division by an epsilon.
        q_el = basix.ufl.quadrature_element(
            domain.basix_cell(), scheme="default", degree=quadrature_degree
        )
        self.Q = fem.functionspace(domain, q_el)
        self._pts = self.Q.element.interpolation_points

        self._vort_fn = fem.Function(self.Q)
        self._vel_fn = fem.Function(self.Q)
        self._gamma_fn = fem.Function(self.Q)
        self._r_fn = fem.Function(self.Q)
        self._z_fn = fem.Function(self.Q)

        self._vort_expr = fem.Expression(vort_sq, self._pts)
        self._vel_expr = fem.Expression(vel_sq, self._pts)
        self._gamma_expr = fem.Expression(gamma, self._pts)
        self._r_fn.interpolate(fem.Expression(r, self._pts))
        self._z_fn.interpolate(fem.Expression(x[1], self._pts))

        # --- integral quantities -------------------------------------------
        md = {"quadrature_degree": max(quadrature_degree, 4)}
        dxr = self.dx(metadata=md)
        self._form_energy = fem.form(0.5 * vel_sq * r * dxr)
        self._form_enstrophy = fem.form(vort_sq * r * dxr)
        self._form_div_l2 = fem.form(div_u**2 * r * dxr)
        self._form_grad_l2 = fem.form(ufl.inner(ufl.grad(u), ufl.grad(u)) * r * dxr)
        self._form_volume = fem.form(fem.Constant(domain, 1.0) * r * dxr)

        # Weak (discrete) divergence residual.  A projection method makes
        # int(div u) q r dx vanish for every pressure test function q; it does NOT
        # make div u vanish pointwise, because the P2/P1 pair is only inf-sup
        # stable, not divergence free.  Reporting the strong norm alone would
        # suggest the projection had failed when it had not, so both are recorded.
        # The 1/r of the divergence is cancelled against the r of the measure.
        q_p = ufl.TestFunction(self._pressure_space(domain))
        div_weighted = u[0].dx(0) * r + u[0] + u[1].dx(1) * r
        self._form_div_weak = fem.form(div_weighted * q_p * dxr)

        self.volume = TWO_PI * self._assemble(self._form_volume)

        # BKM integral is accumulated by the caller through :meth:`accumulate_bkm`.
        self.bkm_integral = 0.0
        self._last_t = None
        self._last_max_vort = None

    # ------------------------------------------------------------------
    @staticmethod
    def _pressure_space(domain):
        """P1 space matching the solver pressure space, for the weak divergence."""
        return fem.functionspace(domain, ("Lagrange", 1))

    def _assemble(self, form):
        local = fem.assemble_scalar(form)
        return float(self.comm.allreduce(local, op=MPI.SUM))

    def cellwise_cfl_dt(self, h_cells, cfl, degree=2):
        """Largest stable step from a *cell-local* CFL condition.

        Returns ``(dt, max_velocity)`` in a single pass over the velocity field.

        A global ``cfl * min(h) / max|u|`` pairs the smallest cell in the mesh with
        the fastest fluid anywhere in the mesh, even when they sit at opposite ends
        of the domain.  On the graded meshes used here the smallest cell is 3.7x
        below the median, so that pairing throws away most of the step size for no
        stability benefit.  Taking the minimum of ``h_cell / max_cell|u|`` over
        cells is both the correct local condition and substantially cheaper.

        Parameters
        ----------
        h_cells : ndarray
            Cell diameters for the locally owned cells, in mesh cell order.
        degree : int
            Velocity polynomial degree; the node spacing is about ``h/degree``.
        """
        self._vel_fn.interpolate(self._vel_expr)
        vals = self._vel_fn.x.array

        n_cells = int(h_cells.size)
        n_qp = int(self._pts.shape[0])
        if n_cells and vals.size >= n_cells * n_qp:
            per_cell_sq = vals[: n_cells * n_qp].reshape(n_cells, n_qp).max(axis=1)
            per_cell = np.sqrt(np.maximum(per_cell_sq, 0.0))
            local_max = float(per_cell.max())
            spacing = h_cells / max(degree, 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dt_cells = cfl * spacing / per_cell
            dt_cells = dt_cells[np.isfinite(dt_cells)]
            local_dt = float(dt_cells.min()) if dt_cells.size else np.inf
        else:
            local_max, local_dt = 0.0, np.inf

        if not np.all(np.isfinite(vals)):
            return float("nan"), float("nan")

        dt = float(self.comm.allreduce(local_dt, op=MPI.MIN))
        vmax = float(self.comm.allreduce(local_max, op=MPI.MAX))
        return dt, vmax

    def max_velocity(self):
        """Cheap probe used every step for the CFL condition.

        ``compute`` evaluates six fields and four integrals; calling it once per
        step just to obtain max|u| dominated the step cost, so the CFL path only
        interpolates the velocity magnitude.
        """
        self._vel_fn.interpolate(self._vel_expr)
        vals = self._vel_fn.x.array
        if not np.all(np.isfinite(vals)):
            return float("nan")
        return float(np.sqrt(max(self._global_max(vals), 0.0)))

    def _global_max(self, values):
        local = float(np.max(values)) if values.size else -np.inf
        return float(self.comm.allreduce(local, op=MPI.MAX))

    # ------------------------------------------------------------------
    def audit(self):
        """Check the invariant this module relies on: no quadrature point on the axis.

        Returns the smallest sampled radius.  If it were zero the L-infinity
        estimates would be evaluating 0/0 and every number below would be suspect,
        so :meth:`compute` refuses to run when it is.
        """
        rvals = self._r_fn.x.array
        local = float(np.min(rvals)) if rvals.size else np.inf
        r_min = float(self.comm.allreduce(local, op=MPI.MIN))
        if not (r_min > 0.0):
            raise RuntimeError(
                "Diagnostic quadrature points include the symmetry axis "
                f"(min sampled r = {r_min!r}). The 1/r terms cannot be evaluated "
                "safely; this is exactly the failure mode of Run 2 (finding B1)."
            )
        return r_min

    # ------------------------------------------------------------------
    def compute(self, t=None, include_location=True):
        """Evaluate all diagnostics for the current state of ``u``.

        Returns a dict with keys ``max_velocity``, ``max_vorticity``,
        ``max_circulation``, ``enstrophy``, ``kinetic_energy``, ``div_u_l2`` and,
        when requested, the ``(r, z)`` position of the vorticity maximum.
        """
        self._vort_fn.interpolate(self._vort_expr)
        self._vel_fn.interpolate(self._vel_expr)
        self._gamma_fn.interpolate(self._gamma_expr)

        vort_sq = self._vort_fn.x.array
        vel_sq = self._vel_fn.x.array

        if not np.all(np.isfinite(vort_sq)) or not np.all(np.isfinite(vel_sq)):
            raise FloatingPointError(
                "Non-finite value in the sampled velocity/vorticity field"
            )

        max_vort = np.sqrt(max(self._global_max(vort_sq), 0.0))
        max_vel = np.sqrt(max(self._global_max(vel_sq), 0.0))
        max_gamma = self._global_max(np.abs(self._gamma_fn.x.array))

        grad_l2 = np.sqrt(max(TWO_PI * self._assemble(self._form_grad_l2), 0.0))
        div_l2 = np.sqrt(max(TWO_PI * self._assemble(self._form_div_l2), 0.0))

        # Weak residual: l2 norm of the vector int(div u) q_i r dx over the P1 basis.
        dvec = fem.assemble_vector(self._form_div_weak)
        dvec.scatter_reverse(la.InsertMode.add)
        owned = dvec.index_map.size_local * dvec.block_size
        local = float(np.sum(dvec.array[:owned] ** 2))
        div_weak = float(np.sqrt(self.comm.allreduce(local, op=MPI.SUM)))

        out = {
            "max_velocity": max_vel,
            "max_vorticity": max_vort,
            "max_circulation": max_gamma,
            "kinetic_energy": TWO_PI * self._assemble(self._form_energy),
            "enstrophy": TWO_PI * self._assemble(self._form_enstrophy),
            "div_u_l2": float(div_l2),
            # Dimensionless: ||div u|| relative to the size of the gradient it is
            # extracted from.  This is the number that says whether the flow is
            # incompressible to discretisation accuracy.
            "div_u_rel": float(div_l2 / grad_l2) if grad_l2 > 0 else 0.0,
            "div_u_weak": div_weak,
        }

        if include_location:
            out.update(self._argmax_location(vort_sq))

        if t is not None:
            out["bkm_integral"] = self.accumulate_bkm(t, max_vort)

        return out

    # ------------------------------------------------------------------
    def _argmax_location(self, vort_sq):
        """(r, z) of the global vorticity maximum, resolved across ranks."""
        if vort_sq.size:
            i = int(np.argmax(vort_sq))
            local = (float(vort_sq[i]), self.comm.rank)
            rz = (float(self._r_fn.x.array[i]), float(self._z_fn.x.array[i]))
        else:
            local = (-np.inf, self.comm.rank)
            rz = (np.nan, np.nan)

        _, owner = self.comm.allreduce(local, op=MPI.MAXLOC)
        r_at, z_at = self.comm.bcast(rz, root=owner)
        return {"r_at_max_vorticity": r_at, "z_at_max_vorticity": z_at}

    # ------------------------------------------------------------------
    def accumulate_bkm(self, t, max_vort):
        """Trapezoidal accumulation of the BKM integral, int ||omega||_inf dt.

        The Beale-Kato-Majda criterion says a singularity at T* forces this
        integral to diverge.  Reporting it directly is more informative than
        eyeballing 1/||omega||: a run that stays finite here has not blown up,
        whatever the vorticity curve looks like.
        """
        if self._last_t is not None and t > self._last_t:
            self.bkm_integral += 0.5 * (max_vort + self._last_max_vort) * (t - self._last_t)
        self._last_t = t
        self._last_max_vort = max_vort
        return self.bkm_integral


def boundary_condition_residual(u, unrolled_dof_sets, comm):
    """Largest boundary-condition violation in the current velocity field.

    R2 never re-applied the boundary conditions after its projection step
    (finding B2) and never checked, so the violation grew unobserved and fed the
    axis blow-up in the vorticity diagnostic.  The solver calls this every logging
    interval and records the result.

    Parameters
    ----------
    unrolled_dof_sets : sequence of int arrays
        Indices into ``u.x.array`` directly, i.e. already unrolled over the block
        size.  The two conventions in DOLFINx are easy to mix up:
        ``locate_dofs_topological(V, ...)`` on a blocked space returns *block*
        indices, while ``locate_dofs_topological((V.sub(i), Vi), ...)`` returns
        indices that are already unrolled.  Resolving that at the call site keeps
        this function unambiguous.
    """
    arr = u.x.array
    local = 0.0
    for idx in unrolled_dof_sets:
        idx = np.asarray(idx, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < arr.size)]
        if idx.size:
            local = max(local, float(np.max(np.abs(arr[idx]))))
    return float(comm.allreduce(local, op=MPI.MAX))
