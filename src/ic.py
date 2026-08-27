"""
ic.py - Initial condition for the axisymmetric cusp study (Run 3).

The analytic fields are unchanged from R1/R2 so the runs stay comparable:

    psi_jet  = J r^2 (f(z)^2 - r^2)^2         (streamfunction, background jet)
    psi_ring = A r^2 exp(-((r-Rv)^2 + z^2)/sigma^2)
    Gamma_0  = S r^2 exp(-((r-Rv)^2 + z^2)/sigma^2)

    u_r = -(1/r) d(psi)/dz,  u_z = (1/r) d(psi)/dr,  u_theta = Gamma_0 / r

Two things are handled better than in R2:

* R2 evaluated ``1/sp.Max(r, 1e-12)`` and then clamped ``r`` to 1e-12 in the numpy
  callback.  Every one of these fields is actually *regular* on the axis
  (``psi`` and ``Gamma_0`` both carry an explicit ``r^2``), so the division can be
  cancelled symbolically instead of being regularised numerically.  We do that with
  ``sympy.cancel`` on ``psi/r`` style expressions, which removes the last place
  where a 1/r could contaminate the axis.

* Finding B12: the Gaussian ring does not vanish on the wall, so the raw
  interpolant violates no-slip by ~2e-3 of the peak.  R2 ignored this.  We measure
  the residual and expose an optional taper that multiplies the ring by a factor
  vanishing at r = f(z), so the initial state is compatible with the boundary
  conditions when the caller asks for it.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

# Study parameters, identical to R1/R2.
DEFAULTS = dict(R0=1.0, H=2.0, k=0.5, J=10.0, A=5.0, S=20.0, Rv=0.5, sigma=0.2)


def symbolic_fields(R0=1.0, H=2.0, k=0.5, J=10.0, A=5.0, S=20.0, Rv=0.5,
                    sigma=0.2, taper=False):
    """Return ``(u_r, u_z, u_theta, Gamma_0, psi, (r, z))`` as sympy expressions.

    All three velocity components are returned in axis-regular form: the explicit
    ``r^2`` prefactors are cancelled against the ``1/r`` of the streamfunction
    relations, so no expression contains a removable singularity at r = 0.
    """
    r, z = sp.symbols("r z", real=True)

    f_z = R0 * sp.cos(sp.pi * z / (2 * H)) * sp.exp(-k * z**2)
    bump = sp.exp(-((r - Rv) ** 2 + z**2) / sigma**2)

    if taper:
        # Vanishes quadratically at the wall r = f(z), keeping the ring smooth and
        # making the initial state compatible with no-slip (finding B12).
        wall = (1 - (r / f_z) ** 2) ** 2
        bump = bump * wall

    psi_jet = J * r**2 * (f_z**2 - r**2) ** 2
    psi_ring = A * r**2 * bump
    psi_tot = psi_jet + psi_ring
    Gamma_0 = S * r**2 * bump

    # u_r = -(1/r) dpsi/dz and u_z = (1/r) dpsi/dr.  Both psi terms carry r^2, so
    # dividing by r leaves a polynomial in r: cancel it symbolically rather than
    # guarding with Max(r, 1e-12) the way R2 did.
    u_r = sp.cancel(sp.expand(-sp.diff(psi_tot, z) / r))
    u_z = sp.cancel(sp.expand(sp.diff(psi_tot, r) / r))
    u_theta = sp.cancel(sp.expand(Gamma_0 / r))

    return u_r, u_z, u_theta, Gamma_0, psi_tot, (r, z)


def velocity_callable(taper=False, **params):
    """Return ``fn(x) -> (3, N)`` suitable for ``dolfinx.fem.Function.interpolate``."""
    p = {**DEFAULTS, **params}
    u_r, u_z, u_theta, _, _, (r, z) = symbolic_fields(taper=taper, **p)

    ur_num = sp.lambdify((r, z), u_r, modules=["numpy"])
    uz_num = sp.lambdify((r, z), u_z, modules=["numpy"])
    ut_num = sp.lambdify((r, z), u_theta, modules=["numpy"])

    def initial_velocity(x):
        rr = np.asarray(x[0], dtype=float)
        zz = np.asarray(x[1], dtype=float)
        out = np.zeros((3, rr.size), dtype=np.float64)
        out[0] = np.broadcast_to(np.asarray(ur_num(rr, zz), dtype=float), rr.shape)
        out[1] = np.broadcast_to(np.asarray(uz_num(rr, zz), dtype=float), rr.shape)
        out[2] = np.broadcast_to(np.asarray(ut_num(rr, zz), dtype=float), rr.shape)
        # The axis conditions u_r = u_theta = 0 hold analytically; enforce them
        # exactly at r = 0 so round-off cannot leak into the axis DOFs.
        on_axis = rr == 0.0
        out[0, on_axis] = 0.0
        out[2, on_axis] = 0.0
        return out

    return initial_velocity


def vorticity_callable(taper=False, **params):
    """Return ``fn(r, z) -> (omega_r, omega_z, omega_theta)`` evaluated analytically.

    Used by the diagnostics regression test: the discrete ``||omega||_inf`` must
    approach this field, not the 1e14-amplified axis noise R2 reported.

    In cylindrical coordinates with no theta dependence:
        omega_r     = -d(u_theta)/dz
        omega_z     =  d(u_theta)/dr + u_theta / r   =  (1/r) d(r u_theta)/dr
        omega_theta =  d(u_r)/dz - d(u_z)/dr

    ``u_theta`` carries a factor r, so ``omega_z`` is written as
    ``d(Gamma)/dr / r`` with the r cancelled symbolically -- regular on the axis.
    """
    p = {**DEFAULTS, **params}
    u_r, u_z, u_theta, Gamma_0, _, (r, z) = symbolic_fields(taper=taper, **p)

    w_r = -sp.diff(u_theta, z)
    w_z = sp.cancel(sp.expand(sp.diff(Gamma_0, r) / r))
    w_t = sp.diff(u_r, z) - sp.diff(u_z, r)

    fr = sp.lambdify((r, z), w_r, modules=["numpy"])
    fz = sp.lambdify((r, z), w_z, modules=["numpy"])
    ft = sp.lambdify((r, z), w_t, modules=["numpy"])

    def omega(rr, zz):
        rr = np.asarray(rr, dtype=float)
        zz = np.asarray(zz, dtype=float)
        return (
            np.broadcast_to(np.asarray(fr(rr, zz), dtype=float), rr.shape),
            np.broadcast_to(np.asarray(fz(rr, zz), dtype=float), rr.shape),
            np.broadcast_to(np.asarray(ft(rr, zz), dtype=float), rr.shape),
        )

    return omega


def reference_extrema(n=1200, taper=False, **params):
    """Sample the exact initial fields inside the domain and return their extrema.

    This is the ground truth the discrete diagnostics are checked against.  With
    the study defaults it gives ``max|u| = 25.211`` and ``max|omega| = 351.60``;
    R2 logged ``482601`` for the latter at its first sample (finding B1).
    """
    p = {**DEFAULTS, **params}
    R0, H, k = p["R0"], p["H"], p["k"]

    rr = np.linspace(0.0, R0, n)
    zz = np.linspace(-H, H, n)
    Rg, Zg = np.meshgrid(rr, zz)
    inside = Rg <= R0 * np.cos(np.pi * Zg / (2 * H)) * np.exp(-k * Zg**2)

    vel = velocity_callable(taper=taper, **p)
    u = vel(np.stack([Rg.ravel(), Zg.ravel()]))
    u = u.reshape(3, *Rg.shape)
    u[:, ~inside] = 0.0

    om = vorticity_callable(taper=taper, **p)(Rg, Zg)
    om = np.stack(om)
    om[:, ~inside] = 0.0

    return {
        "max_velocity": float(np.sqrt((u**2).sum(axis=0)).max()),
        "max_vorticity": float(np.sqrt((om**2).sum(axis=0)).max()),
        "max_omega_r": float(np.abs(om[0]).max()),
        "max_omega_z": float(np.abs(om[1]).max()),
        "max_omega_theta": float(np.abs(om[2]).max()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(reference_extrema(), indent=2))
