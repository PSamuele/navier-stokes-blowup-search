"""
mesh.py - Axisymmetric "apple" domain mesh generator (Run 3).

Fixes, relative to ``src_R2/mesh_R2.py``:

A1  The R2 background field was ``MathEval`` with the string
    ``"0.0001 + 0.0149 * (1.0 - abs(z)/2.0)^3"``.  The 2-D geometry is built in
    the *xy* plane (points are added as ``addPoint(r, z, 0.0)``), so the gmsh
    coordinate ``z`` is identically zero everywhere and the field collapses to the
    constant ``0.0001 + 0.0149 = 0.015``.  Every mesh ever produced by R2 was
    uniform at h = 0.015; the advertised polar refinement never existed.  Here the
    size field is driven by ``Distance`` from the two pole points combined with a
    ``Threshold`` ramp, which does not depend on a coordinate name at all.

A2  R2 set ``Mesh.MeshSizeFromPoints = 0``, making the per-point ``lc`` values it
    carefully computed dead code.  We drive the size exclusively through the
    background field and say so.

A3  R2 sampled the boundary at 400 *uniformly spaced* z values, giving a chord
    length of ~0.01 near the pole while asking for h = 1e-4 there: the geometry
    was 100x coarser than the mesh.  Here the boundary sampling is equidistributed
    against the local target size, and ``num_points`` is derived from ``lc_pole``
    unless the caller overrides it.

A4  R2 used ``addSpline`` (Catmull-Rom), which can overshoot on sparse, steeply
    varying data and produce r < 0.  We use a dense polyline and assert
    non-negativity.

A5  R2 hard-coded ``MeshSizeMax = 0.015``, capping every mesh at the same size and
    making a resolution sweep impossible.  Both bounds now follow the arguments.

A6  ``generate_mesh`` gains ``output_file`` / ``verbosity`` / ``gui`` and returns an
    ``info`` dict -- the API that ``tests/test_mesh.py`` and
    ``scripts_R2/interpolate_mesh_R2.py`` already assumed existed.

The generator additionally *verifies* what gmsh actually produced and raises if the
achieved element size at the poles is far from the request.  That check is what
would have caught A1 on day one.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

try:
    import gmsh
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "gmsh is required to generate the mesh. Install with "
        "`conda install -c conda-forge python-gmsh`."
    ) from exc


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def boundary_radius(z, R0=1.0, H=2.0, k=0.5):
    """Domain boundary r = f(z) = R0 * cos(pi z / 2H) * exp(-k z^2).

    NOTE (see docs/findings.md, finding D1): this profile vanishes *linearly* at the
    poles.  df/dz at z = H equals -(pi R0 / 2H) exp(-k H^2) = -0.106 for the study
    parameters, i.e. the "cusp" is a cone of half-angle about 6 degrees.  The
    exp(-k z^2) factor is smooth and non-zero at z = H; it rescales the body but
    contributes nothing to the narrowing rate at the pole.  Kept unchanged from
    R1/R2 so that Run 3 stays directly comparable with the earlier runs.
    """
    z = np.asarray(z, dtype=float)
    return R0 * np.cos(np.pi * z / (2.0 * H)) * np.exp(-k * z**2)


def boundary_radius_prime(z, R0=1.0, H=2.0, k=0.5):
    """Analytic df/dz, used for arc-length grading of the boundary sampling."""
    z = np.asarray(z, dtype=float)
    c = np.cos(np.pi * z / (2.0 * H))
    s = np.sin(np.pi * z / (2.0 * H))
    e = np.exp(-k * z**2)
    return R0 * e * (-(np.pi / (2.0 * H)) * s - 2.0 * k * z * c)


def target_size(z, lc_pole, lc_boundary, H=2.0, power=2.0):
    """Target element size as a function of height.

    h(z) = lc_pole + (lc_boundary - lc_pole) * (1 - |z|/H)^power

    The shape R2 intended, evaluated correctly.  Used both to grade the boundary
    sampling and to configure the gmsh background field.
    """
    z = np.asarray(z, dtype=float)
    w = np.clip(1.0 - np.abs(z) / H, 0.0, 1.0)
    return lc_pole + (lc_boundary - lc_pole) * w**power


def _graded_boundary_points(lc_pole, lc_boundary, R0, H, k, power, num_points):
    """Sample z so that consecutive chords stay below the local target size.

    Builds a monotone map by integrating ds/h along the boundary curve and then
    inverting it on a uniform grid.  This is what makes the *geometry* as fine as
    the mesh near the poles (finding A3).
    """
    # Dense reference grid, clustered toward the poles via a tanh stretch so the
    # integrand is resolved where f(z) varies fastest.
    xi = np.linspace(-1.0, 1.0, 200001)
    stretch = 2.5
    z_ref = H * np.tanh(stretch * xi) / np.tanh(stretch)

    ds = np.hypot(1.0, boundary_radius_prime(z_ref, R0, H, k))       # |d(r,z)/dz|
    h = target_size(z_ref, lc_pole, lc_boundary, H, power)
    density = ds / h                                                  # points per unit z

    # Cumulative "number of points" coordinate; strictly increasing.
    s = np.concatenate(
        [[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(z_ref))]
    )
    total = float(s[-1])

    if num_points is None:
        num_points = int(np.clip(math.ceil(total), 200, 200000))

    # Invert s(z) on a uniform grid of the s coordinate.
    s_uniform = np.linspace(0.0, total, int(num_points))
    z_pts = np.interp(s_uniform, s, z_ref)
    z_pts[0], z_pts[-1] = -H, H
    z_pts = np.unique(z_pts)
    return z_pts, int(num_points), total


# --------------------------------------------------------------------------
# Mesh generation
# --------------------------------------------------------------------------

def generate_mesh(
    output_file="apple_domain.msh",
    lc_pole=0.002,
    lc_boundary=0.030,
    R0=1.0,
    H=2.0,
    k=0.5,
    num_points=None,
    power=2.0,
    verbosity=0,
    gui=False,
    check=True,
    check_tol=3.0,
):
    """Generate the axisymmetric half-plane mesh and return an ``info`` dict.

    Parameters
    ----------
    output_file : str
        Destination ``.msh`` path.  Parent directories are created.
    lc_pole, lc_boundary : float
        Target element size at the poles (z = +/-H) and at the equator (z = 0).
    R0, H, k : float
        Boundary profile parameters, see :func:`boundary_radius`.
    num_points : int or None
        Boundary sampling points.  ``None`` derives a value from ``lc_pole`` so the
        geometry is never coarser than the mesh.
    power : float
        Exponent of the size ramp between pole and equator.
    check : bool
        Verify the *achieved* element size near the poles against ``lc_pole`` and
        raise ``RuntimeError`` on gross deviation.  This is the regression guard
        for finding A1 -- leave it on.
    check_tol : float
        Allowed multiplicative deviation of the achieved size.

    Returns
    -------
    dict
        Requested parameters plus measured mesh statistics.
    """
    if lc_pole <= 0 or lc_boundary <= 0:
        raise ValueError("lc_pole and lc_boundary must be positive")
    if lc_pole > lc_boundary:
        raise ValueError(
            f"lc_pole ({lc_pole}) must not exceed lc_boundary ({lc_boundary}); "
            "the mesh refines toward the poles"
        )

    out_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(out_dir, exist_ok=True)

    z_pts, n_requested, sampling_budget = _graded_boundary_points(
        lc_pole, lc_boundary, R0, H, k, power, num_points
    )
    r_pts = boundary_radius(z_pts, R0, H, k)

    # A4: guard against a boundary description that leaves the physical half-plane.
    r_pts = np.maximum(r_pts, 0.0)
    if not np.all(np.isfinite(r_pts)):
        raise RuntimeError("Boundary radius evaluation produced non-finite values")

    was_initialized = gmsh.isInitialized()
    if not was_initialized:
        gmsh.initialize()
    else:
        gmsh.clear()

    try:
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.model.add("AppleDomainR3")

        geo = gmsh.model.geo

        # --- points -------------------------------------------------------
        # The characteristic length passed here is deliberately ignored
        # (MeshSizeFromPoints = 0); the background field is the single source of
        # truth.  Stated explicitly so the A2 confusion cannot recur.
        p_south = geo.addPoint(0.0, -H, 0.0, lc_pole)
        p_north = geo.addPoint(0.0, H, 0.0, lc_pole)

        interior = [
            geo.addPoint(float(r), float(z), 0.0, lc_pole)
            for r, z in zip(r_pts[1:-1], z_pts[1:-1])
        ]

        # --- curves -------------------------------------------------------
        axis_line = geo.addLine(p_south, p_north)

        # A4: dense polyline instead of a Catmull-Rom spline.  With the graded
        # sampling above, the chord error sits below the local element size by
        # construction, so a polyline is strictly safer than a spline here.
        wall_pts = [p_south] + interior + [p_north]
        wall_segments = [
            geo.addLine(wall_pts[i], wall_pts[i + 1]) for i in range(len(wall_pts) - 1)
        ]

        loop = geo.addCurveLoop([axis_line] + [-s for s in reversed(wall_segments)])
        surface = geo.addPlaneSurface([loop])

        geo.synchronize()

        gmsh.model.addPhysicalGroup(1, [axis_line], name="SymmetryAxis")
        gmsh.model.addPhysicalGroup(1, wall_segments, name="AppleWall")
        gmsh.model.addPhysicalGroup(2, [surface], name="FluidDomain")

        # --- size field ---------------------------------------------------
        # A1: coordinate-name-free formulation.  Distance from the two pole points
        # feeds a Threshold ramp, so there is no way for the field to silently
        # evaluate against the wrong axis the way the R2 MathEval string did.
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", [p_south, p_north])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_pole)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_boundary)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(2, "DistMax", H)
        gmsh.model.mesh.field.setNumber(2, "Sigmoid", 0)

        lo, hi = float(lc_pole), float(lc_boundary)
        if hi > lo:
            # Reshape the linear Threshold ramp into the intended power law.  The
            # expression is a function of the *field value* F2 only, so it stays
            # independent of coordinate naming.
            gmsh.model.mesh.field.add("MathEval", 3)
            gmsh.model.mesh.field.setString(
                3, "F", f"{lo} + {hi - lo} * ((F2 - {lo})/{hi - lo})^{power}"
            )
            gmsh.model.mesh.field.setAsBackgroundMesh(3)
        else:  # uniform mesh requested
            gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # A5: bounds follow the request instead of being pinned at 0.015.
        gmsh.option.setNumber("Mesh.MeshSizeMin", lo * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", hi * 1.5)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # Delaunay: robust on graded fields

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()

        gmsh.write(output_file)

        stats = _measure(H)

        if gui:  # pragma: no cover - interactive
            gmsh.fltk.run()
    finally:
        if not was_initialized:
            gmsh.finalize()

    info = {
        "output_file": output_file,
        "lc_pole": float(lc_pole),
        "lc_boundary": float(lc_boundary),
        "R0": float(R0),
        "H": float(H),
        "k": float(k),
        "power": float(power),
        "num_points": int(len(z_pts)),
        "num_points_requested": int(n_requested),
        "boundary_sampling_budget": float(sampling_budget),
        **stats,
    }

    if check:
        _verify(info, check_tol)

    return info


def _measure(H):
    """Measure what gmsh actually produced.  Never trust the request."""
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    elem_types, _, elem_nodes = gmsh.model.mesh.getElements(2)
    tris = None
    for etype, enodes in zip(elem_types, elem_nodes):
        if etype == 2:  # 3-node triangle
            tris = np.array(enodes, dtype=np.int64).reshape(-1, 3)
    if tris is None or len(tris) == 0:
        raise RuntimeError("Mesh generation produced no 2-D triangles")

    idx = np.vectorize(tag_to_idx.__getitem__)(tris)
    P = coords[idx]                                   # (ncell, 3, 2)
    edges = np.stack(
        [
            np.linalg.norm(P[:, 1] - P[:, 0], axis=1),
            np.linalg.norm(P[:, 2] - P[:, 1], axis=1),
            np.linalg.norm(P[:, 0] - P[:, 2], axis=1),
        ],
        axis=1,
    )
    h_cell = edges.max(axis=1)
    z_cell = P[:, :, 1].mean(axis=1)

    pole_mask = np.abs(z_cell) > 0.95 * H
    eq_mask = np.abs(z_cell) < 0.05 * H

    return {
        "num_nodes": int(len(coords)),
        "num_elements_2d": int(len(tris)),
        "h_min_actual": float(edges.min()),
        "h_max_actual": float(h_cell.max()),
        "h_median_actual": float(np.median(h_cell)),
        "h_pole_actual": float(np.median(h_cell[pole_mask])) if pole_mask.any() else float("nan"),
        "h_equator_actual": float(np.median(h_cell[eq_mask])) if eq_mask.any() else float("nan"),
        "n_cells_pole_region": int(pole_mask.sum()),
        "r_min": float(coords[:, 0].min()),
        "r_max": float(coords[:, 0].max()),
        "z_min": float(coords[:, 1].min()),
        "z_max": float(coords[:, 1].max()),
    }


def _verify(info, tol):
    """Fail loudly when the achieved resolution does not match the request.

    This is the regression guard for finding A1.  Run against the R2 recipe it
    fires immediately: R2 asked for 1e-4 at the poles and delivered 0.015.
    """
    problems = []

    if info["r_min"] < -1e-12:
        problems.append(f"boundary left the half-plane: r_min = {info['r_min']:.3e} < 0")

    h_pole = info["h_pole_actual"]
    if not np.isfinite(h_pole):
        problems.append("no cells found in the polar region")
    else:
        ratio = h_pole / info["lc_pole"]
        if ratio > tol:
            problems.append(
                f"polar element size is {ratio:.1f}x the request "
                f"(asked {info['lc_pole']:.3e}, achieved {h_pole:.3e}). "
                "The size field is not reaching the poles."
            )

    h_eq = info["h_equator_actual"]
    if np.isfinite(h_eq) and h_eq / info["lc_boundary"] > tol:
        problems.append(
            f"equatorial element size is {h_eq / info['lc_boundary']:.1f}x the request "
            f"(asked {info['lc_boundary']:.3e}, achieved {h_eq:.3e})"
        )

    if problems:
        raise RuntimeError(
            f"Mesh verification failed for {info['output_file']}:\n  - "
            + "\n  - ".join(problems)
        )


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Generate the Run 3 apple-domain mesh.")
    parser.add_argument("--output_file", "-o", default="apple_domain.msh")
    parser.add_argument("--lc_pole", type=float, default=0.002)
    parser.add_argument("--lc_boundary", type=float, default=0.030)
    parser.add_argument("--R0", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=2.0)
    parser.add_argument("--k", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--no_check", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args(argv)

    info = generate_mesh(
        output_file=args.output_file,
        lc_pole=args.lc_pole,
        lc_boundary=args.lc_boundary,
        R0=args.R0,
        H=args.H,
        k=args.k,
        num_points=args.num_points,
        power=args.power,
        verbosity=args.verbosity,
        gui=args.gui,
        check=not args.no_check,
    )
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    main()
