#!/usr/bin/env python3
"""
make_figures.py - Build every README figure that comes from data.

Every static figure in ``media/`` is produced here, from the recorded diagnostics
and the delivered meshes, so none of them is a one-off export that can drift out
of step with the data: run this script and the whole set is rebuilt.

Animations are not produced here.  They are ParaView exports that live beside the
run that generated them -- ``archive/run_0*/`` for the first two attempts and
``results/convergence_aws/<level>/`` for the production study -- and the
documents reference them there rather than through a copy.

Only numpy and matplotlib are used, so this runs outside the FEniCSx
environment -- no dolfinx, no gmsh, no HDF5.

Usage
-----
    python scripts/make_figures.py
    python scripts/make_figures.py --study results/convergence_aws --out media
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Geometry of the domain, identical to the one the study ran on.
R0, H, K = 1.0, 2.0, 0.5


def find_one(directory, pattern, what):
    """Locate a per-level output whose name may carry a tag suffix.

    A convergence level writes blowup_data_<level>.csv; a standalone run writes
    blowup_data.csv. Both are accepted so recorded studies stay readable.
    """
    hits = sorted(glob.glob(os.path.join(directory, pattern)))
    if len(hits) > 1:
        raise SystemExit(f"ambiguous {what} in {directory}: {[os.path.basename(h) for h in hits]}")
    return hits[0] if hits else None


LEVELS = ("coarse", "medium", "fine")
COLOUR = {"coarse": "#c1121f", "medium": "#e08a00", "fine": "#0353a4"}

# The five Richardson/GCI plots analyze_convergence.py already writes.
GCI_PLOTS = {
    "convergence_kinetic_energy.png": "gci-kinetic-energy.png",
    "convergence_max_vorticity.png": "gci-max-vorticity.png",
    "convergence_enstrophy.png": "gci-enstrophy.png",
    "convergence_max_circulation.png": "gci-max-circulation.png",
    "convergence_bkm_integral.png": "gci-bkm-integral.png",
}


def boundary_radius(z):
    """Wall radius f(z). Kept identical to src/mesh.py."""
    return R0 * np.cos(np.pi * z / (2 * H)) * np.exp(-K * z * z)


def read_msh(path):
    """Minimal reader for gmsh 4.1 ASCII: returns (points Nx2, triangles Mx3).

    Enough for drawing. meshio would do this too, but the analysis side of this
    repository deliberately depends on nothing beyond numpy and matplotlib, and
    the format is four lines of bookkeeping.
    """
    with open(path) as fh:
        lines = fh.read().splitlines()

    tag_to_row, coords, tris = {}, [], []
    i = 0
    while i < len(lines):
        section = lines[i].strip()

        if section == "$Nodes":
            n_blocks = int(lines[i + 1].split()[0])
            i += 2
            for _ in range(n_blocks):
                n_in_block = int(lines[i].split()[3])
                i += 1
                tags = [int(lines[i + j]) for j in range(n_in_block)]
                i += n_in_block
                for j, tag in enumerate(tags):
                    x, y, _z = map(float, lines[i + j].split()[:3])
                    tag_to_row[tag] = len(coords)
                    coords.append((x, y))
                i += n_in_block

        elif section == "$Elements":
            n_blocks = int(lines[i + 1].split()[0])
            i += 2
            for _ in range(n_blocks):
                _dim, _tag, etype, n_in_block = map(int, lines[i].split())
                i += 1
                if etype == 2:                       # 3-node triangle
                    for j in range(n_in_block):
                        a, b, c = map(int, lines[i + j].split()[1:4])
                        tris.append((tag_to_row[a], tag_to_row[b], tag_to_row[c]))
                i += n_in_block

        else:
            i += 1

    return np.asarray(coords, dtype=float), np.asarray(tris, dtype=int)


def style():
    """A single look for every figure: white ground, readable on either theme."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 150,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def load(study):
    data = {}
    for name in LEVELS:
        csv = find_one(os.path.join(study, name), "blowup_data*.csv", "diagnostics CSV")
        if csv:
            data[name] = read_csv(csv)
    if not data:
        raise SystemExit(f"no level data found under {study}")
    return data


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  written {path}")
    return path


# --------------------------------------------------------------------------
# 1. The domain is a cone, not a cusp
# --------------------------------------------------------------------------

def fig_cone_not_cusp(out_dir):
    """The visual counterpart to the alpha table.

    Zooming in on the pole never reveals a cusp: the local exponent in
    r ~ c (H - z)^alpha tends to 1.0000, which is a straight cone of constant
    half-angle. Plotting f(z) at three magnifications makes the point that a
    single render cannot -- both a cone and a cusp look sharp to the eye.
    """
    # Slope of the exact tangent cone at the pole, f'(H).
    slope = (np.pi * R0 / (2 * H)) * np.exp(-K * H * H)
    half_angle = np.degrees(np.arctan(slope))

    def local_alpha(d, rel=1e-3):
        """d ln f / d ln d at distance d from the pole."""
        lo, hi = d * (1 - rel), d * (1 + rel)
        return (np.log(boundary_radius(H - hi) / boundary_radius(H - lo))
                / np.log(hi / lo))

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, span in zip(axes, (1.0, 1e-2, 1e-4)):
        d = np.linspace(0, span, 500)
        r = boundary_radius(H - d)
        ax.fill_betweenx(d, -r, r, color="#0353a4", alpha=0.10)
        ax.plot(r, d, color="#0353a4", lw=2.2, label="domain boundary $f(z)$", zorder=3)
        ax.plot(-r, d, color="#0353a4", lw=2.2, zorder=3)
        ax.plot(slope * d, d, color="#c1121f", lw=1.4, ls="--",
                label=f"exact cone, {half_angle:.2f}$\\degree$", zorder=4)
        ax.plot(-slope * d, d, color="#c1121f", lw=1.4, ls="--", zorder=4)

        # Identical limits in units of the window: a perfect cone draws the
        # identical triangle in all three panels, so any curvature stands out.
        ax.set_xlim(-2.2 * slope * span, 2.2 * slope * span)
        ax.set_ylim(span, 0)
        ax.set_xlabel("r")
        ax.set_title(f"zoom: $H-z \\leq$ {span:g}")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
        ax.text(0.5, 0.93, f"local $\\alpha$ = {local_alpha(span):.4f}",
                transform=ax.transAxes, ha="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3cd",
                          edgecolor="#e0c97f"))
    axes[0].set_ylabel("$H-z$   (distance from the pole)")
    axes[0].legend(loc="lower center", fontsize=8.5)

    fig.suptitle(f"Zooming in never reveals a cusp: the profile collapses onto a "
                 f"{half_angle:.2f}$\\degree$ cone", fontweight="bold", y=1.02)
    fig.text(0.5, -0.07,
             "A cusp requires $r \\sim c\\,(H-z)^{\\alpha}$ with $\\alpha > 1$ so the "
             "opening angle closes at the tip.  Here $\\alpha \\to 1.0000$: the half-angle "
             f"is constant at {half_angle:.2f}$\\degree$.\nBoth look sharp at ordinary "
             "render zoom, which is why this is settled numerically rather than visually.",
             ha="center", fontsize=9)
    fig.tight_layout()
    return save(fig, out_dir, "cone-not-cusp.png")


# --------------------------------------------------------------------------
# 2. Where the vorticity maximum actually lives
# --------------------------------------------------------------------------

def fig_vorticity_location(data, out_dir):
    """The refinement went to the poles; the extreme vorticity never did.

    Every recorded sample's argmax of |omega| is plotted over the domain
    outline. They sit on the wall at mid-latitude, not at the engineered
    polar feature the mesh was graded for.
    """
    d = data.get("fine", data[max(data, key=lambda k: len(data[k]))])
    r, z, t = d["r_at_max_vorticity"], d["z_at_max_vorticity"], d["t"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5.2),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    zz = np.linspace(-H, H, 900)
    rr = boundary_radius(zz)
    ax.plot(rr, zz, color="#333333", lw=1.6)
    ax.plot(-rr, zz, color="#333333", lw=1.6)
    ax.fill_betweenx(zz, -rr, rr, color="#eeeeee")
    sc = ax.scatter(r, z, c=t, cmap="viridis", s=16, zorder=3,
                    edgecolors="none", alpha=0.85)
    ax.scatter([0, 0], [H, -H], marker="v", s=90, color="#c1121f", zorder=4)
    ax.annotate("the poles:\nwhere the mesh\nwas refined",
                xy=(0, H), xytext=(0.42, 1.42), fontsize=9, color="#c1121f",
                ha="center", arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.2))
    fig.colorbar(sc, ax=ax, label="t", fraction=0.046, pad=0.04)
    ax.set_xlabel("r"); ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Location of max $|\\omega|$, every recorded sample")

    on_wall = np.abs(r / np.maximum(boundary_radius(z), 1e-12) - 1) < 0.05
    near_axis = r < 0.05
    near_pole = np.abs(z) > 1.8
    bars = [("on the wall\n$r/f(z) > 0.95$", on_wall.mean(), "#0353a4"),
            ("near the axis\n$r < 0.05$", near_axis.mean(), "#7a7a7a"),
            ("near the poles\n$|z| > 1.8$", near_pole.mean(), "#c1121f")]
    bx.bar([b[0] for b in bars], [100 * b[1] for b in bars],
           color=[b[2] for b in bars], width=0.6)
    for i, b in enumerate(bars):
        bx.text(i, 100 * b[1] + 2.5, f"{100 * b[1]:.1f}%", ha="center",
                fontweight="bold", fontsize=11)
    bx.set_ylim(0, 105); bx.set_ylabel("share of samples (%)")
    bx.set_title("The mesh was graded 15:1 toward the poles.\n"
                 "The vorticity maximum is never there.", fontsize=10)
    fig.tight_layout()
    return save(fig, out_dir, "vorticity-location.png")


# --------------------------------------------------------------------------
# 3. Kinetic energy: the null result
# --------------------------------------------------------------------------

def fig_energy_decay(data, study, out_dir):
    """dE/dt <= 0 is exact here, so any rise is numerical by construction."""
    import json

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in LEVELS:
        if name not in data:
            continue
        d = data[name]
        ax.plot(d["t"], d["kinetic_energy"], color=COLOUR[name], lw=1.8, label=name)
        status = find_one(os.path.join(study, name), "status*.json", "status file")
        if status:
            with open(status) as fh:
                st = json.load(fh)
            if st.get("terminated_reason") == "energy_growth":
                tg, eg = st["t_at_energy_min"], st["kinetic_energy_min"]
                ax.scatter([tg], [eg], marker="X", s=110, color=COLOUR[name],
                           zorder=5, edgecolors="white", linewidths=1.2)
                ax.annotate(f"energy guard fires\nreliable to t = {tg:.4f}",
                            xy=(tg, eg), xytext=(0.06, 0.20), textcoords="axes fraction",
                            fontsize=9, color=COLOUR[name], ha="left",
                            arrowprops=dict(arrowstyle="->", color=COLOUR[name],
                                            lw=1.2, connectionstyle="arc3,rad=0.2"))
    ax.set_xlabel("t"); ax.set_ylabel("kinetic energy")
    ax.set_title("Kinetic energy cannot grow in a closed unforced domain")
    ax.legend(title="grid")

    # Per-sample increments: the guard's actual signal.
    for name in LEVELS:
        if name not in data:
            continue
        d = data[name]
        bx.plot(d["t"][1:], np.diff(d["kinetic_energy"]), color=COLOUR[name],
                lw=1.2, label=name)
    bx.axhline(0, color="#333333", lw=1.2, ls="--")
    bx.set_xlabel("t"); bx.set_ylabel("$\\Delta E$ between samples")
    bx.set_title("Anything above the dashed line is unphysical")
    bx.legend(title="grid")
    fig.tight_layout()
    return save(fig, out_dir, "energy-decay.png")


# --------------------------------------------------------------------------
# 4. Two-grid agreement, and where it ends
# --------------------------------------------------------------------------

def fig_medium_vs_fine(data, out_dir):
    """Medium and fine agree on integral quantities and not on pointwise peaks."""
    if not {"medium", "fine"} <= set(data):
        print("  skipped medium-vs-fine (needs both levels)")
        return None
    M, F = data["medium"], data["fine"]
    tg = np.linspace(0.001, min(M["t"][-1], F["t"][-1]), 500)

    quantities = [
        ("kinetic_energy", "kinetic energy", "#0353a4"),
        ("max_circulation", "max $|\\Gamma|$", "#3d8b37"),
        ("bkm_integral", "BKM integral", "#e08a00"),
        ("enstrophy", "enstrophy", "#8338ec"),
        ("max_vorticity", "max $|\\omega|$", "#c1121f"),
    ]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
    for col, label, colour in quantities:
        m, f = np.interp(tg, M["t"], M[col]), np.interp(tg, F["t"], F[col])
        ax.plot(tg, 100 * np.abs(m - f) / np.abs(f), color=colour, lw=1.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t"); ax.set_ylabel("|medium - fine| / fine   (%)")
    ax.set_title("Two-grid difference: integrals converge, peaks do not")
    ax.legend(fontsize=9, ncol=2)

    ke_m = np.interp(tg, M["t"], M["kinetic_energy"])
    ke_f = np.interp(tg, F["t"], F["kinetic_energy"])
    drift = 100 * np.abs(ke_m - ke_f) / ke_f
    bx.plot(tg, drift, color="#0353a4", lw=2)
    bx.fill_between(tg, 0, drift, color="#0353a4", alpha=0.12)
    for mark in (0.2631, tg[-1]):
        bx.axvline(mark, color="#888888", ls=":", lw=1.2)
    bx.annotate(f"{np.interp(0.2631, tg, drift):.2f}% at the\nthree-grid horizon",
                xy=(0.2631, np.interp(0.2631, tg, drift)), xytext=(0.30, 1.1),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    bx.annotate(f"{drift[-1]:.2f}% at T = {tg[-1]:.2f}",
                xy=(tg[-1], drift[-1]), xytext=(0.30, 3.6), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1))
    bx.set_xlabel("t"); bx.set_ylabel("difference (%)")
    bx.set_title("Kinetic energy: the error accumulates with time")
    fig.tight_layout()
    return save(fig, out_dir, "medium-vs-fine-drift.png")


# --------------------------------------------------------------------------
# 5. Resolution against the flow it has to carry
# --------------------------------------------------------------------------

def fig_resolution_vs_flow(data, study, out_dir):
    """The grid is finest where nothing happens and coarsest where everything does."""
    import json

    rows = []
    for name in LEVELS:
        info = os.path.join(study, "meshes", f"apple_{name}.json")
        if not (os.path.exists(info) and name in data):
            continue
        with open(info) as fh:
            mi = json.load(fh)
        d = data[name]
        u = float(np.median(d["max_velocity"]))
        delta = 1.0 / np.sqrt(u * 1.0 / 1e-3)          # laminar wall layer, L/sqrt(Re)
        rows.append((name, mi["h_pole_actual"], mi["h_equator_actual"], delta))
    if not rows:
        print("  skipped resolution-vs-flow (missing mesh metadata)")
        return None

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.4, label="$h$ at the poles", color="#0353a4")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.4, label="$h$ at the equator wall", color="#c1121f")
    ax.plot(x, [r[3] for r in rows], "k--o", lw=1.6, ms=6,
            label="wall layer $\\delta \\sim L/\\sqrt{Re}$")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("length"); ax.legend(fontsize=9)
    ax.set_title("Refinement went to the poles, 15:1")

    cells = [r[3] / r[2] for r in rows]
    bars = bx.bar([r[0] for r in rows], cells, width=0.55,
                  color=["#c1121f" if c < 2 else "#3d8b37" for c in cells])
    bx.axhline(1.0, color="#333333", ls="--", lw=1.2)
    bx.text(len(rows) - 0.45, 1.05, "one cell", fontsize=9, ha="right")
    for b, c in zip(bars, cells):
        bx.text(b.get_x() + b.get_width() / 2, c + 0.03, f"{c:.1f}",
                ha="center", fontweight="bold")
    bx.set_ylabel("cells across the wall layer")
    bx.set_title("Where max $|\\omega|$ lives, even the fine grid\n"
                 "resolves the layer with under one cell", fontsize=10)
    fig.tight_layout()
    return save(fig, out_dir, "resolution-vs-flow.png")


# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 6. The domain, revolved
# --------------------------------------------------------------------------

def fig_domain_3d(out_dir):
    """A cutaway of the volume of revolution, to establish what is being solved."""
    z = np.linspace(-H, H, 260)
    theta = np.linspace(0, 1.5 * np.pi, 160)          # 270 deg: cut away one quarter
    Z, T = np.meshgrid(z, theta)
    Rr = boundary_radius(Z)
    X, Y = Rr * np.cos(T), Rr * np.sin(T)

    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0,
                    antialiased=True, alpha=0.92,
                    facecolors=plt.cm.Blues(0.35 + 0.45 * (Rr / Rr.max())))

    # The exposed meridian section, so the cutaway reads as a solid.
    for ang in (0.0, 1.5 * np.pi):
        rr = boundary_radius(z)
        ax.plot(rr * np.cos(ang), rr * np.sin(ang), z, color="#01315f", lw=1.6)
    ax.plot(np.zeros_like(z), np.zeros_like(z), z, color="#c1121f", lw=1.4,
            ls="--", label="symmetry axis")

    for zz, name in ((H, "pole"), (-H, "pole")):
        ax.scatter([0], [0], [zz], color="#c1121f", s=32, depthshade=False)
    ax.text(0, 0, H + 0.30, "poles: 6.07$\\degree$ cone", color="#c1121f",
            ha="center", fontsize=9)

    ax.set_xticks([]); ax.set_yticks([])          # the shape is the message
    ax.set_zlabel("z")
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("white"); pane.pane.set_alpha(1.0)
    ax.grid(False)
    ax.set_box_aspect((1, 1, 1.7))
    ax.view_init(elev=16, azim=-58)
    ax.set_title("The domain: $f(z) = R_0\\cos(\\pi z/2H)\\,e^{-kz^2}$ revolved\n"
                 "$R_0 = 1$, $H = 2$, $k = 0.5$   (one quarter cut away)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    return save(fig, out_dir, "domain-3d.png")


# --------------------------------------------------------------------------
# 7. The mesh that was asked for versus the mesh that was delivered
# --------------------------------------------------------------------------

def _draw_mesh(ax, pts, tris, xlim=None, ylim=None, lw=0.18, colour="#0353a4"):
    """Draw only the triangles inside the window: the fine mesh has 752k."""
    if xlim and ylim:
        cx, cy = pts[tris].mean(axis=1).T
        pad_x = 0.25 * (xlim[1] - xlim[0])
        pad_y = 0.25 * (ylim[1] - ylim[0])
        keep = ((cx > xlim[0] - pad_x) & (cx < xlim[1] + pad_x) &
                (cy > ylim[0] - pad_y) & (cy < ylim[1] + pad_y))
        tris = tris[keep]
    ax.triplot(pts[:, 0], pts[:, 1], tris, lw=lw, color=colour, alpha=0.85)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return len(tris)


def fig_mesh_comparison(old_msh, new_msh, out_dir):
    """The 150x miss, drawn. Same domain, same requested polar refinement."""
    if not (os.path.exists(old_msh) and os.path.exists(new_msh)):
        print("  skipped mesh-comparison (mesh files not found)")
        return None
    old_p, old_t = read_msh(old_msh)
    new_p, new_t = read_msh(new_msh)

    def cells_near_pole(pts, tris, cut=1.95):
        return int((np.abs(pts[tris].mean(axis=1)[:, 1]) > cut).sum())

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 11.5))
    panels = [
        (axes[0][0], old_p, old_t, "#c1121f", "Runs 1 and 2", None, None),
        (axes[0][1], new_p, new_t, "#0353a4", "Run 3 (coarse level)", None, None),
        (axes[1][0], old_p, old_t, "#c1121f", None, (-0.002, 0.023), (1.86, 2.005)),
        (axes[1][1], new_p, new_t, "#0353a4", None, (-0.002, 0.023), (1.86, 2.005)),
    ]
    for ax, pts, tris, colour, title, xl, yl in panels:
        zoom = xl is not None
        _draw_mesh(ax, pts, tris, xl, yl, lw=0.30 if zoom else 0.12, colour=colour)
        if not zoom:
            ax.set_xlim(-0.05, 1.05); ax.set_ylim(-2.1, 2.1)
            ax.set_title(f"{title}\n{len(tris):,} triangles", fontweight="bold")
            ax.add_patch(plt.Rectangle((-0.002, 1.86), 0.025, 0.145, fill=False,
                                       edgecolor="#111111", lw=1.3, zorder=6))
            ax.set_ylabel("z")
        else:
            n = cells_near_pole(pts, tris)
            ax.set_title(f"at the pole: {n:,} cells with $|z| > 1.95$", fontsize=10)
            ax.set_xlabel("r")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Both meshes were asked for polar refinement. Only one delivered it.",
                 fontweight="bold", fontsize=13, y=0.98)
    fig.text(0.5, 0.055,
             "Runs 1 and 2 loaded the same file, byte for byte (md5 0cba6a4c…). Its size field "
             "referenced the coordinate $z$,\nbut the geometry was built in the xy plane, so the "
             "field collapsed to the constant 0.015 everywhere —\n"
             "150$\\times$ the requested 1e-4. The lower row is the same window in both.",
             ha="center", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    return save(fig, out_dir, "mesh-comparison.png")


# --------------------------------------------------------------------------
# 8. The delivered grading, at the pole
# --------------------------------------------------------------------------

def fig_pole_zoom(msh, out_dir):
    """Three magnifications of the production mesh at the pole."""
    if not os.path.exists(msh):
        print("  skipped pole-zoom (mesh file not found)")
        return None
    pts, tris = read_msh(msh)
    slope = (np.pi * R0 / (2 * H)) * np.exp(-K * H * H)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    for ax, span in zip(axes, (0.40, 0.04, 0.004)):
        r_max = float(boundary_radius(H - span))   # the window's widest point
        xl = (-0.07 * r_max, 1.18 * r_max)
        yl = (H - span, H + 0.04 * span)
        _draw_mesh(ax, pts, tris, xl, yl, lw=0.45)
        d = np.linspace(0, span, 100)
        ax.plot(slope * d, H - d, color="#c1121f", lw=1.6, ls="--", zorder=6,
                label="exact 6.07$\\degree$ cone")
        ax.axvline(0.0, color="#3d8b37", lw=1.4, zorder=6)
        ax.set_title(f"window: {span:g} below the pole")
        ax.set_xlabel("r")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
    axes[0].set_ylabel("z")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Run 3 mesh at the pole: the grading is delivered, and the boundary "
                 "stays a cone all the way down", fontweight="bold", y=1.01)
    fig.tight_layout()
    return save(fig, out_dir, "pole-zoom.png")


def copy_gci_plots(study, out_dir):
    src = os.path.join(study, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    copied = 0
    for name, target in GCI_PLOTS.items():
        p = os.path.join(src, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_dir, target))
            print(f"  copied  {os.path.join(out_dir, target)}")
            copied += 1
    if not copied:
        print(f"  no GCI plots in {src} -- run scripts/analyze_convergence.py first")
    return copied


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", default="results/convergence_aws",
                   help="convergence study directory (default: %(default)s)")
    p.add_argument("--out", default="media", help="media root (default: %(default)s)")
    args = p.parse_args()

    style()
    data = load(args.study)
    print(f"levels found: {', '.join(data)}")

    domain = os.path.join(args.out, "domain")
    conv = os.path.join(args.out, "convergence")

    fig_cone_not_cusp(domain)
    fig_domain_3d(domain)
    fig_mesh_comparison(
        "archive/run_01/code_R1/assets_R1/apple_domain_R1.msh",
        os.path.join(args.study, "meshes", "apple_coarse.msh"), domain)
    fig_pole_zoom(os.path.join(args.study, "meshes", "apple_fine.msh"), domain)
    fig_vorticity_location(data, conv)
    fig_energy_decay(data, args.study, conv)
    fig_medium_vs_fine(data, conv)
    fig_resolution_vs_flow(data, args.study, conv)
    copy_gci_plots(args.study, conv)
    print("done")


if __name__ == "__main__":
    main()
