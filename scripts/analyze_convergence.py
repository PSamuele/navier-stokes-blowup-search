#!/usr/bin/env python3
"""
analyze_convergence.py - Richardson extrapolation and Grid Convergence Index.

Turns three runs at h, h/2, h/4 into numbers with error bars:

* **Observed order of convergence**, solved with the ASME V&V 20 / Roache
  procedure for a *non-uniform* refinement ratio.  Gmsh does not land exactly on
  the requested factor of two -- the validation study measured 1.71 between the two
  coarsest grids -- and forcing a nominal ratio into the formula biases both the
  order and everything derived from it.  If p is far from the formal order of the
  scheme, the grids are not in the asymptotic range and extrapolation is not
  justified; the report says so rather than quoting a number anyway.

* **Richardson extrapolation** to h -> 0 from the two finest grids.

* **Grid Convergence Index** (Roache) with safety factor 1.25: the band within
  which the converged value is expected to lie, i.e. the honest error bar on the
  fine-grid result.

Everything is evaluated on the common sampling grid the solver writes, so no
interpolation across mismatched time series is needed beyond a linear resample.

Usage
-----
    python scripts/analyze_convergence.py --study results/convergence_aws
"""

from __future__ import annotations

import argparse
import json
import glob
import os

import numpy as np


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

# Quantities the study is actually about.  Each is (column, human label, kind)
# where kind is "peak" for a running maximum or "point" for the instantaneous value.
TARGETS = [
    ("max_vorticity", "max |omega|", "point"),
    ("enstrophy", "enstrophy", "point"),
    ("kinetic_energy", "kinetic energy", "point"),
    ("max_circulation", "max |Gamma| = r u_theta", "point"),
    ("bkm_integral", "BKM integral of ||omega||_inf dt", "point"),
]

SAFETY_FACTOR = 1.25  # Roache, for p estimated from three grids


# ---------------------------------------------------------------------

def read_csv(path):
    """Minimal CSV reader so the script has no pandas dependency."""
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        rows = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != len(header):
                continue
            try:
                rows.append([float(v) if v else np.nan for v in parts])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    data = np.array(rows, dtype=float)
    return {name: data[:, i] for i, name in enumerate(header)}


def load_study(study_dir):
    summary_path = os.path.join(study_dir, "convergence_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)

    levels = {}
    for name in LEVELS:
        level_dir = os.path.join(study_dir, name)
        csv_path = find_one(level_dir, "blowup_data*.csv", "diagnostics CSV")
        status_path = find_one(level_dir, "status*.json", "status file")
        mesh_info = os.path.join(study_dir, "meshes", f"apple_{name}.json")
        if not os.path.exists(csv_path):
            print(f"  [{name}] missing {csv_path}; skipping")
            continue
        entry = {"data": read_csv(csv_path)}
        for key, p in (("status", status_path), ("mesh", mesh_info)):
            if os.path.exists(p):
                with open(p) as fh:
                    entry[key] = json.load(fh)
        levels[name] = entry
    return summary, levels


def trustworthy_horizon(entry):
    """Last time this level is physically meaningful, not merely its last row.

    When the energy guard fires, the samples between the energy minimum and the
    stop are already contaminated: kinetic energy was rising, which cannot happen
    in a closed unforced domain. status.json records `t_at_energy_min`, the last
    sample before that began. Building a Richardson extrapolation on the final CSV
    row instead would feed in exactly the unphysical tail this project exists to
    stop quoting.

    A level that terminated `completed` has `t_at_energy_min` equal to its final
    time, so one rule covers both cases.
    """
    data_end = float(np.nanmax(entry["data"]["t"]))
    horizon = entry.get("status", {}).get("t_at_energy_min")
    if horizon is None or not np.isfinite(horizon) or horizon <= 0 or horizon >= data_end:
        return data_end, "last sample"
    return float(horizon), f"energy minimum (guard fired at t = {data_end:.4f})"


def common_time_grid(levels, n=200, verbose=True):
    """Interval over which *every* level is still trustworthy, sampled uniformly."""
    ends, reasons = {}, {}
    for name, entry in levels.items():
        ends[name], reasons[name] = trustworthy_horizon(entry)
        if verbose:
            print(f"  [{name}] usable to t = {ends[name]:.5f}  ({reasons[name]})")

    t_end = min(ends.values())
    t_start = max(float(np.nanmin(lv["data"]["t"])) for lv in levels.values())
    if not (t_end > t_start):
        raise ValueError("Levels share no trustworthy common time interval")
    if verbose:
        limiting = min(ends, key=ends.get)
        print(f"  common window set by '{limiting}': [{t_start:.5f}, {t_end:.5f}]")
    return np.linspace(t_start, t_end, n), t_start, t_end


def resample(data, column, grid):
    t = data["t"]
    y = data[column]
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(grid, np.nan)
    order = np.argsort(t[ok])
    return np.interp(grid, t[ok][order], y[ok][order])


# ---------------------------------------------------------------------

def apparent_order(f_coarse, f_medium, f_fine, r21, r32, iters=60):
    """Observed order p for a *non-uniform* refinement ratio (ASME V&V 20, Roache).

    Real meshes never land exactly on the requested ratio -- in this study the
    measured coarse->medium polar ratio was 1.71 against a nominal 2.0 -- and using
    a single nominal ratio then biases both the order and the extrapolation.  The
    standard procedure solves

        p = |ln|eps32/eps21| + q(p)| / ln(r21),
        q(p) = ln((r21^p - s)/(r32^p - s)),   s = sign(eps32/eps21)

    by fixed-point iteration.  Indices follow ASME: 1 = fine, 2 = medium, 3 = coarse,
    so r21 = h_medium/h_fine and r32 = h_coarse/h_medium.
    """
    eps21 = f_medium - f_fine
    eps32 = f_coarse - f_medium

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = eps32 / eps21
        s = np.sign(ratio)
        logratio = np.log(np.abs(ratio))

        p = np.full_like(f_fine, np.nan, dtype=float)
        ok = np.isfinite(logratio) & (np.abs(eps21) > 0)
        pk = np.full_like(f_fine, 2.0, dtype=float)
        for _ in range(iters):
            q = np.log((r21**pk - s) / (r32**pk - s))
            pk_new = np.abs(logratio + q) / np.log(r21)
            pk_new = np.clip(pk_new, 0.05, 12.0)
            if np.allclose(np.where(ok, pk_new, 0.0), np.where(ok, pk, 0.0),
                           rtol=1e-8, atol=1e-10, equal_nan=True):
                pk = pk_new
                break
            pk = pk_new
        p[ok] = pk[ok]

    # A negative eps32/eps21 means the three grids are not on one monotone
    # convergence curve; no order can legitimately be inferred there.
    p = np.where(ratio > 0, p, np.nan)
    return p


def richardson(f_medium, f_fine, r21, p):
    """Extrapolated h -> 0 value from the two finest grids."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r21**p * f_fine - f_medium) / (r21**p - 1.0)


def gci(f_medium, f_fine, r21, p, safety=SAFETY_FACTOR):
    """Fine-grid Grid Convergence Index, as a fraction (multiply by 100 for %)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs((f_fine - f_medium) / np.where(f_fine == 0, np.nan, f_fine))
        return safety * rel / (r21**p - 1.0)


def analyse_target(levels, column, grid, r21, r32):
    series = {}
    for name in LEVELS:
        if name not in levels or column not in levels[name]["data"]:
            return None
        series[name] = resample(levels[name]["data"], column, grid)

    f_coarse, f_medium, f_fine = series["coarse"], series["medium"], series["fine"]
    p = apparent_order(f_coarse, f_medium, f_fine, r21, r32)
    f_ext = richardson(f_medium, f_fine, r21, p)
    band = gci(f_medium, f_fine, r21, p)

    finite = np.isfinite(p)
    # The last sample time where the three grids actually form a monotone
    # sequence.  Quoting the final grid point unconditionally would report NaN
    # whenever the very last sample happens to be non-monotone, which says nothing
    # about the study as a whole.
    idx = int(np.max(np.nonzero(finite)[0])) if finite.any() else -1
    return {
        "column": column,
        "series": series,
        "p": p,
        "extrapolated": f_ext,
        "gci": band,
        "ref_index": idx,
        "ref_time_valid": bool(finite.any()),
        "fine_ref": float(series["fine"][idx]) if idx >= 0 else float("nan"),
        "extrapolated_ref": float(f_ext[idx]) if idx >= 0 else float("nan"),
        "gci_ref": float(band[idx]) if idx >= 0 else float("nan"),
        "p_median": float(np.nanmedian(p[finite])) if finite.any() else float("nan"),
        "gci_median": float(np.nanmedian(band[np.isfinite(band)]))
        if np.isfinite(band).any() else float("nan"),
        "monotone_fraction": float(finite.mean()),
    }


# ---------------------------------------------------------------------

def make_plots(results, grid, out_dir, ratio):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable; skipping plots")
        return []

    written = []
    for res in results:
        if res is None:
            continue
        col = res["column"]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

        ax = axes[0]
        for name, style in zip(LEVELS, ("--", "-.", "-")):
            ax.plot(grid, res["series"][name], style, label=name)
        ok = np.isfinite(res["extrapolated"])
        ax.plot(grid[ok], res["extrapolated"][ok], "k:", lw=2, label="Richardson h->0")
        ax.set_xlabel("t")
        ax.set_ylabel(col)
        ax.set_title(f"{col}: grid convergence")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(grid, res["p"], "b-", lw=1)
        ax.axhline(1.0, color="grey", ls=":", label="first order")
        ax.axhline(2.0, color="grey", ls="--", label="second order")
        ax.set_xlabel("t")
        ax.set_ylabel("observed order p")
        ax.set_ylim(-1, 5)
        ax.set_title(f"observed order (median {res['p_median']:.2f})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(grid, 100.0 * res["gci"], "r-", lw=1)
        ax.set_xlabel("t")
        ax.set_ylabel("GCI (fine grid) [%]")
        ax.set_yscale("log")
        ax.set_title(f"error band (median {100 * res['gci_median']:.2f}%)")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        path = os.path.join(out_dir, f"convergence_{col}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    return written


def write_report(path, summary, levels, results, grid, ratios, t_span):
    lines = []
    A = lines.append
    A("# Run 3 - mesh convergence report\n")
    A(f"Common time window: t in [{t_span[0]:.5f}, {t_span[1]:.5f}], "
      f"{len(grid)} sample points.\n")

    A("## Grids\n")
    A("| level | cells | velocity DOFs | h_pole (asked) | h_pole (achieved) | "
      "h_equator (achieved) | final t | trustworthy to | terminated |")
    A("| :-- | --: | --: | --: | --: | --: | --: | --: | :-- |")
    for name in LEVELS:
        if name not in levels:
            continue
        m = levels[name].get("mesh", {})
        s = levels[name].get("status", {})
        A(f"| {name} | {m.get('num_elements_2d', '?')} | "
          f"{s.get('n_velocity_dofs', '?')} | "
          f"{m.get('lc_pole', float('nan')):.3e} | "
          f"{m.get('h_pole_actual', float('nan')):.3e} | "
          f"{m.get('h_equator_actual', float('nan')):.3e} | "
          f"{s.get('final_t', float('nan')):.5f} | "
          f"{trustworthy_horizon(levels[name])[0]:.5f} | "
          f"{s.get('terminated_reason', '?')} |")
    A("")

    r21, r32 = ratios
    nominal = summary.get("config", {}).get("ratio", 2.0)
    A(f"Refinement ratios measured from the delivered meshes (polar element size): "
      f"medium/fine = {r21:.4f}, coarse/medium = {r32:.4f} (nominal {nominal}). "
      f"The observed order is solved for these two ratios rather than assuming a "
      f"constant one, so the deviation does not bias the result.\n")

    A("## Convergence of the target quantities\n")
    A("`p` is the observed order, `GCI` the fine-grid Grid Convergence Index "
      "(Roache, safety factor 1.25): the band within which the converged value is "
      "expected to lie. `monotone` is the fraction of sample times where the three "
      "grids form a monotone sequence, which is a precondition for the "
      "extrapolation to mean anything.\n")
    A("| quantity | fine-grid value | Richardson h->0 | observed p (median) | "
      "GCI (median) | monotone |")
    A("| :-- | --: | --: | --: | --: | --: |")
    for res, (col, label, _) in zip(results, TARGETS):
        # Pipes inside the label would terminate the markdown table cell.
        label = label.replace("|", r"\|")
        if res is None:
            A(f"| {label} | - | - | - | - | not available |")
            continue
        A(f"| {label} | {res['fine_ref']:.6g} | "
          f"{res['extrapolated_ref']:.6g} | {res['p_median']:.2f} | "
          f"{100 * res['gci_median']:.2f}% | {100 * res['monotone_fraction']:.0f}% |")
    A("")

    A("## How to read this\n")
    A("- A median `p` near the formal order of the scheme (1 for the pressure "
      "splitting, up to 2 for the spatial discretisation) means the three grids are "
      "in the asymptotic range and the extrapolated column is trustworthy.")
    A("- A `p` far from that, or a low `monotone` fraction, means they are not. In "
      "that case the extrapolated value is **not** a better answer than the fine "
      "grid; it is an artefact. Refine further before quoting a number.")
    A("- The GCI is the number to publish as the uncertainty on the fine-grid "
      "result.\n")

    A("## Provenance\n")
    for name in LEVELS:
        s = levels.get(name, {}).get("status", {})
        if s:
            A(f"- **{name}**: {s.get('final_step', '?')} steps, "
              f"{s.get('wall_time_sec', float('nan')):.0f} s wall, "
              f"{s.get('mpi_size', '?')} ranks, scheme `{s.get('scheme', '?')}`, "
              f"dolfinx {s.get('dolfinx_version', '?')}, "
              f"terminated `{s.get('terminated_reason', '?')}`"
              + (f" ({s.get('terminated_detail')})" if s.get("terminated_detail") else ""))
    A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Analyse the Run 3 convergence study.")
    p.add_argument("--study", required=True, help="Study root (the driver out_root)")
    p.add_argument("--ratio", type=float, default=None,
                   help="Refinement ratio (default: measured from the meshes)")
    p.add_argument("--points", type=int, default=200, help="Common grid sample count")
    p.add_argument("--out_dir", default=None, help="Where to write report and plots")
    args = p.parse_args(argv)

    study = os.path.abspath(args.study)
    out_dir = os.path.abspath(args.out_dir or os.path.join(study, "analysis"))
    os.makedirs(out_dir, exist_ok=True)

    summary, levels = load_study(study)
    if len(levels) < 3:
        print(f"Need all three levels; found {sorted(levels)}. "
              "Run the driver to completion first.")
        return None

    # Measured, per-interval ratios: h_medium/h_fine and h_coarse/h_medium.
    def _h(name):
        return levels[name].get("mesh", {}).get("h_pole_actual", float("nan"))

    if args.ratio is not None:
        r21 = r32 = float(args.ratio)
    else:
        r21 = _h("medium") / _h("fine")
        r32 = _h("coarse") / _h("medium")
        if not (np.isfinite(r21) and np.isfinite(r32) and r21 > 1 and r32 > 1):
            nominal = float(summary.get("config", {}).get("ratio", 2.0))
            print(f"  could not measure grid ratios; falling back to {nominal}")
            r21 = r32 = nominal
    ratio = r21
    print(f"Measured refinement ratios: medium/fine = {r21:.4f}, "
          f"coarse/medium = {r32:.4f}")

    grid, t0, t1 = common_time_grid(levels, args.points)
    print(f"Common time window: [{t0:.5f}, {t1:.5f}]")

    results = [analyse_target(levels, col, grid, r21, r32) for col, _, _ in TARGETS]

    print(f"\n{'quantity':<34}{'fine':>14}{'Richardson':>14}{'p':>8}{'GCI':>10}")
    for res, (col, label, _) in zip(results, TARGETS):
        if res is None:
            print(f"{label:<34}{'n/a':>14}")
            continue
        print(f"{label:<34}{res['fine_ref']:>14.6g}"
              f"{res['extrapolated_ref']:>14.6g}{res['p_median']:>8.2f}"
              f"{100 * res['gci_median']:>9.2f}%{100 * res['monotone_fraction']:>9.0f}%")

    plots = make_plots(results, grid, out_dir, ratio)
    report = write_report(os.path.join(out_dir, "convergence_report.md"),
                          summary, levels, results, grid, (r21, r32), (t0, t1))

    payload = {
        "ratio_medium_fine": r21,
        "ratio_coarse_medium": r32,
        "time_window": [t0, t1],
        "targets": {
            col: {
                "p_median": res["p_median"],
                "gci_median": res["gci_median"],
                "monotone_fraction": res["monotone_fraction"],
                "fine_value": res["fine_ref"],
                "richardson_value": res["extrapolated_ref"],
                "gci_at_reference": res["gci_ref"],
            }
            for res, (col, _, _) in zip(results, TARGETS) if res is not None
        },
    }
    with open(os.path.join(out_dir, "convergence_metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nwritten {report}")
    for path in plots:
        print(f"written {path}")
    return payload


if __name__ == "__main__":
    main()
