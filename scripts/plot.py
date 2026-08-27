#!/usr/bin/env python3
"""
plot.py - Diagnostic plots for a single Run 3 solver output.

Reads ``<out_dir>/blowup_data*.csv`` and draws the quantities the study is about.

Two of these panels exist specifically because Run 2 had no way to notice it was
producing nonsense:

* **BKM integral.**  The Beale-Kato-Majda criterion says a singularity at T* forces
  ``int ||omega||_inf dt`` to diverge.  Plotting that integral directly is far more
  informative than plotting ``1/||omega||`` and looking for a straight line -- a
  straight line there is equally consistent with a numerical instability, which is
  exactly the trap Run 2 fell into.

* **CFL and boundary-condition residual.**  If either drifts, the numbers in the
  other panels are not physics.  Run 2 tracked neither.

Usage
-----
    python scripts/plot.py --out_dir results/convergence_aws/fine
    python scripts/plot.py --out_dir A --out_dir B --labels coarse fine
"""

from __future__ import annotations

import argparse
import json
import glob
import os

import numpy as np


def read_csv(path):
    with open(path) as fh:
        header = fh.readline().strip().split(",")
        rows = []
        for line in fh:
            parts = line.strip().split(",")
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


def load(out_dir):
    hits = sorted(glob.glob(os.path.join(out_dir, "blowup_data*.csv")))
    if not hits:
        raise SystemExit(f"no blowup_data*.csv in {out_dir}")
    csv = read_csv(hits[0])
    status = {}
    sp = os.path.join(out_dir, "status.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            status = json.load(fh)
    return csv, status


PANELS = [
    ("max_vorticity", "max |omega|", True),
    ("enstrophy", "enstrophy", True),
    ("bkm_integral", "BKM  int ||omega||_inf dt", False),
    ("kinetic_energy", "kinetic energy", False),
    ("max_circulation", "max |Gamma| = r u_theta", False),
    ("max_velocity", "max |u|", False),
    ("cfl", "CFL attained", False),
    ("div_u_rel", "||div u|| / ||grad u||", True),
    ("bc_residual", "boundary-condition residual", True),
]


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot Run 3 solver diagnostics.")
    p.add_argument("--out_dir", action="append", required=True,
                   help="Solver output directory; repeat to overlay several runs")
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--save", default=None, help="Output PNG (default: <first out_dir>/diagnostics.png)")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for i, d in enumerate(args.out_dir):
        csv, status = load(d)
        label = (args.labels[i] if args.labels and i < len(args.labels)
                 else status.get("label") or os.path.basename(os.path.normpath(d)))
        runs.append((label, csv, status))

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for ax, (col, title, logy) in zip(axes.ravel(), PANELS):
        drawn = False
        for label, csv, _ in runs:
            if col not in csv:
                continue
            y = csv[col]
            ok = np.isfinite(y)
            if col in ("div_u_rel", "bc_residual"):
                ok &= y > 0          # log axis cannot show an exact zero
            if ok.sum() < 2:
                continue
            ax.plot(csv["t"][ok], y[ok], lw=1.2, label=label)
            drawn = True
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("t")
        ax.grid(alpha=0.3)
        if logy and drawn:
            ax.set_yscale("log")
        if not drawn:
            ax.text(0.5, 0.5, "no positive data\n(quantity is exactly zero)",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                    color="grey")
        if len(runs) > 1 and drawn:
            ax.legend(fontsize=7)

    subtitle = "   |   ".join(
        f"{lab}: {st.get('terminated_reason', '?')} at t={st.get('final_t', float('nan')):.4f}"
        for lab, _, st in runs
    )
    fig.suptitle(f"Run 3 diagnostics\n{subtitle}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    save = args.save or os.path.join(args.out_dir[0], "diagnostics.png")
    fig.savefig(save, dpi=140)
    plt.close(fig)
    print(f"written {save}")

    for label, csv, st in runs:
        print(f"\n--- {label} ---")
        print(f"  terminated       : {st.get('terminated_reason', '?')}"
              + (f"  ({st.get('terminated_detail')})" if st.get("terminated_detail") else ""))
        print(f"  final t / steps  : {st.get('final_t', float('nan')):.6f} / {st.get('final_step', '?')}")
        for col in ("max_vorticity", "max_velocity", "enstrophy", "bkm_integral"):
            if col in csv and np.isfinite(csv[col]).any():
                print(f"  {col:<17}: final {csv[col][-1]:.6g}   peak {np.nanmax(csv[col]):.6g}")
        if "bc_residual" in csv:
            print(f"  {'bc_residual':<17}: max {np.nanmax(csv['bc_residual']):.3e}")
    return save


if __name__ == "__main__":
    main()
