#!/usr/bin/env python3
"""
run_convergence.py - Drive the three-resolution mesh convergence study.

The point of the study is to obtain values that can be *trusted*, which means
knowing the discretisation error rather than hoping it is small.  That requires
three grids related by a constant refinement ratio, so Richardson extrapolation
and the Grid Convergence Index are defined.

Design constraints this script enforces
---------------------------------------
1. **One length parameter per level.**  Every length in the mesh (polar size and
   equatorial size alike) is scaled by the same factor ``ratio``.  If only the
   polar size were refined, "h" would be ambiguous and the observed order of
   convergence would be meaningless.

2. **The achieved h is measured, not assumed.**  Run 2 asked for h = 1e-4 at the
   poles and silently received 0.015 (finding A1).  Every level records the mesh
   size gmsh actually produced, and the *effective* refinement ratio is computed
   from those numbers and written into the summary.

3. **A common output time grid.**  All levels sample diagnostics at the same
   ``sample_dt``, so the three runs can be compared point by point without
   interpolating across wildly different step sequences.

4. **Resumable.**  A level whose ``status.json`` says it completed is skipped, so
   an interrupted study (or a spot instance reclaim) can be continued with the
   same command.

Usage
-----
    python scripts/run_convergence.py --config configs/aws_production.json
    python scripts/run_convergence.py --config configs/local_validation.json --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import mesh  # noqa: E402


LEVELS = ("coarse", "medium", "fine")


# ---------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------

DEFAULT_CONFIG = {
    "label": "run_03_convergence",
    "out_root": "results/convergence",
    # Coarsest level.  Finer levels divide these by `ratio` per step.
    "lc_pole_coarse": 2.0e-3,
    "grading": 15.0,          # lc_boundary / lc_pole, held FIXED across levels
    "ratio": 2.0,             # refinement ratio between consecutive levels
    "power": 2.0,
    "R0": 1.0, "H": 2.0, "k": 0.5,
    # Solver
    "T": 0.55,
    "nu": 1.0e-3,
    "cfl": 0.5,
    "scheme": "ipcs",
    "dt_policy": "cfl",       # "cfl": dt proportional to h; "fixed": same dt on all levels
    "dt_fixed": None,         # used when dt_policy == "fixed"
    "dt_min": 1.0e-9,
    "dt_max": 5.0e-3,
    "max_velocity": 1.0e6,
    "max_energy_growth": 0.01,
    "sample_dt": 1.0e-3,      # common time grid for all levels
    "xdmf_dt": 1.0e-2,
    "checkpoint_dt": 5.0e-2,
    "quadrature_degree": 6,
    "np": 8,                  # MPI ranks per level
    "log_interval": 500,
}


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path) as fh:
            cfg.update(json.load(fh))
    return cfg


def level_sizes(cfg):
    """Return ``[(name, lc_pole, lc_boundary), ...]`` from coarse to fine."""
    out = []
    for i, name in enumerate(LEVELS):
        lc_pole = cfg["lc_pole_coarse"] / (cfg["ratio"] ** i)
        out.append((name, lc_pole, lc_pole * cfg["grading"]))
    return out


# ---------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------

def build_meshes(cfg, out_root, force=False, only=None):
    """Generate the three meshes and record the size gmsh actually delivered."""
    mesh_dir = os.path.join(out_root, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    infos = {}

    for name, lc_pole, lc_boundary in level_sizes(cfg):
        if only and name not in only:
            continue
        path = os.path.join(mesh_dir, f"apple_{name}.msh")
        info_path = os.path.join(mesh_dir, f"apple_{name}.json")

        if os.path.exists(path) and os.path.exists(info_path) and not force:
            with open(info_path) as fh:
                infos[name] = json.load(fh)
            print(f"[mesh:{name}] reusing {path}")
            continue

        print(f"[mesh:{name}] generating lc_pole={lc_pole:.3e} lc_boundary={lc_boundary:.3e} ...")
        t0 = time.time()
        info = mesh.generate_mesh(
            output_file=path,
            lc_pole=lc_pole,
            lc_boundary=lc_boundary,
            R0=cfg["R0"], H=cfg["H"], k=cfg["k"],
            power=cfg["power"],
            verbosity=0,
            check=True,
        )
        info["generation_sec"] = time.time() - t0
        with open(info_path, "w") as fh:
            json.dump(info, fh, indent=2)
        infos[name] = info
        print(
            f"[mesh:{name}] {info['num_elements_2d']} cells, "
            f"h_pole {info['h_pole_actual']:.3e} (asked {lc_pole:.3e}), "
            f"h_eq {info['h_equator_actual']:.3e}, {info['generation_sec']:.1f}s"
        )

    _report_effective_ratio(cfg, infos)
    return mesh_dir, infos


def _report_effective_ratio(cfg, infos):
    """The ratio that Richardson uses must come from the delivered meshes."""
    print("\n--- effective refinement ratios (measured, not requested) ---")
    for a, b in zip(LEVELS[:-1], LEVELS[1:]):
        if a in infos and b in infos:
            rp = infos[a]["h_pole_actual"] / infos[b]["h_pole_actual"]
            re = infos[a]["h_equator_actual"] / infos[b]["h_equator_actual"]
            print(f"  {a} -> {b}:  polar {rp:.4f}   equatorial {re:.4f}   (requested {cfg['ratio']})")
            if abs(rp - cfg["ratio"]) / cfg["ratio"] > 0.15:
                print(f"    WARNING: polar ratio deviates >15% from the request; "
                      f"Richardson extrapolation will be biased.")
    print()


def solver_command(cfg, name, mesh_path, level_dir, info, restart=False):
    cmd = []
    if cfg["np"] and cfg["np"] > 1:
        cmd += ["mpirun", "-np", str(cfg["np"])]
        if cfg.get("bind_to"):
            cmd += ["--bind-to", cfg["bind_to"]]
        if cfg.get("oversubscribe"):
            cmd += ["--oversubscribe"]
    cmd += [
        sys.executable, os.path.join(ROOT, "src", "solver.py"),
        "--mesh", mesh_path,
        "--out_dir", level_dir,
        "--label", f"{cfg['label']}:{name}",
        "--tag", name,
        "-T", str(cfg["T"]),
        "--nu", str(cfg["nu"]),
        "--cfl", str(cfg["cfl"]),
        "--scheme", cfg["scheme"],
        "--dt_min", str(cfg["dt_min"]),
        "--dt_max", str(cfg["dt_max"]),
        "--max_velocity", str(cfg["max_velocity"]),
        "--max_energy_growth", str(cfg["max_energy_growth"]),
        "--sample_dt", str(cfg["sample_dt"]),
        "--xdmf_dt", str(cfg["xdmf_dt"]),
        "--checkpoint_dt", str(cfg["checkpoint_dt"]),
        "--quadrature_degree", str(cfg["quadrature_degree"]),
        "--log_interval", str(cfg["log_interval"]),
    ]

    if cfg["dt_policy"] == "fixed":
        # Same dt on every level: the observed order is then purely spatial.
        dt = cfg["dt_fixed"]
        if dt is None:
            raise ValueError('dt_policy "fixed" requires "dt_fixed" in the config')
        cmd += ["--no_adaptive_dt", "--num_steps", str(int(round(cfg["T"] / dt)))]
    else:
        # dt follows the CFL condition, so it shrinks with h.  Space and time are
        # refined together; the observed order mixes both contributions.
        cmd += ["--adaptive_dt"]

    if restart:
        cmd += ["--restart"]
    return cmd


def run_level(cfg, name, mesh_path, out_root, info, benchmark=False, dry_run=False,
              restart=False):
    level_dir = os.path.join(out_root, name)
    os.makedirs(level_dir, exist_ok=True)
    status_path = os.path.join(level_dir, "status.json")

    if not benchmark and not restart and os.path.exists(status_path):
        with open(status_path) as fh:
            st = json.load(fh)
        if st.get("terminated_reason") == "completed":
            print(f"[run:{name}] already completed at t={st['final_t']:.4f}; skipping")
            return st
        print(f"[run:{name}] previous attempt ended as "
              f"'{st.get('terminated_reason')}'; re-running")

    cfg_run = dict(cfg)
    if benchmark:
        # Short probe: enough steps to measure s/step without producing output.
        cfg_run["T"] = cfg["sample_dt"] * 2
        cfg_run["xdmf_dt"] = 1e9
        cfg_run["checkpoint_dt"] = 1e9
        cfg_run["log_interval"] = 10
        level_dir = os.path.join(out_root, "_benchmark", name)
        os.makedirs(level_dir, exist_ok=True)

    cmd = solver_command(cfg_run, name, mesh_path, level_dir, info, restart=restart)
    print(f"[run:{name}] {' '.join(cmd)}")
    if dry_run:
        return None

    log_path = os.path.join(level_dir, f"solver_{name}.log")
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            # Flush both sinks on every line.  A long run is normally watched
            # through `| tee`, and a pipe makes Python block-buffer its output --
            # so without these flushes the driver looks frozen for many minutes
            # while it is in fact working, and the solver log lags behind reality.
            log.write(line)
            log.flush()
            if any(tag in line for tag in ("step", "---", "!!", "Error", "error")):
                print(f"  [{name}] {line.rstrip()}", flush=True)
        proc.wait()
    wall = time.time() - t0

    st_path = os.path.join(level_dir, "status.json")
    if not os.path.exists(st_path):
        print(f"[run:{name}] FAILED (exit {proc.returncode}); see {log_path}")
        return {"terminated_reason": "process_failed", "returncode": proc.returncode,
                "log": log_path}
    with open(st_path) as fh:
        st = json.load(fh)
    st["driver_wall_sec"] = wall
    print(f"[run:{name}] {st['terminated_reason']} at t={st['final_t']:.5f} "
          f"after {st['final_step']} steps ({wall:.1f}s)")
    return st


# ---------------------------------------------------------------------
# benchmark / cost projection
# ---------------------------------------------------------------------

def estimate_memory_gb(velocity_dofs, ranks):
    """Rough total resident memory for one level.

    Fitted from measurements at 287k and 1.14M velocity DOFs on 4 ranks
    (780 MB and 2191 MB total): a fixed per-process cost for Python plus DOLFINx,
    and about 1.66 kB per velocity DOF for the matrices, vectors and the AMG
    hierarchy.  Good enough to choose an instance size; it is not a substitute for
    watching the real thing.
    """
    per_rank_overhead_mb = 76.0
    per_dof_mb = 1.656e-3
    total_mb = per_rank_overhead_mb * max(ranks, 1) + per_dof_mb * velocity_dofs
    return total_mb / 1024.0


def benchmark_study(cfg, mesh_dir, infos, out_root, levels=None):
    """Measure s/step per level and project the full-study wall time and cost.

    Run this before committing a cloud instance: the cost of the fine level scales
    like h^-3 (cells like h^-2, steps like h^-1), so a factor-2 error in the
    per-step estimate is a factor-2 error in the bill.
    """
    print("\n=== benchmark: measuring cost per step on each mesh ===")
    levels = levels or list(LEVELS)
    rows = []
    for name, lc_pole, _ in level_sizes(cfg):
        if name not in levels:
            # Honour --levels here too.  The production fine level needs about 8 GB,
            # more than a laptop usually has, so benchmarking it must be opt-in.
            print(f"  [{name}] skipped (not selected by --levels)")
            continue
        mesh_path = os.path.join(mesh_dir, f"apple_{name}.msh")
        st = run_level(cfg, name, mesh_path, out_root, infos[name], benchmark=True)
        if not st or st.get("final_step", 0) < 1:
            print(f"  [{name}] benchmark produced no steps; skipping projection")
            continue
        sec_per_step = st["wall_time_sec"] / max(st["final_step"], 1)
        # Average step, not the last one.  The final step of any run is truncated to
        # land exactly on T, so quoting it understates dt by an arbitrary factor and
        # inflates the projected step count with it.
        dt_typ = st["final_t"] / max(st["final_step"], 1)
        rows.append({
            "level": name,
            "cells": infos[name]["num_elements_2d"],
            "velocity_dofs": st["n_velocity_dofs"],
            "sec_per_step": sec_per_step,
            "dt": dt_typ,
            "ram_gb": estimate_memory_gb(st["n_velocity_dofs"], cfg["np"]),
        })

    print("\n--- projected full study ---")
    print(f"{'level':<8}{'cells':>10}{'velDOF':>12}{'s/step':>10}{'dt':>12}"
          f"{'steps':>12}{'hours':>10}{'RAM GB':>9}")
    total_h = 0.0
    for row in rows:
        if not row["dt"]:
            continue
        steps = cfg["T"] / float(row["dt"])
        hours = steps * row["sec_per_step"] / 3600.0
        row["projected_steps"] = steps
        row["projected_hours"] = hours
        total_h += hours
        print(f"{row['level']:<8}{row['cells']:>10}{row['velocity_dofs']:>12}"
              f"{row['sec_per_step']:>10.3f}{float(row['dt']):>12.2e}"
              f"{steps:>12.0f}{hours:>10.1f}{row['ram_gb']:>9.1f}")
    print(f"{'TOTAL':<8}{'':>10}{'':>12}{'':>10}{'':>12}{'':>12}{total_h:>10.1f}")

    peak = max((r.get("ram_gb", 0.0) for r in rows), default=0.0)
    if peak:
        print(f"\nPeak memory (estimated): {peak:.1f} GB for the largest level "
              f"benchmarked. Allow 1.5x headroom when choosing an instance.")

    rate = cfg.get("usd_per_hour")
    if rate:
        print(f"\nprojected cost at ${rate}/h on {cfg['np']} ranks: "
              f"${total_h * rate:,.0f}  ({total_h:.0f} h)")
    print("Note: measured on this machine. Re-run --benchmark on the target "
          "instance before committing to a budget.\n")

    bench_path = os.path.join(out_root, "benchmark.json")
    with open(bench_path, "w") as fh:
        json.dump({"config": cfg, "levels": rows, "total_hours": total_h}, fh, indent=2)
    print(f"written {bench_path}")
    return rows


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main(argv=None):
    # Under `| tee` stdout is a pipe, which Python block-buffers; reconfigure to
    # line buffering so progress is visible as it happens rather than in bursts.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:  # pragma: no cover - very old Python
        pass

    p = argparse.ArgumentParser(
        description="Run the Run 3 three-resolution mesh convergence study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="JSON config file")
    p.add_argument("--out_root", default=None, help="Override the config output root")
    p.add_argument("--levels", nargs="*", default=None, choices=LEVELS,
                   help="Subset of levels to run (default: all)")
    p.add_argument("--np", type=int, default=None, help="Override MPI ranks per level")
    p.add_argument("-T", "--t_final", type=float, default=None, help="Override final time")
    p.add_argument("--benchmark", action="store_true",
                   help="Measure s/step and project the total cost, then exit")
    p.add_argument("--meshes_only", action="store_true", help="Generate meshes and exit")
    p.add_argument("--force_mesh", action="store_true", help="Regenerate meshes")
    p.add_argument("--restart", action="store_true", help="Resume levels from checkpoints")
    p.add_argument("--dry_run", action="store_true", help="Print commands without running")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.out_root:
        cfg["out_root"] = args.out_root
    if args.np:
        cfg["np"] = args.np
    if args.t_final:
        cfg["T"] = args.t_final

    out_root = os.path.abspath(cfg["out_root"])
    os.makedirs(out_root, exist_ok=True)

    print("=" * 74)
    print(f" Run 3 convergence study: {cfg['label']}")
    print("=" * 74)
    for name, lp, lb in level_sizes(cfg):
        print(f"  {name:<8} lc_pole={lp:.3e}  lc_boundary={lb:.3e}")
    print(f"  ratio={cfg['ratio']}  grading={cfg['grading']}  T={cfg['T']}  "
          f"dt_policy={cfg['dt_policy']}  np={cfg['np']}")
    print(f"  out_root={out_root}")
    print()

    with open(os.path.join(out_root, "study_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    mesh_dir, infos = build_meshes(cfg, out_root, force=args.force_mesh,
                                   only=args.levels)
    if args.meshes_only:
        return {"meshes": infos}

    if args.benchmark:
        benchmark_study(cfg, mesh_dir, infos, out_root, levels=args.levels)
        return {"meshes": infos}

    levels = args.levels or list(LEVELS)
    results = {}
    for name in levels:
        mesh_path = os.path.join(mesh_dir, f"apple_{name}.msh")
        results[name] = run_level(cfg, name, mesh_path, out_root, infos[name],
                                  dry_run=args.dry_run, restart=args.restart)

    summary = {
        "config": cfg,
        "meshes": infos,
        "levels": results,
        "effective_ratio": {
            f"{a}->{b}": infos[a]["h_pole_actual"] / infos[b]["h_pole_actual"]
            for a, b in zip(LEVELS[:-1], LEVELS[1:])
            if a in infos and b in infos
        },
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = os.path.join(out_root, "convergence_summary.json")
    if not args.dry_run:
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwritten {summary_path}")
        print("Next: python scripts/analyze_convergence.py --study "
              f"{out_root}")
    return summary


if __name__ == "__main__":
    main()
