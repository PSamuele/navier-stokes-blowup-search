# Run 3 — the production campaign

This is the run the repository is about, and the only one whose code is live: it
is `src/` at the repository root, not a frozen copy. Runs 1 and 2 are archived
under [`archive/`](../archive/) as evidence.

## What was run

A three-resolution mesh convergence study on an AWS `c6i.4xlarge`, 8 MPI ranks,
`ν = 1e-3`, `T = 0.55`, CFL 0.5, IPCS with checked linear solves. All lengths are
scaled by the same factor between levels, so a single `h` identifies each grid —
a precondition for Richardson extrapolation to be defined at all.

| level | cells | velocity DOFs | h at the poles | outcome | final t | trustworthy to | wall time |
| :-- | --: | --: | --: | :-- | --: | --: | --: |
| coarse | 47,134 | 287,205 | 2.137e-3 | `energy_growth` | 0.2680 | **0.2631** | 5.1 min |
| medium | 188,462 | 1,139,571 | 1.077e-3 | **`completed`** | **0.5500** | 0.5500 | 1.6 h |
| fine | 752,803 | 4,534,410 | 5.467e-4 | **`completed`** | **0.5500** | 0.5500 | **14.0 h** |

Measured refinement ratios from the delivered meshes: **1.985** and **1.969**
against a nominal 2, so the extrapolation is well conditioned.

Two of the three grids reached `T` with **strictly decreasing kinetic energy** —
zero rising samples out of 550 on each — and never tripped the energy guard. Only
the coarse grid broke down, at `t = 0.2631`.

## What is here

```
convergence_aws/                      the production study
├── {coarse,medium,fine}/
│   ├── blowup_data_<level>.csv       diagnostics on a common time grid
│   ├── status_<level>.json           provenance: mesh, solver, versions, stop reason
│   ├── run_meta_<level>.json         invocation record
│   ├── solver_<level>.log            per-level log
│   ├── velocity_<level>.xdmf/.h5     field output (HDF5 not in git, see below)
│   ├── vortex_blowup_<level>.gif     ParaView animation of that level
│   └── checkpoints/
├── meshes/apple_<level>.json         what gmsh actually delivered, measured
├── analysis/                         Richardson, GCI, plots (regenerate, see below)
└── convergence_summary.json          study configuration and mesh generation record
validation_local/                     the same study at laptop scale, run end to end first
```

**Every per-level file carries its level in the name.** Three levels write into
sibling directories, so without the suffix their files would be named identically
and stop being distinguishable the moment one is downloaded, quoted or archived
away from its directory. The solver does this itself through `--tag`, which the
convergence driver sets from the level name; a standalone run writes the plain
names. Study-wide files — `convergence_summary.json`, `study_config.json`, and
everything under `analysis/` — cover all three levels and stay generic.

Note that `velocity_<level>.xdmf` references its HDF5 companion by name, so the
two must be renamed together.

## What is not here

The HDF5 velocity fields (`velocity_<level>.h5`: 17 MB / 129 MB / 510 MB) and the
`.msh` binaries are excluded from git — they exceed GitHub's limits and the
meshes are reproducible anyway, since `src/mesh.py` is deterministic and verifies
what gmsh delivered. The fields are archived separately; the small `.json` files
recording each mesh's *measured* properties are kept here.

## Regenerating the analysis

```bash
python scripts/analyze_convergence.py --study results/convergence_aws
```

```bash
python scripts/make_figures.py
```

Neither needs FEniCSx — numpy and matplotlib are enough. The first rewrites
`analysis/`, the second rebuilds every data figure under `media/`.

→ **What the numbers mean, including where they do not converge:**
[`docs/convergence.md`](../docs/convergence.md)
→ **How the campaign was deployed and costed:** [`deploy/`](../deploy/)
