# Running the Run 3 convergence study on AWS

Target: **16 vCPU**, budget **€200**. The projected cost is well inside that — the
point of the procedure below is to *confirm* that on the real instance before
committing, rather than trusting an estimate made on a laptop.

---

## 1. Choose the instance

| | recommendation |
| :-- | :-- |
| Instance | **`c6i.4xlarge`** — 16 vCPU (8 physical cores + hyperthreading), 32 GiB RAM, Ice Lake |
| Region | whichever is cheapest for you; on-demand is ≈ **$0.68/h** in `eu-west-1` |
| AMI | Ubuntu Server 22.04 or 24.04 LTS (x86_64) |
| Storage | **60 GB gp3** — the study writes ≈ 3 GB, the conda environment ≈ 6 GB, the rest is headroom |
| Pricing | on-demand is simplest. Spot is ≈ 70 % cheaper and the study is resumable, but a reclaim mid-level costs you that level's progress back to its last checkpoint |

**Memory is the binding constraint, not CPU.** Measured on two levels and fitted
(≈ 76 MB per rank plus 1.66 kB per velocity DOF, which reproduces both measurements
exactly):

| level | cells | velocity DOFs | RAM @ 8 ranks | RAM @ 16 ranks |
| :-- | --: | --: | --: | --: |
| coarse | 47,134 | 287,205 | 1.1 GB | 1.7 GB |
| medium | 188,462 | 1,139,571 | 2.4 GB | 3.0 GB |
| fine | 752,803 | 4,534,410 | **7.8 GB** | **8.5 GB** |

32 GiB gives roughly 3.5x headroom on the fine level. Do not go below 16 GiB — the
fine level was developed against an 8 GB WSL and **crashed it**, which is exactly how
the number above was pinned down.

---

## 2. Set up

```bash
ssh ubuntu@<instance>
git clone https://github.com/PSamuele/Navier_Stokes_Cusp_Study.git ~/navier-stokes-cusp-study
cd ~/navier-stokes-cusp-study
git checkout run-03-corrected-solver-and-convergence-study   # until it is merged into main
bash ./deploy/aws_setup.sh
conda activate fenicsx-env
```

`aws_setup.sh` installs `build-essential` first, and that is not incidental: FFCx
**JIT-compiles every variational form at run time**, so without a working `gcc` the
solver dies at the first `fem.form()` with a `cffi.VerificationError`. The script
verifies the compiler, DOLFINx, PETSc and HYPRE before returning.

Then confirm the port is healthy:

```bash
cd ~/navier-stokes-cusp-study/.
python -m pytest tests -q          # expect: 49 passed
```

---

## 3. Measure before you commit

### Pick the rank count

`c6i.4xlarge` advertises 16 vCPUs but has **8 physical cores**. For a
memory-bandwidth-bound FEM solve, 16 ranks is not automatically faster than 8, and on
the development laptop 4 ranks gave only a 1.5x speedup over 1. Measure it:

```bash
bash deploy/run_aws.sh --ranks-sweep
```

Put the winner in `"np"` in `configs/aws_production.json`.

### Project the cost

```bash
bash deploy/run_aws.sh --benchmark
```

This runs a handful of real steps on the coarse and medium meshes and prints s/step,
the projected step count, wall time and memory per level. The fine level is excluded
because it needs ~8 GB; **multiply the medium row by 8** for its estimate (4x the cells,
2x the steps — cost scales as `h^-3`).

For reference, the same benchmark on the development laptop (4 ranks, WSL, 8 GB):

| level | cells | velocity DOFs | s/step | dt | steps | hours |
| :-- | --: | --: | --: | --: | --: | --: |
| coarse | 47,134 | 287,205 | 1.11 | 2.22e-4 | 2,475 | 0.8 |
| medium | 188,462 | 1,139,571 | 3.62 | 1.18e-4 | 4,675 | 4.7 |
| fine *(extrapolated)* | 752,803 | 4,534,410 | ≈14.5 | ≈5.9e-5 | ≈9,350 | ≈38 |
| **total** | | | | | | **≈43 h** |

A c6i instance with more and faster cores should come in several times below that, so
expect **roughly 10–20 h, i.e. $7–15** — and in practice less, because the energy guard
stops each level before `T` (see section 1bis of `AWS_OPERATIONS.md`), whereas this
projection assumes every level runs the full way. Treat the on-instance benchmark as the real
number; if it projects more than about 100 h, stop and reconsider `T` or the resolution
triple rather than paying for it.

Note `dt` halves from coarse to medium, as it must: the cell-local CFL condition is set
by the jet core near the equator, and the measured `dt` ratio between grids differing by
exactly 2 is **1.98**.

---

## 4. Run it

```bash
tmux new -s ns3
bash deploy/run_aws.sh
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t ns3`. Without tmux the run dies
with your SSH session.

Monitor from a second shell:

```bash
tail -f ./results/convergence_aws/*/solver.log
watch -n30 'tail -3 ./results/convergence_aws/*/blowup_data_*.csv'
```

### If it stops

The solver stops deliberately rather than producing nonsense, and always writes
`status.json` saying why:

| `terminated_reason` | meaning | what to do |
| :-- | :-- | :-- |
| `completed` | reached `T` | proceed to the analysis |
| `energy_growth` | kinetic energy rose above its running minimum, so the grid no longer resolves the flow. `status.json` records `t_at_energy_min` — the reliable horizon | **the expected outcome, not a failure.** Use that horizon, or refine. See section 1bis of AWS_OPERATIONS.md |
| `cfl_below_dt_min` | the stability limit fell below `--dt_min` | **this is the Run 2 failure mode, caught.** The flow is genuinely trying to develop scales the grid cannot carry. Refine, or accept `t_final` as the reliable horizon |
| `velocity_limit_exceeded` | max\|u\| passed `--max_velocity` | same reading: something is running away. Inspect the CSV before raising the limit |
| `linear_solver_diverged` | a PETSc solve failed (e.g. `DIVERGED_PC_FAILED`) | try `--pc_projection jacobi` or `--pc_momentum asm`; report which stage from the message |
| `non_finite_state` | NaN/Inf reached the field | should not happen now; keep the log |

To continue after an interruption:

```bash
bash deploy/run_aws.sh --resume
```

Completed levels are skipped; a partially finished level restarts from its last
checkpoint. Restart validates the rank count and the mesh fingerprint, so resuming with
a different `-np` fails loudly instead of silently scrambling the state.

---

## 5. Collect the results

```bash
python scripts/analyze_convergence.py \
    --study results/convergence_aws
```

Then pull back the small artefacts — the report, plots, CSVs and metadata are a few MB;
the XDMF/HDF5 fields are the multi-GB part and are usually not worth transferring:

```bash
# from your machine
rsync -avz --exclude='*.h5' --exclude='*.npy' --exclude='*.xdmf' \
    ubuntu@<instance>:~/navier-stokes-cusp-study/./results/convergence_aws/ \
    ./results/convergence_aws/
```

**Then terminate the instance.** A forgotten `c6i.4xlarge` costs about $16/day.

---

## 6. Reading the report

`analysis/convergence_report.md` gives, per quantity: the fine-grid value, the
Richardson extrapolation to `h -> 0`, the observed order `p`, the **GCI** and the
monotone fraction.

* Quote the **fine-grid value ± GCI**. That is the defensible number.
* An observed `p` near the formal order (1 for the pressure splitting, up to 2 spatially)
  with a high monotone fraction means the grids are in the asymptotic range and the
  extrapolation is meaningful.
* A `p` far from that, or a low monotone fraction, means they are not — the extrapolated
  value is then an artefact, **not** a better answer than the fine grid. Refine further
  before quoting anything.
* Watch the **BKM integral** `∫ ||omega||_inf dt`. A finite-time singularity at `T*`
  requires it to diverge; a run where it stays bounded has not blown up, however
  dramatic the vorticity curve looks. See finding D1 in `../docs/findings.md` for why this
  geometry is a cone rather than a cusp, and what that does to the expectation.

---

## 7. Budget summary

| item | estimate |
| :-- | --: |
| compute, `c6i.4xlarge` on-demand, ≈15 h | ≈ $10 |
| EBS, 60 GB gp3 for ~2 days | ≈ $0.4 |
| data transfer out (few hundred MB) | ≈ $0.05 |
| **total** | **≈ $11** |

Even at 5x the projected wall time this stays under $60, comfortably inside €200. The
`--benchmark` step exists so you never find that out the expensive way.
