# Run 2 — archived, superseded, **do not run**

The second attempt: the same problem moved onto AWS with 16 MPI ranks and
adaptive time-stepping, intended as the high-resolution counterpart to
[Run 1](../run_01/). Preserved here as **evidence**, not as software. Nothing has
been modified since the run.

## The resolution increase never happened

`main_R2.py` loads `assets/apple_domain.msh`. That file is **byte-for-byte
identical to Run 1's mesh**:

```
0cba6a4c2eb0baa64c19dbde072f180b   run_01/code_R1/assets_R1/apple_domain_R1.msh
0cba6a4c2eb0baa64c19dbde072f180b   run_02/code_R2/assets_R2/apple_domain_R2.msh
```

The mesh generator drives element size with a `MathEval` field referencing the
coordinate `z`, but the geometry is built in the **xy plane**, so `z` is
identically zero at every node and the field collapses to the constant `0.015`
everywhere. The advertised polar refinement was never delivered — 150× off the
requested `1e-4`. The only real differences between Runs 1 and 2 are fixed versus
adaptive `dt`, and 1 versus 16 ranks.

*(A second file, `code_R2/apple_domain_R2.msh`, has a different checksum and is
890 KB rather than 930 KB. It is not the one the solver loads and appears to be an
unused leftover.)*

## What it produced

| | |
| :-- | :-- |
| integrated to | `t = 0.55` (39,018 diagnostic samples) |
| final max \|u\| | **9.4 × 10²¹** |
| final max \|ω\| | **6.7 × 10²⁸** |
| published claim | a "reliability threshold" at `t = 0.2095` |

The run continued past total numerical breakdown to `|u| = 10²²` because nothing
in the code could detect it: no NaN guard, no divergence check, and a `dt` clamp
that silently abandons the CFL condition past `|u| ≈ 50`.

The published threshold at `t = 0.2095` is triggered by a **0.665 % change in
velocity across one 25 µs sample**, in a window where neighbouring growth rates
read −56 and +42 and the vorticity is *decreasing*. It is sample-to-sample noise
differentiated over a tiny `dt`, not a physical transition.

## Four further defects, each verified numerically

- Vorticity evaluated as `u_θ/(r + 1e-14)` at DG1 vertices on the axis:
  **1373× error on the exact analytic initial condition**, before any physics.
- The projection step is solved with **no boundary conditions** at all, so the
  corrected velocity satisfies neither no-slip nor the axis conditions.
- The scheme is **Chorin**, not the IPCS the documentation describes.
- The momentum matrix is reassembled only when `dt` changes, but it carries the
  velocity as a coefficient — so it sat **stale for 62.8 %** of the run.

## Files

- `code_R2/` — solver, mesh generator, analysis scripts, tests as they ran
  (the tests reference modules that do not exist and error at collection)
- `results_R2/` — diagnostics CSV, per-rank checkpoints, ParaView media
- `results_R2/media_R2/vortex_blowup_R2.gif` — reused in the main README; it is
  spectacular and it is an artefact

→ **Full analysis with the numbers:** [`docs/findings.md`](../../docs/findings.md)
→ **Corrected implementation:** [`src/`](../../src/)

Do not copy from these files. The corrected versions of everything here live in
`src/`, and the regression tests in `tests/` pin each defect so it cannot return.
