# Run 1 — archived, superseded, **do not run**

The first attempt: a local, single-core solve of the axisymmetric Navier–Stokes
equations in the compressive "apple" domain. Preserved here as **evidence**, not
as software. The code and the recorded data are exactly as they ran.

## What it produced

| | |
| :-- | :-- |
| integrated to | `t = 0.55` (111 diagnostic samples) |
| final max \|u\| | **11.85** — plausible |
| final max \|ω\| | **3.12 × 10⁶** — not plausible on a mesh with `h = 0.015` |

That pair is the tell. With `h = 0.015` the largest vorticity the discretisation
can represent is roughly `u/h ≈ 10³`. The reported value exceeds it by three
orders of magnitude while the velocity stays ordinary, so the vorticity number is
measuring the diagnostic, not the flow.

## Why it cannot be repaired from this data

- **The vorticity diagnostic divides by `r + 1e-14` at cell vertices that sit
  exactly on the symmetry axis**, amplifying round-off by up to `10¹⁴`.
- **Its mesh is byte-for-byte identical to Run 2's** (md5
  `0cba6a4c2eb0baa64c19dbde072f180b`). The refinement Run 2 was supposed to add
  over Run 1 never existed — see the note in [`../run_02/`](../run_02/).
- The requested polar element size was never delivered: `0.015` against a target
  of `1e-4`, off by a factor of 150.

## Files

- `code_R1/` — solver, mesh generator, initial conditions as they ran
- `results_R1/` — diagnostics CSV, checkpoints, ParaView media
- `results_R1/media_R1/vortex_blowup_R1.gif` — reused in the main README

→ **Full analysis with the numbers:** [`docs/findings.md`](../../docs/findings.md)
→ **Corrected implementation:** [`src/`](../../src/)

Do not copy from these files. The corrected versions of everything here live in
`src/`, and the regression tests in `tests/` pin each defect so it cannot return.
