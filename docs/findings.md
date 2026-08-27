# Findings — how Runs 1 and 2 produced confident wrong answers

A catalogue of every defect found in the first two attempts, with the measurement
that establishes each one. Nothing here is inferred from reading the code alone:
each entry is something that was run and compared against a known answer.

The code these findings describe is preserved unmodified under
[`archive/`](../archive/). The corrected implementation is [`src/`](../src/), and
[`tests/`](../tests/) contains a regression test for every item marked **bold**.

---

## 1. The three that invalidate the published result

### A1 — The mesh was never refined

`src_R2/mesh_R2.py` drives the element size with

```python
gmsh.model.mesh.field.setString(1, "F", "0.0001 + 0.0149 * (1.0 - abs(z)/2.0)^3")
```

The 2-D geometry is built in the **xy plane** — points are added as
`addPoint(r, z, 0.0)` — so the gmsh coordinate `z` is identically zero at every
node. The field therefore evaluates to the constant `0.0001 + 0.0149 = 0.015`
everywhere.

Measured on the shipped mesh:

| | value |
| :-- | :-- |
| node `z` range | `0.0` to `0.0` — confirms the field argument is always zero |
| median cell size, whole domain | `0.015` |
| median cell size, \|z\| > 1.95 (the "cusp") | `0.015` |
| cells in \|z\| > 1.95 | **12** |
| requested polar size | `0.0001` — never delivered, off by a factor of **150** |

`Mesh.MeshSizeFromPoints = 0` disabled the per-point `lc` grading the function also
computed, so nothing else could rescue it.

**Runs 1 and 2 used the same mesh file, byte for byte:**

```
0cba6a4c2eb0baa64c19dbde072f180b   run_01/code_R1/assets_R1/apple_domain_R1.msh
0cba6a4c2eb0baa64c19dbde072f180b   run_02/code_R2/assets_R2/apple_domain_R2.msh
```

10,029 nodes, 19,485 triangles, and `main_R2.py` loads exactly that path
(`mesh_file="assets/apple_domain.msh"`). The only real differences between the two
runs are fixed versus adaptive `dt`, and 1 versus 16 MPI ranks — not resolution.

*A second file, `run_02/code_R2/apple_domain_R2.msh`, has a different checksum and
is 890 KB against 930 KB. The solver does not load it; it appears to be an unused
leftover.*

### B1 — The quantity measured was not vorticity

`main_R2.py` computes

```python
r_safe  = r + 1e-14
omega_z = u_n[2].dx(0) + u_n[2] / r_safe
W_DG    = fem.functionspace(domain, ("DG", 1))
vort_expr = fem.Expression(vorticity_sq, W_DG.element.interpolation_points)
```

The interpolation points of a DG1 element are the **cell vertices**, and every cell
touching the symmetry axis has vertices at exactly `r = 0`. There `u_θ` should be
zero but is only zero to solver tolerance — and Run 2 never re-applied the boundary
conditions after its projection step (B2), so it was not even that. The quotient
divides a round-off-level number by `1e-14`, amplifying it by `10¹⁴`.

| | max \|ω\| |
| :-- | --: |
| exact analytic initial condition | **351.64** |
| first row of `blowup_data_R2.csv` (t ≈ 2e-6) | **482,601** |
| ratio | **1373×, before a single physical event** |

It gets worse further in. At `t = 0.2095` the logged max velocity is **20.3**
(initial: 25.2, i.e. the flow has barely changed) while the logged max vorticity is
**3.3 × 10⁸**. On a mesh with `h = 0.015` the largest representable vorticity is
about `u/h ≈ 1.4 × 10³`. The reported value exceeds it by five orders of magnitude.

The corrected diagnostic, on the same analytic field, returns **352.59** against the
exact 351.64 — a 0.27 % error.

### C1 — The reliability threshold is noise

`find_instability.py` flags the first sample where `d(log₁₀ V)/dt > 100`. At the
sample it selects:

| sample | t | max velocity | growth rate | max vorticity |
| :-- | --: | --: | --: | --: |
| n−2 | 0.209455 | 20.1435 | −56.5 | 3.79e8 |
| n−1 | 0.209480 | 20.1921 | +42.2 | 3.55e8 |
| **n** | **0.209505** | **20.3263** | **+116.5** | **3.32e8** |
| n+1 | 0.209530 | 20.4509 | +108.0 | 3.09e8 |

The trigger is a **0.665 % change in velocity across one 25 µs sample**, in a window
where the neighbouring growth rates read −56 and +42 and the vorticity is
*decreasing*. It is sample-to-sample noise differentiated over a tiny `dt`, not a
physical transition.

The genuine numerical breakdown happens between `t ≈ 0.30` (max\|u\| = 32.6, still
sane) and `t ≈ 0.40` (max\|u\| = 3.5 × 10⁹). The run continued to
**max\|u\| = 9.4 × 10²¹** at `t = 0.55`, across 39,018 recorded samples, without
stopping.

---

## 2. Full catalogue

### A — Mesh generation (`src_R2/mesh_R2.py`)

| # | Finding |
| :-- | :-- |
| **A1** | `MathEval` field references `z`, but the geometry lies in the xy plane, so the field is the constant 0.015. **The advertised polar refinement never existed.** |
| A2 | The per-point `lc` grading is dead code: `Mesh.MeshSizeFromPoints = 0`. |
| A3 | The boundary is sampled at 400 **uniformly spaced** z values — chord ≈ 0.01 near the pole against a target `h` of 1e-4. The geometry was 100× coarser than the mesh it was meant to carry. |
| A4 | `addSpline` (Catmull–Rom) through sparse, steeply varying points can overshoot to `r < 0`. |
| A5 | `MeshSizeMax = 0.015` is hard-coded, capping every mesh identically and making a resolution sweep impossible. |
| A6 | No `output_file` / `verbosity` / `gui` parameters and no return value, although `tests/test_mesh.py` and `scripts_R2/interpolate_mesh_R2.py` both call it as if it had them. `H` is hard-coded as `2.0` inside the field string. |

### B — Solver (`main_R2.py`)

| # | Finding |
| :-- | :-- |
| **B1** | Vorticity evaluated as `u_θ/(r + 1e-14)` at DG1 vertices, which sit on the axis. Invalidates every vorticity number the project reported. |
| **B2** | Step 3 (the projection) is solved with **no boundary conditions** — no `apply_lifting`, no `set_bc`. The corrected velocity satisfies neither no-slip on the wall nor `u_r = u_θ = 0` on the axis. This is what feeds B1. |
| **B3** | The scheme is **Chorin**, not IPCS: the predictor has no `grad(p^n)` term and the pressure is never accumulated. The documentation described IPCS. |
| **B4** | `dx_min = 0.0001` is hard-coded and unrelated to the mesh in use, whose true minimum edge is `1.633e-3` — 16× larger. |
| **B5** | `dt` is clamped by `max(min(dt_new, 0.005), 1e-6)`. Past \|u\| ≈ 50 the CFL condition is silently abandoned. **This is the mechanism behind the reported "blow-up".** No NaN or divergence guard: the run continued to \|u\| = 10²². |
| B6 | Pressure pinned by a Dirichlet condition at the interior point (0,0). A point constraint is not H¹-admissible in 2-D and degrades the AMG preconditioner. |
| B7 | `try: setType(HYPRE) except: setType(GAMG)` — `PCSetType` does not fail for a missing package (the failure appears later in `PCSetUp`), so the fallback can never fire. The bare `except` also swallows `KeyboardInterrupt`. |
| B8 | On restart `XDMFFile(..., "w")` truncates the existing series; `un_file`/`pn_file` are computed and never used; `restart` is hard-coded `False`; per-rank checkpoints record nothing about the rank count or mesh they belong to. |
| **B9** | No CLI at all. `run_solver()` takes no arguments and hard-codes `out_dir = "results"`, so the documented command — `python main_R2.py --out_dir … --mesh …` — silently ignores both flags. |
| B10 | Every rank prints and writes checkpoints every 10 steps; XDMF shares the diagnostic cadence (hence a 9.4 GB HDF5 file). With adaptive `dt`, "every 10 steps" is non-uniform sampling in time, so runs at different resolutions cannot be compared. |
| B11 | Maxima are taken over DG1 nodal values; for a P2 velocity the true L∞ is not attained at DG1 nodes, systematically under-reporting the very quantity being studied. |
| B12 | The initial condition violates no-slip (the ring Gaussians do not vanish at the wall, ≈2e-3 of peak) and is never projected onto a discretely divergence-free field. |
| **B13** | `A1` is reassembled **only when `dt` changes** — but `a1` carries `u_n` as a coefficient through the linearised advection, so it must be reassembled every step. `dt` sat within `np.isclose`'s tolerance of its 1e-6 floor for **62.8 %** of the recorded samples, so for most of the run the momentum operator was frozen at a stale velocity field. |

### C — Analysis and tooling

| # | Finding |
| :-- | :-- |
| **C1** | The published reliability threshold `t = 0.2095` is triggered by a 0.665 % velocity wiggle across one sample (see §1). |
| C2 | `interpolate_mesh_R2.py` imports `src.mesh` (the module is `src_R2/mesh_R2.py`) and calls `generate_mesh(output_file=…, verbosity=…, gui=…)` with parameters that do not exist. It also assumes DOLFINx orders P2 DOFs as `[gmsh vertices…, gmsh edge midpoints…]`, which is **not** the DOLFINx layout — DOLFINx reorders and repartitions — so any restart built this way would be scrambled. `est_step = t*681180` is a magic constant, and the restart pressure is set to zero. |
| C3 | `analyze_output_R2.py` calls `interpolation_points()` as a method; in DOLFINx 0.10 it is a property, so the script cannot run. It also swallows the DOLFINx import with a bare `except: pass` and then uses `MPI` unconditionally. |
| C4 | `analyze_blowup_R2.py` reads `blowup_data_RUN_AWS.csv`, which does not exist. |
| C5 | The repository's `tests/` reference `src.mesh`, `main`, `scripts.interpolate_mesh` and `runs/run_02_standard_aws/` — none of which exist. **Every test errors at collection.** They do, however, describe the intended API, which the rewrite implements. |
| C6 | `conditions_R2.py` is dead scaffolding that divides by `r` with no guard and is never imported. |

### D — Model and documentation

| # | Finding |
| :-- | :-- |
| **D1** | **The domain is not a cusp.** Near `z = H`, `f(z) = R₀ cos(πz/2H) exp(−kz²)` vanishes *linearly*: `f'(H) = −(πR₀/2H)exp(−kH²) = −0.106`. That is a **cone of half-angle 6.07°**. The `exp(−kz²)` factor is smooth and non-zero at `z = H` — it rescales the body but contributes nothing to the narrowing rate at the pole. So "narrows exponentially near the stagnation point z = H" and "the stiffness parameter k ≫ 1 dictates the severity of the polar cusp" are both wrong, and the study runs with `k = 0.5`, not `k ≫ 1`. A genuine cusp needs `f` vanishing faster than linearly, e.g. `f ∝ (H−z)^α` with `α > 1`, or `exp(−c/(H−z))`. |
| D2 | The Run 1 versus Run 2 resolution contrast describes a difference that does not exist (see A1). |
| D3 | The documentation described IPCS (the code implements Chorin, B3) and "BiCGSTAB with an ILU preconditioner" (the code uses BJACOBI). |
| D4 | Image and command paths pointed at `runs/run_01_coarse/…` and `runs/run_02_aws/…`; the directories are named `run_01` and `run_02`, so every image link was broken. |
| D5 | None of the quantities the theory section is about — circulation `Γ = r·u_θ`, the stretching term `Σ = r⁻³ ∂_z(Γ²)`, enstrophy — was ever computed by the solver. |

---

## 3. The pattern

Sorting the catalogue by how it would have been caught:

| how it would have been found | findings |
| :-- | :-- |
| **Comparing a delivered artefact against what was requested** | A1, A3, A5, B4 |
| **Comparing a computed quantity against a known exact answer** | B1, B11, D1 |
| **Checking a return code the library does not raise on** | B7, and the `DIVERGED_PC_FAILED` case found during the rewrite |
| **Checking a conservation law that must hold** | B5, B13 |
| Reading the code | A2, A4, A6, B6, B8, B9, B10, B12, C2–C6, D2–D5 |

The first four rows are the ones that mattered — they are the findings that
invalidate results rather than merely annoy. **None of them is visible by reading
the source.** Every one requires running something and comparing it to an
independently known answer.

That is the argument for the verification apparatus in
[`methodology.md`](methodology.md): the defects that change your conclusions are
precisely the ones code review cannot see.
