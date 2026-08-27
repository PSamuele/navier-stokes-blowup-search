# Methodology — what the rewrite does, and why

Each choice below is motivated on general grounds: a known class of CFD pitfall,
not a patch for a specific past mistake. The mistakes are catalogued separately in
[`findings.md`](findings.md); this document is meant to be readable by someone who
never looks at those.

---

## Mesh generation — [`src/mesh.py`](../src/mesh.py)

**The problem being solved.** Symbolic element-size fields are written in a
coordinate system that the geometry may not share. A field can evaluate to a
constant, or to something unintended, without gmsh complaining — the mesh is
generated successfully, it is simply not the mesh that was asked for. Nothing in
the output announces this.

**What is done instead:**

- The size field is built from a `Distance` from the two pole points plus a
  `Threshold` ramp. **No coordinate name appears anywhere**, so a field that
  silently references the wrong axis is structurally impossible.
- Boundary sampling is equidistributed against the local target size and clustered
  toward the poles, with the point count derived from `lc_pole`. The polyline
  geometry is never coarser than the mesh it has to carry.
- A dense polyline replaces the Catmull–Rom spline, which can overshoot to `r < 0`
  through sparse, steeply varying points. `r >= 0` is asserted.
- Size bounds follow the arguments instead of being pinned to a hard-coded cap.

**And then the generator measures what gmsh actually produced and raises if the
achieved polar size misses the request.** This is the part worth copying. Run
against the old recipe, it fires immediately.

| requested `lc_pole` | achieved (median cell size at \|z\| > 0.95H) | ratio |
| --: | --: | --: |
| 4.0e-3 | 4.02e-3 | 1.00 |
| 8.0e-3 | 8.02e-3 | 1.00 |
| 5.0e-4 | 5.47e-4 | 1.09 |
| *old recipe: 1.0e-4* | *1.5e-2* | ***150*** |

Guarded by `test_polar_refinement_is_actually_delivered` and
`test_verification_rejects_a_mesh_that_missed_its_target`.

---

## Axis-safe diagnostics — [`src/diagnostics.py`](../src/diagnostics.py)

**The problem being solved.** In cylindrical coordinates the azimuthal vorticity
carries a `u_θ/r` term with a *removable* singularity at `r = 0`: the exact field
is regular there because `u_θ` vanishes like `r`. Numerically it is not removable,
because `u_θ` is only zero to solver tolerance. Regularising with `u_θ/(r + ε)`
converts a well-posed limit into a division of noise by `ε` — an amplification of
up to `1/ε`, applied exactly at the points a discontinuous element samples.

**What is done instead:**

- Every point-sampled quantity is evaluated at **interior quadrature points**,
  where `r > 0` strictly. A Gauss point of a triangle has all barycentric
  coordinates positive, so it can only sit on the axis if all three vertices do —
  which cannot happen for a non-degenerate cell. **No epsilon is needed and none
  is used.**
- `Diagnostics.audit()` asserts the property on the actual mesh rather than
  assuming it, and `compute()` refuses to run if the assertion fails.
- Integral quantities use the axisymmetric measure `r dx`, against which the `1/r`
  terms cancel analytically.
- Maxima are taken consistently with the element order, rather than at nodes where
  a P2 field does not attain its true L∞.

**Validation against ground truth.** On the exact analytic initial condition the
diagnostic returns **352.59** against an analytic **351.64** — 0.27 %. The same
formula written the old way returns **482,601** on the same field, a factor of
1373. Both are in the test suite, the second as
`test_the_r2_formula_reproduces_the_bug`, so the diagnosis is demonstrated rather
than asserted.

The module also computes the quantities the theory is actually about, which the
earlier code never did: enstrophy, circulation `Γ = r·u_θ`, kinetic energy, both
the strong and weak divergence residual, the location of the vorticity maximum,
and the running **BKM integral** `∫‖ω‖_∞ dt` — the quantity the Beale–Kato–Majda
criterion is stated in terms of, and more informative than eyeballing `1/‖ω‖`.

---

## Checked linear solves — [`src/solver.py`](../src/solver.py)

**The problem being solved.** PETSc does not raise an exception when a
preconditioner breaks down. It sets a negative converged reason, leaves whatever
it has in the solution vector — often `Inf` — and returns normally. A solver loop
that does not inspect that reason will propagate the failure into every subsequent
step, producing a long run full of numbers rather than an error.

**This happened for real during the rewrite,** which is why it is here rather than
in a list of hypothetical risks. On the production mesh the r-weighted mass matrix
has diagonal entries down to **6.7 × 10⁻¹⁵** in the cells touching the axis. ILU
inside block Jacobi fails there, and the projection returned reason **−11
(`DIVERGED_PC_FAILED`)** with an `Inf` velocity field.

**What is done instead:** the projection uses SOR, which cannot break down in this
way, and `check_ksp` aborts on any negative converged reason. Guarded by
`test_check_ksp_raises_on_a_diverged_solve`.

---

## Cell-local CFL — [`src/solver.py`](../src/solver.py)

**The problem being solved.** The usual global estimate `min(h) / max|u|` pairs
the smallest cell in the mesh with the fastest fluid anywhere in the domain, even
when they are at opposite ends of it. On a strongly graded mesh — which any
resolution study of a local feature will have — that is needlessly pessimistic by
a large factor.

**What is done instead:** `min over cells of (h_cell / |u|_cell)`, the correct
local condition. Measured on these meshes it gives a **6.1× larger step at the
same true CFL**: the same run took 15 steps instead of 92, 10.0 s instead of
52.7 s.

It also produces a `dt` that scales as `h` — measured ratio **1.98** between two
grids differing by exactly 2 — which is what a convergence study needs in order for
the time discretisation to refine along with the space discretisation.

The limiting cell turns out to sit in the jet core near the equator, not at the
poles. That is an early hint of the resolution mismatch documented in
[`convergence.md`](convergence.md).

---

## The energy guard

**The problem being solved.** A simulation can stop being physical without
violating any of the conditions a solver normally checks. Loss of spatial
resolution is the common case: the CFL number stays exactly on target, no solve
diverges, no value is `NaN`, and the answer is nonetheless wrong. A velocity
threshold requires guessing a scale; a CFL guard cannot see it at all.

**What is done instead.** In a closed domain with no-slip walls and no body force,
the exact solution obeys

```
dE/dt = −2ν ∫ |D(u)|² ≤ 0
```

strictly. So **any growth in kinetic energy is numerical by definition.** This
needs no arbitrary constant and no knowledge of the flow. The guard stops the run
at the first sample more than 1 % above the running minimum, and reports the
energy minimum as the reliable horizon.

**That distinction turned out to matter.** Running the validation grids to
`T = 0.55`:

| grid | h_pole | cells | energy minimum | guard fires | max\|u\| just after |
| :-- | --: | --: | --: | --: | --: |
| coarse | 1.37e-2 | 3,811 | t = 0.1223 | **t = 0.1262** | 25.5 → 480 |
| medium | 8.02e-3 | 14,780 | t = 0.1502 | **t = 0.1522** | — |

The coarse sequence past its minimum reads +0.29 %, +1.37 %, +4.23 %, +14.95 %,
+47.7 %, +278.9 % — **with the CFL number sitting exactly on its 0.5 target the
whole time.** The breakdown is loss of spatial resolution as the ring compresses,
not a time-step instability, so neither a CFL guard nor a velocity threshold
catches the onset.

Guarded by `test_energy_guard_stops_an_unphysical_run`.

**Why this is the most transferable idea here.** It is a conservation law the
discretisation is *not* constructed to satisfy exactly, used as an independent
check on the discretisation. Any closed system with a monotone invariant admits
the same construction, and it costs one reduction per sample.

---

## Convergence machinery — [`scripts/`](../scripts/)

`run_convergence.py` generates three grids with **all lengths scaled by the same
factor**, so a single `h` identifies each level — a precondition for Richardson
extrapolation to be defined at all. It records the *achieved* sizes and the
effective ratio, runs the levels on a common output time grid so they can be
compared sample for sample, is resumable, and has a `--benchmark` mode that
projects wall time, cost and memory before an instance is committed.

`analyze_convergence.py` computes the observed order with the **ASME V&V 20 /
Roache procedure for a non-uniform refinement ratio**. This matters: gmsh does not
land exactly on the requested factor of two — the validation study measured **1.71**
between the two coarsest grids and 1.99 between the two finest — and forcing a
nominal ratio into the formula biases the order and everything derived from it.

It then reports Richardson extrapolation to `h → 0`, the **Grid Convergence Index**
(Roache, safety factor 1.25) as the error bar, and the fraction of sample times at
which the three grids are actually monotone — which is the precondition for any of
it to mean anything. Where the observed order is far from the formal order of the
scheme, or the monotone fraction is low, the report **says the extrapolated value
is an artefact** rather than quoting it as a better answer.

Each level is truncated at its `t_at_energy_min` rather than at its last written
row: the samples between the energy minimum and the stop are already contaminated,
and feeding them to the extrapolation would import exactly the unphysical tail the
guard exists to exclude.

---

## Verification performed

```bash
conda activate fenicsx-env && python -m pytest tests -q
```

**50 tests, all passing**, about 40 seconds. The ones that encode findings:

| test | guards |
| :-- | :-- |
| `test_polar_refinement_is_actually_delivered` | A1 — fails by 150× against the old recipe |
| `test_verification_rejects_a_mesh_that_missed_its_target` | the mesh guard itself fires |
| `test_vorticity_matches_the_analytic_initial_condition` | B1 — 352.59 against an analytic 351.64 |
| `test_the_r2_formula_reproduces_the_bug` | confirms the old recipe really does explode on axis noise |
| `test_boundary_conditions_hold_after_the_projection` | B2 — residual < 1e-10 |
| `test_solver_stops_instead_of_violating_cfl` | B5 — stops rather than clamping |
| `test_energy_guard_stops_an_unphysical_run` | loss of resolution, which CFL cannot see |
| `test_check_ksp_raises_on_a_diverged_solve` | silent `DIVERGED_PC_FAILED` |
| `test_domain_is_a_cone_not_a_cusp_at_the_poles` | D1 |

Beyond the suite:

- The domain volume of revolution computes to **4.284858** against an exact
  **4.285398** — 1.3e-4 relative, i.e. the polygonal boundary error. This
  validates the r-weighted quadrature the whole diagnostics module rests on.
- Results are identical on 1 and 4 MPI ranks, and were confirmed identical on 8
  and 16 ranks during the production benchmark (`dt` 4.975e-05, max\|u\| 25.202 on
  both).
- The full three-level study was run end to end at validation scale before any
  cloud time was spent — see [`../results/validation_local/`](../results/validation_local/).

**What is not verified.** There is no Method of Manufactured Solutions here, so the
*formal order of accuracy of this implementation* has never been established
independently. The observed orders in [`convergence.md`](convergence.md) therefore
have no reference to be compared against: `p = 1.39` for kinetic energy is
plausible for a first-order pressure splitting with second-order space, but
"plausible" is not "verified". This is the single largest gap in the verification,
and it is cheap to close — MMS runs locally in minutes and needs no cloud time.
