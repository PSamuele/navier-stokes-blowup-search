# Run 3 - mesh convergence report

Common time window: t in [0.00055, 0.02000], 200 sample points.

## Grids

| level | cells | velocity DOFs | h_pole (asked) | h_pole (achieved) | h_equator (achieved) | final t | trustworthy to | terminated |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| coarse | 3811 | 23922 | 1.600e-02 | 1.374e-02 | 6.869e-02 | 0.02000 | 0.02000 | completed |
| medium | 14780 | 90609 | 8.000e-03 | 8.024e-03 | 3.448e-02 | 0.02000 | 0.02000 | completed |
| fine | 59298 | 359643 | 4.000e-03 | 4.031e-03 | 1.688e-02 | 0.02000 | 0.02000 | completed |

Refinement ratios measured from the delivered meshes (polar element size): medium/fine = 1.9905, coarse/medium = 1.7130 (nominal 2.0). The observed order is solved for these two ratios rather than assuming a constant one, so the deviation does not bias the result.

## Convergence of the target quantities

`p` is the observed order, `GCI` the fine-grid Grid Convergence Index (Roache, safety factor 1.25): the band within which the converged value is expected to lie. `monotone` is the fraction of sample times where the three grids form a monotone sequence, which is a precondition for the extrapolation to mean anything.

| quantity | fine-grid value | Richardson h->0 | observed p (median) | GCI (median) | monotone |
| :-- | --: | --: | --: | --: | --: |
| max \|omega\| | 744.435 | 919.596 | 1.21 | 10.21% | 66% |
| enstrophy | 21044.8 | 21044.8 | 3.33 | 0.02% | 52% |
| kinetic energy | 175.032 | 175.125 | 2.48 | 0.04% | 100% |
| max \|Gamma\| = r u_theta | 5.72212 | 5.7225 | 2.74 | 0.03% | 65% |
| BKM integral of \|\|omega\|\|_inf dt | 7.29846 | 7.29842 | 1.18 | 4.97% | 74% |

## How to read this

- A median `p` near the formal order of the scheme (1 for the pressure splitting, up to 2 for the spatial discretisation) means the three grids are in the asymptotic range and the extrapolated column is trustworthy.
- A `p` far from that, or a low `monotone` fraction, means they are not. In that case the extrapolated value is **not** a better answer than the fine grid; it is an artefact. Refine further before quoting a number.
- The GCI is the number to publish as the uncertainty on the fine-grid result.

## Provenance

- **coarse**: 35 steps, 3 s wall, 4 ranks, scheme `ipcs`, dolfinx 0.10.0, terminated `completed`
- **medium**: 75 steps, 23 s wall, 4 ranks, scheme `ipcs`, dolfinx 0.10.0, terminated `completed`
- **fine**: 165 steps, 190 s wall, 4 ranks, scheme `ipcs`, dolfinx 0.10.0, terminated `completed`
