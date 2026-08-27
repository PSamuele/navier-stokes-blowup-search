# Run 3 - mesh convergence report

Common time window: t in [0.00024, 0.26310], 200 sample points.

## Grids

| level | cells | velocity DOFs | h_pole (asked) | h_pole (achieved) | h_equator (achieved) | final t | trustworthy to | terminated |
| :-- | --: | --: | --: | --: | --: | --: | --: | :-- |
| coarse | 47134 | 287205 | 2.000e-03 | 2.137e-03 | 3.206e-02 | 0.26804 | 0.26310 | energy_growth |
| medium | 188462 | 1139571 | 1.000e-03 | 1.077e-03 | 1.566e-02 | 0.55000 | 0.55000 | completed |
| fine | 752803 | 4534410 | 5.000e-04 | 5.467e-04 | 7.777e-03 | 0.55000 | 0.55000 | completed |

Refinement ratios measured from the delivered meshes (polar element size): medium/fine = 1.9692, coarse/medium = 1.9848 (nominal 2.0). The observed order is solved for these two ratios rather than assuming a constant one, so the deviation does not bias the result.

## Convergence of the target quantities

`p` is the observed order, `GCI` the fine-grid Grid Convergence Index (Roache, safety factor 1.25): the band within which the converged value is expected to lie. `monotone` is the fraction of sample times where the three grids form a monotone sequence, which is a precondition for the extrapolation to mean anything.

| quantity | fine-grid value | Richardson h->0 | observed p (median) | GCI (median) | monotone |
| :-- | --: | --: | --: | --: | --: |
| max \|omega\| | 2156.47 | 2155.35 | 1.15 | 6.93% | 74% |
| enstrophy | 91231.6 | 91372.1 | 1.82 | 0.40% | 52% |
| kinetic energy | 152.337 | 152.491 | 1.39 | 0.16% | 100% |
| max \|Gamma\| = r u_theta | 5.40361 | 5.43324 | 1.00 | 0.86% | 92% |
| BKM integral of \|\|omega\|\|_inf dt | 636.183 | 637.514 | 0.98 | 8.85% | 92% |

## How to read this

- A median `p` near the formal order of the scheme (1 for the pressure splitting, up to 2 for the spatial discretisation) means the three grids are in the asymptotic range and the extrapolated column is trustworthy.
- A `p` far from that, or a low `monotone` fraction, means they are not. In that case the extrapolated value is **not** a better answer than the fine grid; it is an artefact. Refine further before quoting a number.
- The GCI is the number to publish as the uncertainty on the fine-grid result.

## Provenance

- **coarse**: 1424 steps, 308 s wall, 8 ranks, scheme `ipcs`, dolfinx 0.10.0, terminated `energy_growth` (kinetic energy rose 1.21% above its running minimum (146.307 against 144.562 at t = 0.263101) by t = 0.268045. Energy cannot increase in this configuration, so the solution has stopped being physical -- almost always because the grid can no longer resolve the flow. Refine, or treat t = 0.263101 as the reliable horizon.)
- **medium**: 6780 steps, 5767 s wall, 8 ranks, scheme `ipcs`, dolfinx 0.10.0, terminated `completed`
- **fine**: 14216 steps, 50374 s wall, 8 ranks, scheme `ipcs`, dolfinx 0.10.0, terminated `completed`
