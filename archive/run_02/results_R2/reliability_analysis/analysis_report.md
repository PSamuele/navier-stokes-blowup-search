# Reliability Analysis of Run 2

This folder contains the automated analysis to determine the exact point where the mathematical reliability of the simulation breaks down due to the FEniCSx DOF permutation bug and the subsequent von Neumann instability.

## Methodology

A physical finite-time singularity follows a power-law asymptotic scaling (e.g., $1/\omega \propto (T^* - t)$). In contrast, a von Neumann instability (CFL violation) typically manifests as exponential error growth, which appears as a sudden spike in the logarithmic growth rate of the velocity field $d(\log V)/dt$.

By analyzing the `blowup_data_R2.csv` logs, we scanned for the first time step where the exponential growth rate exceeds a physically plausible threshold (e.g., $100$ decades per second), indicating that the numerical error has completely overtaken the physical advection-diffusion dynamics.

## Findings

The script identified the point of reliability failure at:
- **Time ($t$):** `0.2095 s`
- **Max Velocity:** `2.03e+01 m/s`
- **Max Vorticity:** `3.32e+08 1/s`

**Conclusion:**
Up to $t pprox 0.2095 s$, the simulation reliably captures the extreme physical accumulation of enstrophy forced by the cuspidal geometry. Beyond this point, the CFL violation causes the kinetic energy to grow exponentially due to numerical artifacts rather than Navier-Stokes mechanics. All data slices and projections after $t = 0.2095 s$ should be considered mathematically invalid.
