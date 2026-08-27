# How this was built

Writing the code was the cheap part. That is the whole reason this repository is
organised around verification rather than around the solver.

I chose the problem and the geometry, specified the physics and the parameter
point, sized and paid for the infrastructure, ran the three campaigns, and read the
output. AI assistance did a large share of the implementation — and produced, in
the first two attempts, all five silent failures catalogued in
[`findings.md`](findings.md). It also did much of the forensic work later, under
direction, and wrote the tests that pin each finding.

What closed the loop was knowing what the numbers should look like. Run 2 reported
a maximum velocity of 20 beside a maximum vorticity of 10⁸. Nothing flagged that
pair, and it is only obviously wrong to someone who has spent time with real flow
fields. Deciding it was wrong, deciding to spend real money re-running rather than
publishing, and choosing which invariants were worth checking in the first place —
those are the judgements the apparatus was built around, and they are the part no
amount of generation supplies.

---

## Sparring, not delegation

The work ran as a continuous argument, and it ran in both directions. Two exchanges
stand for the rest; both are visible in the repository, so neither has to be taken
on trust.

**The geometry was being apologised for, and should have been argued.** The domain
is a 6.07° cone rather than the cusp it was believed to be, and the write-up
treated that as a shortfall to disclose. I rejected that framing: a modelling
choice that survives three campaigns needs a justification, not a confession. Working
the question through inverted the conclusion entirely. A true cusp is **not a
Lipschitz domain**, so the standard well-posedness theory does not apply to it; a
loss of regularity observed at the tip could not be attributed to the fluid rather
than to the boundary's own geometric singularity; and no sequence of grids with a
common refinement factor exists there, which leaves the Richardson and GCI
machinery undefined. The cone is the domain in which a negative answer means
something. That is now [the argument in the
README](../README.md#why-a-cone-is-the-right-domain-and-a-cusp-would-not-be), and
it changed what the study claims rather than merely how it reads.

**The literature was being read to fit the result.** The null result had been
framed as confirming that constant-viscosity Navier–Stokes cannot go singular —
tidy, and convenient. I asked for every citation to be checked at source rather
than summarised from memory. It was wrong: Hou (2023) reports potentially singular
behaviour with *uniform* viscosity, and the regularising effect belongs to a
different paper under different initial conditions. The conclusion was rewritten
around what actually separates the regimes — initial data tuned to a self-similar
profile and dynamic rescaling, not the value of ν. The honest version is weaker as
a headline and considerably harder to attack.

Neither exchange would have produced anything on its own. The search, the
derivation and the rewriting happened at a speed I could not have matched by hand;
the judgement that the framing was wrong in the first place is not something the
tool volunteered.

## The part worth generalising

None of the five failures in [`findings.md`](findings.md) raised an error. None of
them would have been caught by reading the code. They were:

- a size field that evaluated to a constant because the geometry lived in a
  different plane than the field's variable;
- a diagnostic that divided round-off by `1e-14` at points where the exact answer
  is a removable limit;
- a threshold detector firing on differentiated sampling noise;
- a matrix reassembled on the wrong condition;
- a projection step missing its boundary conditions.

Every one produces plausible-looking numbers. Several produce *dramatic*
plausible-looking numbers, which is worse, because a spectacular result invites
less scepticism than a boring one.

What caught them was, in every case, **comparison against something independently
known**:

| defect | what it was compared against |
| :-- | :-- |
| mesh field degeneracy | the element size actually delivered, versus the size requested |
| vorticity amplification | the exact analytic initial condition |
| stale momentum matrix | a conservation law that must hold |
| clamped time step | the CFL condition, checked rather than enforced by clamping |
| unchecked solves | the solver's own converged reason |

That is the whole method. It is not sophisticated. It is just work that nobody does
when writing code is expensive, because the code itself absorbs all the effort.

**When code generation gets cheap, verification is the only remaining cost, and
skipping it produces wrong answers faster than before.** That is the argument this
repository exists to make concretely rather than rhetorically.

One asymmetry is worth naming, because it explains why the human half is not
interchangeable. Run 2 reported a maximum velocity of 20 alongside a maximum
vorticity of 10⁸. Nothing flagged that pair; it is only obviously wrong to someone
who has looked at enough real flow fields to know what those two numbers do
together. That came from working with Ansys and OpenFOAM, from validating CFD
against measured data, and from a manometer bank on a Venturi tube. Domain
knowledge is what turns a plausible number into a suspicious one, and no amount of
generation substitutes for it.

---

## A category distinction

There is an established use of AI in exactly this research area, and it is a
different thing from what happened here.

Wang, Lai, Gómez-Serrano and Buckmaster (Phys. Rev. Lett. **130**, 244002, 2023)
used **physics-informed neural networks as the numerical method**: the network
represents the self-similar blow-up profile and is trained against the PDE
residual. The AI *is* the solver, and what validates it is the residual it
achieves.

Here, an assistant wrote conventional finite-element code in FEniCSx. The AI
*wrote* the solver, and what validates it is a test suite and a convergence study.

Both are legitimate. They are not the same claim, they carry different risks, and
they are validated by different means. Presenting this work as though it belonged
to the first category would be a category error — and one a reader in this field
would spot immediately.

---

## What this does not excuse

Being transparent about AI use is not a substitute for verification, and this
document should not be read as one. The relevant defences of the work are:

- **50 regression tests**, one per finding, in [`tests/`](../tests/)
- **Comparison against exact answers**: analytic vorticity to 0.27 %, domain
  volume to 1.3e-4
- **Rank-independence**: identical results on 1, 4, 8 and 16 MPI ranks
- **A convergence study that reports its own failures** — three of five quantities
  are flagged as not in the asymptotic range, with the monotone fraction that
  says so

And one honest gap, stated in [`methodology.md`](methodology.md): there is no
Method of Manufactured Solutions, so the formal order of accuracy of this
implementation has never been established independently. The observed orders have
no reference to be judged against. That is the largest remaining hole in the
verification, and it is not one that transparency about tooling fills.
