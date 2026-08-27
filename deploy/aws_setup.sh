#!/usr/bin/env bash
#
# aws_setup.sh - Prepare a fresh Ubuntu instance to run the Run 3 convergence study.
#
# Tested against the environment the study was developed in: DOLFINx 0.10.0,
# PETSc 3.25.4 with HYPRE, gmsh 4.15.2, OpenMPI 5.0.x, Python 3.10.
#
# Usage (on the instance):
#     bash aws_setup.sh
#     conda activate fenicsx-env
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-fenicsx-env}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$HOME/miniforge3}"
REPO_DIR="${REPO_DIR:-$HOME/navier-stokes-cusp-study}"

echo "=== Run 3 environment setup ==="
echo "    conda prefix : $MINIFORGE_DIR"
echo "    environment  : $ENV_NAME"
echo "    repo         : $REPO_DIR"
echo

# --- system packages -------------------------------------------------------
# A compiler is NOT optional: FFCx JIT-compiles every variational form at run time.
# Without it the solver fails at the first fem.form() with a cffi VerificationError.
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential git curl tmux

# --- miniforge -------------------------------------------------------------
if [ ! -d "$MINIFORGE_DIR" ]; then
    echo "--- installing miniforge ---"
    curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash /tmp/miniforge.sh -b -p "$MINIFORGE_DIR"
    rm -f /tmp/miniforge.sh
fi
# shellcheck disable=SC1091
source "$MINIFORGE_DIR/etc/profile.d/conda.sh"

# --- solver environment ----------------------------------------------------
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "--- creating conda environment '$ENV_NAME' (this takes a few minutes) ---"
    if [ -f "$REPO_DIR/./environment.yml" ]; then
        conda env create -n "$ENV_NAME" -f "$REPO_DIR/./environment.yml"
    else
        conda create -y -n "$ENV_NAME" -c conda-forge \
            python=3.10 fenics-dolfinx=0.10.0 petsc \
            openmpi gmsh python-gmsh numpy sympy matplotlib h5py pytest
    fi
fi

conda activate "$ENV_NAME"

# --- verification ----------------------------------------------------------
echo
echo "=== verifying the environment ==="
python - <<'PY'
import sys
import dolfinx, ufl, basix, gmsh
from petsc4py import PETSc
v = PETSc.Sys.getVersionInfo()
print(f"  python   {sys.version.split()[0]}")
print(f"  dolfinx  {dolfinx.__version__}")
print(f"  ufl      {ufl.__version__}")
print(f"  basix    {basix.__version__}")
print(f"  petsc    {v['major']}.{v['minor']}.{v['subminor']}")

# HYPRE (BoomerAMG) is what makes the pressure Poisson tractable at this size.
# Probe it on a real SPD tridiagonal matrix: BoomerAMG segfaults outright when
# handed a matrix with no entries, so an empty 4x4 placeholder crashes the check
# rather than answering it.
n = 10
A = PETSc.Mat().createAIJ([n, n], nnz=3)
A.setUp()
for i in range(n):
    A.setValue(i, i, 2.0)
    if i > 0:
        A.setValue(i, i - 1, -1.0)
    if i < n - 1:
        A.setValue(i, i + 1, -1.0)
A.assemble()
pc = PETSc.PC().create(); pc.setOperators(A)
try:
    pc.setType("hypre"); pc.setUp()
    print("  hypre    available")
except Exception as exc:
    print(f"  hypre    NOT AVAILABLE ({exc}); the solver will fall back to GAMG")
PY

echo
echo "  JIT compiler check (FFCx needs a working gcc):"
python - <<'PY'
import ufl
from mpi4py import MPI
import dolfinx
from dolfinx import fem
m = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
V = fem.functionspace(m, ("Lagrange", 2, (3,)))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
fem.form(ufl.inner(u, v) * ufl.dx)
print("  JIT      OK")
PY

echo
echo "=== hardware ==="
echo "  vCPUs    : $(nproc)"
echo "  physical : $(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s}' | tr -d ' ')"
echo "  memory   : $(free -g | awk '/^Mem:/{print $2" GB"}')"
echo
echo "Setup complete. Next:"
echo "    conda activate $ENV_NAME"
echo "    cd $REPO_DIR/."
echo "    python -m pytest tests -q"
echo "    bash deploy/run_aws.sh --benchmark"
