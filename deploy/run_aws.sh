#!/usr/bin/env bash
#
# run_aws.sh - Launch the Run 3 convergence study on the instance.
#
#   bash run_aws.sh --benchmark     measure s/step and project cost, then stop
#   bash run_aws.sh --ranks-sweep   time the coarse level at 8 and 16 ranks
#   bash run_aws.sh                 run the full study (use inside tmux)
#   bash run_aws.sh --resume        continue an interrupted study
#
# The study is resumable: a level whose status.json says "completed" is skipped,
# so re-running after an interruption picks up where it stopped.
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN03="$(dirname "$HERE")"
CODE="$RUN03"
CONFIG="${CONFIG:-$CODE/configs/aws_production.json}"
OUT_ROOT="${OUT_ROOT:-$RUN03/results/convergence_aws}"
NP="${NP:-8}"

cd "$CODE"

case "${1:-}" in
  --benchmark)
    echo "=== benchmarking coarse and medium (the fine level needs ~9 GB) ==="
    python scripts/run_convergence.py \
        --config "$CONFIG" --out_root "$OUT_ROOT" --np "$NP" \
        --levels coarse medium --benchmark
    echo
    echo "The fine level costs roughly 8x the medium level: 4x the cells and"
    echo "2x the steps. Multiply the medium row by 8 for the fine estimate."
    ;;

  --ranks-sweep)
    # c6i.4xlarge exposes 16 vCPUs = 8 physical cores plus hyperthreading.
    # For a memory-bandwidth-bound FEM solve, 16 ranks is not automatically
    # faster than 8. Measure instead of assuming.
    echo "=== rank sweep on the coarse level ==="
    python scripts/run_convergence.py --config "$CONFIG" \
        --out_root "$OUT_ROOT" --levels coarse --meshes_only
    MESH="$OUT_ROOT/meshes/apple_coarse.msh"
    for N in 8 16; do
        echo "--- $N ranks ---"
        /usr/bin/time -f "  wall: %e s   peak RSS: %M kB" \
          mpirun -np "$N" --bind-to core --oversubscribe \
          python src/solver.py --mesh "$MESH" \
            --out_dir "/tmp/ranksweep_$N" -T 0.004 \
            --log_interval 1000 --sample_dt 1e9 --xdmf_dt 1e9 --checkpoint_dt 1e9 \
          2>&1 | grep -E "completed|wall:|peak"
    done
    echo
    echo "Set \"np\" in $CONFIG to whichever won, then run the study."
    ;;

  --resume)
    echo "=== resuming the study (completed levels are skipped) ==="
    python scripts/run_convergence.py \
        --config "$CONFIG" --out_root "$OUT_ROOT" --np "$NP"
    python scripts/analyze_convergence.py --study "$OUT_ROOT"
    ;;

  *)
    echo "=== full study: coarse, medium, fine ==="
    echo "    config  : $CONFIG"
    echo "    out_root: $OUT_ROOT"
    echo "    ranks   : $NP"
    echo
    echo "This runs for hours. If you are not inside tmux, stop now and use:"
    echo "    tmux new -s ns3 'bash $0'"
    echo
    python scripts/run_convergence.py \
        --config "$CONFIG" --out_root "$OUT_ROOT" --np "$NP"
    python scripts/analyze_convergence.py --study "$OUT_ROOT"
    echo
    echo "Report: $OUT_ROOT/analysis/convergence_report.md"
    ;;
esac
