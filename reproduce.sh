#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
if [[ "${1:-}" == "--refresh-online" ]]; then
  shift
  python realdata/materialize_dev.py --metadata realdata --out realdata/dev "$@"
elif [[ "${1:-}" == "--cached-raw" ]]; then
  python realdata/materialize_dev.py --metadata realdata --out realdata/dev "$@"
elif (( $# == 0 )); then
  python realdata/install_snapshot.py
else
  echo "Usage: $0 [--refresh-online | --cached-raw PATH]" >&2
  exit 2
fi
(cd realdata && python -m unittest -v test_materialize_dev.py)
for exp in kriging_dev historical_dev analog_dev cold_dev; do
  (cd "$exp" && python run.py && python -m unittest -v test_run.py)
done
(cd analog_dev && python analyze.py)
python verify_results.py
