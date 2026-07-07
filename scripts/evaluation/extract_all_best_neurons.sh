#!/bin/bash

# check if argument provided
if [ -z "$1" ]; then
  echo "Usage: $0 <number of neurons>"
  exit 1
fi

NUM_NEURONS=$1

# Run extract_best_neurons.py for all pickle files in results/ga
for pickle in results/ga/*.pkl; do
  base=$(basename "$pickle" .pkl)
  out_dir="results/best_neurons/${base}"
  out="${out_dir}/${NUM_NEURONS}x_${base}.pkl"
  python3 scripts/evaluation/extract_best_neurons.py "$pickle" "$NUM_NEURONS" "$out" --write-config &
done

# wait for all background jobs
wait
