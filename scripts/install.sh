#!/bin/bash
# Copies the gyms into a unitree_rl_mjlab checkout so its train.py and play.py can find them.
# Usage: ./scripts/install.sh /path/to/unitree_rl_mjlab
set -e
kit=${1:?path to unitree_rl_mjlab}
here=$(cd "$(dirname "$0")/.." && pwd)
for g in base strike defence fighter spar; do
  mkdir -p "$kit/src/tasks/$g"
  cp "$here"/gyms/$g/*.py "$kit/src/tasks/$g/"
done
cp "$here"/scripts/*.py "$kit/"
echo "installed gyms into $kit/src/tasks"
