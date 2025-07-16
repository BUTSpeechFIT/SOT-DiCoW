#!/bin/bash

set -x

if [ $# -eq 0 ]; then
  echo "No extra config provided, using default config."
fi

# Enable opening multiple files
ulimit -n 4096

# Enable to save bigger files
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

source $(dirname "${BASH_SOURCE[0]}")/../../local_paths.sh

# if [ "$(hostname -d)" == "cerit-sc.cz" -o "$(hostname -d)" == "fzu.cz" -o "$(hostname -d)" == "metacentrum.cz" ]; then
#     echo "Move data to SCRATCHDIR"
#     DATA_DIR="$SCRATCHDIR/data"
#     ${SRC_ROOT}/scripts/data/move_to_dir.sh $DATA_DIR
#     export MANIFEST_DIR=$DATA_DIR/manifests
# fi

# For some reason, if I don't use eval, argparse treats array of args separated by space as a string separated by space (i.e. the args don't get parsed correctly)
if [ ! -z "$N_GPUS" ] && [ "$N_GPUS" -gt 1 ]; then
    eval "torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS" "${SRC_ROOT}/src/main.py" "$@"
else
    eval "python" "${SRC_ROOT}/src/main.py" "$@"
fi
