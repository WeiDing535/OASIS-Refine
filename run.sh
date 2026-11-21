#!/bin/sh
# Entry point for the standalone oasis-refine deployment.
#
#   ./run.sh --mtz input.mtz --pdb input.pdb --out-dir out [options]
#   PYTHON=/path/to/venv/bin/python ./run.sh --help
#
# Only gemmi / numpy / scipy from PyPI are needed.  The restraint dictionary
# is bundled in minirefmac/data/ -- no external files are read at runtime.
set -e
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
cd "$HERE"
exec "$PYTHON" -m minirefmac.cli "$@"
