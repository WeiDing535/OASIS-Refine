#!/bin/sh
# Entry point for the standalone oasis-refine deployment.
#
#   ./run.sh --mtz input.mtz --pdb input.pdb --out-dir out [options]
#   PYTHON=/path/to/venv/bin/python ./run.sh --help
#
# Only gemmi / numpy / scipy from PyPI are needed.  The restraint dictionary
# is bundled in minirefmac/data/ -- no external files are read at runtime.
#
# Interpreter precedence: $PYTHON > local .venv/bin/python > python3.
set -e
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
else
    PY=python3
fi
cd "$HERE"
exec "$PY" -m minirefmac "$@"
