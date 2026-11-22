#!/bin/sh
# pack_release.sh -- build a distributable archive with the CORRECT layout.
#
#   ./pack_release.sh                 # dist/oasis-refine-<ver>.tar.gz + fresh MANIFEST
#   ./pack_release.sh --zip           # also write a .zip
#   ./pack_release.sh --out /tmp/x.tar.gz
#
# Why this exists: a flat `zip -j` / Finder "Compress" of this project silently
# loses the package layout (minirefmac/ subdirs, test_data/7enm/) and leaves two
# different __init__.py files fighting for one name -- the delivered tree then
# fails with "No module named 'minirefmac'".  Always ship with this script (or
# `tar czf`, never `zip -j`).
#
# The archive contains no .venv, no compiled _simd.so (built on first use on the
# target machine), no test outputs and no VCS metadata.  MANIFEST.sha256 is
# regenerated inside the archive and verified by extracting it again.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE"

OUT=""
DO_ZIP=no
while [ $# -gt 0 ]; do
    case "$1" in
        --zip)        DO_ZIP=yes ;;
        --out)        shift; OUT=${1:-} ;;
        --out=*)      OUT=${1#--out=} ;;
        -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

VER=$(sed -n 's/^__version__ *= *"\(.*\)".*/\1/p' minirefmac/__init__.py)
[ -n "$VER" ] || { echo "cannot read __version__ from minirefmac/__init__.py" >&2; exit 1; }
NAME="oasis-refine-$VER"
[ -n "$OUT" ] || OUT="dist/$NAME.tar.gz"

# sha256 helper: GNU coreutils vs macOS/BSD
if command -v sha256sum >/dev/null 2>&1; then
    SHA="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
    SHA="shasum -a 256"
else
    echo "no sha256sum/shasum found" >&2; exit 1
fi

STAGE=$(mktemp -d "${TMPDIR:-/tmp}/pack.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT INT TERM
DEST="$STAGE/$NAME"
mkdir -p "$DEST"

echo "staging $NAME ..."
tar -cf - \
    --exclude=.venv --exclude=__pycache__ --exclude='*.pyc' --exclude='*.so' \
    --exclude=linux_test_out --exclude=oasis_gui_runs --exclude=dist --exclude=.git \
    --exclude='*.egg-info' \
    --exclude=build --exclude='.pytest_cache' --exclude='.ruff_cache' \
    . 2>/dev/null | tar -xf - -C "$DEST"

# Drop files that must not ship, then sanity-check the layout.
rm -f "$DEST/MANIFEST.sha256" "$DEST/minirefmac/_simd.so"
for req in minirefmac/__init__.py minirefmac/core.py minirefmac/data/__init__.py \
           minirefmac/data/monlib_bundle.zip minirefmac/_simd.c \
           test_data/7enm/input.mtz test_data/7enm/input.pdb \
           check_install.py check_outputs.py run.sh run_linux_test.sh \
           pyproject.toml requirements.txt; do
    [ -f "$DEST/$req" ] || { echo "missing from the package: $req" >&2; exit 1; }
done

echo "regenerating MANIFEST.sha256 ..."
( cd "$DEST" && find . -type f ! -name MANIFEST.sha256 -print \
    | sed 's|^\./||' | LC_ALL=C sort \
    | while IFS= read -r f; do $SHA "$f"; done > MANIFEST.sha256 )
N=$(wc -l < "$DEST/MANIFEST.sha256" | tr -d ' ')
echo "  $N files hashed"

mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$STAGE" "$NAME"
echo "wrote $OUT"

if [ "$DO_ZIP" = yes ]; then
    ZIP="${OUT%.tar.gz}.zip"
    if command -v zip >/dev/null 2>&1; then
        ( cd "$STAGE" && zip -qr "$OLDPWD/$ZIP" "$NAME" )
        echo "wrote $ZIP"
    else
        echo "zip not installed -- skipped" >&2
    fi
fi

# ---- verify the archive the way the user will use it ----------------------
CHECK=$(mktemp -d "${TMPDIR:-/tmp}/verify.XXXXXX")
trap 'rm -rf "$STAGE" "$CHECK"' EXIT INT TERM
tar -xzf "$OUT" -C "$CHECK"
( cd "$CHECK/$NAME" && $SHA -c MANIFEST.sha256 >/dev/null ) \
    || { echo "MANIFEST verification FAILED" >&2; exit 1; }
echo "MANIFEST verified after re-extraction"

PY=""
for cand in "$HERE/.venv/bin/python" python3; do
    if command -v "$cand" >/dev/null 2>&1 \
        && "$cand" -c 'import gemmi, numpy, scipy' >/dev/null 2>&1; then
        PY="$cand"; break
    fi
done
if [ -n "$PY" ]; then
    ( cd "$CHECK/$NAME" && PYTHONPATH=. "$PY" -c \
        "import minirefmac, minirefmac.cli, minirefmac.selftest; print('import OK, version', minirefmac.__version__)" )
else
    echo "note: no interpreter with gemmi/numpy/scipy found -- import check skipped"
fi

echo ""
echo "archive : $OUT"
echo "size    : $(du -h "$OUT" | cut -f1)"
echo "sha256  : $($SHA "$OUT" | awk '{print $1}')"
echo ""
echo "Ship it, then on the target machine:"
echo "  tar xzf $(basename "$OUT") && cd $NAME"
echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
echo "  ./run_linux_test.sh"
