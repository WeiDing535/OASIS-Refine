#!/bin/sh
# One-shot Linux validation for the standalone oasis-refine deployment.
#
#   ./run_linux_test.sh                 # full: integrity + selftest + short refine
#   ./run_linux_test.sh --checks-only  # fast: integrity + selftest, no refine
#   PYTHON=/path/to/venv/bin/python ./run_linux_test.sh
#
# No external suite, no external dictionary CIFs and no local binaries are needed.
# Dependencies are NOT installed by this script -- see README_LINUX.md.
set -u

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE" || exit 1

MODE=full
case "${1:-}" in
    ""|--full)       MODE=full ;;
    --checks-only)   MODE=checks ;;
    -h|--help)
        sed -n '2,9p' "$0"
        exit 0 ;;
    *) echo "unknown option: $1 (use --help)" >&2; exit 2 ;;
esac

# Pick an interpreter: explicit PYTHON > local .venv > system python3.
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
else
    PY=python3
fi

MTZ="$HERE/test_data/7enm/input.mtz"
PDB="$HERE/test_data/7enm/input.pdb"
OUT="$HERE/linux_test_out"

step() { printf '\n==================== %s ====================\n' "$1"; }
fail() { printf '\nFAILED: %s\n' "$1" >&2; exit 1; }

step "0/4  interpreter and dependencies"
command -v "$PY" >/dev/null 2>&1 || fail "python not found: $PY (set PYTHON=/path/to/python)"
"$PY" -c 'import sys, platform; print("python  :", sys.version.split()[0], platform.system(), platform.machine())' || exit 1
if ! "$PY" -c 'import gemmi, numpy, scipy' >/dev/null 2>&1; then
    echo "" >&2
    echo "Missing gemmi / numpy / scipy for $PY." >&2
    echo "Install them first, e.g.:" >&2
    echo "  python3 -m venv .venv" >&2
    echo "  .venv/bin/pip install -r requirements.txt" >&2
    echo "  ./run_linux_test.sh" >&2
    echo "" >&2
    fail "dependencies not installed"
fi
"$PY" -c 'import gemmi,numpy,scipy; print("deps    : gemmi %s / numpy %s / scipy %s" % (gemmi.__version__,numpy.__version__,scipy.__version__))'
[ -f "$MTZ" ] || fail "missing test data: $MTZ"
[ -f "$PDB" ] || fail "missing test data: $PDB"
echo "data    : test_data/7enm (input.mtz + input.pdb)"

# $OUT is this script's own scratch directory (always $HERE/linux_test_out).
# A fresh run resets it so old logs cannot mask a failure.
rm -rf "$OUT"
mkdir -p "$OUT/selftest" "$OUT/refine" || fail "cannot create $OUT"

step "1/4  deployment integrity (self-contained bundled dictionary)"
"$PY" check_install.py > "$OUT/check_install.txt" 2>&1 || { cat "$OUT/check_install.txt"; fail "check_install.py"; }
cat "$OUT/check_install.txt"
grep -q "RESULT     : PASS" "$OUT/check_install.txt" || fail "check_install.py did not report PASS"

step "2/4  build restraints from the bundled dictionary on real data"
"$PY" check_install.py --mtz "$MTZ" --pdb "$PDB" > "$OUT/restraints.txt" 2>&1 || { cat "$OUT/restraints.txt"; fail "restraints on 7enm"; }
cat "$OUT/restraints.txt"
grep -q "RESULT     : PASS" "$OUT/restraints.txt" || fail "restraints check did not report PASS"

step "3/4  numerical selftest (8 checks, ~1-2 min)"
"$PY" -m minirefmac.selftest \
    --mtz "$MTZ" --pdb "$PDB" \
    --out-dir "$OUT/selftest" \
    --json "$OUT/selftest.json" > "$OUT/selftest.log" 2>&1 || { cat "$OUT/selftest.log"; fail "selftest"; }
cat "$OUT/selftest.log"
grep -q "SELFTEST PASS" "$OUT/selftest.log" || fail "selftest did not report PASS"

if [ "$MODE" = full ]; then
    step "4/4  short end-to-end refinement (1 cycle x 3 L-BFGS iters)"
    PYTHON="$PY" ./run.sh \
        --mtz "$MTZ" --pdb "$PDB" \
        --out-dir "$OUT/refine" \
        --ncyc 1 --maxiter 3 > "$OUT/refine.log" 2>&1 || { cat "$OUT/refine.log"; fail "refine"; }
    cat "$OUT/refine.log"
else
    step "4/4  short refinement SKIPPED (--checks-only)"
fi

printf '\n========================================================\n'
echo "ALL LINUX CHECKS PASSED   (mode: $MODE)"
echo "logs / outputs : $OUT"
printf '========================================================\n'
exit 0
