#!/bin/sh
# One-shot Linux validation for the standalone oasis-refine deployment.
#
#   ./run_linux_test.sh                 # full: env + integrity + units + selftest + refine + outputs
#   ./run_linux_test.sh --checks-only   # skip the end-to-end refinement (fast)
#   ./run_linux_test.sh --kernels       # also verify numpy / fused / SIMD give the same R
#   ./run_linux_test.sh --full --kernels
#   PYTHON=/path/to/venv/bin/python ./run_linux_test.sh     (or --python PATH)
#
# No external suite, no external dictionary CIFs and no local binaries are needed.
# Set MINIREFMAC_TEST_OUT to an absolute path for a separate output directory.
# Dependencies are NOT installed by this script -- see README_LINUX.md.
set -u

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$HERE" || exit 1

MODE=full
KERNELS=no
PY_OVERRIDE=${PYTHON:-}
while [ $# -gt 0 ]; do
    case "$1" in
        ""|--full)      MODE=full ;;
        --checks-only)  MODE=checks ;;
        --kernels)      KERNELS=yes ;;
        --python)       shift; PY_OVERRIDE=${1:-} ;;
        --python=*)     PY_OVERRIDE=${1#--python=} ;;
        -h|--help)      sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (use --help)" >&2; exit 2 ;;
    esac
    shift
done

# Pick an interpreter: --python / $PYTHON > local .venv > system python3.
if [ -n "$PY_OVERRIDE" ]; then
    PY="$PY_OVERRIDE"
elif [ -x "$HERE/.venv/bin/python" ]; then
    PY="$HERE/.venv/bin/python"
else
    PY=python3
fi

MTZ="$HERE/test_data/7enm/input.mtz"
PDB="$HERE/test_data/7enm/input.pdb"
OUT="${MINIREFMAC_TEST_OUT:-$HERE/linux_test_out}"
case "$OUT" in
    /*) ;;
    *) echo "MINIREFMAC_TEST_OUT must be an absolute path" >&2; exit 2 ;;
esac
SUMMARY="$OUT/summary.txt"
T0=$(date +%s)

step() { printf '\n==================== %s ====================\n' "$1"; }
fail() {
    printf '\nFAILED: %s\n' "$1" >&2
    printf 'FAIL  %s\n' "$1" >> "$SUMMARY"
    printf '\n---- summary (%s) ----\n' "$SUMMARY" >&2
    cat "$SUMMARY" >&2
    exit 1
}
record() { printf 'PASS  %-42s %4ss\n' "$1" "$2" >> "$SUMMARY"; }
elapsed() { echo $(( $(date +%s) - $1 )); }

# ---------------------------------------------------------------- 0
step "0/7  interpreter and dependencies"
mkdir -p "$OUT" || exit 1
: > "$SUMMARY" || exit 1
command -v "$PY" >/dev/null 2>&1 || fail "python not found: $PY (use --python PATH)"

"$PY" - <<'EOF' || exit 1
import sys
print("python  :", sys.version.split()[0], "|", sys.implementation.name)
if sys.version_info < (3, 9):
    sys.exit("minirefmac needs Python >= 3.9")
EOF
if ! "$PY" -c 'import gemmi, numpy, scipy' >/dev/null 2>&1; then
    echo ""
    echo "Missing gemmi / numpy / scipy for $PY.  Install them first:"
    echo "  python3 -m venv .venv"
    echo "  .venv/bin/pip install -r requirements.txt"
    echo "  ./run_linux_test.sh"
    echo ""
    fail "dependencies not installed"
fi
"$PY" - <<'EOF' | tee "$OUT/env.txt"
import os, platform, sys
import gemmi, numpy, scipy
print("deps    : gemmi %s / numpy %s / scipy %s"
      % (gemmi.__version__, numpy.__version__, scipy.__version__))
print("machine : %s %s, %d cores, %.1f GB RAM"
      % (platform.system(), platform.machine(), os.cpu_count() or 0,
         (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
         if hasattr(os, "sysconf") else float("nan")))
try:
    from minirefmac import simdkernel
    ok, why = simdkernel.probe()
    print("kernel  : auto -> %s (%s)" % ("simd" if ok else "numpy", why))
except Exception as exc:                      # pragma: no cover
    print("kernel  : probe failed: %s" % exc)
EOF
[ -f "$MTZ" ] || fail "missing test data: $MTZ"
[ -f "$PDB" ] || fail "missing test data: $PDB"
echo "data    : test_data/7enm (input.mtz + input.pdb)"
record "environment + dependencies" "$(elapsed $T0)"

# A fresh run resets the output tree so old logs cannot mask a failure.
rm -rf "$OUT/selftest" "$OUT/refine" "$OUT/kernels"
mkdir -p "$OUT/selftest" "$OUT/refine" || fail "cannot create $OUT subdirs"

# ---------------------------------------------------------------- 1
step "1/7  deployment integrity (self-contained bundled dictionary)"
S=$(date +%s)
"$PY" check_install.py > "$OUT/check_install.txt" 2>&1 \
    || { cat "$OUT/check_install.txt"; fail "check_install.py"; }
cat "$OUT/check_install.txt"
grep -q "RESULT     : PASS" "$OUT/check_install.txt" \
    || fail "check_install.py did not report PASS"
record "deployment integrity" "$(elapsed $S)"

# ---------------------------------------------------------------- 2
step "2/7  build restraints from the bundled dictionary on real data"
S=$(date +%s)
"$PY" check_install.py --mtz "$MTZ" --pdb "$PDB" > "$OUT/restraints.txt" 2>&1 \
    || { cat "$OUT/restraints.txt"; fail "restraints on 7enm"; }
cat "$OUT/restraints.txt"
grep -q "RESULT     : PASS" "$OUT/restraints.txt" \
    || fail "restraints check did not report PASS"
record "restraints from bundled dictionary" "$(elapsed $S)"

# ---------------------------------------------------------------- 3
step "3/7  unit tests (fast, stdlib unittest)"
S=$(date +%s)
"$PY" -m unittest discover -s tests -v > "$OUT/unit_tests.log" 2>&1 \
    || { cat "$OUT/unit_tests.log"; fail "unit tests"; }
tail -3 "$OUT/unit_tests.log"
record "unit tests" "$(elapsed $S)"

# ---------------------------------------------------------------- 4
step "4/7  numerical selftest (10 checks, ~1-2 min)"
S=$(date +%s)
"$PY" -m minirefmac.selftest \
    --mtz "$MTZ" --pdb "$PDB" \
    --out-dir "$OUT/selftest" \
    --json "$OUT/selftest.json" > "$OUT/selftest.log" 2>&1 \
    || { cat "$OUT/selftest.log"; fail "selftest"; }
cat "$OUT/selftest.log"
grep -q "SELFTEST PASS" "$OUT/selftest.log" || fail "selftest did not report PASS"
record "numerical selftest (10 checks)" "$(elapsed $S)"

# ---------------------------------------------------------------- 5
if [ "$MODE" = full ]; then
    step "5/7  short end-to-end refinement (1 cycle x 3 L-BFGS iters)"
    S=$(date +%s)
    "$PY" -m minirefmac \
        --mtz "$MTZ" --pdb "$PDB" \
        --out-dir "$OUT/refine" \
        --ncyc 1 --maxiter 3 > "$OUT/refine.log" 2>&1 \
        || { cat "$OUT/refine.log"; fail "refine"; }
    cat "$OUT/refine.log"
    grep -q "^RESULT" "$OUT/refine.log" || fail "refine printed no RESULT line"
    record "short refinement (1 cycle)" "$(elapsed $S)"

    # ------------------------------------------------------------ 6
    step "6/7  output validation (refined.mtz / refined.pdb / report.json)"
    S=$(date +%s)
    "$PY" check_outputs.py --mtz "$MTZ" --pdb "$PDB" \
        --out-dir "$OUT/refine" --ncyc 1 > "$OUT/check_outputs.txt" 2>&1 \
        || { cat "$OUT/check_outputs.txt"; fail "output validation"; }
    cat "$OUT/check_outputs.txt"
    grep -q "RESULT     : PASS" "$OUT/check_outputs.txt" \
        || fail "check_outputs.py did not report PASS"
    record "output validation" "$(elapsed $S)"
else
    step "5/7  refinement SKIPPED (--checks-only)"
    step "6/7  output validation SKIPPED (--checks-only)"
fi

# ---------------------------------------------------------------- 7
if [ "$KERNELS" = yes ]; then
    step "7/7  kernel parity (all available kernels must agree)"
    S=$(date +%s)
    KERNEL_LIST="numpy fused"
    if "$PY" -c 'from minirefmac.simdkernel import probe; import sys; ok, why = probe(); print("SIMD:", why); sys.exit(0 if ok else 1)'; then
        KERNEL_LIST="$KERNEL_LIST simd"
    else
        echo "SKIP  SIMD kernel unavailable" | tee -a "$SUMMARY"
    fi
    for K in $KERNEL_LIST; do
        mkdir -p "$OUT/kernels/$K"
        MINIREFMAC_STRICT_KERNEL=1 "$PY" -m minirefmac --mtz "$MTZ" --pdb "$PDB" \
            --out-dir "$OUT/kernels/$K" --ncyc 1 --maxiter 1 \
            --kernel "$K" > "$OUT/kernels/$K.log" 2>&1 \
            || { cat "$OUT/kernels/$K.log"; fail "kernel $K"; }
        grep -E "^RESULT" "$OUT/kernels/$K.log"
    done
    "$PY" - "$OUT/kernels" $KERNEL_LIST <<'EOF' || fail "kernels disagree (see above)"
import json, os, sys
root = sys.argv[1]
vals = {}
for k in sys.argv[2:]:
    p = os.path.join(root, k, "report.json")
    if not os.path.exists(p):
        sys.exit("missing %s" % p)
    r = json.load(open(p))
    if r["kernel"] != k:
        sys.exit("requested %s, used %s" % (k, r["kernel"]))
    vals[k] = (r["final"]["r_work"], r["final"]["r_free"], r["final"]["negll"])
ref = vals["numpy"]
bad = False
for k, v in vals.items():
    d_r, d_rf, d_ll = (abs(v[0] - ref[0]), abs(v[1] - ref[1]),
                       abs(v[2] - ref[2]) / max(abs(ref[2]), 1.0))
    flag = "ok " if (d_r < 1e-6 and d_rf < 1e-6 and d_ll < 1e-9) else "DIFF"
    if flag != "ok ":
        bad = True
    print("  %-6s R=%.6f Rfree=%.6f -LL=%.3f   dR=%.1e dRfree=%.1e dLL=%.1e  %s"
          % (k, v[0], v[1], v[2], d_r, d_rf, d_ll, flag))
sys.exit(1 if bad else 0)
EOF
    record "kernel parity ($KERNEL_LIST)" "$(elapsed $S)"
else
    step "7/7  kernel parity SKIPPED (add --kernels)"
fi

printf '\n========================================================\n'
echo "ALL LINUX CHECKS PASSED   (mode: $MODE, kernels: $KERNELS)"
printf 'total wall time : %ss\n' "$(elapsed $T0)"
printf 'logs / outputs  : %s\n' "$OUT"
printf '========================================================\n'
{
    echo ""
    echo "RESULT: ALL PASSED  (mode=$MODE kernels=$KERNELS total=$(elapsed $T0)s)"
} >> "$SUMMARY"
cat "$SUMMARY"
exit 0
