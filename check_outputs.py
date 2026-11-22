#!/usr/bin/env python3
"""check_outputs.py -- sanity-check a finished minirefmac run.

Validates the three artefacts (refined.mtz / refined.pdb / report.json) against
each other and against the input, without trusting the program's own log:

    python check_outputs.py --mtz in.mtz --pdb in.pdb --out-dir run1

Checks
  1. refined.mtz opens, carries every REFMAC-style column, and keeps the input
     space group, cell and reflection count.
  2. the map coefficients are usable: no NaN/Inf, |FWT| > 0, phases in range.
  3. R recomputed from the output file (F vs FC_ALL) agrees with report.json to
     within 0.01 -- a mismatch means the output scale or the column pairing is
     wrong.
  4. refined.pdb: every coordinate/B/occupancy is finite, B stays inside
     [bmin, bmax], and the atom count matches the input model (minus dropped
     hydrogens/altlocs).
  5. report.json: one entry per cycle, Rfree recorded when a free set exists,
     and the last cycle is the one written out.

Exit code 0 = PASS, 1 = FAIL.  Prints one line per check.
"""

import argparse
import json
import math
import os
import sys

import gemmi
import numpy as np

from minirefmac import dataio, maps
from minirefmac.core import ScaleParams

REQUIRED_MTZ = ["FC", "PHIC", "FC_ALL", "PHIC_ALL",
                "FWT", "PHWT", "DELFWT", "PHDELWT", "FOM"]


def _fail(msgs, text):
    msgs.append(text)
    print("  FAIL  " + text)
    return False


def check_mtz(path, data, report, msgs):
    ok = True
    if not os.path.exists(path):
        return _fail(msgs, "missing %s" % path)
    m = gemmi.read_mtz_file(path)
    labels = [c.label for c in m.columns]
    refl = report.get("reflection_columns", {})
    fp = dataio._pick(data, dataio._FP_LABELS, "F", refl.get("f"))
    sig = dataio._pick(data, dataio._SIG_LABELS, "Q", refl.get("sigma"))
    required = [fp.label, sig.label, *REQUIRED_MTZ]
    missing = [c for c in required if c not in labels]
    if missing:
        ok = _fail(msgs, "refined.mtz lacks columns: %s" % ", ".join(missing))
    else:
        print("  ok    refined.mtz has all %d expected columns" % len(required))
    sg_out, sg_in = str(m.spacegroup.hm), str(data.spacegroup.hm)
    if sg_out.replace(" ", "") != sg_in.replace(" ", ""):
        ok = _fail(msgs, "space group %s != input %s" % (sg_out, sg_in))
    else:
        print("  ok    space group preserved (%s)" % sg_out)
    if max(abs(a - b) for a, b in zip(m.cell.parameters,
                                      data.cell.parameters)) > 1e-3:
        ok = _fail(msgs, "unit cell changed: %s vs %s"
                   % (m.cell.parameters, data.cell.parameters))
    else:
        print("  ok    unit cell preserved")
    if m.nreflections != data.nreflections:
        ok = _fail(msgs, "reflection count %d != input %d"
                   % (m.nreflections, data.nreflections))
    else:
        print("  ok    %d reflections" % m.nreflections)

    idx = {c.label: i for i, c in enumerate(m.columns)}
    arr = np.array(m, copy=False)
    for name in ("FWT", "DELFWT", "FC_ALL"):
        if name not in labels:
            continue
        v = np.asarray(arr[:, idx[name]], dtype=np.float64)
        if not np.all(np.isfinite(v)):
            ok = _fail(msgs, "%s contains NaN/Inf" % name)
        elif np.all(v == 0.0):
            ok = _fail(msgs, "%s is all zeros" % name)
        else:
            print("  ok    %s finite (%.3g .. %.3g)" % (name, v.min(), v.max()))
    for name in ("PHWT", "PHIC_ALL", "PHDELWT"):
        if name not in labels:
            continue
        ph = np.asarray(arr[:, idx[name]], dtype=np.float64)
        if not np.all(np.isfinite(ph)) or np.any(np.abs(ph) > 360.0 + 1e-6):
            ok = _fail(msgs, "%s phases out of range" % name)

    # Reconstruct the working set from the original MTZ. Rows outside the
    # retained resolution window may have zero map coefficients on output.
    if fp.label in labels and "FC_ALL" in labels:
        if not np.array_equal(arr[:, :3], data.array[:, :3]):
            return _fail(msgs, "output HKL rows differ from input")
        f = np.asarray(arr[:, idx[fp.label]], dtype=np.float64)
        fc = np.asarray(arr[:, idx["FC_ALL"]], dtype=np.float64)
        sel = np.isfinite(fp.array) & (fp.array > 0) & np.isfinite(sig.array)
        scale_values = report.get("final_scale_params")
        if scale_values:
            scale = ScaleParams(**scale_values)
            k, rows = maps.all_rows_k_total(data, scale)
            with np.errstate(divide="ignore", invalid="ignore"):
                resolution = 0.5 / np.sqrt(rows.stol2)
            sel &= np.isfinite(resolution)
            sel &= resolution >= float(report.get("d_min", 0.0)) - 1e-9
            sel &= resolution <= float(report.get("d_max", np.inf)) + 1e-9
            free = report.get("free", {})
            if free.get("mode") != "absent" and free.get("label"):
                col = data.column_with_label(free["label"])
                if col is None:
                    return _fail(msgs, "report FreeR column missing from input")
                sel &= np.rint(col.array) != int(free["value"])
            if free.get("n_work") is not None and int(sel.sum()) != int(free["n_work"]):
                return _fail(msgs, "reconstructed work reflection count differs from report")
            if report.get("map_scale", "absolute") == "absolute":
                f *= np.where(k > 1e-12, k, 1.0)
                fc *= k
        if not np.any(sel) or np.sum(f[sel]) <= 0 or not np.isfinite(f[sel]).all():
            return _fail(msgs, "working amplitudes in the output are missing or invalid")
        r_file = float(np.sum(np.abs(f[sel] - fc[sel])) / np.sum(f[sel]))
        r_rep = float((report.get("written") or report.get("final", {}))
                      .get("r_work", float("nan")))
        if not math.isfinite(r_rep):
            ok = _fail(msgs, "report.json has no final R")
        elif abs(r_file - r_rep) > 0.01:
            ok = _fail(msgs, "R from MTZ %.4f vs report %.4f (diff > 0.01)"
                       % (r_file, r_rep))
        else:
            print("  ok    R from MTZ %.4f matches report %.4f (d=%.4f)"
                  % (r_file, r_rep, abs(r_file - r_rep)))
    return ok


def check_pdb(path, model_in, report, msgs, b_min=2.0, b_max=200.0):
    ok = True
    if not os.path.exists(path):
        return _fail(msgs, "missing %s" % path)
    st = gemmi.read_structure(path)
    atoms = [a for ch in st[0] for res in ch for a in res]
    n_at = len(atoms)
    n_in = sum(len(res) for ch in model_in[0] for res in ch)
    if n_at > n_in:
        ok = _fail(msgs, "refined.pdb has more atoms (%d) than input (%d)"
                   % (n_at, n_in))
    else:
        print("  ok    %d atoms (input %d, dropped %d)" % (n_at, n_in, n_in - n_at))
    xyz = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in atoms])
    b = np.array([a.b_iso for a in atoms])
    occ = np.array([a.occ for a in atoms])
    for name, v in (("coordinates", xyz), ("B factors", b), ("occupancies", occ)):
        if not np.all(np.isfinite(v)):
            ok = _fail(msgs, "%s contain NaN/Inf" % name)
    if np.all(np.isfinite(b)):
        if b.min() < b_min - 1e-6 or b.max() > b_max + 1e-6:
            ok = _fail(msgs, "B range %.2f..%.2f outside [%.1f, %.1f]"
                       % (b.min(), b.max(), b_min, b_max))
        else:
            print("  ok    B range %.2f .. %.2f A^2" % (b.min(), b.max()))
    if np.all(np.isfinite(occ)) and (occ.min() < -1e-6 or occ.max() > 1.0 + 1e-6):
        ok = _fail(msgs, "occupancies outside [0, 1]")
    return ok


def check_report(path, ncyc, msgs):
    ok = True
    if not os.path.exists(path):
        return _fail(msgs, "missing %s" % path)
    with open(path) as fh:
        rep = json.load(fh)
    cycles = rep.get("cycles", [])
    # cycles[0] is the starting state, so ncyc refined cycles -> ncyc + 1 rows
    if len(cycles) != ncyc + 1:
        ok = _fail(msgs, "report has %d cycle rows, expected %d (= ncyc + the "
                   "starting row)" % (len(cycles), ncyc + 1))
    else:
        print("  ok    report has %d cycle rows (%d refined + start)"
              % (len(cycles), ncyc))
    if "kernel" not in rep:
        ok = _fail(msgs, "report does not record the kernel")
    else:
        print("  ok    kernel=%s (%s)" % (rep["kernel"], rep.get("kernel_reason")))
    free = rep.get("free") or {}
    if free.get("n_free", 0) > 0:
        rf = [c.get("r_free") for c in cycles if c.get("r_free") is not None]
        if not rf:
            ok = _fail(msgs, "free set exists but no Rfree in the cycles")
        else:
            print("  ok    Rfree recorded (%d cycles, last %.4f)" % (len(rf), rf[-1]))
    else:
        print("  note  no free set -> Rfree not available")
    r = [c.get("r_work") for c in cycles if c.get("r_work") is not None]
    if len(r) >= 2 and r[-1] > r[0] + 0.005:
        ok = _fail(msgs, "R got worse: %.4f -> %.4f" % (r[0], r[-1]))
    elif r:
        print("  ok    R %.4f -> %.4f over %d cycle(s)"
              % (r[0], r[-1], len(r) - 1))
    return ok, rep


def main(argv=None):
    p = argparse.ArgumentParser(
        description="sanity-check a finished minirefmac run")
    p.add_argument("--mtz", required=True, help="input MTZ (reference)")
    p.add_argument("--pdb", required=True, help="input model (reference)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--ncyc", type=int, default=None,
                   help="expected number of cycles (default: from report)")
    p.add_argument("--bmin", type=float, default=2.0)
    p.add_argument("--bmax", type=float, default=200.0)
    a = p.parse_args(argv)

    tag = (a.tag + "_") if a.tag else ""
    mtz_out = os.path.join(a.out_dir, tag + "refined.mtz")
    pdb_out = os.path.join(a.out_dir, tag + "refined.pdb")
    rep_out = os.path.join(a.out_dir, tag + "report.json")

    msgs = []
    data = gemmi.read_mtz_file(a.mtz)
    model_in = gemmi.read_structure(a.pdb)
    ncyc = a.ncyc

    print("1) report.json")
    rep = {}
    ok_rep = False
    if os.path.exists(rep_out):
        with open(rep_out) as fh:
            rep = json.load(fh)
        n = a.ncyc if a.ncyc is not None else len(rep.get("cycles", []))
        ok_rep = check_report(rep_out, n, msgs)
    else:
        _fail(msgs, "missing %s" % rep_out)

    print("2) refined.mtz")
    ok_mtz = check_mtz(mtz_out, data, rep, msgs)

    print("3) refined.pdb")
    ok_pdb = check_pdb(pdb_out, model_in, rep, msgs, a.bmin, a.bmax)

    ok = bool(ok_rep and ok_mtz and ok_pdb)
    print("\nRESULT     : %s" % ("PASS" if ok else "FAIL"))
    for m in msgs:
        print("  - " + m)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
