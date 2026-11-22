#!/usr/bin/env python3
"""weight_scan.py -- Rfree-vs-weight study driver (research tool, not used at runtime).

Runs the CLI over a ladder of w_xray values (or over the built-in weight
controllers) on one data set, then prints a table and writes a CSV so the
weight/Rfree trade-off can be judged from data instead of taste:

    python -m minirefmac.tools.weight_scan \
        --mtz test_data/7enm/input.mtz --pdb test_data/7enm/input.pdb \
        --ncyc 5 --maxiter 10 --weights 0.3 0.6 1 2 4 8 16 --out-dir /tmp/scan7

    # controllers instead of fixed weights
    python -m minirefmac.tools.weight_scan ... --modes rmsd:0.008 rmsd:0.012 heuristic

Each run gets its own subdirectory; the table reports R/Rfree at the last cycle,
the best Rfree reached, and the final geometry rms deviations.
"""

import argparse
import csv
import json
import os
import subprocess
import sys


def _run(cmd, log_path):
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return proc.returncode


def _read(out_dir):
    path = os.path.join(out_dir, "report.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def main(argv=None):
    p = argparse.ArgumentParser(description="Rfree-vs-weight study driver")
    p.add_argument("--mtz", required=True)
    p.add_argument("--pdb", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ncyc", type=int, default=5)
    p.add_argument("--maxiter", type=int, default=10)
    p.add_argument("--weights", nargs="*", type=float, default=[],
                   help="fixed w_xray values to scan")
    p.add_argument("--modes", nargs="*", default=[],
                   help="weight controllers, e.g. rmsd:0.008 rmsd:0.012 heuristic")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="extra CLI args passed through to every run")
    a = p.parse_args(argv)

    if not a.weights and not a.modes:
        a.weights = [0.3, 0.6, 1.0, 2.0, 4.0, 8.0]
    os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    jobs = [("w=%.4g" % w, ["--weight", "%.10g" % w]) for w in a.weights]
    for spec in a.modes:
        if spec.startswith("rmsd"):
            target = spec.split(":", 1)[1] if ":" in spec else "0.02"
            jobs.append(("rmsd=%s" % target,
                         ["--weight-mode", "rmsd", "--rmsd-target", target]))
        else:
            jobs.append((spec, ["--weight-mode", spec]))

    for label, extra in jobs:
        slug = label.replace("=", "").replace(".", "p").replace(" ", "")
        out = os.path.join(a.out_dir, slug)
        cmd = [a.python, "-m", "minirefmac", "--mtz", a.mtz, "--pdb", a.pdb,
               "--out-dir", out, "--ncyc", str(a.ncyc),
               "--maxiter", str(a.maxiter), "--no-keep-best", "--quiet"]
        cmd += extra + list(a.extra)
        print("running %-12s -> %s" % (label, out), flush=True)
        rc = _run(cmd, os.path.join(a.out_dir, slug + ".log"))
        rep = _read(out)
        if rc != 0 or rep is None:
            print("  FAILED (see %s.log)" % slug, flush=True)
            continue
        cyc = rep["cycles"]
        last = cyc[-1]
        best = min((c for c in cyc if c.get("r_free") is not None),
                   key=lambda c: c["r_free"])
        rows.append(dict(
            label=label, ncyc=len(cyc) - 1,
            w_final=last["w_xray"],
            r_first=cyc[0]["r_work"], r_last=last["r_work"],
            rf_first=cyc[0]["r_free"], rf_last=last["r_free"],
            rf_best=best["r_free"], rf_best_cycle=int(best["cycle"]),
            rmsd_bond=last["rmsd_bond"], rmsd_angle=last["rmsd_angle"],
            rmsz_bond=last["rmsz_bond"], r_at_best=best["r_work"],
        ))

    if not rows:
        print("no successful runs", file=sys.stderr)
        return 1

    hdr = ("%-12s %8s %8s %8s %8s %6s %8s %8s %7s"
           % ("run", "R_last", "Rfree_l", "Rfree_b", "@cyc", "w_end",
              "rmsBOND", "rmsANGL", "rmsZ"))
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda d: d["rf_best"]):
        print("%-12s %8.4f %8.4f %8.4f %8d %6.3g %8.4f %8.4f %7.3f"
              % (r["label"], r["r_last"], r["rf_last"], r["rf_best"],
                 r["rf_best_cycle"], r["w_final"], r["rmsd_bond"],
                 r["rmsd_angle"], r["rmsz_bond"]))

    csv_path = os.path.join(a.out_dir, "scan.csv")
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\nwrote %s" % csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
