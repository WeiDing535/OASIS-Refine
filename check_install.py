#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-check for a standalone oasis-refine deployment.

    python check_install.py                      # dependencies + dictionary
    python check_install.py --mtz X.mtz --pdb X.pdb   # + build restraints

Exit status 0 means the deployment is complete and fully self-contained.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mtz", default=None)
    ap.add_argument("--pdb", default=None)
    a = ap.parse_args()

    ok = True
    print("deploy root :", HERE)

    ext = [p for p in sys.path if "monomers" in p.lower()
           or "mon_lib" in p.lower()]
    print("external dictionary paths in sys.path :",
          ext if ext else "NONE")

    try:
        import gemmi
        import numpy
        import scipy
        print("deps       : gemmi %s / numpy %s / scipy %s"
              % (gemmi.__version__, numpy.__version__, scipy.__version__))
    except ImportError as exc:
        print("FAIL: missing dependency:", exc)
        print("      fix: pip install -r requirements.txt")
        return 1

    try:
        import minirefmac
        from minirefmac import monlib
    except ImportError as exc:
        print("FAIL: cannot import minirefmac:", exc)
        return 1
    print("package    :", minirefmac.__file__)
    if not os.path.abspath(minirefmac.__file__).startswith(HERE):
        print("FAIL: minirefmac was imported from outside this deployment")
        ok = False

    bundle = monlib.BUNDLE_PATH
    if not os.path.exists(bundle):
        print("FAIL: bundled dictionary missing:", bundle)
        return 1
    print("dictionary : %s (%.1f MB)"
          % (bundle, os.path.getsize(bundle) / 1024.0 / 1024.0))
    try:
        bundle_obj = monlib.bundled()
        if not bundle_obj.available():
            print("FAIL: bundled dictionary cannot be opened")
            return 1
        info = bundle_obj.info
        n = len(info.get("residues", ()))
        print("             %d residues, built %s, gemmi %s"
              % (n, info.get("built", "?"), info.get("gemmi_version", "?")))
        if n < 1000:
            ok = False
    except Exception as exc:
        print("FAIL: cannot read dictionary manifest:", exc)
        return 1

    lic = os.path.join(HERE, "minirefmac", "data",
                       "LICENSE-CCP4-MONOMERS.txt")
    print("licence    :", "present" if os.path.exists(lic) else "MISSING")
    if not os.path.exists(lic):
        ok = False

    if a.mtz or a.pdb:
        if not (a.mtz and a.pdb):
            print("FAIL: --mtz and --pdb must be given together")
            return 1
        from minirefmac import dataio, geometry
        d = dataio.load_data(a.mtz, a.pdb)
        R = geometry.Restraints(d)
        f, gx, gb, st = R.calc(d.xyz, d.b_iso)
        print("restraints : %d bonds / %d angles / %d planes / %d torsions"
              % (R.info["n_bond"], R.info["n_angle"],
                 R.info["n_plane_groups"], R.info["n_torsion"]))
        print("             f=%.6f  rmsd_bond=%.6f"
              % (f, st["rmsd_bond"]))

    print("RESULT     :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
