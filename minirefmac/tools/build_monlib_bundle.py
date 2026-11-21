# -*- coding: utf-8 -*-
"""BUILD-TIME ONLY -- bundle monomer-library restraints into minirefmac.

Run this ONCE on a machine that has the source chemical-component dictionary.
The generated zip is the only dictionary artifact minirefmac reads at runtime.

    python minirefmac/tools/build_monlib_bundle.py --src /path/to/monomers \
        --out minirefmac/data/monlib_bundle.zip

Only the five restraint loops actually consumed by ``geometry.Restraints``
are kept (bonds / angles / planes / chirs / torsions); coordinates, 2D
diagrams, chemical descriptors and atom-level metadata are dropped.  That
reduces a 677 MB CIF tree to a few MB.  The zip holds one small JSON per
residue plus ``LINK_TRANS.json`` and ``MANIFEST.json`` so runtime lookup stays
O(1) and only the residues present in the model are decoded.
"""

import argparse
import json
import os
import sys
import time
import zipfile

import gemmi

# Keep keys short: ~35k residues, so every byte matters.
# b = bonds, a = angles, p = planes, c = chirs, t = torsions
_BUNDLE_VERSION = 1


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _chir_name(sign):
    nm = getattr(sign, "name", None)
    if nm is None:
        nm = str(sign).split(".")[-1]
    return str(nm)


def extract_rt(rt):
    """Serialise the five restraint loops of a gemmi ChemComp.Restraints."""
    rec = {
        "b": [[b.id1.atom, b.id2.atom, _num(b.value), _num(b.esd)]
              for b in rt.bonds],
        "a": [[a.id1.atom, a.id2.atom, a.id3.atom,
               _num(a.value), _num(a.esd)] for a in rt.angles],
        "p": [[str(p.label), [x.atom for x in p.ids], _num(p.esd)]
              for p in rt.planes],
        "c": [[c.id1.atom, c.id2.atom, c.id3.atom, c.id_ctr.atom,
               _chir_name(c.sign)] for c in rt.chirs],
        "t": [[str(t.label), t.id1.atom, t.id2.atom, t.id3.atom, t.id4.atom,
               _num(t.value), _num(t.esd), int(t.period)]
              for t in rt.torsions],
    }
    return rec


def find_comp_block(path):
    """Reproduce geometry._load_library's block selection exactly."""
    doc = gemmi.cif.read(path)
    blocks = list(doc)
    if not blocks:
        return None
    want = None
    stem = os.path.splitext(os.path.basename(path))[0]
    blk = None
    for b in blocks:
        if b.name.lower() == "comp_" + stem.lower():
            blk = b
            break
    if blk is None:
        for b in blocks:
            if not b.name.lower().endswith("list"):
                blk = b
                break
    if blk is None:
        blk = blocks[-1]
    want = blk.name
    for b in blocks:
        if b.name.lower().startswith("comp_") and not b.name.lower().endswith("list"):
            want = b.name
            break
    return blk, want


def extract_link_trans(monlib_path):
    """Pull the TRANS peptide-link definition out of mon_lib_list.cif."""
    path = os.path.join(monlib_path, "list", "mon_lib_list.cif")
    if not os.path.exists(path):
        return None
    doc = gemmi.cif.read(path)
    blk = doc.find_block("link_TRANS")
    if blk is None:
        return None
    out = {}
    bb = blk.find(["_chem_link_bond.value_dist",
                   "_chem_link_bond.value_dist_esd"])
    if len(bb):
        out["bond"] = [_num(bb[0][0]), _num(bb[0][1])]
    atags = ["_chem_link_angle.atom_1_comp_id",
             "_chem_link_angle.atom_id_1",
             "_chem_link_angle.atom_2_comp_id",
             "_chem_link_angle.atom_id_2",
             "_chem_link_angle.atom_3_comp_id",
             "_chem_link_angle.atom_id_3",
             "_chem_link_angle.value_angle",
             "_chem_link_angle.value_angle_esd"]
    ta = blk.find(atags)
    out["angles"] = [[r[0], r[1], r[2], r[3], r[4], r[5],
                      _num(r[6]), _num(r[7])] for r in ta]
    tt = blk.find(["_chem_link_tor.id", "_chem_link_tor.value_angle",
                   "_chem_link_tor.value_angle_esd",
                   "_chem_link_tor.period"])
    out["torsions"] = [[r[0], _num(r[1]), _num(r[2]), int(float(r[3]))]
                       for r in tt]
    tp = blk.find(["_chem_link_plane.plane_id",
                   "_chem_link_plane.atom_comp_id",
                   "_chem_link_plane.atom_id",
                   "_chem_link_plane.dist_esd"])
    out["planes"] = [[r[0], r[1], r[2], _num(r[3])] for r in tp]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True,
                    help="source monomer-library directory (build-time only)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "monlib_bundle.zip"))
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N CIFs (smoke test)")
    a = ap.parse_args(argv)

    if not os.path.isdir(a.src):
        print("ERROR: monomer library not found:", a.src, file=sys.stderr)
        return 2

    paths = []
    for sub in sorted(os.listdir(a.src)):
        d = os.path.join(a.src, sub)
        if len(sub) != 1 or not os.path.isdir(d):
            continue                      # skip list/, root-level global CIFs
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".cif"):
                paths.append(os.path.join(d, fn))
    if a.limit:
        paths = paths[:a.limit]
    print("monomer CIFs to process:", len(paths), flush=True)

    t0 = time.time()
    n_ok = n_skip = 0
    names = []
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for i, p in enumerate(paths):
            rn = os.path.splitext(os.path.basename(p))[0].upper()
            try:
                blk, _blockname = find_comp_block(p)
                cc = gemmi.make_chemcomp_from_block(blk)
                if cc is None or cc.rt is None:
                    n_skip += 1
                    continue
                rec = extract_rt(cc.rt)
                zf.writestr(rn + ".json",
                            json.dumps(rec, separators=(",", ":")))
                names.append(rn)
                n_ok += 1
            except Exception:
                n_skip += 1
            if (i + 1) % 5000 == 0:
                print("  %d/%d  ok=%d skip=%d  %.1fs"
                      % (i + 1, len(paths), n_ok, n_skip,
                         time.time() - t0), flush=True)

        link = extract_link_trans(a.src)
        if link is not None:
            zf.writestr("LINK_TRANS.json",
                        json.dumps(link, separators=(",", ":")))
        manifest = dict(
            version=_BUNDLE_VERSION,
            source="monomer library restraint loops",
            source_path=os.path.abspath(a.src),
            gemmi_version=gemmi.__version__,
            n_residues=len(names),
            link_trans=(link is not None),
            built=time.strftime("%Y-%m-%d %H:%M:%S"),
            residues=names,
        )
        zf.writestr("MANIFEST.json", json.dumps(manifest,
                                               separators=(",", ":")))

    print("wrote %s  (%d residues, %d skipped, %.1f s, %.2f MB)"
          % (a.out, n_ok, n_skip, time.time() - t0,
             os.path.getsize(a.out) / 1024.0 / 1024.0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
