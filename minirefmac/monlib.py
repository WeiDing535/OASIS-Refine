# -*- coding: utf-8 -*-
"""Bundled monomer-dictionary access -- fully self-contained at runtime.

At BUILD time ``tools/build_monlib_bundle.py`` extracts the five restraint
loops (bonds / angles / planes / chirs / torsions) plus the ``link_TRANS``
peptide-link definition into ``data/monlib_bundle.zip``.  At RUNTIME this
module reads that zip and rebuilds lightweight objects with exactly the
attribute surface ``geometry.Restraints`` expects (``b.id1.atom``,
``a.value``, ``pl.ids``, ``ch.sign``, ``tt.period`` ...).  No external files
are needed for the default bundled path. Explicit component CIFs can
be supplied by the caller.

Only the residues actually present in the model are decoded.  A single
module-level ``Monlib`` instance keeps the zip handle open and caches decoded
residues, so repeated ``Restraints`` construction is cheap.

The source dictionary is distributed under LGPL-3.0; the licence text and
provenance notice ship next to the bundle in ``data/``.
"""

import json
import math
import os
import zipfile
from types import SimpleNamespace as _NS

import gemmi

BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "monlib_bundle.zip")

# Fallbacks if the bundle is somehow unavailable (e.g. a source checkout that
# did not run the build script).  Values match the link_TRANS peptide link.
_FALLBACK_LINK = dict(
    bond=[1.337, 0.011],
    angles=[["1", "CA", "1", "C", "2", "N", 115.917, 1.50],
            ["1", "O", "1", "C", "2", "N", 123.469, 1.50],
            ["1", "C", "2", "N", "2", "CA", 122.095, 1.76],
            ["1", "C", "2", "N", "2", "H", 119.176, 1.83]],
    torsions=[["omega", 180.0, 5.0, 0]],
    planes=[["plan-1", "1", "CA", 0.020],
            ["plan-1", "1", "C", 0.020],
            ["plan-1", "1", "O", 0.020],
            ["plan-1", "2", "N", 0.020]],
)


def _ns(**kw):
    return _NS(**kw)


def _atom(name):
    return _ns(atom=name)


def _rebuild_rt(rec):
    """Rebuild a gemmi-Restraints-shaped object from a bundle record."""
    return _ns(
        bonds=[_ns(id1=_atom(r[0]), id2=_atom(r[1]),
                   value=float(r[2]), esd=float(r[3])) for r in rec.get("b", ())],
        angles=[_ns(id1=_atom(r[0]), id2=_atom(r[1]), id3=_atom(r[2]),
                    value=float(r[3]), esd=float(r[4])) for r in rec.get("a", ())],
        planes=[_ns(label=r[0], ids=[_atom(x) for x in r[1]],
                    esd=float(r[2])) for r in rec.get("p", ())],
        chirs=[_ns(id1=_atom(r[0]), id2=_atom(r[1]), id3=_atom(r[2]),
                   id_ctr=_atom(r[3]), sign=_ns(name=r[4]))
               for r in rec.get("c", ())],
        torsions=[_ns(label=r[0], id1=_atom(r[1]), id2=_atom(r[2]),
                      id3=_atom(r[3]), id4=_atom(r[4]),
                      value=float(r[5]), esd=float(r[6]),
                      period=int(r[7])) for r in rec.get("t", ())],
    )


class Monlib:
    """Lazy, cached reader for the bundled monomer dictionary."""

    def __init__(self, path=None):
        self.path = path or BUNDLE_PATH
        self._zip = None
        self._names = None
        self._link = None
        self._cache = {}
        self.info = {}          # MANIFEST.json, filled on first _open()

    # -- internals -----------------------------------------------------
    def _open(self):
        if self._zip is None:
            if not os.path.exists(self.path):
                return None
            self._zip = zipfile.ZipFile(self.path, "r")
            try:
                mf = json.loads(self._zip.read("MANIFEST.json"))
                self._names = frozenset(mf.get("residues", ()))
                self.info = mf
            except Exception:
                self._names = frozenset()
                self.info = {}
        return self._zip

    def available(self):
        return self._open() is not None

    # -- public API ----------------------------------------------------
    def get(self, name):
        """Return a restraints object for one residue name, or None."""
        key = str(name).upper()
        if key in self._cache:
            return self._cache[key]
        zf = self._open()
        if zf is None or self._names is None or key not in self._names:
            return None
        try:
            rec = json.loads(zf.read(key + ".json"))
        except (KeyError, ValueError):
            return None
        rt = _rebuild_rt(rec)
        self._cache[key] = rt
        return rt

    def load(self, kinds):
        """Bulk lookup: ``{resname: restraints}`` for the names present."""
        out = {}
        for rn in kinds:
            rt = self.get(rn)
            if rt is not None:
                out[rn] = rt
        return out

    def link_trans(self):
        """Raw ``link_TRANS`` record (bond / angles / torsions / planes)."""
        if self._link is None:
            zf = self._open()
            if zf is not None:
                try:
                    self._link = json.loads(zf.read("LINK_TRANS.json"))
                except Exception:
                    self._link = dict(_FALLBACK_LINK)
            else:
                self._link = dict(_FALLBACK_LINK)
        return self._link


_BUNDLE = Monlib()


def bundled():
    """The process-wide bundled dictionary reader."""
    return _BUNDLE



def load_component_cifs(paths):
    """Read externally supplied intramonomer restraints by component ID."""
    components = {}
    for path in paths:
        try:
            doc = gemmi.cif.read_file(str(path))
            parsed = []
            for block in doc:
                if not block.find_loop("_chem_comp_atom.atom_id"):
                    continue
                comp = gemmi.make_chemcomp_from_block(block)
                cif_id = block.find_value("_chem_comp.id")
                name = (gemmi.cif.as_string(cif_id) if cif_id else comp.name).upper()
                if not name or not comp.atoms:
                    raise ValueError("component ID or atoms missing")
                rt = comp.rt
                if any(not math.isfinite(b.value) or b.value <= 0
                       or not math.isfinite(b.esd) for b in rt.bonds):
                    raise ValueError("invalid bond target or uncertainty for %s" % name)
                if any(not math.isfinite(a.value) or not math.isfinite(a.esd)
                       for a in rt.angles):
                    raise ValueError("invalid angle target or uncertainty for %s" % name)
                parsed.append((name, rt))
            if not parsed:
                raise ValueError("no chemical component with an atom loop")
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError("invalid component CIF %s: %s" % (path, exc)) from exc
        for name, restraints in parsed:
            if name in components:
                raise ValueError("duplicate component %s in restraint CIFs" % name)
            components[name] = restraints
    return components
