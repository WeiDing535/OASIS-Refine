# -*- coding: utf-8 -*-
"""Self-contained runtime data for minirefmac.

``monlib_bundle.zip``
    Compact per-residue restraint records (bonds, angles, planes,
    chiralities, torsions) plus the ``link_TRANS`` peptide-link definition,
    extracted once at build time from the source chemical-component
    dictionary by ``tools/build_monlib_bundle.py``.  Coordinates, 2D
    diagrams, chemical descriptors and atom metadata are dropped, reducing
    a 677 MB CIF tree to ~48 MB.  Runtime lookup is O(1) and decodes only
    residues present in the model.  The dictionary is always read from this
    bundle.

Licence
    The source dictionary is distributed under the GNU Lesser General Public
    License v3 (LGPL-3.0).  The full text ships alongside as
    ``LICENSE-CCP4-MONOMERS.txt``.  The bundle is a derived *data* artifact
    containing no executable code.  Please retain this notice and the licence
    text when redistributing minirefmac.
"""

BUNDLE_FILENAME = "monlib_bundle.zip"
LICENCE_FILENAME = "LICENSE-CCP4-MONOMERS.txt"
