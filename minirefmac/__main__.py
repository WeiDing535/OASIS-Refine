"""Allow ``python -m minirefmac`` as well as ``python -m minirefmac.cli``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
