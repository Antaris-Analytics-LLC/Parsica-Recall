"""python -m parsica_recall <command>: the same as the parsica-recall command."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
