"""Put the project root on sys.path.

The operator scripts (costfloor.py, fi2010.py, gapstudy.py, build_dataset.py)
live at the root rather than inside src/lobforge, deliberately: they are not
importable library code and they are kept out of the Docker image. That choice
costs nothing except that pytest, which only adds tests/ to sys.path, cannot
see them. One insert fixes it for every test module.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
