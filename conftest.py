"""Root conftest.py — adds src/ to sys.path so tests can import modules directly.

Tests use bare imports like ``from build_features import enrich_features``.
This file ensures that works after the source files were moved into src/.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
