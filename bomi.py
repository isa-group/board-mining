"""Compatibility wrapper for the package implementation.

Existing notebooks in this repository import ``bomi`` from the repository root.
The installable package implementation lives under ``src/bomi``.
"""

from src.bomi import *  # noqa: F401
