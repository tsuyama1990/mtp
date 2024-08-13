"""Glue package to convert MLIP-2 to Python.

This package provides Python bindings for the MLIP-2 software.
"""

from .core import MTPCalactor, PyConfiguration

__all__ = ["MTPCalactor", "PyConfiguration"]
