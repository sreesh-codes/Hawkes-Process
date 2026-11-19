"""Alias module for the deliverable name ``sttests.py``.

The statistical-testing code lives in :mod:`analysis.stat_tests` (a clearer
name); this thin shim re-exports it so the Phase-5 deliverable path
``analysis/sttests.py`` also resolves.
"""
from analysis.stat_tests import (  # noqa: F401
    TESTED_METRICS, bootstrap_ci, paired_tests, plot_significance, run_all_tests,
)

__all__ = ["TESTED_METRICS", "bootstrap_ci", "paired_tests", "run_all_tests",
           "plot_significance"]
