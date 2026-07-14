"""COPC export bounded context: streaming octree build + COPC container write.

This context turns a finished LAZ deliverable (from ``prep`` or ``reconcile``)
into a Cloud-Optimized Point Cloud (``.copc.laz``) whose declared LAS-header
bounds and COPC octree cube are consistent *by construction* — the fix for
``docs/bugs/pdal-copc-xyz-bounds-flat-terrain.md``. It never reaches
out to a distribution portal and never re-fetches: it is a pure transform over
a single input cloud, streamed in bounded memory.
"""

from __future__ import annotations
