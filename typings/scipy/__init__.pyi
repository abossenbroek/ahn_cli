"""Minimal type stub package for the untyped ``scipy`` API the reconcile verb uses.

``scipy`` ships no usable typing for these members under pyright strict, so this
partial stub declares only ``scipy.spatial.cKDTree`` (the kNN primitive) and
``scipy.interpolate.LinearNDInterpolator`` (Delaunay-linear interpolation). It is
typing infrastructure, not a faithful reproduction of the library, and lives
under ``typings/`` (ruff-excluded) so its surface is never linted as first-party
source.
"""
