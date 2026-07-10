"""The ``prep`` bounded context: data *transform and export*.

Responsibility: turn cached raw source tiles into finished deliverables. This
context owns clipping, classification filtering, dedup, decimation, mosaicking,
raster/point-cloud export, and writing the provenance sidecar -- everything
downstream of acquisition. It consumes what the ``fetch`` context produced and
never reaches out to a portal itself, so the two contexts stay separate.

This is a context skeleton: it declares the boundary only and holds no logic
yet.
"""

from __future__ import annotations
