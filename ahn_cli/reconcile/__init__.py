"""Reconcile context: interpolate the AHN cloud onto the orthophoto grid.

Turns a fetched orthophoto and AHN point cloud -- on different native grids --
into a single coloured point cloud sampled on the ortho's grid, emitted in the
TouchDesigner-friendly ``laz``/``ply``/``pt``/``exr`` formats.
"""
