# Stub for ``laspy.copc``: the CopcReader surface used to verify a written file
# reopens as a valid COPC, and to read only the octree nodes that intersect a
# 2-D bounding box (``query``) for a clipped export.

from typing import Any

class Bounds:
    def __init__(self, mins: Any, maxs: Any) -> None: ...

class CopcReader:
    header: Any
    @classmethod
    def open(cls, path: str) -> Any: ...
    def query(self, bounds: Any = ..., resolution: Any = ...) -> Any: ...
