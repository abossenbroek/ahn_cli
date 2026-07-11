# Stub for ``laspy.copc``: only the CopcReader surface the copc-context
# tests use to verify a written file reopens as a valid COPC.

from typing import Any

class CopcReader:
    header: Any
    @classmethod
    def open(cls, path: str) -> Any: ...
