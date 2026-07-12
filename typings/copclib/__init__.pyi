# Local type stubs for the untyped ``copclib`` (copc-lib pybind11) surface the
# ``copc`` bounded context depends on.
#
# ``copclib`` is a compiled pybind11 module with no ``py.typed`` marker, so
# under ``pyright`` strict every access resolves to *unknown*. As with the
# vendored ``laspy`` stubs, only the symbols actually used are declared, so a
# typo in a top-level name still fails type-checking, while the genuinely
# dynamic pybind11 objects (mutable config views, char vectors) are ``Any``.

from typing import Any

class Vector3:
    def __init__(self, x: float, y: float, z: float) -> None: ...

class VoxelKey:
    d: int
    x: int
    y: int
    z: int
    def __init__(self, d: int, x: int, y: int, z: int) -> None: ...

class VectorChar:
    def __init__(self, data: Any) -> None: ...
    def __len__(self) -> int: ...

class CopcInfo:
    center_x: float
    center_y: float
    center_z: float
    halfsize: float
    spacing: float

class CopcConfigWriter:
    copc_info: CopcInfo
    las_header: Any
    def __init__(
        self,
        point_format_id: int,
        scale: tuple[float, float, float] = ...,
        offset: tuple[float, float, float] = ...,
        wkt: str = ...,
        has_extended_stats: bool = ...,
    ) -> None: ...

class FileWriter:
    copc_config: CopcConfigWriter
    def __init__(self, file_path: str, config: CopcConfigWriter) -> None: ...
    def AddNode(
        self, key: VoxelKey, uncompressed_data: VectorChar
    ) -> Any: ...
    def Close(self) -> None: ...
