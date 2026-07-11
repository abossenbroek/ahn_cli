# Local type stubs for the untyped ``laspy`` API surface WP10 depends on.
#
# ``laspy`` ships no ``py.typed`` marker, so under ``pyright`` strict every
# access resolves to an *unknown* type and raises ``reportMissingTypeStubs`` /
# ``reportUnknownMemberType``. Excluding the consuming module from pyright is
# forbidden by the epic's coverage/typing governance, so instead this stub keeps
# first-party ``prep`` logic fully strict-checked while modelling the genuinely
# dynamic ``laspy`` objects (dimension access by name, scaled views) as ``Any``.
#
# Only the symbols WP10 actually uses are declared, so a typo in a top-level
# ``laspy`` name still fails type-checking; later work packages extend this stub
# rather than silencing the whole library. This directory is vendored third-party
# typing infrastructure, so it is excluded from the first-party ruff lint.

from typing import Any

class LaspyException(Exception): ...

class LasHeader:
    offsets: Any
    scales: Any
    point_format: Any
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

class ScaleAwarePointRecord:
    array: Any
    point_format: Any
    @classmethod
    def zeros(
        cls, point_count: int, *, header: Any
    ) -> ScaleAwarePointRecord: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

class LasData:
    points: Any
    def __init__(self, header: Any) -> None: ...
    def write(
        self,
        destination: Any,
        do_compress: Any = ...,
        laz_backend: Any = ...,
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

def open(source: Any, *args: Any, **kwargs: Any) -> Any: ...
