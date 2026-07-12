# Local type stubs for the untyped ``meshoptimizer`` (pybind11 codec) surface
# the ``tiles3d`` game profile depends on.
#
# ``meshoptimizer`` ships a compiled extension with no ``py.typed`` marker, so
# under ``pyright`` strict every access resolves to *unknown*. As with the
# vendored ``copclib``/``laspy`` stubs, only the four functions actually used
# are declared, so a typo in a top-level name still fails type-checking while
# the genuinely dynamic pybind11 buffers stay ``ndarray``.

import numpy as np
import numpy.typing as npt

def encode_vertex_buffer(
    vertices: npt.NDArray[np.uint8],
    vertex_count: int,
    vertex_size: int,
) -> bytes: ...
def decode_vertex_buffer(
    vertex_count: int,
    vertex_size: int,
    buffer: bytes,
) -> npt.NDArray[np.generic]: ...
def encode_index_buffer(
    indices: npt.NDArray[np.uint32],
    index_count: int,
    vertex_count: int,
) -> bytes: ...
def decode_index_buffer(
    index_count: int,
    index_size: int,
    buffer: bytes,
) -> npt.NDArray[np.generic]: ...
