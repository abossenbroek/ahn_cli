"""Pure ``KHR_mesh_quantization`` quantizer for tile-local RTC geometry.

Turns the float32 RTC vertices :func:`ahn_cli.tiles3d.mesh.build_tile_mesh`
produces (glTF y-up, tile-local) into per-axis ``uint16`` integers that a
glTF runtime dequantizes through the node's ``scale``/``translation``
only — the ``KHR_mesh_quantization`` contract. The transform is a
per-axis affine over the tile's *actual* per-axis data extents::

    translation[a] = min(vertices[:, a])
    scale[a]       = (max(vertices[:, a]) - min(vertices[:, a])) / 65535
    q[:, a]        = round_half_even((vertices[:, a] - translation[a]) / scale[a])
    dequant[:, a]  = q[:, a] * scale[a] + translation[a]

with ``q`` clamped to ``[0, 65535]``. For a terrain tile those extents
*are* the pixel span in X/Y and the height range in Z; they are derived
from the data, never from grid metadata.

Rounding is round-half-even (banker's rounding, via :func:`numpy.rint`).

Error bound: the worst-case round-trip error on axis ``a`` is half a
quantization step,

    ``|dequant[:, a] - source[:, a]| <= scale[a] / 2 == extent[a] / 65535 / 2``.

At 8 cm / 256 px tiles the XY extent is ~20.5 m, so the XY bound is
~0.16 mm; the Z bound follows the tile's own height range.
:func:`position_error_bound` exports this formula so callers assert
against it, never a literal.

Degenerate axis (a flat tile axis or a single vertex, extent 0): a zero
``scale`` is invalid glTF, so ``scale[a]`` is set to :data:`EPSILON_SCALE`
and ``translation[a]`` to ``min[a]``. Every value on that axis equals
``min[a]``, so ``q`` is 0 and dequantizes to exactly ``min[a]`` — the
round-trip error is 0 and the node scale stays non-zero.

Texture coordinates use core-glTF normalized ``uint16`` (no extension):
``q = round_half_even(uv * 65535)``, dequant ``uv' = q / 65535``, so the
worst-case UV error is ``1 / 65535 / 2``. Inputs must already lie in
``[0, 1]`` (texel-centre UVs do); anything outside is a
:class:`~ahn_cli.tiles3d.errors.Tiles3dError`.

Pure module, no I/O: all intermediate math is float64 and every output is
a deterministic function of the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "EPSILON_SCALE",
    "UINT16_MAX",
    "QuantizedAxis",
    "QuantizedPositions",
    "axis_error_bound",
    "dequantize_axis",
    "dequantize_positions",
    "position_error_bound",
    "quantize_axis",
    "quantize_positions",
    "quantize_uvs",
]

UINT16_MAX = 65535
"""Largest quantization level; each axis is stored in ``[0, 65535]``."""

EPSILON_SCALE = 1e-9
"""Substitute per-axis scale (metres/unit) for a zero-extent axis: keeps the
glTF node scale non-zero while every value quantizes to 0 (exact round-trip)."""

_MATRIX_NDIM = 2
"""A vertex/UV block is a 2-D ``(n, width)`` array."""

Axis3 = tuple[float, float, float]
"""A per-axis (x, y, z) triple of floats."""


@dataclass(frozen=True, eq=False)
class QuantizedPositions:
    """Per-axis ``uint16`` positions plus their glTF dequant transform.

    Contract (fields):
        - ``ints``: ``(n, 3)`` uint16 quantized vertices, each component
          in ``[0, 65535]``.
        - ``scale``: per-axis ``KHR_mesh_quantization`` node scale.
        - ``translation``: per-axis node translation (the axis minimum).

    Invariants:
        - ``dequant = ints * scale + translation`` reconstructs the
          source to within :func:`position_error_bound` per axis.
        - Every ``scale`` component is strictly positive
          (:data:`EPSILON_SCALE` on a zero-extent axis), so the transform
          is valid glTF.
        - ``eq=False``: wraps an array, so instances compare by identity.
    """

    ints: npt.NDArray[np.uint16]
    scale: Axis3
    translation: Axis3


def quantize_positions(
    vertices: npt.NDArray[np.floating],
) -> QuantizedPositions:
    """Quantize RTC vertices to per-axis ``uint16`` (KHR_mesh_quantization).

    Contract:
        - ``vertices``: ``(n, 3)`` float array (``n >= 1``), all finite,
          in tile-local RTC space.
        - Returns a :class:`QuantizedPositions` whose ``ints`` and
          ``(scale, translation)`` are pure deterministic functions of
          ``vertices``: per-axis ``translation = min``,
          ``scale = extent / 65535`` (or :data:`EPSILON_SCALE` when the
          axis extent is 0), and ``ints = round_half_even((v -
          translation) / scale)`` clamped to ``[0, 65535]``.

    Failure modes:
        - Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if
          ``vertices`` is not ``(n, 3)`` with ``n >= 1`` or holds any
          non-finite value.
    """
    data = _finite_2d(vertices, 3, "positions")
    lo = data.min(axis=0)
    extent = data.max(axis=0) - lo
    scale = np.where(extent > 0.0, extent / UINT16_MAX, EPSILON_SCALE)
    ints = np.clip(np.rint((data - lo) / scale), 0, UINT16_MAX).astype(
        np.uint16,
    )
    return QuantizedPositions(
        ints=ints,
        scale=(float(scale[0]), float(scale[1]), float(scale[2])),
        translation=(float(lo[0]), float(lo[1]), float(lo[2])),
    )


def dequantize_positions(
    quantized: QuantizedPositions,
) -> npt.NDArray[np.float64]:
    """Reconstruct float64 RTC vertices from a :class:`QuantizedPositions`.

    Contract:
        - Applies the stored glTF transform ``v = ints * scale +
          translation`` per axis and returns an ``(n, 3)`` float64 array.
        - Inverse to within :func:`position_error_bound`; a zero-extent
          axis reconstructs its single value exactly.
    """
    scale = np.asarray(quantized.scale, dtype=np.float64)
    translation = np.asarray(quantized.translation, dtype=np.float64)
    return quantized.ints.astype(np.float64) * scale + translation


def position_error_bound(scale: Axis3) -> Axis3:
    """Return the per-axis worst-case round-trip error of a position.

    Contract:
        - Returns ``scale[a] / 2`` per axis — half a quantization step,
          equivalently ``extent[a] / 65535 / 2``. A dequantized position
          differs from its source by at most this on each axis.
        - The verifier asserts ``|dequant - source| <= bound`` against
          this exported formula, never a literal.
    """
    return (scale[0] / 2.0, scale[1] / 2.0, scale[2] / 2.0)


@dataclass(frozen=True, eq=False)
class QuantizedAxis:
    """A single axis' ``uint16`` levels plus its affine dequant transform.

    Contract (fields):
        - ``ints``: ``(n,)`` uint16 quantized values, each in ``[0, 65535]``.
        - ``scale``: metres per level (:data:`EPSILON_SCALE` on a zero-extent
          axis), always strictly positive.
        - ``offset``: the axis minimum (the affine translation).

    Invariants:
        - ``dequant = ints * scale + offset`` reconstructs the source to
          within :func:`axis_error_bound`.
        - ``eq=False``: wraps an array, so instances compare by identity.
    """

    ints: npt.NDArray[np.uint16]
    scale: float
    offset: float


def quantize_axis(values: npt.NDArray[np.floating]) -> QuantizedAxis:
    """Quantize one axis of values to ``uint16`` (the position scheme, 1-D).

    Contract:
        - ``values``: a ``(n,)`` float array (``n >= 1``), all finite. Uses
          exactly the per-axis affine of :func:`quantize_positions`:
          ``offset = min``, ``scale = extent / 65535`` (or
          :data:`EPSILON_SCALE` when the extent is 0), and
          ``ints = round_half_even((v - offset) / scale)`` clamped to
          ``[0, 65535]`` — so a value quantized here matches the same value
          quantized as one column of :func:`quantize_positions`.

    Failure modes:
        - Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if ``values``
          is not ``(n,)`` with ``n >= 1`` or holds any non-finite value.
    """
    data = _finite_1d(values, "axis")
    lo = float(data.min())
    extent = float(data.max()) - lo
    scale = extent / UINT16_MAX if extent > 0.0 else EPSILON_SCALE
    ints = np.clip(np.rint((data - lo) / scale), 0, UINT16_MAX).astype(
        np.uint16,
    )
    return QuantizedAxis(ints=ints, scale=scale, offset=lo)


def dequantize_axis(quantized: QuantizedAxis) -> npt.NDArray[np.float64]:
    """Reconstruct float64 values from a :class:`QuantizedAxis`.

    Contract:
        - Returns ``ints * scale + offset`` as a ``(n,)`` float64 array,
          inverse to within :func:`axis_error_bound`; a zero-extent axis
          reconstructs its single value exactly.
    """
    return quantized.ints.astype(np.float64) * quantized.scale + (
        quantized.offset
    )


def axis_error_bound(scale: float) -> float:
    """Return the worst-case round-trip error of a single quantized axis.

    Contract:
        - Returns ``scale / 2`` — half a quantization step, equivalently
          ``extent / 65535 / 2``. A dequantized value differs from its
          source by at most this. The verifier asserts against this
          exported formula, never a literal.
    """
    return scale / 2.0


def quantize_uvs(uvs: npt.NDArray[np.floating]) -> npt.NDArray[np.uint16]:
    """Quantize texel-centre UVs to core-glTF normalized ``uint16``.

    Contract:
        - ``uvs``: ``(n, 2)`` float array (``n >= 1``), all finite, every
          component in ``[0, 1]``.
        - Returns ``(n, 2)`` uint16 ``= round_half_even(uv * 65535)``; a
          runtime dequantizes via ``uv' = q / 65535`` (normalized
          accessor, no extension). Worst-case error ``1 / 65535 / 2``.

    Failure modes:
        - Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if ``uvs``
          is not ``(n, 2)`` with ``n >= 1``, holds a non-finite value, or
          any component lies outside ``[0, 1]``.
    """
    data = _finite_2d(uvs, 2, "uvs")
    if data.min() < 0.0 or data.max() > 1.0:
        msg = "quantize uvs: every component must lie in [0, 1]."
        raise Tiles3dError(msg)
    return np.clip(np.rint(data * UINT16_MAX), 0, UINT16_MAX).astype(
        np.uint16
    )


def _finite_1d(
    array: npt.NDArray[np.floating], name: str
) -> npt.NDArray[np.float64]:
    """Validate an ``(n,)`` finite float vector; return it as float64.

    Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` when the shape is
    not ``(n,)`` with ``n >= 1`` or any value is non-finite.
    """
    data = np.asarray(array, dtype=np.float64)
    if data.ndim != 1 or data.shape[0] < 1:
        msg = (
            f"quantize {name}: expected a (n,) array with n >= 1, "
            f"got shape {data.shape}."
        )
        raise Tiles3dError(msg)
    if not np.isfinite(data).all():
        msg = f"quantize {name}: every value must be finite."
        raise Tiles3dError(msg)
    return data


def _finite_2d(
    array: npt.NDArray[np.floating], width: int, name: str
) -> npt.NDArray[np.float64]:
    """Validate an ``(n, width)`` finite float block; return it as float64.

    Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` when the shape is
    not ``(n, width)`` with ``n >= 1`` or any value is non-finite.
    """
    data = np.asarray(array, dtype=np.float64)
    if (
        data.ndim != _MATRIX_NDIM
        or data.shape[0] < 1
        or data.shape[1] != width
    ):
        msg = (
            f"quantize {name}: expected an (n, {width}) array with "
            f"n >= 1, got shape {data.shape}."
        )
        raise Tiles3dError(msg)
    if not np.isfinite(data).all():
        msg = f"quantize {name}: every value must be finite."
        raise Tiles3dError(msg)
    return data
