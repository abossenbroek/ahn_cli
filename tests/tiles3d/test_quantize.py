"""Tests for the pure KHR_mesh_quantization quantizer."""

from __future__ import annotations

import numpy as np
import pytest

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.quantize import (
    EPSILON_SCALE,
    UINT16_MAX,
    dequantize_positions,
    position_error_bound,
    quantize_positions,
    quantize_uvs,
)


def _grid(width: int, height: int, seed: int) -> np.ndarray:
    """Build a deterministic ``(n, 3)`` float32 RTC-like vertex block."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-20.0, 20.0, width, dtype=np.float64)
    y = np.linspace(-15.0, 15.0, height, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    z = rng.uniform(-5.0, 40.0, (height, width))
    return np.column_stack([xx.ravel(), yy.ravel(), z.ravel()]).astype(
        np.float32,
    )


def test_transform_follows_khr_semantics() -> None:
    """The transform is per-axis min for translation, extent / 65535 scale."""
    verts = _grid(7, 5, 1)
    qp = quantize_positions(verts)
    data = verts.astype(np.float64)
    for axis in range(3):
        lo = float(data[:, axis].min())
        hi = float(data[:, axis].max())
        assert qp.translation[axis] == lo
        assert qp.scale[axis] == (hi - lo) / UINT16_MAX


def test_ints_are_uint16_in_range() -> None:
    """Every quantized component is a uint16 within [0, 65535]."""
    qp = quantize_positions(_grid(9, 6, 2))
    assert qp.ints.dtype == np.uint16
    assert qp.ints.shape == (54, 3)
    assert int(qp.ints.min()) >= 0
    assert int(qp.ints.max()) <= UINT16_MAX


def test_extremes_map_to_endpoints() -> None:
    """The per-axis min quantizes to 0 and the max to 65535."""
    qp = quantize_positions(_grid(8, 8, 3))
    assert int(qp.ints.min(axis=0).max()) == 0
    for axis in range(3):
        assert int(qp.ints[:, axis].min()) == 0
        assert int(qp.ints[:, axis].max()) == UINT16_MAX


def test_round_half_even_is_used() -> None:
    """A value exactly on a half-step rounds to the even neighbour."""
    # extent 2.0 over 65534 steps is contrived; use a direct half case:
    # one axis spans [0, 2], the mid vertex 0.5 * scale lands on x.5.
    verts = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    qp = quantize_positions(verts)
    # scale = 2/65535; (1-0)/scale = 32767.5 -> banker's rounds to 32768.
    assert int(qp.ints[2, 0]) == 32768


def test_round_trip_within_documented_bound() -> None:
    """|dequant - source| never exceeds position_error_bound per axis."""
    for seed in range(200):
        verts = _grid(6, 5, seed)
        qp = quantize_positions(verts)
        deq = dequantize_positions(qp)
        source = verts.astype(np.float64)
        bound = np.asarray(position_error_bound(qp.scale))
        diff = np.abs(deq - source)
        slack = 8.0 * np.finfo(np.float64).eps * (np.abs(source) + 1.0)
        assert np.all(diff <= bound + slack)


def test_idempotent_in_int_domain() -> None:
    """Re-quantizing a dequantized block reproduces the same ints."""
    for seed in range(200):
        qp = quantize_positions(_grid(6, 5, seed))
        again = quantize_positions(dequantize_positions(qp))
        assert np.array_equal(again.ints, qp.ints)


def test_deterministic() -> None:
    """Same input yields identical ints and identical transform."""
    verts = _grid(7, 7, 9)
    a = quantize_positions(verts)
    b = quantize_positions(verts)
    assert np.array_equal(a.ints, b.ints)
    assert a.scale == b.scale
    assert a.translation == b.translation


def test_position_error_bound_is_half_a_step() -> None:
    """The exported bound is exactly scale / 2 per axis."""
    scale = (4.0e-4, 6.0e-4, 1.0e-3)
    assert position_error_bound(scale) == (2.0e-4, 3.0e-4, 5.0e-4)


def test_zero_range_axis_uses_epsilon_scale() -> None:
    """A flat axis gets EPSILON_SCALE, ints 0, and exact round-trip."""
    verts = np.array(
        [[0.0, 5.0, 10.0], [4.0, 5.0, 40.0], [8.0, 5.0, 25.0]],
        dtype=np.float32,
    )
    qp = quantize_positions(verts)
    assert qp.scale[1] == EPSILON_SCALE
    assert np.all(qp.ints[:, 1] == 0)
    deq = dequantize_positions(qp)
    assert np.all(deq[:, 1] == 5.0)


def test_single_vertex_span() -> None:
    """A single vertex is flat on every axis: all ints 0, exact recon."""
    verts = np.array([[3.0, -2.0, 17.0]], dtype=np.float32)
    qp = quantize_positions(verts)
    assert qp.scale == (EPSILON_SCALE, EPSILON_SCALE, EPSILON_SCALE)
    assert np.all(qp.ints == 0)
    assert np.array_equal(dequantize_positions(qp), verts.astype(np.float64))


def test_quantized_positions_compares_by_identity() -> None:
    """``eq=False``: distinct results never compare equal."""
    verts = _grid(4, 4, 0)
    assert quantize_positions(verts) != quantize_positions(verts)


def test_quantize_positions_rejects_wrong_ndim() -> None:
    """A 1-D array is not an (n, 3) block."""
    with pytest.raises(Tiles3dError, match="positions"):
        quantize_positions(np.zeros(3, dtype=np.float32))


def test_quantize_positions_rejects_wrong_width() -> None:
    """An (n, 2) array is not an (n, 3) block."""
    with pytest.raises(Tiles3dError, match="positions"):
        quantize_positions(np.zeros((4, 2), dtype=np.float32))


def test_quantize_positions_rejects_empty() -> None:
    """An (0, 3) array has no vertices to quantize."""
    with pytest.raises(Tiles3dError, match="positions"):
        quantize_positions(np.zeros((0, 3), dtype=np.float32))


def test_quantize_positions_rejects_non_finite() -> None:
    """A NaN or inf vertex is refused."""
    verts = _grid(4, 4, 0)
    verts[0, 2] = np.nan
    with pytest.raises(Tiles3dError, match="finite"):
        quantize_positions(verts)


def test_quantize_uvs_round_trips_within_bound() -> None:
    """Texel-centre UVs dequantize (q / 65535) within half a step."""
    rng = np.random.default_rng(4)
    uvs = rng.uniform(0.0, 1.0, (128, 2)).astype(np.float32)
    ints = quantize_uvs(uvs)
    assert ints.dtype == np.uint16
    deq = ints.astype(np.float64) / UINT16_MAX
    bound = 1.0 / UINT16_MAX / 2.0
    assert np.all(np.abs(deq - uvs.astype(np.float64)) <= bound + 1e-12)


def test_quantize_uvs_maps_endpoints() -> None:
    """0 maps to 0 and 1 maps to 65535 exactly."""
    ints = quantize_uvs(np.array([[0.0, 1.0]], dtype=np.float32))
    assert int(ints[0, 0]) == 0
    assert int(ints[0, 1]) == UINT16_MAX


def test_quantize_uvs_rejects_below_zero() -> None:
    """A UV component below 0 is out of the normalized range."""
    with pytest.raises(Tiles3dError, match=r"\[0, 1\]"):
        quantize_uvs(np.array([[-0.01, 0.5]], dtype=np.float32))


def test_quantize_uvs_rejects_above_one() -> None:
    """A UV component above 1 is out of the normalized range."""
    with pytest.raises(Tiles3dError, match=r"\[0, 1\]"):
        quantize_uvs(np.array([[0.5, 1.01]], dtype=np.float32))


def test_quantize_uvs_rejects_wrong_width() -> None:
    """An (n, 3) array is not a UV block."""
    with pytest.raises(Tiles3dError, match="uvs"):
        quantize_uvs(np.zeros((4, 3), dtype=np.float32))


def test_quantize_uvs_rejects_non_finite() -> None:
    """A non-finite UV is refused before the range check."""
    with pytest.raises(Tiles3dError, match="finite"):
        quantize_uvs(np.array([[np.inf, 0.5]], dtype=np.float32))
