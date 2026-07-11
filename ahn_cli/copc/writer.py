"""Copc-context container writing: a typed façade over copclib.

Everything the original PDAL bug was made of is nailed down here:

- Nodes are handed to copclib as **raw pre-packed PDRF byte buffers**, so the
  int32 coordinates chosen upstream land in the file untouched — there is no
  second quantization path whose rounding could disagree with the header.
- The LAS header min/max are set (after all nodes are written) from the exact
  quantized extremes of the *written* points, decoded with the same
  ``int * scale + offset`` float64 expression every reader uses — bit-equal
  to the decoded points, and always at least 1 m inside the cube faces.
- Points are sorted by GPS time inside each node (``copc-validator`` warns on
  unsorted nodes) and the COPC info VLR's ``gpstime_minimum/maximum`` — which
  the copclib binding never fills — are patched in place after ``Close()``,
  at the byte offsets the COPC 1.0 spec fixes for the (mandatory first) info
  VLR.

The written SRS must be WKT1 (``WKT1_GDAL``): the proj4js parser inside
``copc-validator`` cannot read WKT2 strings.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import copclib
import numpy as np
from pyproj import CRS

from ahn_cli.copc.octree import CopcError

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.copc.octree import BuildPlan, NodeKey

RGB_POINT_FORMAT = 7
BARE_POINT_FORMAT = 6

_CORE_FIELDS = [
    ("X", "<i4"),
    ("Y", "<i4"),
    ("Z", "<i4"),
    ("intensity", "<u2"),
    ("returns", "u1"),
    ("flags", "u1"),
    ("classification", "u1"),
    ("user_data", "u1"),
    ("scan_angle", "<i2"),
    ("point_source_id", "<u2"),
    ("gps_time", "<f8"),
]
_RGB_FIELDS = [("red", "<u2"), ("green", "<u2"), ("blue", "<u2")]
_PDRF6_DTYPE = np.dtype(_CORE_FIELDS)
_PDRF7_DTYPE = np.dtype(_CORE_FIELDS + _RGB_FIELDS)

_LAS14_HEADER_SIZE = 375
_VLR_HEADER_SIZE = 54
_VLR_USER_ID_OFFSET = 2  # after the u16 ``reserved`` field
_COPC_USER_ID = b"copc"
_GPS_MIN_FIELD_OFFSET = 56  # center xyz + halfsize + spacing + hier off/size
_MAX_RETURNS = 15
_SYSTEM_IDENTIFIER = "ahn_cli"
_CREATION_DAY = 1  # pinned (2020-01-01), matching reconcile's LAZ writer
_CREATION_YEAR = 2020


def rd_new_wkt() -> str:
    """Return EPSG:28992 (RD New) as WKT1 for the written SRS VLR."""
    return CRS.from_epsg(28992).to_wkt("WKT1_GDAL")


class CopcNodeWriter:
    """Streaming COPC writer: construct, ``add_node`` per node, ``finish``.

    Contract:
        - ``point_format_id`` is 6 (bare) or 7 (RGB); node ``records`` use
          :data:`ahn_cli.copc.scatter.RECORD_DTYPE` and are written once per
          octree node, any order, memory proportional to one node.
        - ``finish()`` seals the file: header bounds from the written
          extremes, per-return histogram, GPS-range VLR patch. Returns the
          written point count.

    Invariants:
        - Deterministic: node payloads are sorted by (gps_time, input order).
    """

    def __init__(
        self,
        path: Path,
        plan: BuildPlan,
        *,
        point_format_id: int,
        wkt: str,
        generating_software: str = "ahn_cli copc",
    ) -> None:
        """Open ``path`` for writing with the plan's exact cube geometry."""
        config = copclib.CopcConfigWriter(
            point_format_id=point_format_id,
            scale=(plan.scale, plan.scale, plan.scale),
            offset=plan.offsets,
            wkt=wkt,
        )
        half = float(plan.side_m) / 2.0
        info = config.copc_info
        info.center_x = plan.offsets[0] + half
        info.center_y = plan.offsets[1] + half
        info.center_z = plan.offsets[2] + half
        info.halfsize = half
        info.spacing = float(plan.side_m) / plan.sample_grid
        header = config.las_header
        header.generating_software = generating_software
        # Pinned like reconcile's LAZ writer: byte-deterministic output must
        # carry no timestamps or host metadata (copclib itself writes 0/0).
        header.system_identifier = _SYSTEM_IDENTIFIER
        header.creation_day = _CREATION_DAY
        header.creation_year = _CREATION_YEAR
        try:
            self._writer = copclib.FileWriter(str(path), config)
        except RuntimeError as exc:
            msg = f"cannot open {path} for COPC writing: {exc}"
            raise CopcError(msg) from exc
        self._path = path
        self._plan = plan
        self._dtype = (
            _PDRF7_DTYPE
            if point_format_id == RGB_POINT_FORMAT
            else _PDRF6_DTYPE
        )
        self._count = 0
        self._by_return = np.zeros(_MAX_RETURNS, dtype=np.int64)
        self._int_mins = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
        self._int_maxs = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
        self._gps_min = np.inf
        self._gps_max = -np.inf

    def add_node(self, key: NodeKey, records: npt.NDArray[np.void]) -> None:
        """Write one octree node from scatter records (must be non-empty)."""
        if records.shape[0] == 0:
            msg = (
                f"node {key} has zero points; copc-validator warns on "
                "zero-point nodes so the builder must never emit them"
            )
            raise CopcError(msg)
        order = np.argsort(records["gps_time"], kind="stable")
        ordered = records[order]
        packed = self._pack(ordered)
        self._writer.AddNode(
            copclib.VoxelKey(key.level, key.x, key.y, key.z),
            copclib.VectorChar(np.frombuffer(packed, dtype=np.int8)),
        )
        self._track(ordered)

    def finish(self) -> int:
        """Seal the file (header bounds, histogram, GPS patch); return count."""
        if self._count == 0:
            msg = "refusing to finish a COPC file with no points written"
            raise CopcError(msg)
        header = self._writer.copc_config.las_header
        offsets = self._plan.offsets
        scale = self._plan.scale
        header.min = copclib.Vector3(
            float(self._int_mins[0]) * scale + offsets[0],
            float(self._int_mins[1]) * scale + offsets[1],
            float(self._int_mins[2]) * scale + offsets[2],
        )
        header.max = copclib.Vector3(
            float(self._int_maxs[0]) * scale + offsets[0],
            float(self._int_maxs[1]) * scale + offsets[1],
            float(self._int_maxs[2]) * scale + offsets[2],
        )
        header.points_by_return = [int(n) for n in self._by_return]
        self._writer.Close()
        if (self._gps_min, self._gps_max) != (0.0, 0.0):
            patch_gps_range(self._path, self._gps_min, self._gps_max)
        return self._count

    def _pack(self, records: npt.NDArray[np.void]) -> bytes:
        """Pack scatter records into raw PDRF 6/7 point bytes."""
        packed = np.zeros(records.shape[0], dtype=self._dtype)
        packed["X"] = records["x"]
        packed["Y"] = records["y"]
        packed["Z"] = records["z"]
        packed["intensity"] = records["intensity"]
        packed["returns"] = records["return_number"] | (
            records["number_of_returns"] << 4
        )
        packed["classification"] = records["classification"]
        packed["user_data"] = records["user_data"]
        packed["scan_angle"] = records["scan_angle"]
        packed["point_source_id"] = records["point_source_id"]
        packed["gps_time"] = records["gps_time"]
        if self._dtype is _PDRF7_DTYPE:
            packed["red"] = records["red"]
            packed["green"] = records["green"]
            packed["blue"] = records["blue"]
        return packed.tobytes()

    def _track(self, records: npt.NDArray[np.void]) -> None:
        """Fold one node's records into the file-level accumulators."""
        self._count += records.shape[0]
        histogram = np.bincount(
            records["return_number"].astype(np.int64),
            minlength=_MAX_RETURNS + 1,
        )
        self._by_return += histogram[1 : _MAX_RETURNS + 1]
        for axis, name in enumerate(("x", "y", "z")):
            values = records[name].astype(np.int64)
            self._int_mins[axis] = min(
                int(self._int_mins[axis]), int(values.min())
            )
            self._int_maxs[axis] = max(
                int(self._int_maxs[axis]), int(values.max())
            )
        gps = records["gps_time"]
        self._gps_min = min(self._gps_min, float(gps.min()))
        self._gps_max = max(self._gps_max, float(gps.max()))


def patch_gps_range(path: Path, low: float, high: float) -> None:
    """Write ``gpstime_minimum/maximum`` into the COPC info VLR in place.

    The copclib binding does not expose these two CopcInfo fields, but the
    COPC 1.0 spec fixes the info VLR as the first VLR (record data at byte
    ``375 + 54``) with the GPS pair 56 bytes into the record. Leaving them
    zero fails ``copc-validator``'s ``gpsTime`` bounds check whenever real
    GPS times are present.
    """
    with path.open("r+b") as handle:
        handle.seek(_LAS14_HEADER_SIZE + _VLR_USER_ID_OFFSET)
        user_id = handle.read(len(_COPC_USER_ID))
        if user_id != _COPC_USER_ID:
            msg = (
                f"{path} does not carry the COPC info VLR first; "
                "refusing to patch its GPS time range"
            )
            raise CopcError(msg)
        handle.seek(
            _LAS14_HEADER_SIZE + _VLR_HEADER_SIZE + _GPS_MIN_FIELD_OFFSET
        )
        handle.write(struct.pack("<dd", low, high))
