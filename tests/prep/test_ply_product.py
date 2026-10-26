"""Tests for the PLY product: lossless export plus its provenance sidecar."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from ahn_cli.prep.decimate import PoissonThinning
from ahn_cli.prep.ply import CoordMode, PlyExportError, ShareThinning
from ahn_cli.prep.ply_product import (
    ClassOrigin,
    PlyProductRequest,
    export_ply_product,
)
from ahn_cli.provenance.product import (
    SCHEMA_VERSION,
    read_product_sidecar,
    sidecar_path,
)
from tests.prep.ply_helpers import make_cloud, read_ply, write_cloud

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_T0 = datetime(2026, 10, 13, 12, 0, 0, tzinfo=timezone.utc)


def _sha(path: Path) -> str:
    """Return the sha256 hex of ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clock() -> datetime:
    """Return a fixed start time."""
    return _T0


def _ticker() -> Callable[[], float]:
    """Return a timer yielding 10.0 then 12.3456 (wall time 2.346)."""
    ticks = iter([10.0, 12.3456])
    return lambda: next(ticks)


def _block(record: dict[str, object], key: str) -> dict[str, Any]:
    """Return one top-level sidecar block as a plain dict."""
    return cast("dict[str, Any]", record[key])


def test_sidecar_is_written_beside_the_ply_and_matches_the_files(
    tmp_path: Path,
) -> None:
    """Every output carries a schema-valid sidecar whose hashes and counts hold."""
    cloud = make_cloud(300)
    src = tmp_path / "src.laz"
    write_cloud(src, cloud)
    out = tmp_path / "out.ply"

    stats = export_ply_product(
        PlyProductRequest(
            source=src,
            output=out,
            include_classes=(2, 6),
            class_origin=ClassOrigin.EXPLICIT,
            coords=CoordMode.FLOAT32_CENTRED,
        ),
        clock=_clock,
        timer=_ticker(),
    )

    side = sidecar_path(out)
    assert side.name == "out.ply.provenance.json"
    record = read_product_sidecar(side)  # validates the schema
    assert record["schema"] == SCHEMA_VERSION
    source = _block(record, "input")
    assert source["sha256"] == _sha(src)
    assert source["bytes"] == src.stat().st_size
    assert source["point_count"] == 300
    assert source["is_copc"] is False
    product = _block(record, "output")
    assert product["sha256"] == _sha(out)
    assert product["bytes"] == out.stat().st_size
    assert product["offset_xyz"] == list(stats.offset_xyz)
    assert product["coords"] == "float32-centred"
    ply = read_ply(out)
    names = [prop["name"] for prop in product["properties"]]
    assert names == list(ply.vertex.dtype.names or ())
    keep = np.isin(cloud.classification, [2, 6])
    assert _block(record, "counts") == {  # sidecar counts == PLY vertex count
        "source": 300,
        "scanned": 300,
        "selected": int(keep.sum()),
        "out": len(ply.vertex),
    }
    filters = _block(record, "filters")
    assert filters["class_include"] == [2, 6]
    assert filters["class_origin"] == "explicit"
    assert filters["bbox"] is None
    assert filters["thinning"] == {"mode": "none"}
    classes_out = dict(_block(record, "classes")["out"])
    assert set(classes_out) == {2, 6}
    assert sum(classes_out.values()) == int(keep.sum())
    assert _block(record, "run") == {
        "started_utc": "2026-10-13T12:00:00+00:00",
        "wall_seconds": 2.346,
    }
    assert _block(record, "tool")["laspy"] != "unknown"


def test_sidecar_records_each_thinning_rule_and_the_bbox(
    tmp_path: Path,
) -> None:
    """Spacing and share rules, the bbox and upstream filters are recorded."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(200))
    box = (194_000.0, 443_000.0, 194_080.0, 443_080.0)
    spacing = tmp_path / "s.ply"
    share = tmp_path / "h.ply"

    export_ply_product(
        PlyProductRequest(
            source=src,
            output=spacing,
            thinning=PoissonThinning(radius=3.0, seed=5),
            bbox=box,
            upstream={"stage": "prep", "include_classes": [2]},
        )
    )
    export_ply_product(
        PlyProductRequest(
            source=src,
            output=share,
            thinning=ShareThinning(share=0.5, seed=2),
            class_origin=ClassOrigin.EXPLICIT,
        )
    )

    a = _block(read_product_sidecar(sidecar_path(spacing)), "filters")
    b = _block(read_product_sidecar(sidecar_path(share)), "filters")
    assert a["thinning"] == {
        "mode": "spacing",
        "algorithm": "poisson-disk",
        "spacing_m": 3.0,
        "seed": 5,
    }
    assert a["bbox"] == list(box)
    assert a["upstream"] == {"stage": "prep", "include_classes": [2]}
    assert a["class_include"] is None
    assert a["class_origin"] == "all"
    assert b["thinning"] == {"mode": "share", "share": 0.5, "seed": 2}
    assert b["class_origin"] == "explicit"


def test_rebuild_is_byte_identical_except_the_run_block(
    tmp_path: Path,
) -> None:
    """Same input gives the same PLY bytes and the same sidecar sans run."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(250))
    first = tmp_path / "a.ply"
    second = tmp_path / "b.ply"

    export_ply_product(PlyProductRequest(source=src, output=first))
    export_ply_product(PlyProductRequest(source=src, output=second))

    assert first.read_bytes() == second.read_bytes()
    one = json.loads(sidecar_path(first).read_text())
    two = json.loads(sidecar_path(second).read_text())
    for record in (one, two):
        record.pop("run")
        record["output"].pop("name")
    assert one == two


def test_a_failed_export_writes_no_sidecar(tmp_path: Path) -> None:
    """An empty selection raises and leaves no sidecar behind."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(20))
    out = tmp_path / "out.ply"

    with pytest.raises(PlyExportError):
        export_ply_product(
            PlyProductRequest(source=src, output=out, include_classes=(18,))
        )

    assert not sidecar_path(out).exists()
