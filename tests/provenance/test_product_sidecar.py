"""Tests for the product provenance sidecar: schema, codec and probes."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from typing import TYPE_CHECKING, Any

import pytest

from ahn_cli.provenance import product as product_module
from ahn_cli.provenance.product import (
    SCHEMA_VERSION,
    ProductProvenanceError,
    collect_tool_versions,
    fingerprint,
    product_record_to_bytes,
    read_product_sidecar,
    run_text,
    sidecar_path,
    validate_product_record,
    write_product_sidecar,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _record() -> dict[str, Any]:
    """Return a minimal valid record (every key, correct leaf types)."""
    return {
        "schema": SCHEMA_VERSION,
        "tool": {
            "ahn_cli": "0.3.5",
            "ahn_cli_commit": None,
            "ahn_cli_dirty": None,
            "python": "3.12.0",
            "laspy": "2.6.1",
            "lazrs": None,
            "numpy": "2.3.2",
            "pdal_python": None,
            "pdal_cli": None,
        },
        "input": {
            "path": "/x/in.laz",
            "sha256": "0" * 64,
            "bytes": 10,
            "point_count": 3,
            "point_format": 8,
            "las_version": "1.4",
            "is_copc": False,
            "las_scale": [0.001, 0.001, 0.001],
            "las_offset": [0.0, 0.0, 0.0],
            "mins": [0.0, 0.0, 0.0],
            "maxs": [1.0, 1.0, 1.0],
        },
        "filters": {
            "class_include": [2, 6],
            "class_origin": "default",
            "bbox": None,
            "thinning": {"mode": "none"},
            "upstream": None,
        },
        "counts": {"source": 3, "scanned": 3, "selected": 2, "out": 2},
        "classes": {"scanned": [[2, 3]], "out": [[2, 2]]},
        "output": {
            "name": "o.ply",
            "format": "ply",
            "sha256": "1" * 64,
            "bytes": 5,
            "coords": "float64",
            "offset_xyz": [0.0, 0.0, 0.0],
            "offset_rule": "world = vertex + offset_xyz",
            "properties": [{"name": "x", "type": "double"}],
            "colour": None,
        },
        "run": {
            "started_utc": "2026-10-13T12:00:00+00:00",
            "wall_seconds": 1.5,
        },
    }


def _no_output(argv: list[str], cwd: Path | None = None) -> str | None:
    """Stand in for run_text when no probe tool answers."""
    del argv, cwd


def _no_binary(name: str) -> str | None:
    """Stand in for shutil.which when the binary is absent."""
    del name


def _pdal_binary(name: str) -> str | None:
    """Stand in for shutil.which when pdal is installed."""
    del name
    return "/bin/pdal"


def test_sidecar_path_appends_the_suffix(tmp_path: Path) -> None:
    """The sidecar is named after the full product file name."""
    assert sidecar_path(tmp_path / "a.ply").name == "a.ply.provenance.json"


def test_valid_record_round_trips_deterministically(tmp_path: Path) -> None:
    """Write then read is the identity; bytes are sorted and newline-ended."""
    record = _record()
    product = tmp_path / "o.ply"

    path = write_product_sidecar(product, record)

    assert path == sidecar_path(product)
    assert read_product_sidecar(path) == record
    data = path.read_bytes()
    assert data.endswith(b"}\n")
    assert data == product_record_to_bytes(copy.deepcopy(record))
    keys = list(json.loads(data))
    assert keys == sorted(keys)


def _drop_counts(record: dict[str, Any]) -> None:
    record.pop("counts")


def _add_extra(record: dict[str, Any]) -> None:
    record["extra"] = 1


def _mistype_out(record: dict[str, Any]) -> None:
    record["counts"]["out"] = "2"


def _mistype_thinning(record: dict[str, Any]) -> None:
    record["filters"]["thinning"] = []


def _output_not_object(record: dict[str, Any]) -> None:
    record["output"] = [1]


def _wrong_schema(record: dict[str, Any]) -> None:
    record["schema"] = "other/9"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_counts, "missing=\\['counts'\\]"),
        (_add_extra, "unknown=\\['extra'\\]"),
        (_mistype_out, "counts.out"),
        (_mistype_thinning, "thinning"),
        (_output_not_object, "must be a JSON object"),
        (_wrong_schema, "unsupported sidecar schema"),
    ],
)
def test_validation_rejects_each_kind_of_malformation(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Missing/unknown keys, mistyped leaves, non-objects, a wrong schema."""
    record = _record()
    mutate(record)

    with pytest.raises(ProductProvenanceError, match=message):
        validate_product_record(record)


def test_validation_rejects_a_non_object() -> None:
    """The top level must itself be an object."""
    with pytest.raises(ProductProvenanceError, match="JSON object"):
        validate_product_record(["not", "an", "object"])


def test_read_rejects_invalid_json(tmp_path: Path) -> None:
    """Garbage bytes raise the typed error, not a JSON traceback."""
    bad = tmp_path / "bad.provenance.json"
    bad.write_text("{nope")

    with pytest.raises(ProductProvenanceError, match="not valid JSON"):
        read_product_sidecar(bad)


def test_fingerprint_is_sha256_and_size(tmp_path: Path) -> None:
    """The fingerprint matches hashlib over the same bytes."""
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1000)

    assert fingerprint(path) == (
        hashlib.sha256(b"abc" * 1000).hexdigest(),
        3000,
    )


def test_tool_versions_reports_the_environment() -> None:
    """The tool block names ahn_cli, laspy, numpy, python and git state."""
    tool = collect_tool_versions()

    assert tool["laspy"] != "unknown"
    assert tool["numpy"]
    assert tool["python"]
    assert tool["ahn_cli"]
    assert set(tool) == set(_record()["tool"])


def test_tool_versions_degrade_to_none_without_git_or_pdal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No git checkout and no PDAL: those fields are None, nothing raises."""
    monkeypatch.setattr(product_module, "run_text", _no_output)
    monkeypatch.setattr(product_module.shutil, "which", _no_binary)

    tool = collect_tool_versions()

    assert tool["ahn_cli_commit"] is None
    assert tool["ahn_cli_dirty"] is None
    assert tool["pdal_cli"] is None


def test_tool_versions_read_the_pdal_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PDAL line is picked out of the multi-line --version banner."""
    banner = "-----\npdal 2.10.2 (git-version: Release)\n-----"

    def fake(argv: list[str], cwd: Path | None = None) -> str | None:
        del cwd
        if argv[0] == "git":
            return "abc123" if argv[1] == "rev-parse" else ""
        return banner

    monkeypatch.setattr(product_module, "run_text", fake)
    monkeypatch.setattr(product_module.shutil, "which", _pdal_binary)

    tool = collect_tool_versions()

    assert tool["pdal_cli"] == "pdal 2.10.2 (git-version: Release)"
    assert tool["ahn_cli_commit"] == "abc123"
    assert tool["ahn_cli_dirty"] is False


def test_pdal_without_a_banner_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pdal binary that prints nothing yields None, not an error."""
    monkeypatch.setattr(product_module, "run_text", _no_output)
    monkeypatch.setattr(product_module.shutil, "which", _pdal_binary)

    assert collect_tool_versions()["pdal_cli"] is None


def test_run_text_returns_output_or_none() -> None:
    """A working argv returns stdout; a missing binary returns None."""
    assert run_text([sys.executable, "-c", "print('hi')"]) == "hi"
    assert run_text(["/no/such/binary-xyz"]) is None
