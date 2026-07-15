"""Tests for the bounded-context skeletons and the domain public API."""

from ahn_cli import domain, fetch, prep, tiles3d


def test_fetch_context_documents_acquisition_responsibility() -> None:
    """The ``fetch`` context declares its acquisition boundary."""
    assert fetch.__doc__ is not None
    assert "acquisition" in fetch.__doc__


def test_prep_context_documents_transform_responsibility() -> None:
    """The ``prep`` context declares its transform/export boundary."""
    assert prep.__doc__ is not None
    assert "transform" in prep.__doc__


def test_tiles3d_context_documents_export_responsibility() -> None:
    """The ``tiles3d`` context declares its 3D Tiles export boundary."""
    assert tiles3d.__doc__ is not None
    assert "3D Tiles" in tiles3d.__doc__


def test_domain_public_api_exports_the_value_objects() -> None:
    """The domain package re-exports every value object it owns."""
    assert set(domain.__all__) == {
        "BBox",
        "GeoTransform",
        "Generation",
        "PixelGrid",
        "Product",
        "ProgressCallback",
        "Provenance",
        "Tile",
        "Vintage",
        "ensure_valid_bbox",
    }
    for name in domain.__all__:
        assert hasattr(domain, name)
