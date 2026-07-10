"""Tests for the bounded-context skeletons and the domain public API."""

from ahn_cli import domain, fetch, prep


def test_fetch_context_documents_acquisition_responsibility() -> None:
    """The ``fetch`` context declares its acquisition boundary."""
    assert fetch.__doc__ is not None
    assert "acquisition" in fetch.__doc__


def test_prep_context_documents_transform_responsibility() -> None:
    """The ``prep`` context declares its transform/export boundary."""
    assert prep.__doc__ is not None
    assert "transform" in prep.__doc__


def test_domain_public_api_exports_the_value_objects() -> None:
    """The domain package re-exports every value object it owns."""
    assert set(domain.__all__) == {
        "BBox",
        "Generation",
        "Product",
        "Provenance",
        "Tile",
        "Vintage",
        "ensure_valid_bbox",
    }
    for name in domain.__all__:
        assert hasattr(domain, name)
