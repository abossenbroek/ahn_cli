import pytest
from click import ClickException

from ahn_cli.validator import (
    validate_exclusive_args,
    validate_geojson,
)


class TestValidator:
    def test_validate_geojson_valid(self, tmp_path):
        """Test validation of valid GeoJSON file."""
        geojson_file = tmp_path / "test.geojson"
        geojson_file.write_text("{}")

        result = validate_geojson(str(geojson_file))
        assert result == str(geojson_file)

    def test_validate_geojson_json_extension(self, tmp_path):
        """Test validation of JSON file with .json extension."""
        json_file = tmp_path / "test.json"
        json_file.write_text("{}")

        result = validate_geojson(str(json_file))
        assert result == str(json_file)

    def test_validate_geojson_none(self):
        """Test validation when geojson is None."""
        result = validate_geojson(None)
        assert result is None

    def test_validate_geojson_file_not_exists(self):
        """Test validation when GeoJSON file doesn't exist."""
        with pytest.raises(
            ClickException, match="GeoJSON file does not exist"
        ):
            validate_geojson("/path/to/nonexistent.geojson")

    def test_validate_geojson_invalid_extension(self, tmp_path):
        """Test validation when file has invalid extension."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.write_text("{}")

        with pytest.raises(
            ClickException, match="File must have .geojson or .json extension"
        ):
            validate_geojson(str(invalid_file))

    def test_validate_exclusive_args_one_option(self):
        """Test validation when exactly one option is provided."""
        # Only city
        validate_exclusive_args(None, "amsterdam", None)

        # Only bbox
        validate_exclusive_args([1, 2, 3, 4], None, None)

        # Only geojson
        validate_exclusive_args(None, None, "test.geojson")

    def test_validate_exclusive_args_no_options(self):
        """Test validation when no options are provided."""
        with pytest.raises(
            ClickException, match="You must specify exactly one of"
        ):
            validate_exclusive_args(None, None, None)

    def test_validate_exclusive_args_multiple_options(self):
        """Test validation when multiple options are provided."""
        # City and bbox
        with pytest.raises(
            ClickException, match="You must specify exactly one of"
        ):
            validate_exclusive_args([1, 2, 3, 4], "amsterdam", None)

        # City and geojson
        with pytest.raises(
            ClickException, match="You must specify exactly one of"
        ):
            validate_exclusive_args(None, "amsterdam", "test.geojson")

        # Bbox and geojson
        with pytest.raises(
            ClickException, match="You must specify exactly one of"
        ):
            validate_exclusive_args([1, 2, 3, 4], None, "test.geojson")

        # All three
        with pytest.raises(
            ClickException, match="You must specify exactly one of"
        ):
            validate_exclusive_args([1, 2, 3, 4], "amsterdam", "test.geojson")
