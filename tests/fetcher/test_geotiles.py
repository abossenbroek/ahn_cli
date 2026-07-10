import os
import unittest

from ahn_cli.fetcher.geotiles import (
    ahn_subunit_indices_of_bbox,
    ahn_subunit_indices_of_city,
    ahn_subunit_indices_of_geojson,
)


class TestGeoTile(unittest.TestCase):
    def test_ahn_subunit_indices_of_city(self) -> None:
        tiles = ahn_subunit_indices_of_city("Delft")
        expected = [
            "37EZ1_03",
            "37EZ1_04",
            "37EZ1_05",
            "37EN1_04",
            "37EN1_05",
            "37EN1_08",
            "37EN1_09",
            "37EN1_10",
            "37EN1_12",
            "37EN1_13",
            "37EN1_14",
            "37EN1_15",
            "37EN1_17",
            "37EN1_18",
            "37EN1_19",
            "37EN1_20",
            "37EN1_23",
            "37EN1_24",
            "37EN1_25",
            "37EZ2_01",
            "37EZ2_02",
            "37EZ2_03",
            "37EZ2_08",
            "37EN2_01",
            "37EN2_02",
            "37EN2_06",
            "37EN2_07",
            "37EN2_11",
            "37EN2_12",
            "37EN2_16",
            "37EN2_17",
            "37EN2_21",
            "37EN2_22",
            "37EN2_23",
        ]
        self.assertEqual(tiles, expected)

    def test_ahn_subunit_indices_of_bbox(self) -> None:
        bbox = [
            84592.705048133007949,
            444443.127025160647463,
            86312.074818017281359,
            446712.346010794688482,
        ]
        tiles = ahn_subunit_indices_of_bbox(bbox)
        expected = [
            "37EN1_15",
            "37EN1_20",
            "37EN1_25",
            "37EN2_11",
            "37EN2_12",
            "37EN2_16",
            "37EN2_17",
            "37EN2_21",
            "37EN2_22",
        ]

        self.assertListEqual(tiles, expected)

    def test_ahn_subunit_indices_of_geojson(self) -> None:
        # Get the path to the test GeoJSON file
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        geojson_path = os.path.join(
            test_dir, "testdata", "sample_polygon.geojson"
        )

        tiles = ahn_subunit_indices_of_geojson(geojson_path)

        # The sample polygon covers a small area around (85000-86000, 445000-446000)
        # We expect it to intersect with some tiles
        self.assertIsInstance(tiles, list)
        self.assertGreater(len(tiles), 0)  # Should find at least one tile

        # All returned values should be strings (tile IDs)
        for tile in tiles:
            self.assertIsInstance(tile, str)

    def test_ahn_subunit_indices_of_geojson_multipolygon(self) -> None:
        # Test with multiple polygons
        test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        geojson_path = os.path.join(
            test_dir, "testdata", "multipolygon.geojson"
        )

        tiles = ahn_subunit_indices_of_geojson(geojson_path)

        # Should find tiles for both polygons
        self.assertIsInstance(tiles, list)
        self.assertGreater(len(tiles), 0)

        # All returned values should be strings (tile IDs)
        for tile in tiles:
            self.assertIsInstance(tile, str)

    def test_ahn_subunit_indices_of_geojson_different_crs(self) -> None:
        # Test with WGS84 GeoJSON (will be transformed to EPSG:28992)
        import json
        import tempfile

        wgs84_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [4.351, 52.01],  # Roughly Delft area in WGS84
                                [4.361, 52.01],
                                [4.361, 52.02],
                                [4.351, 52.02],
                                [4.351, 52.01],
                            ]
                        ],
                    },
                }
            ],
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".geojson", delete=False
        ) as f:
            json.dump(wgs84_geojson, f)
            temp_path = f.name

        try:
            tiles = ahn_subunit_indices_of_geojson(temp_path)
            self.assertIsInstance(tiles, list)
            self.assertGreater(len(tiles), 0)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
