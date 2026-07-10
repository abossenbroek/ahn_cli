"""Integration tests for GeoJSON functionality."""

import os
import tempfile
import unittest
from unittest.mock import patch

import laspy
import numpy as np

from ahn_cli import config
from ahn_cli.process import process


class TestGeoJSONIntegration(unittest.TestCase):
    """Test the complete GeoJSON pipeline."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "output.laz")

        # Create a test GeoJSON file
        self.geojson_path = os.path.join(self.temp_dir, "test_area.geojson")
        geojson_content = """{
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [85000, 445000],
                        [85100, 445000],
                        [85100, 445100],
                        [85000, 445100],
                        [85000, 445000]
                    ]]
                }
            }],
            "crs": {"type": "name", "properties": {"name": "EPSG:28992"}}
        }"""
        with open(self.geojson_path, "w") as f:
            f.write(geojson_content)

        # Create mock LAZ file
        self.mock_laz_path = os.path.join(self.temp_dir, "mock_tile.laz")
        self._create_mock_laz()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_mock_laz(self):
        """Create a mock LAZ file with test data."""
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = [85000.0, 445000.0, 0.0]
        header.scales = [0.01, 0.01, 0.01]

        las = laspy.LasData(header)

        # Create points within the test area
        num_points = 1000
        x = np.random.uniform(85000, 85100, num_points)
        y = np.random.uniform(445000, 445100, num_points)
        z = np.random.uniform(0, 10, num_points)

        las.x = x
        las.y = y
        las.z = z

        # Set some classifications
        las.classification = np.random.choice([2, 6], size=num_points)

        las.write(self.mock_laz_path)

    @patch(
        "ahn_cli.manipulator.ptc_handler.PntCHandler.clip_by_arbitrary_polygon"
    )
    @patch("ahn_cli.fetcher.request.Fetcher.fetch")
    @patch("ahn_cli.fetcher.geotiles.ahn_subunit_indices_of_geojson")
    def test_geojson_pipeline_basic(
        self, mock_indices, mock_fetch, mock_clip
    ):
        """Test basic GeoJSON pipeline without verification."""
        # Mock the tile selection
        mock_indices.return_value = ["37EN1_15"]

        # Mock the fetcher to return our test file
        mock_fetch.return_value = {"test_url": self.mock_laz_path}

        # Mock the clip method to do nothing (points already in correct area)
        mock_clip.return_value = None

        # Run the process
        cfg = config.Config()
        process(
            cfg.geotiles_base_url,
            cfg.city_polygon_file,
            self.output_path,
            city_name=None,
            geojson=self.geojson_path,
            no_verify=True,  # Skip verification for this test
        )

        # Verify output file was created
        self.assertTrue(os.path.exists(self.output_path))

        # Verify we can read the output
        with laspy.open(self.output_path) as f:
            self.assertGreater(f.header.point_count, 0)

    @patch(
        "ahn_cli.manipulator.ptc_handler.PntCHandler.clip_by_arbitrary_polygon"
    )
    @patch("ahn_cli.fetcher.request.Fetcher.fetch")
    @patch("ahn_cli.fetcher.geotiles.ahn_subunit_indices_of_geojson")
    def test_geojson_with_class_filtering(
        self, mock_indices, mock_fetch, mock_clip
    ):
        """Test GeoJSON pipeline with class filtering."""
        # Mock the tile selection
        mock_indices.return_value = ["37EN1_15"]

        # Mock the fetcher to return our test file
        mock_fetch.return_value = {"test_url": self.mock_laz_path}

        # Mock the clip method to do nothing (points already in correct area)
        mock_clip.return_value = None

        # Run the process with class filtering (only ground points)
        cfg = config.Config()
        process(
            cfg.geotiles_base_url,
            cfg.city_polygon_file,
            self.output_path,
            city_name=None,
            include_classes=[2],  # Only ground points
            geojson=self.geojson_path,
            no_verify=True,
        )

        # Verify output file was created
        self.assertTrue(os.path.exists(self.output_path))

        # Verify only ground points are in output
        with laspy.open(self.output_path) as f:
            las = f.read()
            self.assertTrue(np.all(las.classification == 2))

    @patch(
        "ahn_cli.manipulator.ptc_handler.PntCHandler.clip_by_arbitrary_polygon"
    )
    @patch("ahn_cli.fetcher.request.Fetcher.fetch")
    @patch("ahn_cli.fetcher.geotiles.ahn_subunit_indices_of_geojson")
    def test_geojson_with_verification(
        self, mock_indices, mock_fetch, mock_clip
    ):
        """Test GeoJSON pipeline with verification enabled."""
        # Mock the tile selection
        mock_indices.return_value = ["37EN1_15"]

        # Mock the fetcher to return our test file
        mock_fetch.return_value = {"test_url": self.mock_laz_path}

        # Mock the clip method to do nothing (points already in correct area)
        mock_clip.return_value = None

        # Run the process with verification
        cfg = config.Config()
        process(
            cfg.geotiles_base_url,
            cfg.city_polygon_file,
            self.output_path,
            city_name=None,
            geojson=self.geojson_path,
            no_verify=False,  # Enable verification
            verify_pdal=False,  # Skip PDAL for testing
            bbox_tolerance=10.0,
        )

        # Verify output file was created
        self.assertTrue(os.path.exists(self.output_path))

        # The verification should have passed
        with laspy.open(self.output_path) as f:
            self.assertGreater(f.header.point_count, 0)

    @patch("ahn_cli.fetcher.request.Fetcher.fetch")
    @patch("ahn_cli.fetcher.geotiles.ahn_subunit_indices_of_geojson")
    def test_geojson_no_tiles_found(self, mock_indices, mock_fetch):
        """Test GeoJSON pipeline when no tiles are found."""
        # Mock no tiles found
        mock_indices.return_value = []

        # Run the process
        cfg = config.Config()
        process(
            cfg.geotiles_base_url,
            cfg.city_polygon_file,
            self.output_path,
            city_name=None,
            geojson=self.geojson_path,
            no_verify=True,
        )

        # Verify no output file was created
        self.assertFalse(os.path.exists(self.output_path))

    @patch("ahn_cli.fetcher.request.logging.warning")
    @patch("ahn_cli.fetcher.request.ahn_subunit_indices_of_geojson")
    def test_geojson_many_tiles_warning(self, mock_indices, mock_warning):
        """Test warning when many tiles will be downloaded."""
        # Mock many tiles found
        mock_indices.return_value = [f"tile_{i}" for i in range(60)]

        # Create the fetcher which should trigger the warning
        from ahn_cli.fetcher.request import Fetcher

        Fetcher("http://test.com/", geojson_file=self.geojson_path)

        # Verify warning was logged
        mock_warning.assert_called_with(
            "This will download 60 tiles. This may take significant time and disk space."
        )


if __name__ == "__main__":
    unittest.main()
