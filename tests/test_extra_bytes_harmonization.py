"""Regression test for merging AHN tiles with differing extra byte layouts.

Reproduces the bug where process() raised "Incompatible point formats
found" when input LAZ tiles had different extra byte specifications, and
verifies the header-harmonization fix merges them instead.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import laspy
import numpy as np

from ahn_cli import config
from ahn_cli.process import _harmonize_headers, process


class TestExtraBytesHarmonization(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "output.laz")

        self.geojson_path = os.path.join(self.temp_dir, "test_area.geojson")
        geojson_content = """{
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [194000, 443000],
                        [195500, 443000],
                        [195500, 444500],
                        [194000, 444500],
                        [194000, 443000]
                    ]]
                }
            }],
            "crs": {"type": "name", "properties": {"name": "EPSG:28992"}}
        }"""
        with open(self.geojson_path, "w") as f:
            f.write(geojson_content)

        self.file_no_extra = os.path.join(self.temp_dir, "tile1.laz")
        self.file_with_extra = os.path.join(self.temp_dir, "tile2.laz")
        self._create_laz_without_extra_bytes(self.file_no_extra)
        self._create_laz_with_extra_bytes(self.file_with_extra)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir)

    def _create_laz_without_extra_bytes(self, filepath: str) -> None:
        header = laspy.LasHeader(point_format=2, version="1.2")
        header.offsets = [0, 0, 0]
        header.scales = [0.01, 0.01, 0.01]

        las = laspy.LasData(header)
        n_points = 100
        las.x = np.random.uniform(194000, 195000, n_points)
        las.y = np.random.uniform(443000, 444000, n_points)
        las.z = np.random.uniform(0, 10, n_points)
        las.classification = np.random.choice([2, 6], n_points)
        las.write(filepath)

    def _create_laz_with_extra_bytes(self, filepath: str) -> None:
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.offsets = [0, 0, 0]
        header.scales = [0.01, 0.01, 0.01]
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="confidence",
                type=np.float32,
                description="Point confidence value",
            )
        )

        las = laspy.LasData(header)
        n_points = 100
        las.x = np.random.uniform(194500, 195500, n_points)
        las.y = np.random.uniform(443500, 444500, n_points)
        las.z = np.random.uniform(0, 10, n_points)
        las.classification = np.random.choice([2, 6], n_points)
        las.confidence = np.random.uniform(0, 1, n_points)
        las.write(filepath)

    def test_harmonize_headers_produces_superset_of_extra_dims(self):
        header = _harmonize_headers(
            [self.file_no_extra, self.file_with_extra]
        )
        self.assertEqual(header.point_format.id, 6)
        self.assertEqual(
            list(header.point_format.extra_dimension_names), ["confidence"]
        )

    def test_harmonize_headers_single_file_already_having_extra_dims(self):
        # Regression test: a single file whose own extra dims are already
        # present on its header must not have them re-added, which raised
        # "field 'X' occurs more than once" when building the point dtype.
        header = _harmonize_headers([self.file_with_extra])
        self.assertEqual(header.point_format.id, 6)
        self.assertEqual(
            list(header.point_format.extra_dimension_names), ["confidence"]
        )
        # dtype() must not raise on duplicate fields
        header.point_format.dtype()

    @patch(
        "ahn_cli.manipulator.ptc_handler.PntCHandler.clip_by_arbitrary_polygon"
    )
    @patch("ahn_cli.fetcher.request.Fetcher.fetch")
    @patch("ahn_cli.fetcher.geotiles.ahn_subunit_indices_of_geojson")
    def test_process_merges_tiles_with_mismatched_extra_bytes(
        self, mock_indices, mock_fetch, mock_clip
    ):
        mock_indices.return_value = ["37EN1_15", "37EN1_16"]
        mock_fetch.return_value = {
            "tile1": self.file_no_extra,
            "tile2": self.file_with_extra,
        }
        mock_clip.return_value = None

        cfg = config.Config()
        process(
            cfg.geotiles_base_url,
            cfg.city_polygon_file,
            self.output_path,
            city_name=None,
            geojson=self.geojson_path,
            no_verify=True,
        )

        self.assertTrue(os.path.exists(self.output_path))
        with laspy.open(self.output_path) as f:
            self.assertEqual(f.header.point_format.id, 6)
            self.assertEqual(
                list(f.header.point_format.extra_dimension_names),
                ["confidence"],
            )
            las = f.read()
            self.assertEqual(len(las.points), 200)


if __name__ == "__main__":
    unittest.main()
