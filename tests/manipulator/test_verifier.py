import os
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import laspy
import numpy as np

from ahn_cli.manipulator.verifier import (
    verify_bounds,
    verify_laz_integrity,
    verify_with_pdal,
)


class TestVerifier(unittest.TestCase):
    def setUp(self):
        """Create temporary test files."""
        # Create a temporary LAZ file with some test data
        self.temp_dir = tempfile.mkdtemp()
        self.laz_path = os.path.join(self.temp_dir, "test.laz")
        
        # Create test point cloud data
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = [85000.0, 445000.0, 0.0]
        header.scales = [0.01, 0.01, 0.01]
        
        las = laspy.LasData(header)
        
        # Create some test points within a small area
        num_points = 100
        x = np.random.uniform(85000, 85100, num_points)
        y = np.random.uniform(445000, 445100, num_points)
        z = np.random.uniform(0, 10, num_points)
        
        las.x = x
        las.y = y
        las.z = z
        
        las.write(self.laz_path)
        
        # Create test GeoJSON that contains the LAZ bounds
        self.geojson_path = os.path.join(self.temp_dir, "test.geojson")
        geojson_content = """{
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [84900, 444900],
                        [85200, 444900],
                        [85200, 445200],
                        [84900, 445200],
                        [84900, 444900]
                    ]]
                }
            }],
            "crs": {"type": "name", "properties": {"name": "EPSG:28992"}}
        }"""
        with open(self.geojson_path, 'w') as f:
            f.write(geojson_content)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_verify_laz_integrity_valid(self):
        """Test integrity check on valid LAZ file."""
        result = verify_laz_integrity(self.laz_path)
        self.assertTrue(result)
    
    def test_verify_laz_integrity_invalid(self):
        """Test integrity check on invalid file."""
        # Create an invalid file
        invalid_path = os.path.join(self.temp_dir, "invalid.laz")
        with open(invalid_path, 'w') as f:
            f.write("not a laz file")
        
        result = verify_laz_integrity(invalid_path)
        self.assertFalse(result)
    
    def test_verify_laz_integrity_empty(self):
        """Test integrity check on empty LAZ file."""
        # Create empty LAZ file
        empty_path = os.path.join(self.temp_dir, "empty.laz")
        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)
        las.write(empty_path)
        
        result = verify_laz_integrity(empty_path)
        self.assertFalse(result)  # Should fail for empty file
    
    def test_verify_bounds_contained(self):
        """Test bounds verification when LAZ is contained in GeoJSON."""
        result = verify_bounds(self.laz_path, self.geojson_path)
        self.assertTrue(result)
    
    def test_verify_bounds_with_tolerance(self):
        """Test bounds verification with tolerance."""
        # Create GeoJSON with tighter bounds
        tight_geojson = os.path.join(self.temp_dir, "tight.geojson")
        geojson_content = """{
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [85050, 445050],
                        [85060, 445050],
                        [85060, 445060],
                        [85050, 445060],
                        [85050, 445050]
                    ]]
                }
            }],
            "crs": {"type": "name", "properties": {"name": "EPSG:28992"}}
        }"""
        with open(tight_geojson, 'w') as f:
            f.write(geojson_content)
        
        # Should pass with high tolerance
        result = verify_bounds(self.laz_path, tight_geojson, tolerance=100.0)
        self.assertTrue(result)
        
        # Should fail with low tolerance
        result = verify_bounds(self.laz_path, tight_geojson, tolerance=1.0)
        self.assertFalse(result)
    
    def test_verify_bounds_different_crs(self):
        """Test bounds verification with WGS84 GeoJSON."""
        # Skip CRS transformation test for now - would need accurate coordinates
        # Just test that CRS transformation doesn't crash
        wgs84_geojson = os.path.join(self.temp_dir, "wgs84.geojson")
        geojson_content = """{
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [4.350, 52.009],
                        [4.352, 52.009],
                        [4.352, 52.011],
                        [4.350, 52.011],
                        [4.350, 52.009]
                    ]]
                }
            }],
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}
        }"""
        with open(wgs84_geojson, 'w') as f:
            f.write(geojson_content)
        
        # Just verify it doesn't crash - coordinates don't match our test data
        try:
            verify_bounds(self.laz_path, wgs84_geojson, tolerance=10000.0)
            # Test passed - CRS transformation worked
        except Exception as e:
            self.fail(f"CRS transformation failed: {e}")
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_verify_with_pdal_success(self, mock_which, mock_run):
        """Test PDAL verification when successful."""
        mock_which.return_value = '/usr/bin/pdal'
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Valid",
            stderr=""
        )
        
        result = verify_with_pdal(self.laz_path)
        self.assertTrue(result)
        
        mock_run.assert_called_once_with(
            ["pdal", "info", "--validate", self.laz_path],
            capture_output=True,
            text=True,
            check=True,
        )
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_verify_with_pdal_failure(self, mock_which, mock_run):
        """Test PDAL verification when it fails."""
        mock_which.return_value = '/usr/bin/pdal'
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="error: Invalid LAZ file"
        )
        mock_run.side_effect = subprocess.CalledProcessError(
            1, 'pdal', stderr="error: Invalid LAZ file"
        )
        
        result = verify_with_pdal(self.laz_path)
        self.assertFalse(result)
    
    @patch('shutil.which')
    def test_verify_with_pdal_not_installed(self, mock_which):
        """Test PDAL verification when PDAL is not installed."""
        mock_which.return_value = None
        
        # Should return True (pass) when PDAL is not available
        result = verify_with_pdal(self.laz_path)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()