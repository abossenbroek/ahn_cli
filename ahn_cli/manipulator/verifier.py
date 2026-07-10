"""
Verification module for LAZ file integrity and bounds checking.
"""

import logging
import shutil
import subprocess

import geopandas as gpd
import laspy
from shapely.geometry import box


def verify_laz_integrity(output_path: str) -> bool:
    """
    Performs a basic integrity check on a LAZ file by attempting to open it
    and read its header.

    Args:
        output_path: Path to the LAZ file to verify.

    Returns:
        bool: True if the file can be opened and read, False otherwise.
    """
    try:
        with laspy.open(output_path) as f:
            # Reading the header is an implicit check
            _ = f.header
            # Try to read point count as additional validation
            point_count = f.header.point_count
            if point_count == 0:
                logging.warning("LAZ file has no points")
                return False
        return True
    except Exception as e:
        logging.error(f"LAZ integrity check failed: {e}")
        return False


def verify_bounds(
    output_path: str, geojson_path: str, tolerance: float = 10.0
) -> bool:
    """
    Verifies that the output LAZ file's bounding box is contained within
    the input GeoJSON polygon bounds.

    Args:
        output_path: Path to the LAZ file to verify.
        geojson_path: Path to the GeoJSON file containing the input polygon(s).
        tolerance: Maximum allowed difference in meters between bounds.

    Returns:
        bool: True if bounds are within tolerance, False otherwise.
    """
    try:
        # 1. Get bounds from output LAZ file
        with laspy.open(output_path) as f:
            header = f.header
            # The CRS of AHN data is EPSG:28992
            laz_bounds = (
                header.mins[0],
                header.mins[1],
                header.maxs[0],
                header.maxs[1],
            )
            laz_box = box(*laz_bounds)

        # 2. Get bounds from input GeoJSON
        gdf = gpd.read_file(geojson_path)
        # Transform to Dutch national grid if needed
        if gdf.crs != "EPSG:28992":
            gdf_28992 = gdf.to_crs("EPSG:28992")
        else:
            gdf_28992 = gdf

        # Get the unified geometry bounds
        unified_geom = gdf_28992.geometry.union_all()
        geojson_bounds = unified_geom.bounds  # (minx, miny, maxx, maxy)
        geojson_box = box(*geojson_bounds)

        # 3. Check if LAZ bounds are contained within GeoJSON bounds (with tolerance)
        # Calculate the maximum difference between bounds
        max_diff = max(
            abs(laz_bounds[0] - geojson_bounds[0]),
            abs(laz_bounds[1] - geojson_bounds[1]),
            abs(laz_bounds[2] - geojson_bounds[2]),
            abs(laz_bounds[3] - geojson_bounds[3]),
        )

        # Check containment with tolerance
        buffered_geojson = geojson_box.buffer(tolerance)
        contained = buffered_geojson.contains(laz_box)

        # Calculate coverage percentage
        intersection_area = laz_box.intersection(geojson_box).area
        coverage_pct = (intersection_area / geojson_box.area) * 100

        # Log detailed information
        logging.info("Verifying output bounding box against GeoJSON input...")
        logging.info(
            f"GeoJSON bbox: [{geojson_bounds[0]:.2f}, {geojson_bounds[1]:.2f}, "
            f"{geojson_bounds[2]:.2f}, {geojson_bounds[3]:.2f}]"
        )
        logging.info(
            f"LAZ bbox:     [{laz_bounds[0]:.2f}, {laz_bounds[1]:.2f}, "
            f"{laz_bounds[2]:.2f}, {laz_bounds[3]:.2f}]"
        )

        if not contained:
            logging.warning(
                f"LAZ bounding box exceeds GeoJSON bounds by up to {max_diff:.2f}m"
            )
            if max_diff > tolerance:
                logging.error(
                    f"Bbox difference {max_diff:.2f}m exceeds tolerance {tolerance}m"
                )
                return False
            else:
                logging.info(
                    f"Bbox difference {max_diff:.2f}m is within tolerance {tolerance}m ✓"
                )
        else:
            logging.info(
                "LAZ bounding box is fully contained within GeoJSON bounds ✓"
            )

        logging.info(f"Coverage: {coverage_pct:.1f}% of input area covered ✓")

        return True

    except Exception as e:
        logging.error(f"Bounds verification failed: {e}")
        return False


def verify_with_pdal(output_path: str) -> bool:
    """
    Uses PDAL to perform advanced validation on the output LAZ file.

    Args:
        output_path: Path to the LAZ file to verify.

    Returns:
        bool: True if PDAL validation passes, False otherwise.
    """
    # Check if PDAL is available
    if not shutil.which("pdal"):
        logging.warning(
            "`pdal` command not found. Skipping PDAL verification. "
            "Install PDAL for advanced validation."
        )
        return True  # Return True for optional check

    try:
        # Run PDAL info with --validate flag
        result = subprocess.run(
            ["pdal", "info", "--validate", output_path],
            capture_output=True,
            text=True,
            check=True,
        )

        # Check if output contains validation errors
        if "error" in result.stderr.lower():
            logging.error(f"PDAL validation errors:\n{result.stderr}")
            return False

        logging.info("PDAL verification successful ✓")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(
            f"PDAL verification failed with exit code {e.returncode}"
        )
        if e.stderr:
            logging.error(f"PDAL error output:\n{e.stderr}")
        return False
    except FileNotFoundError:
        logging.warning("PDAL command not found. Skipping PDAL verification.")
        return True  # Return True for optional check
    except Exception as e:
        logging.error(f"PDAL verification failed: {e}")
        return False
