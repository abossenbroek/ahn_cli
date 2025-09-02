import logging
import os

import laspy
import numpy as np
from laspy.lasappender import LasAppender
from tqdm import tqdm

from ahn_cli.fetcher.request import Fetcher
from ahn_cli.manipulator.preview import previewer
from ahn_cli.manipulator.ptc_handler import PntCHandler


def _validate_headers(files: list[str]) -> laspy.LasHeader:
    """
    Validates that all LAZ files have compatible headers and returns a
    master header with the combined spatial extent.

    Raises:
        ValueError: If no files provided
        RuntimeError: If files have incompatible point formats
    """
    if not files:
        raise ValueError("No files to process.")

    # Use the first file's header as the reference
    with laspy.open(files[0]) as f:
        master_header = f.header

    ref_point_format = master_header.point_format
    ref_dims = sorted(ref_point_format.extra_dimension_names)

    for file_path in files[1:]:
        with laspy.open(file_path) as f:
            current_header = f.header
            current_point_format = current_header.point_format
            current_dims = sorted(current_point_format.extra_dimension_names)

            if ref_point_format.id != current_point_format.id or ref_dims != current_dims:
                raise RuntimeError(
                    f"Incompatible point formats found.\n"
                    f"File '{files[0]}' has format {ref_point_format.id} with dims {ref_dims}.\n"
                    f"File '{file_path}' has format {current_point_format.id} with dims {current_dims}."
                )

            # Update master header bounds to enclose all files
            master_header.maxs = np.maximum(master_header.maxs, current_header.maxs)
            master_header.mins = np.minimum(master_header.mins, current_header.mins)

    return master_header


def process(
    base_url: str,
    city_polygon_path: str,
    output_path: str,
    city_name: str,
    include_classes: list[int] | None = None,
    exclude_classes: list[int] | None = None,
    no_clip_city: bool | None = False,
    clip_file: str | None = None,
    epsg: int | None = None,
    decimate: int | None = None,
    bbox: list[float] | None = None,
    geojson: str | None = None,
    preview: bool | None = False,
) -> None:
    ahn_fetcher = Fetcher(base_url, city_name, bbox, geojson)
    fetched_files = ahn_fetcher.fetch()

    files = list(fetched_files.values())
    if not files:
        logging.info("No files found for the given area.")
        return

    # Perform pre-flight header validation
    try:
        global_header = _validate_headers(files)
    except (RuntimeError, ValueError) as e:
        logging.error(f"Header validation failed: {e}")
        # Clean up downloaded files before exiting
        for file in files:
            os.remove(file)
        return

    for i, file in enumerate(
        tqdm(files, desc="Processing files", unit="file", total=len(files))
    ):
        logging.info("Start processing downloaded files...")
        with laspy.open(file) as las:
            # Calculate offset for this file relative to global header
            header = las.header
            offset = global_header.offsets - header.offsets

            p_handler = PntCHandler(
                las.read(),
                city_polygon_path,
                city_name,
                epsg if epsg is not None else 4326,
            )

            if bbox is not None:
                p_handler.clip_by_bbox(bbox)
            if include_classes is not None and len(include_classes) > 0:
                p_handler.include(include_classes)
            if exclude_classes is not None and len(exclude_classes) > 0:
                p_handler.exclude(exclude_classes)
            if not no_clip_city and city_name is not None:
                p_handler.clip()
            if clip_file is not None:
                p_handler.clip_by_arbitrary_polygon(clip_file)
            if decimate is not None:
                p_handler.decimate(decimate)

            with laspy.open(
                output_path, mode="w" if i == 0 else "a", header=global_header
            ) as writer:
                points = p_handler.points().points
                if len(points) == 0:
                    continue
                points.x = points.x - offset[0]
                points.y = points.y - offset[1]
                points.z = points.z - offset[2]

                if isinstance(writer, laspy.LasWriter):
                    writer.write_points(points)
                if isinstance(writer, LasAppender):
                    writer.append_points(points)

    for file in files:
        os.remove(file)

    if preview:
        print("Previewing output file...")
        previewer(output_path)
