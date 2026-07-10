import logging
import os

import laspy
import numpy as np
from laspy.lasappender import LasAppender
from tqdm import tqdm

from ahn_cli.fetcher.request import Fetcher
from ahn_cli.manipulator.preview import previewer
from ahn_cli.manipulator.ptc_handler import PntCHandler
from ahn_cli.manipulator.verifier import (
    verify_bounds,
    verify_laz_integrity,
    verify_with_pdal,
)


def _harmonize_headers(files: list[str]) -> laspy.LasHeader:
    """
    Pre-scans all LAZ files and builds a master header whose extra
    dimensions are the superset of every input file's extra dimensions,
    upgrading the point format/version to support them if needed.

    This allows merging AHN tiles that specify different extra byte
    layouts, instead of failing with "Incompatible point formats found."

    Raises:
        ValueError: If no files provided
    """
    if not files:
        raise ValueError("No files to process.")

    base_header: laspy.LasHeader | None = None
    harmonized_extra_dims: dict[str, laspy.point.dims.DimensionInfo] = {}

    for file_path in files:
        with laspy.open(file_path) as f:
            header = f.header
            if base_header is None:
                base_header = header
            else:
                base_header.maxs = np.maximum(base_header.maxs, header.maxs)
                base_header.mins = np.minimum(base_header.mins, header.mins)

            for eb_struct in header.point_format.extra_dimensions:
                name = eb_struct.name
                if name not in harmonized_extra_dims:
                    harmonized_extra_dims[name] = eb_struct
                elif (
                    harmonized_extra_dims[name].dtype != eb_struct.dtype
                    or harmonized_extra_dims[name].num_elements
                    != eb_struct.num_elements
                ):
                    logging.warning(
                        f"Conflicting extra dimension '{name}': "
                        f"existing type={harmonized_extra_dims[name].dtype}, "
                        f"new type={eb_struct.dtype}. "
                        "Using first encountered definition."
                    )

    assert base_header is not None
    master_header = base_header

    if harmonized_extra_dims and master_header.point_format.id < 6:
        logging.info(
            f"Upgrading point format from {master_header.point_format.id} to 6 "
            "and LAS version to 1.4 to support merged extra dimensions."
        )
        upgraded_header = laspy.LasHeader(point_format=6, version="1.4")
        upgraded_header.offsets = master_header.offsets
        upgraded_header.scales = master_header.scales
        upgraded_header.maxs = master_header.maxs
        upgraded_header.mins = master_header.mins
        master_header = upgraded_header

    existing_dims = set(master_header.point_format.extra_dimension_names)
    for dim_info in harmonized_extra_dims.values():
        if dim_info.name in existing_dims:
            continue
        master_header.add_extra_dim(
            laspy.ExtraBytesParams(
                name=dim_info.name,
                type=dim_info.dtype,
                description=dim_info.description,
            )
        )

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
    no_verify: bool | None = False,
    verify_pdal: bool | None = False,
    bbox_tolerance: float = 10.0,
    strict_bbox_check: bool | None = False,
) -> None:
    ahn_fetcher = Fetcher(base_url, city_name, bbox, geojson)
    fetched_files = ahn_fetcher.fetch()

    files = list(fetched_files.values())
    if not files:
        logging.info("No files found for the given area.")
        return

    # Pre-scan files and harmonize extra dimensions across tiles
    global_header = _harmonize_headers(files)

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
            if geojson is not None:
                p_handler.clip_by_arbitrary_polygon(geojson)
            if decimate is not None:
                p_handler.decimate(decimate)

            in_points = p_handler.points().points
            if len(in_points) == 0:
                continue

            # Build points matching the harmonized format, copying over
            # only the dimensions present in both the source and the
            # merged output (handles files with differing extra bytes).
            out_points = laspy.ScaleAwarePointRecord.zeros(
                len(in_points), header=global_header
            )
            for dim_name in in_points.point_format.dimension_names:
                if dim_name in out_points.point_format.dimension_names:
                    out_points[dim_name] = in_points[dim_name]

            out_points.x = out_points.x - offset[0]
            out_points.y = out_points.y - offset[1]
            out_points.z = out_points.z - offset[2]

            with laspy.open(
                output_path, mode="w" if i == 0 else "a", header=global_header
            ) as writer:
                if isinstance(writer, laspy.LasWriter):
                    writer.write_points(out_points)
                if isinstance(writer, LasAppender):
                    writer.append_points(out_points)

    for file in files:
        os.remove(file)

    # Perform verification if enabled
    if not no_verify:
        logging.info("Verifying output file...")

        # Basic LAZ integrity check
        if not verify_laz_integrity(output_path):
            raise RuntimeError("Output LAZ file validation failed")

        # Bounds verification for GeoJSON input
        if geojson and not verify_bounds(
            output_path, geojson, bbox_tolerance
        ):
            if strict_bbox_check:
                raise RuntimeError(
                    f"Bounding box verification failed - difference exceeds {bbox_tolerance}m tolerance"
                )

        # Optional PDAL verification
        if verify_pdal and not verify_with_pdal(output_path):
            logging.warning("PDAL verification failed but continuing")

    if preview:
        print("Previewing output file...")
        previewer(output_path)
