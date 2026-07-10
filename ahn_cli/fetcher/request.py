# DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE MOVED
# Legacy pre-7rad module, pending migration into the new bounded contexts.
import warnings

warnings.warn(
    "ahn_cli.fetcher.request is a deprecated pre-7rad module; logic must move into the new bounded contexts",
    DeprecationWarning,
    stacklevel=2,
)

import logging
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from ahn_cli.fetcher.geotiles import (
    ahn_subunit_indices_of_bbox,
    ahn_subunit_indices_of_city,
    ahn_subunit_indices_of_geojson,
)


class Fetcher:
    """
    Fetcher class for fetching AHN data.

    Args:
        base_url (str): The base URL for fetching AHN data.
        city_name (str | None): The name of the city for which to fetch AHN data.
        bbox (list[float] | None, optional): The bounding box coordinates [minx, miny, maxx, maxy]
            for a specific area of interest. Defaults to None.
        geojson_file (str | None, optional): Path to GeoJSON file containing polygon(s)
            for a specific area of interest. Defaults to None.

    Raises:
        ValueError: If the base URL is invalid.

    Attributes:
        base_url (str): The base URL for fetching AHN data.
        city_name (str | None): The name of the city for which to fetch AHN data.
        bbox (list[float] | None): The bounding box coordinates [minx, miny, maxx, maxy]
            for a specific area of interest.
        geojson_file (str | None): Path to GeoJSON file for a specific area of interest.
        urls (list[str]): The constructed URLs for fetching AHN data.

    Methods:
        fetch: Fetches AHN data.
        _check_valid_url: Checks if the base URL is valid.
        _construct_urls: Constructs the URLs for fetching AHN data.
    """

    def __init__(
        self,
        base_url: str,
        city_name: str | None = None,
        bbox: list[float] | None = None,
        geojson_file: str | None = None,
    ):
        if not self._check_valid_url(base_url):
            raise ValueError("Invalid URL")
        self.base_url = base_url
        self.city_name = city_name
        self.bbox = bbox
        self.geojson_file = geojson_file
        self.urls = self._construct_urls()

    def fetch(self) -> dict:
        """
        Fetches AHN data.

        Returns:
            dict: A dictionary containing the fetched AHN data, where the keys are the URLs
            and the values are the temporary file names where the data is stored.
        """
        logging.info("Start fetching AHN data")
        logging.info(f"Fetching {len(self.urls)} tiles")

        def req(
            url: str, nth: int, results: dict, lock: Lock, pbar: tqdm
        ) -> None:
            res = requests.get(url, stream=True)
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w+b", suffix=".laz"
            ) as temp_file:
                for chunk in tqdm(
                    res.iter_content(chunk_size=500 * 1024 * 1024),
                    desc="writing a file",
                ):
                    temp_file.write(chunk)
                with lock:
                    results[url] = temp_file.name
            pbar.update(1)

        results: dict = {}
        lock = threading.Lock()
        with tqdm(total=len(self.urls)) as pbar:
            pbar.set_description("Fetching AHN data")
            with ThreadPoolExecutor(max_workers=8) as executor:
                for i, url in enumerate(self.urls):
                    executor.submit(req, url, i, results, lock, pbar)
        return results

    def _check_valid_url(self, url: str) -> bool:
        """
        Checks if the base URL is valid.

        Args:
            url (str): The base URL to check.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc, result.path])
        except ValueError:
            return False

    def _construct_urls(self) -> list[str]:
        """
        Constructs the URLs for fetching AHN data.

        Returns:
            list[str]: A list of URLs for fetching AHN data.
        """
        if self.bbox:
            tiles_indices = ahn_subunit_indices_of_bbox(self.bbox)
        elif self.geojson_file:
            tiles_indices = ahn_subunit_indices_of_geojson(self.geojson_file)
        else:
            tiles_indices = ahn_subunit_indices_of_city(self.city_name)

        # Warn if downloading many tiles
        if len(tiles_indices) > 50:
            logging.warning(
                f"This will download {len(tiles_indices)} tiles. "
                "This may take significant time and disk space."
            )

        urls = []
        for tile_index in tiles_indices:
            urls.append(os.path.join(self.base_url + f"{tile_index}.LAZ"))
        return urls
