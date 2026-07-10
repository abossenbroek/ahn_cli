"""The ``fetch`` bounded context: data *acquisition*.

Responsibility: turn an area of interest into raw, cached source tiles. This
context owns portal clients, tile discovery, downloading, checksumming, and
content-addressed caching -- everything up to and including materialising
untouched source bytes on disk. It never transforms or exports pixel/point
data; that is the ``prep`` context's job, and the two are kept separate.

This is a context skeleton: it declares the boundary only and holds no logic
yet.
"""

from __future__ import annotations
