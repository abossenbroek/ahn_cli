"""The :class:`Product` value object: the kind of dataset being handled.

A product is a *closed* set of dataset kinds the pipeline knows how to fetch
and prepare. Because the set is fixed and small, it is modelled as an
:class:`~enum.Enum`: every member is an immutable value object with
deterministic equality and hashing, and -- unlike a validated string wrapper --
an illegal product is *unrepresentable* rather than merely rejected at runtime.

Downstream code MUST branch on the enum members (``Product.AHN_POINT_CLOUD``),
never on their string values, to keep the domain free of stringly-typed
product switches.
"""

from __future__ import annotations

from enum import Enum


class Product(Enum):
    """A dataset kind produced or consumed by the pipeline.

    Contract:
        - Membership is closed: exactly the four kinds below exist, so an
          invalid product cannot be constructed.
        - ``value`` is the member's stable canonical code, intended as the
          identity WP3 will serialise; it is not for control-flow switching.

    Invariants:
        - Instances are immutable and compare by identity/value, so they are
          safe as dictionary keys and set members.

    Members:
        AHN_POINT_CLOUD: The AHN LAZ elevation *point cloud* product.
        ORTHO: A Beeldmateriaal orthophoto (RGB image) product.
        DSM: A Digital Surface Model *raster* (AHN-derived, distinct from the
            point cloud).
        VIIRS: A VIIRS night-lights raster product.
    """

    AHN_POINT_CLOUD = "ahn_point_cloud"
    ORTHO = "ortho"
    DSM = "dsm"
    VIIRS = "viirs"
