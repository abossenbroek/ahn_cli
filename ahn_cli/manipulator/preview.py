# DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE MOVED
# Legacy pre-7rad module, pending migration into the new bounded contexts.
import warnings

warnings.warn(
    "ahn_cli.manipulator.preview is a deprecated pre-7rad module; logic must move into the new bounded contexts",
    DeprecationWarning,
    stacklevel=2,
)

import laspy
import polyscope as ps


def previewer(filepath: str) -> None:
    # Read LAS file
    las = laspy.read(filepath)
    points = las.xyz

    ps.init()
    ps.register_point_cloud("AHN Point Cloud", points)

    ps.show()
