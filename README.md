# AHN CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version: 0.2.1](https://img.shields.io/badge/Version-0.2.1-green.svg)](https://github.com/HideBa/ahn-cli/releases/tag/v0.2.1)
[![CICD Status: Passing](https://img.shields.io/badge/CICD-Passing-brightgreen.svg)](https://github.com/HideBa/ahn-cli/actions)

## Description

AHN CLI is a command-line interface tool designed for the effortless downloading of AHN (Actueel Hoogtebestand Nederland) point cloud data for specific cities and classification classes.

## Features

- Download point cloud data for specific Dutch cities
- Filter by classification classes (ground, buildings, water, etc.)
- Clip to city boundaries, bounding boxes, or custom GeoJSON polygons
- Decimate point clouds for faster processing
- Preview point clouds in 3D viewer
- **NEW**: GeoJSON polygon support for custom area selection
- **NEW**: LAZ file verification with configurable tolerance
- **NEW**: Optional PDAL integration for advanced validation

## Installation

Install AHN CLI using pip:

```bash
pip install ahn_cli
```

Or with uv:

```bash
uv tool install ahn_cli
```

## Usage

To utilize the AHN CLI, execute the following command with the appropriate options:

```shell
Options:
 -c, --city <city_name>        Download point cloud data for the specified city.
 -o, --output <file>           Designate the output file for the downloaded data.
 -i, --include-class <class>   Include specific point cloud classes in the download,
                               specified in a comma-separated list. Available classes:
                               0:Created, never classified; 1:Unclassified; 2:Ground;
                               6:Building; 9:Water; 14:High tension; 26:Civil structure.
 -e, --exclude-class <class>   Exclude specific point cloud classes from the download,
                               specified in a comma-separated list. Available classes as above.
 -d, --decimate <step>         Decimate the point cloud data by the specified step.
 -ncc, --no-clip-city          Avoid clipping the point cloud data to the city boundary.
 -cf, --clip-file <file>       Provide a file path for a clipping boundary file to clip
                               the point cloud data to a specified area.
 -e, --epsg <epsg>             Set the EPSG code for user's clip file.
 -b, --bbox <bbox>             Specify a bounding box to clip the point cloud data. It should be comma-separated list with minx,miny,maxx,maxy
                               centered on the city polygon.
 -g, --geojson <file>          Specify a GeoJSON file containing polygon(s) to clip the point cloud data.
 -p, --preview                 Preview the point cloud data in a 3D viewer.
 --no-verify                   Skip LAZ file verification after processing.
 --verify-pdal                 Use PDAL for additional LAZ file verification (requires PDAL).
 --bbox-tolerance <meters>     Maximum allowed difference in meters between LAZ bounds and input GeoJSON (default: 10.0).
 --strict-bbox-check           Fail if bounding box verification exceeds tolerance.
 -h, --help [category]         Show help information. Optionally specify a category for
                               detailed help on a specific command.
 -v, --version                 Display the version number of the AHN CLI and exit.
```

### Usage Examples

**Download Point Cloud Data for Delft with All Classification Classes:**

```
ahn_cli -c delft -o ./delft.laz
```

**To Include or Exclude Specific Classes:**

```
ahn_cli -c delft -o ./delft.laz -i 1,2
```

**For Non-Clipped, Rectangular-Shaped Data:**

```
ahn_cli -c delft -o ./delft.laz -i 1,2 -ncc
```

**To Decimate City-Scale Point Cloud Data:**

```
ahn_cli -c delft -o ./delft.laz -i 1,2 -d 2
```

**Specify a Bounding box for clipping:**

If you specify a `b`, it will clip the point cloud data with specified bounding box.
```
ahn_cli -o ./delft.laz -i 1,2 -b 194198.0,443461.0,194594.0,443694.0
```

**Download Point Cloud Data for Custom GeoJSON Polygon:**

Use a GeoJSON file containing one or more polygons to define a custom area of interest:
```
ahn_cli -g my_area.geojson -o ./custom_area.laz
```

**Download with GeoJSON and Enable Verification:**

Enable LAZ file verification with configurable tolerance for bounding box checks:
```
ahn_cli -g my_area.geojson -o ./verified_area.laz --bbox-tolerance 15.0
```

**Download with Strict Verification:**

Use strict verification that fails if the output bounding box doesn't match the input GeoJSON within tolerance:
```
ahn_cli -g my_area.geojson -o ./strict_area.laz --strict-bbox-check
```


## Reporting Issues

Encountering issues or bugs? We greatly appreciate your feedback. Please report any problems by opening an issue on our GitHub repository. Be as detailed as possible in your report, including steps to reproduce the issue, the expected outcome, and the actual result. This information will help us address and resolve the issue more efficiently.

## Contributing

Your contributions are welcome! If you're looking to contribute to the AHN CLI project, please first review our Contribution Guidelines. Whether it's fixing bugs, adding new features, or improving documentation, we value your help.

### Local development setup

```bash
# Install dependencies (creates the uv-managed virtualenv)
make install

# Install the pre-commit hooks (strict ruff lint + format, typos, pyright);
# they run automatically on every `git commit`.
uv run pre-commit install

# Run the full gate locally (lint, typos, pyright, tests + 100% coverage,
# format-check) — this is exactly what CI runs:
make check
```

To get started:

- Fork the repository on GitHub.
- Clone your forked repository to your local machine.
- Create a new branch for your contribution.
- Make your changes and commit them with clear, descriptive messages.
  Push your changes to your fork.
- Submit a pull request to our repository, providing details about your changes and the value they add to the project.
- We look forward to reviewing your contributions and potentially merging them into the project!
