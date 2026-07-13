# Overview: from Dutch national geodata to streamable 3D terrain

`ahn_cli` turns **open Dutch national elevation and imagery data** into finished,
reproducible deliverables — coloured point clouds, position maps, Cloud-Optimized
Point Clouds, OGC 3D Tiles, and a compact packed terrain archive a game engine can
stream. This page explains the domain concepts the tool works with (AHN, orthophotos,
NAP, RD New) and the formats it produces (notably the `AHNP` pack), so you can judge
where the package fits your own project.

If you just want to run it, see the [README](../README.md). If you are writing a
consumer for the output formats, the normative byte specs live in [`docs/specs/`](specs/).

## The one-paragraph version

The Netherlands publishes, as open data, a nationwide airborne-LiDAR elevation model
(**AHN**) and nationwide aerial photography (**orthophotos**). Each is authoritative
and free, but each arrives as thousands of raw tiles in a national coordinate system,
in formats built for GIS rather than for rendering. `ahn_cli` fetches the tiles for an
area you care about, cleans and combines them — draping the photography's colour over
the elevation — and exports the result in formats meant to be *consumed*: by
TouchDesigner, by point-cloud tools, by 3D-Tiles viewers, or by a custom game engine
via the `AHNP` pack this project defines.

## The source layers

### AHN — the elevation

**AHN** stands for *Actueel Hoogtebestand Nederland* — "Current Elevation File of the
Netherlands." It is a national, LiDAR-derived elevation dataset covering the entire
country, published through PDOK (the Dutch national spatial-data portal) and mirrored by
GeoTiles.nl. It comes in **generations** — successive national surveys: AHN3 and AHN4 are
the established, fully published products, with AHN5 rolling out. `ahn_cli` auto-selects
the newest generation available for your area or lets you pin one with `--ahn`.

AHN ships as two kinds of product this tool uses:

- **Point cloud** (`.LAZ`) — the raw measurement: millions of `(x, y, z)` points per
  tile, each tagged with a **classification class** describing what the laser hit.
  `ahn_cli` can filter to the classes you want:

  | class | meaning |
  |------:|---------|
  | 0 | created, never classified |
  | 1 | unclassified |
  | 2 | ground |
  | 6 | building |
  | 9 | water |
  | 14 | high tension (power lines) |
  | 26 | civil structure |

  So "ground + buildings only" is `-i 2,6`; the water class lets you cut canals and the
  sea, and so on.

- **DSM** (Digital Surface Model) — a regular raster grid (~0.5 m spacing) giving a
  single height per cell. It is derived from the point cloud and is cheaper to sample
  than the raw points, which is why the fast paths (position maps, the elevation grid
  for 3D tiles) read the DSM.

Heights in AHN are **NAP** heights (see coordinate systems below), i.e. metres relative
to Dutch mean sea level — not metres above the ellipsoid.

### Orthophotos — the imagery ("ortho maps")

An **orthophoto** is aerial photography that has been *orthorectified*: geometric
distortion from camera tilt and terrain relief has been removed, so every pixel sits at
its true map position and the image can be laid directly onto a map or a 3D surface like
a texture. (A raw aerial photo cannot — buildings lean, scale varies across the frame.)

`ahn_cli` pulls orthophotos from **Beeldmateriaal**, the Dutch national open aerial-
imagery program (RGB, ~8 cm/pixel nationally, ~5 cm on selected parcels, **CC-BY 4.0** —
attribution required, recorded automatically in `provenance.json`). The orthophoto is
what gives the terrain its *look*: `reconcile` and the 3D-tiles export drape these pixels
as colour/texture over the AHN elevation, so a rooftop is not just a height but the
actual photographed roof.

### Other layers

- **VIIRS** — satellite night-lights imagery, imported from an externally produced
  GeoTIFF, for artists who want a nocturnal/illumination layer.
- **DTM vs DSM** — AHN also distinguishes a *terrain* model (bare earth, buildings and
  vegetation removed) from the *surface* model (everything); this tool works from the
  DSM/point cloud and lets class filtering approximate a bare-earth view.

## Coordinate systems (why heights and positions "just line up")

Dutch geodata lives in national reference systems, and the pipeline leans on them so it
never has to reproject imagery against elevation:

| system | EPSG | what it is | used for |
|--------|------|------------|----------|
| RD New / Amersfoort | 28992 | the Dutch national grid (metres) | orthophotos, bounding boxes, tile identity |
| RD New + NAP | 7415 | 28992 horizontally, **NAP** height vertically | the AHN DSM/point cloud |
| WGS 84 geodetic / ECEF | 4979 / 4978 | global lon/lat/height and Earth-centred XYZ | the 3D-Tiles / game output |

**NAP** (*Normaal Amsterdams Peil*) is the Dutch vertical datum — the reference "zero"
for heights, historically Amsterdam mean sea level. Because the orthophoto grid
(EPSG:28992) and the AHN grid (EPSG:7415) share the exact same horizontal system, a photo
pixel and an elevation sample at the same X/Y are *the same spot on the ground* — only
the Z (the NAP height) is semantically new. That is what lets `reconcile` interpolate
heights onto photo pixels with no reprojection, and it is the reason the outputs are
crisp rather than smeared.

## The pipeline

Each subcommand is one stage; every stage reads the previous stage's output from a **site
directory** on disk and writes its own plus an updated `provenance.json`. There is no
hidden in-memory handoff — you can stop, inspect, and resume anywhere.

```
fetch ──► prep ──► reconcile ──► copc        (Cloud-Optimized Point Cloud)
  │         │          └───────► tiles3d      (3D Tiles / AHNP pack)
  │         └──────────────────► export-positions   (positions.exr for TouchDesigner)
  └► import-viirs
```

| stage | turns … | into … |
|-------|---------|--------|
| `fetch` | an area (city / bbox / GeoJSON) | raw cached AHN + DSM + ortho tiles |
| `prep` | raw tiles | a filtered, de-duplicated, optionally thinned `pointcloud.laz`/`.ply` |
| `reconcile` | cloud + ortho | one cloud coloured from the photo, or an `.exr` height grid on the photo's pixel grid |
| `export-positions` | the DSM | a deterministic `positions.exr` position map |
| `copc` | a pipeline LAZ | a validator-green `.copc.laz` (streamable point cloud) |
| `tiles3d` | ortho + reconciled heights | an OGC 3D Tiles 1.1 tileset, or a packed terrain archive |

## The output formats (what you actually consume)

`ahn_cli` deliberately ends in *consumer* formats, not GIS formats:

- **`pointcloud.laz` / `.ply`** — for point-cloud tools and TouchDesigner.
- **`positions.exr`** — a 3-channel float position map; feed it to a TouchDesigner
  point/instancing network.
- **`.copc.laz`** — a **Cloud-Optimized Point Cloud**: a LAZ whose points are organized
  into an octree so a viewer can stream only the region and detail it needs, without
  downloading the whole cloud.
- **OGC 3D Tiles 1.1** (`strict` profile) — a standard quadtree of glTF terrain tiles
  draped with the ortho, loadable by CesiumJS and other 3D-Tiles viewers.
- **The `AHNP` pack** (`game` and `heightfield` profiles) — see below.

### `.hf` and the `AHNP` pack — terrain for a streaming engine

The standard 3D-Tiles output is portable but heavy and JSON-indexed, which is not ideal
for a game that streams terrain at flight speed. So this project defines two compact,
engine-oriented formats:

- **`.hf` — a heightfield chunk.** One tile's elevation, stored as a fixed little-endian
  header plus a single zstd-compressed frame of 12-bit-quantized `uint16` NAP-height
  levels (with a documented ≤ 25 mm error bound — an order of magnitude inside AHN's own
  accuracy). Vertex positions, texture coordinates, and triangle connectivity are
  *implicit*: the runtime rebuilds them from the tile's geographic footprint, so only the
  heights and a sibling JPEG texture are stored. Normative spec:
  [`docs/specs/2026-07-12-heightfield-chunk-format.md`](specs/2026-07-12-heightfield-chunk-format.md).

- **`AHNP` — the pack container** (`tiles.hfp`, magic bytes `AHNP` = "AHN Pack"). Instead
  of scattering thousands of small tile files across a directory, both engine profiles
  bundle every tile's content blobs (heightfield `.hf` + JPEG, or a quantized-glTF game
  tile) into **one self-describing archive**. Its header is a *binary scene index* — a
  level directory plus one 96-byte entry per tile carrying that tile's geographic region,
  level-of-detail error, and byte offsets — so a runtime opens the pack, reads the index
  once, and then streams individual tiles by ranged reads with **no JSON parsing at
  runtime**. A content-derived `dataset_id` lets a client detect a changed dataset in a
  single header read, and layered CRC-32 / SHA-256 integrity guards every load. Normative
  spec: [`docs/specs/2026-07-12-hfp-pack-format.md`](specs/2026-07-12-hfp-pack-format.md).

A companion Rust crate, [`rust/ahn-heightfield`](../rust/ahn-heightfield), decodes both
`.hf` chunks and `AHNP` packs against those specs (not against the Python source), so a
game written in Rust can consume the output directly.

## Who this is for

- **Technical artists / TouchDesigner** — real Dutch terrain and imagery as point clouds
  and position maps, deterministic and reproducible.
- **3D / web geospatial** — standards-compliant 3D Tiles and COPC for existing viewers.
- **Game / simulation developers** — the `AHNP` pack + the Rust crate give a streamable,
  integrity-checked terrain format with genuine survey-grade elevation and photographic
  texture, no GIS stack required at runtime.
- **Researchers** — a reproducible acquisition-to-deliverable pipeline with a provenance
  record at every step, over an authoritative national dataset.

Everything is built on **open data** (AHN, and Beeldmateriaal under CC-BY 4.0): attribute
the sources, and the deliverables are yours to use.
