# tiles3d profile external-validation record (2026-07-12)

Companion to `2026-07-11-tiles3d-design.md`'s conformance section (which
recorded Cesium's `3d-tiles-validator` reporting 0/0/0 on the **strict**
profile). This file records the same external conformance pass for the two
new lossy profiles — **game** and **heightfield** — added in the
compression-profiles epic.

These are **manual, one-shot** runs (they need Node + network for `npx`),
**not** part of `make test`. This document is the deliverable; the built-in
per-profile verifier (`ahn_cli/tiles3d/verify*.py`, run unconditionally at
the end of every build) is the in-CI gate.

## Environment

- Node `v24.15.0` (from `~/.local/share/nvm/v24.15.0`).
- `3d-tiles-validator` `0.6.1` (Cesium, via `npx --yes 3d-tiles-validator`).
- `gltf-validator` `2.0.0-dev.3.10` (Khronos glTF-Validator; the npm package
  exposes a `validateBytes` **API**, not a CLI bin, so it is driven by a
  three-line Node script — see below — rather than `npx gltf-validator`,
  which fails with "could not determine executable to run").

## Sample tilesets

Built from a 64×64 smooth synthetic ortho/EXR pair at `tile_pixels=32`
(one root + four leaves) with **real** geodesy, one per profile:

```python
build_tiles3d(ortho, heights, out/"game", tile_pixels=32, profile=Profile.GAME)
build_tiles3d(ortho, heights, out/"heightfield", tile_pixels=32, profile=Profile.HEIGHTFIELD)
```

## 3d-tiles-validator

### Game profile — PASS (0 errors, 0 warnings, 5 infos)

```
npx --yes 3d-tiles-validator --tilesetFile <out>/game/tileset.json
```

```json
{ "numErrors": 0, "numWarnings": 0, "numInfos": 5 }
```

All five infos are the two **expected, benign** kinds, repeated across the
tiles that carry them:

- `Cannot validate an extension as it is not supported by the validator:
  'EXT_meshopt_compression'.` — expected: the validator has no meshopt
  decoder, so it declines to inspect the compressed streams. The glb is
  still valid glTF 2.0 (confirmed independently by the Khronos validator
  below). This is the documented generic-validator limitation from
  coordinator resolution 5(a).
- `Image has non-power-of-two dimensions: 32x33.` (and `33x32`, `33x33`) —
  informational only. Terrain tiles share a 1-pixel boundary column/row
  with their neighbours, so a leaf's sampled span is one larger than the
  stride; NPOT textures are legal in glTF 2.0 / WebGL2.

### Heightfield profile — PASS (0 errors, 0 warnings, 0 infos)

```
npx --yes 3d-tiles-validator --tilesetFile <out>/heightfield/tileset.json
```

```json
{ "numErrors": 0, "numWarnings": 0, "numInfos": 0 }
```

Recorded honestly per coordinator resolution 5(b): the tileset **structure**
is fully valid 3D Tiles 1.1. Note the content URIs point at vendor `.hf`
chunks (`"uri": "tiles/0-0-0.hf"`, …); `3d-tiles-validator` `0.6.1` has no
content validator registered for that extension, so it validates the
tileset JSON and **silently skips** the `.hf` payloads rather than flagging
them — hence 0 infos, not the "unknown content type" info the brief
anticipated. The `.hf` bytes are covered instead by the project's own
`verify_heightfield.py` and by the normative format spec
(`2026-07-12-heightfield-chunk-format.md`) the Rust consumer codes against.

## glTF-Validator (Khronos)

Driven via the node API:

```js
// validate_glb.mjs
import { readFileSync } from 'fs';
import validator from 'gltf-validator';
const bytes = new Uint8Array(readFileSync(process.argv[2]));
const report = await validator.validateBytes(bytes, {
  uri: process.argv[2],
  externalResourceFunction: () => Promise.reject(new Error('no external resources')),
});
console.log(JSON.stringify(report.issues, null, 2));
```

```
node validate_glb.mjs <out>/game/tiles/0-0-0.glb
```

Root tile `0-0-0.glb` — PASS (0 errors, 0 warnings, 2 infos):

```json
{
  "numErrors": 0, "numWarnings": 0, "numInfos": 2,
  "messages": [
    { "code": "UNSUPPORTED_EXTENSION",
      "message": "Cannot validate an extension as it is not supported by the validator: 'EXT_meshopt_compression'.",
      "severity": 2, "pointer": "/extensionsUsed/0" },
    { "code": "IMAGE_NPOT_DIMENSIONS",
      "message": "Image has non-power-of-two dimensions: 33x33.",
      "severity": 2, "pointer": "/images/0" }
  ]
}
```

Leaf tile `1-0-0.glb` (32×32, power-of-two texture) — PASS (0 errors, 0
warnings, 1 info): only the `UNSUPPORTED_EXTENSION` meshopt info remains.

Both infos are severity 2 (INFO) and are the same expected pair as above:
the validator cannot decode `EXT_meshopt_compression`, and NPOT textures are
legal. **No structural glTF errors or warnings** in either the container,
the accessors/bufferViews, or the JSON.

The heightfield profile writes no glb (its content is the `.hf` chunk plus a
sibling JPEG), so the glTF validator does not apply to it.

## Summary

| Artifact | Tool | Errors | Warnings | Infos (all expected/benign) |
|---|---|---:|---:|---|
| game `tileset.json` | 3d-tiles-validator 0.6.1 | 0 | 0 | 5 (meshopt-unsupported, NPOT) |
| heightfield `tileset.json` | 3d-tiles-validator 0.6.1 | 0 | 0 | 0 (`.hf` content not inspected) |
| game `0-0-0.glb` | gltf-validator 2.0.0-dev.3.10 | 0 | 0 | 2 (meshopt-unsupported, NPOT) |
| game `1-0-0.glb` (leaf) | gltf-validator 2.0.0-dev.3.10 | 0 | 0 | 1 (meshopt-unsupported) |

Both lossy profiles are externally conformant: zero errors, zero warnings.
The only messages are informational — the generic validators cannot decode
the ratified `EXT_meshopt_compression` extension or inspect the vendor `.hf`
content type, exactly the caveats coordinator resolution 5 called out. The
project's own end-of-build verifier covers what the generic tools cannot.
