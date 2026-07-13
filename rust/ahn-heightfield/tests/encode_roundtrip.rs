//! The optional `encode` feature: semantic round-trip contract.
//!
//! The whole file is gated on the `encode` feature, so a default-feature
//! `cargo test` build never sees it. Every check is **semantic** round-trip —
//! `decode(encode(x))` reproduces the identical quantized levels and header
//! fields — never byte parity with the Python producer (the specs scope byte
//! determinism to Python alone).
#![cfg(feature = "encode")]

mod common;

use ahn_heightfield::{
    encode_chunk, quantize_levels, ChunkFields, Heightfield, HfError, ABSOLUTE_ERROR_CAP_M,
    MAX_QUANTIZED_LEVEL,
};

/// A deterministic finite ramp of `width * height` heights whose extent stays
/// within the 25 mm error cap (so a valid chunk encodes).
fn ramp(width: u32, height: u32) -> Vec<f64> {
    (0..(width as u64 * height as u64))
        .map(|i| i as f64 * 0.001)
        .collect()
}

/// Encodes `heights` at the given grid, decodes the result, and asserts the full
/// semantic round-trip: dimensions, the derived quantizer, every level, the
/// header floats, and the per-sample dequantization bound.
fn assert_round_trips(width: u32, height: u32, heights: &[f64]) {
    let rtc = [1.0, 2.0, 3.0];
    let region = [0.11, 0.22, 0.33, 0.44, 0.0, 100.0];
    let bytes = encode_chunk(ChunkFields {
        width,
        height,
        rtc_centre: rtc,
        region,
        heights,
    })
    .expect("encodes a within-cap tile");

    let tile = Heightfield::decode(&bytes).expect("decodes its own output");
    assert_eq!((tile.width(), tile.height()), (width, height));

    let (z_offset, z_scale, levels) = quantize_levels(heights);
    assert_eq!(tile.levels(), levels.as_slice(), "levels round-trip");
    assert_eq!(tile.header().z_offset, z_offset, "z_offset round-trips");
    assert_eq!(tile.header().z_scale, z_scale, "z_scale round-trips");
    assert_eq!(tile.header().rtc_centre, rtc, "rtc_centre round-trips");
    assert_eq!(tile.header().region, region, "region round-trips");
    assert!(levels.iter().all(|&l| l <= MAX_QUANTIZED_LEVEL));

    // Every dequantized height is within the exported bound (z_scale / 2) of its
    // source sample — the spec's worst-case round-trip error.
    let bound = z_scale / 2.0;
    for r in 0..height {
        for c in 0..width {
            let source = heights[(r * width + c) as usize];
            let recovered = tile.dequantize_at(r, c);
            assert!(
                (recovered - source).abs() <= bound + 1e-9,
                "sample ({r},{c}) off by more than the bound",
            );
        }
    }
}

#[test]
fn round_trips_over_a_range_of_sizes() {
    // Includes 1×1 (flat, epsilon scale), 2×2, a non-square grid, and the
    // largest production tile 257×257.
    for (w, h) in [(1u32, 1u32), (2, 2), (3, 5), (5, 3), (257, 257)] {
        let heights = ramp(w, h);
        assert_round_trips(w, h, &heights);
    }
}

#[test]
fn flat_tile_stores_zeros_and_keeps_a_positive_scale() {
    let heights = [7.5; 9];
    let bytes = encode_chunk(ChunkFields {
        width: 3,
        height: 3,
        rtc_centre: [0.0, 0.0, 0.0],
        region: [0.1, 0.2, 0.3, 0.4, 7.5, 7.5],
        heights: &heights,
    })
    .expect("flat tile encodes");
    let tile = Heightfield::decode(&bytes).expect("decodes");
    assert_eq!(tile.levels(), &[0u16; 9]);
    assert!(tile.header().z_scale > 0.0);
    // A flat tile dequantizes back to exactly z_offset (error 0).
    assert_eq!(tile.dequantize_at(1, 1), 7.5);
}

// ---- cap refusal ----------------------------------------------------------

#[test]
fn refuses_a_tile_past_the_absolute_error_cap() {
    // Height extent 300 m > 204.75 m, so z_scale / 2 exceeds the 0.025 m cap.
    let heights = [0.0, 300.0, 150.0, 75.0];
    let err = encode_chunk(ChunkFields {
        width: 2,
        height: 2,
        rtc_centre: [0.0, 0.0, 0.0],
        region: [0.1, 0.2, 0.3, 0.4, 0.0, 300.0],
        heights: &heights,
    })
    .unwrap_err();
    match err {
        HfError::ErrorCapExceeded { z_scale, bound_m } => {
            assert!(bound_m > ABSOLUTE_ERROR_CAP_M);
            assert_eq!(bound_m, z_scale / 2.0);
        }
        other => panic!("expected ErrorCapExceeded, got {other:?}"),
    }
}

#[test]
fn accepts_a_tile_exactly_at_the_cap() {
    // Extent 204.75 m ⇒ z_scale = 0.05, bound = 0.025 m, exactly the cap (the
    // reject is strictly `> cap`, so this is accepted).
    let heights = [0.0, 204.75];
    let bytes = encode_chunk(ChunkFields {
        width: 2,
        height: 1,
        rtc_centre: [0.0, 0.0, 0.0],
        region: [0.1, 0.2, 0.3, 0.4, 0.0, 204.75],
        heights: &heights,
    })
    .expect("a tile exactly at the cap is valid");
    let tile = Heightfield::decode(&bytes).expect("decodes");
    assert_eq!(tile.header().z_scale / 2.0, ABSOLUTE_ERROR_CAP_M);
}

// ---- input validation -----------------------------------------------------

#[test]
fn rejects_zero_dimension() {
    let err = encode_chunk(ChunkFields {
        width: 0,
        height: 4,
        rtc_centre: [0.0, 0.0, 0.0],
        region: [0.1, 0.2, 0.3, 0.4, 0.0, 1.0],
        heights: &[],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        HfError::ZeroDimension {
            width: 0,
            height: 4
        }
    ));
}

#[test]
fn rejects_a_sample_count_that_does_not_match_the_grid() {
    let err = encode_chunk(ChunkFields {
        width: 2,
        height: 2,
        rtc_centre: [0.0, 0.0, 0.0],
        region: [0.1, 0.2, 0.3, 0.4, 0.0, 1.0],
        heights: &[0.0, 1.0, 2.0], // 3 samples for a 4-vertex grid
    })
    .unwrap_err();
    assert!(matches!(
        err,
        HfError::PlaneLengthMismatch {
            expected: 8,
            actual: 6
        }
    ));
}

#[test]
fn rejects_a_non_finite_header_float() {
    let err = encode_chunk(ChunkFields {
        width: 2,
        height: 1,
        rtc_centre: [0.0, 0.0, 0.0],
        region: [0.1, 0.2, 0.3, 0.4, f64::NAN, 1.0],
        heights: &[0.0, 1.0],
    })
    .unwrap_err();
    assert!(matches!(
        err,
        HfError::NonFiniteHeaderField {
            field: "region[4]",
            ..
        }
    ));
}

// ---- cross-language sanity: real fixture-derived source values ------------

#[test]
fn dequantized_fixture_heights_re_encode_to_the_same_levels() {
    // Take every committed heightfield tile, decode it, dequantize its levels
    // back to source NAP heights, then re-encode those heights with the crate
    // and decode again. Because a non-flat tile spans the full [0, 4095] range
    // by construction (its own min/max map to 0 and 4095), the requantization
    // is the identity on levels — a strong cross-language fidelity check on
    // real AHN data, not synthetic ramps.
    //
    // This asserts SEMANTIC equality of levels only. It deliberately does NOT
    // assert byte parity with the committed Python `.hf` bytes (the frame bytes
    // are a libzstd-build property, scoped to the Python producer).
    let chunks = common::heightfield_chunks();
    assert!(!chunks.is_empty(), "fixture yielded no chunks");

    for chunk in &chunks {
        let original = Heightfield::decode(&chunk.hf).expect("fixture tile decodes");
        let heights: Vec<f64> = (0..original.height())
            .flat_map(|r| (0..original.width()).map(move |c| (r, c)))
            .map(|(r, c)| original.dequantize_at(r, c))
            .collect();

        let bytes = encode_chunk(ChunkFields {
            width: original.width(),
            height: original.height(),
            rtc_centre: original.header().rtc_centre,
            region: original.header().region,
            heights: &heights,
        })
        .expect("fixture-derived heights re-encode");
        let round_tripped = Heightfield::decode(&bytes).expect("re-encoded tile decodes");

        assert_eq!(
            round_tripped.levels(),
            original.levels(),
            "tile ({},{},{}) levels diverged on re-encode",
            chunk.level,
            chunk.tx,
            chunk.ty,
        );
    }
}
