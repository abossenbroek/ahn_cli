//! Shared archive-layer types.
//!
//! This module currently defines the foundational public types the `AHNP`
//! pack reader is built on — the [`ReadAt`] positioned-read trait, the
//! [`TileKey`] lookup key, and the [`BlobSlot`] tag. The `Archive<R>` reader
//! itself, together with `PackHeader`, `Entry`, `LevelRun` and the pack
//! constants, is added in a later phase; the types here are stable and used
//! by both the error surface and that future reader.

use std::cmp::Ordering;
use std::io;

/// Positioned, cursor-independent reads over a pack's backing storage.
///
/// This mirrors the platform positioned-read extension traits
/// ([`std::os::unix::fs::FileExt::read_at`] /
/// `std::os::windows::fs::FileExt::seek_read`): every read names its own
/// `offset`, so there is no shared cursor and concurrent tile loads need no
/// lock. It takes `&self`, not `&mut self`, for exactly that reason.
///
/// The trait is **not sealed** — implement it for custom transports (HTTP
/// range reads, test mocks, an mmap slice). Only [`ReadAt::read_at`] is
/// required; [`ReadAt::read_exact_at`] is provided.
///
/// # Examples
///
/// A byte slice is the simplest backing store:
///
/// ```
/// use ahn_heightfield::ReadAt;
///
/// let data: &[u8] = b"AHNP....payload";
/// let mut buf = [0u8; 4];
/// let n = ReadAt::read_at(&data, &mut buf, 0)?;
/// assert_eq!(&buf[..n], b"AHNP");
/// # Ok::<(), std::io::Error>(())
/// ```
pub trait ReadAt {
    /// Reads into `buf` starting at `offset`, cursor-independent.
    ///
    /// Returns the number of bytes read (`0` at end of input); may read fewer
    /// than `buf.len()` bytes.
    ///
    /// # Errors
    ///
    /// Returns any [`std::io::Error`] the underlying storage reports.
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

    /// Reads exactly `buf.len()` bytes starting at `offset`, looping over
    /// short reads.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::UnexpectedEof`] if the storage ends before
    /// `buf` is filled, or any error [`ReadAt::read_at`] reports. Provided; do
    /// not override unless a faster exact-read path exists.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let mut filled = 0usize;
        while filled < buf.len() {
            let got = self.read_at(&mut buf[filled..], offset + filled as u64)?;
            if got == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            filled += got;
        }
        Ok(())
    }
}

#[cfg(unix)]
impl ReadAt for std::fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl ReadAt for std::fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::windows::fs::FileExt::seek_read(self, buf, offset)
    }
}

impl ReadAt for &[u8] {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(self.len());
        let src = &self[start..];
        let n = src.len().min(buf.len());
        buf[..n].copy_from_slice(&src[..n]);
        Ok(n)
    }
}

impl<T: ReadAt + ?Sized> ReadAt for &T {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        (**self).read_at(buf, offset)
    }
}

/// Which blob slot of a pack entry an error or lookup refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlobSlot {
    /// The primary blob (`.hf` chunk or `.glb`).
    Primary,
    /// The texture blob (`.jpg`), present only for heightfield packs.
    Texture,
}

/// A pack lookup key: a tile's `(level, tx, ty, tz)` coordinates.
///
/// Field order is `(level, tx, ty, tz)` to match the on-disk entry layout,
/// but [`Ord`] uses the spec's sort key `(level, tz, ty, tx)` — a **different**
/// order. `Ord`/`PartialOrd` are therefore hand-written, not derived, so this
/// mismatch cannot be introduced silently. `tz` is `0` in this format version.
///
/// # Examples
///
/// ```
/// use ahn_heightfield::TileKey;
///
/// let a = TileKey { level: 1, tx: 1, ty: 0, tz: 0 };
/// let b = TileKey { level: 1, tx: 0, ty: 1, tz: 0 };
/// // Sorted by (level, tz, ty, tx): b's ty is larger, so a < b.
/// assert!(a < b);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    /// Quadtree level (`0` = root).
    pub level: u32,
    /// Tile column index at this level.
    pub tx: u32,
    /// Tile row index at this level.
    pub ty: u32,
    /// Tile depth index (always `0` in this version).
    pub tz: u32,
}

impl Ord for TileKey {
    /// Orders by the spec's `(level, tz, ty, tx)` sort key — **not** the
    /// struct's `(level, tx, ty, tz)` field order.
    fn cmp(&self, other: &Self) -> Ordering {
        (self.level, self.tz, self.ty, self.tx).cmp(&(other.level, other.tz, other.ty, other.tx))
    }
}

impl PartialOrd for TileKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
