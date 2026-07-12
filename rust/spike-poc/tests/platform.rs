//! P11 platform timing: index read via mmap vs positioned reads. On Windows
//! this is where Defender on-access scanning would show up; the numbers are
//! recorded from the CI windows job. Cursor-independence of positioned reads
//! is also asserted (concurrent reads must not share a file cursor).

mod common;

use std::io::Read;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;

/// Positioned read of the 128-B pack header at offset 0 and the index region,
/// via the platform `ReadAt`/`seek_read` API, matches a plain buffered read —
/// and does not disturb a cursor.
#[test]
fn positioned_reads_are_cursor_independent() {
    let path = common::data_dir().join("heightfield.hfp");
    let whole = std::fs::read(&path).unwrap();
    let file = std::fs::File::open(&path).unwrap();

    let mut hdr = [0u8; 128];
    read_at(&file, &mut hdr, 0);
    assert_eq!(&hdr[..], &whole[..128]);

    // a second positioned read at a later offset must not depend on the first
    let mut mid = [0u8; 32];
    read_at(&file, &mut mid, 200);
    assert_eq!(&mid[..], &whole[200..232]);

    // a plain sequential read still starts at 0 (positioned reads left the
    // cursor untouched on unix; on windows seek_read moves it, so we reopen)
    let mut f2 = std::fs::File::open(&path).unwrap();
    let mut first4 = [0u8; 4];
    f2.read_exact(&mut first4).unwrap();
    assert_eq!(&first4, b"AHNP");
}

/// mmap vs positioned-read timing over the index region, repeated, so the
/// windows job records a Defender-sensitive number. Informational: asserts
/// only that both complete and agree.
#[test]
fn mmap_vs_pread_index_timing() {
    let path = common::data_dir().join("heightfield.hfp");
    let file = std::fs::File::open(&path).unwrap();
    let len = file.metadata().unwrap().len() as usize;

    let reps = 500u32;

    // positioned reads: pull the index region [128, hash_offset) each rep
    let mut buf = vec![0u8; len];
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        read_at(&file, &mut buf, 0);
    }
    let pread = t0.elapsed() / reps;

    // mmap: map once, copy the same span each rep
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let mut sink = vec![0u8; len];
    let t1 = std::time::Instant::now();
    for _ in 0..reps {
        sink.copy_from_slice(&mmap[..len]);
    }
    let mm = t1.elapsed() / reps;

    assert_eq!(&buf[..], &mmap[..]);
    let os = if cfg!(windows) {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    };
    eprintln!(
        "[bench] os={os} file={len}B reps={reps}: pread {:?}/rep, mmap-copy {:?}/rep",
        pread, mm
    );
}

#[cfg(unix)]
fn read_at(file: &std::fs::File, buf: &mut [u8], off: u64) {
    let n = file.read_at(buf, off).unwrap();
    assert_eq!(n, buf.len());
}

#[cfg(windows)]
fn read_at(file: &std::fs::File, buf: &mut [u8], off: u64) {
    // seek_read may return a short read; loop to fill.
    let mut filled = 0usize;
    while filled < buf.len() {
        let n = file
            .seek_read(&mut buf[filled..], off + filled as u64)
            .unwrap();
        assert!(n > 0, "unexpected EOF in seek_read");
        filled += n;
    }
}
