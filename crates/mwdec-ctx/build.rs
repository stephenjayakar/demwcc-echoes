//! Hash of the sources that decide a TypeDb's content (the context builder, the core types,
//! the object reader, the compiler driver): part of the TypeDb cache key, so a cached TypeDb
//! never outlives a change to how it is built (concurrent builds share one cache directory).
use std::path::{Path, PathBuf};

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            rs_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

fn main() {
    let here = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let crates = here.parent().unwrap().to_path_buf();
    let mut files = vec![];
    for c in ["mwdec-ctx", "mwdec-core", "mwdec-obj", "mwdec-mwcc"] {
        let d = crates.join(c).join("src");
        println!("cargo:rerun-if-changed={}", d.display());
        rs_files(&d, &mut files);
    }
    files.sort();
    let mut h: u64 = 0xcbf29ce484222325;
    for f in &files {
        let rel = f.strip_prefix(&crates).unwrap_or(f).to_string_lossy().replace(std::path::MAIN_SEPARATOR, "/");
        let text = std::fs::read(f).unwrap_or_default();
        // (line endings differ between checkouts)
        for b in rel.bytes().chain([0u8]).chain(text.into_iter().filter(|&b| b != b'\r')).chain([0u8]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    println!("cargo:rustc-env=MWDEC_CTX_INPUTS_HASH={h:016x}");
}
