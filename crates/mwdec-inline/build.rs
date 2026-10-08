//! Hash of the sources that decide what a probe's template is (the lifter, the type context,
//! the core IR types and template extraction): part of the template cache key, so cached
//! templates never outlive a change to how probes are lifted.
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
    for c in ["mwdec-lift", "mwdec-ctx", "mwdec-core"] {
        let d = crates.join(c).join("src");
        println!("cargo:rerun-if-changed={}", d.display());
        rs_files(&d, &mut files);
    }
    for f in ["template.rs", "stmts.rs", "probe.rs", "cflow.rs", "ser.rs"] {
        let p = here.join("src").join(f);
        println!("cargo:rerun-if-changed={}", p.display());
        files.push(p);
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
    println!("cargo:rustc-env=MWDEC_TEMPLATE_INPUTS_HASH={h:016x}");
}
