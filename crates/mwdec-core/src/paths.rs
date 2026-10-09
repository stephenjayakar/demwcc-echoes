//! Default locations of the project checkout and scratch space.
//!
//! Runtime env vars win (`MWDEC_ROOT`, `MWDEC_WORK_BASE`); otherwise the build-time defaults
//! `MWDEC_DEFAULT_ROOT` / `MWDEC_DEFAULT_WORK_BASE` (set e.g. in a local `.cargo/config.toml`
//! `[env]` table); otherwise the current directory / `./mwdec-work`.

use std::path::PathBuf;

/// The decomp project checkout (read-only input: build dir, headers, compilers, configure.py).
pub fn project_root() -> PathBuf {
    if let Some(p) = std::env::var_os("MWDEC_ROOT") {
        return PathBuf::from(p);
    }
    PathBuf::from(option_env!("MWDEC_DEFAULT_ROOT").unwrap_or("."))
}

/// Base directory for scratch output (compiles, caches, eval logs).
pub fn work_base() -> PathBuf {
    if let Some(p) = std::env::var_os("MWDEC_WORK_BASE") {
        return PathBuf::from(p);
    }
    PathBuf::from(option_env!("MWDEC_DEFAULT_WORK_BASE").unwrap_or("mwdec-work"))
}

/// `work_base()/<sub>`.
pub fn work_dir(sub: &str) -> PathBuf {
    work_base().join(sub)
}

/// Fingerprint of the project's headers: path, size and modification time of every file under
/// the include directories (`include`, `libc`, `extern`, `build/G2ME01/include`) and of the
/// header files under `src` (FNV-1a 64 of the sorted entries). Caches of results derived from
/// headers (TypeDbs, compiled objects) are keyed by it, so an edited header never serves an old
/// result. Computed once per process and root.
pub fn header_fingerprint(root: &std::path::Path) -> u64 {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<Vec<(PathBuf, u64)>>> = OnceLock::new();
    let c = CACHE.get_or_init(Default::default);
    if let Some((_, h)) = c.lock().unwrap().iter().find(|(r, _)| r == root) {
        return *h;
    }
    fn walk(base: &std::path::Path, d: &std::path::Path, headers_only: bool, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            let Ok(m) = e.metadata() else { continue };
            if m.is_dir() {
                walk(base, &p, headers_only, out);
                continue;
            }
            if headers_only && !p.extension().is_some_and(|x| matches!(x.to_str(), Some("h" | "hpp" | "inc" | "hxx"))) {
                continue;
            }
            let t = m.modified().ok().and_then(|t| t.duration_since(std::time::SystemTime::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
            out.push(format!("{}\u{1}{}\u{1}{t}", p.strip_prefix(base).unwrap_or(&p).to_string_lossy().replace('\\', "/"), m.len()));
        }
    }
    let mut entries: Vec<String> = vec![];
    for (sub, headers_only) in [("include", false), ("libc", false), ("extern", false), ("build/G2ME01/include", false), ("src", true)] {
        walk(root, &root.join(sub), headers_only, &mut entries);
    }
    entries.sort();
    let mut h: u64 = 0xcbf29ce484222325;
    for e in &entries {
        for b in e.bytes().chain([b'\n']) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    c.lock().unwrap().push((root.to_path_buf(), h));
    h
}
