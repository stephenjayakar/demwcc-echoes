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
