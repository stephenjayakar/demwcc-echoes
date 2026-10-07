//! Run mwcceppc.exe on a source string and return the object bytes.

use crate::flags::{profile_flags, Profile};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};


static COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct Compiler {
    /// Project root (read-only): compilers and headers come from here.
    pub root: PathBuf,
    /// Scratch directory for sources/objects.
    pub work: PathBuf,
    pub profile: Profile,
    /// Extra flags appended after the profile flags (before -lang is irrelevant: MWCC takes the last).
    pub extra: Vec<String>,
    /// Override compiler version dir (e.g. "GC/2.6").
    pub version: Option<String>,
    /// Exact unit cflags (e.g. from build.ninja via mwdec-project) replacing the profile flags.
    /// Relative `-i`/`-I` include paths are made absolute against `root`.
    pub cflags: Option<Vec<String>>,
}

impl Default for Compiler {
    fn default() -> Self {
        Compiler {
            root: mwdec_core::paths::project_root(),
            work: std::env::var_os("MWDEC_ORACLE_WORK").map(PathBuf::from).unwrap_or_else(|| mwdec_core::paths::work_dir("mwdec-oracle")),
            profile: Profile::Game,
            extra: Vec::new(),
            version: None,
            cflags: None,
        }
    }
}

pub struct CompileOutput {
    pub object: Vec<u8>,
    /// Compiler diagnostics (warnings), if any.
    pub messages: String,
}

impl Compiler {
    pub fn exe(&self) -> PathBuf {
        let v = self.version.clone().unwrap_or_else(|| self.profile.version().to_string());
        self.root.join("build/compilers").join(v).join("mwcceppc.exe")
    }

    pub fn flags(&self) -> Vec<String> {
        let mut f = match &self.cflags {
            None => profile_flags(self.profile, &self.root),
            Some(cf) => {
                let mut out = Vec::with_capacity(cf.len());
                let mut abs_next = false;
                for a in cf {
                    if abs_next && !std::path::Path::new(a).is_absolute() {
                        out.push(self.root.join(a).to_string_lossy().replace('\\', "/"));
                    } else {
                        out.push(a.clone());
                    }
                    abs_next = a == "-i" || a == "-I" || a == "-ir";
                }
                out
            }
        };
        f.extend(self.extra.iter().cloned());
        f
    }

    /// Compile `source` (whole TU text). `ext` is "cpp" or "c" (only affects the temp file name).
    pub fn compile(&self, source: &str) -> Result<CompileOutput> {
        std::fs::create_dir_all(&self.work).with_context(|| format!("creating {}", self.work.display()))?;
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let stem = format!("o{}_{}", std::process::id(), n);
        let ext = if self.profile.is_c() { "c" } else { "cpp" };
        let src = self.work.join(format!("{stem}.{ext}"));
        let obj = self.work.join(format!("{stem}.o"));
        std::fs::write(&src, source)?;
        let res = self.compile_file(&src, &obj);
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(self.work.join(format!("{stem}.d")));
        let out = res?;
        let bytes = std::fs::read(&obj).with_context(|| format!("reading {}", obj.display()))?;
        let _ = std::fs::remove_file(&obj);
        Ok(CompileOutput { object: bytes, messages: out })
    }

    /// Compile an existing file to `obj`; returns compiler output text.
    pub fn compile_file(&self, src: &Path, obj: &Path) -> Result<String> {
        let exe = self.exe();
        let out = Command::new(&exe)
            .args(self.flags())
            .arg("-c")
            .arg(src)
            .arg("-o")
            .arg(obj)
            .current_dir(&self.work)
            .output()
            .with_context(|| format!("running {}", exe.display()))?;
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        if !out.status.success() || !obj.exists() {
            bail!("compile failed ({}):\n{}", out.status, text.trim());
        }
        Ok(text)
    }
}
