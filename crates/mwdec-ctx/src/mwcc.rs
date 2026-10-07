//! Private minimal compiler driver (until `mwdec-mwcc` provides one): run mwcceppc.exe
//! (through sjiswrap like the project's `mwcc_sjis` rule) with cwd = project root.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;


/// Project root: `$MWDEC_PROJECT_ROOT` or the frozen default.
pub fn project_root() -> PathBuf {
    std::env::var_os("MWDEC_PROJECT_ROOT").map(PathBuf::from).unwrap_or_else(mwdec_core::paths::project_root)
}

pub struct MwccOutput {
    pub success: bool,
    pub output: String,
}

/// Remove flags that conflict with our own invocation (-c, -o X, -MMD, -E, -maxerrors N).
pub fn sanitize_flags(cflags: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < cflags.len() {
        let f = cflags[i].as_str();
        match f {
            "-c" | "-MMD" | "-MD" | "-E" | "-EP" | "-g" => {}
            "-o" | "-maxerrors" => i += 1,
            _ => out.push(cflags[i].clone()),
        }
        i += 1;
    }
    out
}

/// Shared `mwdec-mwcc` driver per project root (its process pool caps concurrent compilers at 6).
pub fn driver(root: &Path) -> std::sync::Arc<mwdec_mwcc::Mwcc> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static DRIVERS: OnceLock<Mutex<HashMap<PathBuf, Arc<mwdec_mwcc::Mwcc>>>> = OnceLock::new();
    let map = DRIVERS.get_or_init(Default::default);
    map.lock()
        .unwrap()
        .entry(root.to_path_buf())
        .or_insert_with(|| Arc::new(mwdec_mwcc::Mwcc::new(root, &mwdec_mwcc::default_work(), 6)))
        .clone()
}

/// Compile (or with `-E` in `flags`, preprocess) the TU text `source` in `dir` through the
/// `mwdec-mwcc` driver. Ok(Ok(output bytes)), Ok(Err(compiler messages)) when the compiler
/// rejects the input, Err for I/O failures and timeouts.
pub fn compile_text(root: &Path, source: &str, flags: &[String], dir: &Path) -> Result<std::result::Result<Vec<u8>, String>> {
    let drv = driver(root);
    match drv.compile_tu(source, flags, None, dir) {
        Ok(c) => {
            let _ = std::fs::remove_file(&c.obj_path);
            Ok(Ok(c.obj.to_vec()))
        }
        Err(mwdec_mwcc::MwccError::Compile { messages, .. }) => Ok(Err(messages)),
        Err(e) => Err(anyhow::anyhow!("{e}")),
    }
}

/// Run the compiler directly (through sjiswrap when present); `args` are complete
/// (flags + inputs + outputs). Kept for ad-hoc experiments; the pipeline uses `compile_text`.
pub fn run(root: &Path, args: &[String]) -> Result<MwccOutput> {
    let mwcc = root.join("build/compilers/GC/2.7/mwcceppc.exe");
    let sjis = root.join("build/tools/sjiswrap.exe");
    let mut cmd = if sjis.exists() {
        let mut c = Command::new(&sjis);
        c.arg(&mwcc);
        c
    } else {
        Command::new(&mwcc)
    };
    cmd.args(args).current_dir(root);
    let out = cmd.output().with_context(|| format!("running {}", mwcc.display()))?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok(MwccOutput { success: out.status.success(), output: text })
}

/// Lines (1-based) of `file_text` that the compiler reported errors on. MWCC echoes the
/// offending source line (`#      12: <text>`) before `#   Error:`; we only accept a line number
/// whose echoed text matches our file, so errors inside headers are not misattributed.
pub fn error_lines(output: &str, file_text: &str) -> Vec<usize> {
    let lines: Vec<&str> = file_text.lines().collect();
    let out_lines: Vec<&str> = output.lines().collect();
    let mut res = Vec::new();
    for (k, l) in out_lines.iter().enumerate() {
        let t = l.trim_start_matches('#').trim_start();
        let Some(colon) = t.find(": ") else { continue };
        let Ok(n) = t[..colon].trim().parse::<usize>() else { continue };
        let echoed = t[colon + 2..].trim();
        let is_err = out_lines.get(k + 1).is_some_and(|nx| nx.contains("Error:"));
        if !is_err || n == 0 || n > lines.len() {
            continue;
        }
        if lines[n - 1].trim() == echoed || (echoed.len() > 20 && lines[n - 1].trim().starts_with(&echoed[..20])) {
            res.push(n);
        }
    }
    res.sort();
    res.dedup();
    res
}

pub fn path_arg(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}
