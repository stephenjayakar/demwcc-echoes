//! Automatic header-only context for units without a source file (`mwdec harvest`).
//!
//! The context is a list of `#include` lines of headers under `include/` that define the
//! classes named by the module's symbols (relocation targets of the unit's code and of the
//! module's data objects: callees, vtables, globals), plus a few base headers. Only headers are
//! read (to index which header defines which class); never a `.cpp`/`.c` file. Compiler flags
//! are the project's REL game-unit flags (the most common flag set of REL C++ units).
//!
//! Each header is test-compiled alone once (cached on disk); the combined context drops headers
//! the compiler names in errors until it compiles. Contexts are cached per unit on disk.
use anyhow::{bail, Result};
use mwdec_core::*;
use mwdec_mwcc::Mwcc;
use mwdec_project::Project;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Base headers always included when present: `MWDEC_AUTOCTX_BASE_HEADERS` (comma-separated,
/// runtime env, else the build-time default of the same name), else just `types.h`.
fn base_headers() -> Vec<String> {
    let v = std::env::var("MWDEC_AUTOCTX_BASE_HEADERS").ok().or_else(|| option_env!("MWDEC_AUTOCTX_BASE_HEADERS").map(String::from)).unwrap_or_else(|| "types.h".into());
    v.split(',').map(|h| h.trim().to_string()).filter(|h| !h.is_empty()).collect()
}

/// class/struct name -> headers (relative to include/) that define it; REL flags; caches.
pub struct HeaderIndex {
    pub defs: BTreeMap<String, Vec<String>>,
    root: PathBuf,
    cache_dir: PathBuf,
    /// header -> compiles alone with the REL flags
    header_ok: Mutex<HashMap<String, bool>>,
    rel_flags: Mutex<Option<Vec<String>>>,
}

fn walk(dir: &Path, base: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, base, out);
        } else if let Some(ext) = p.extension().and_then(|x| x.to_str()) {
            if ext == "h" || ext == "hpp" {
                if let Ok(rel) = p.strip_prefix(base) {
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
}

/// Names of classes/structs/unions/enums *defined* (with a body) in a header text.
fn defined_types(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    for kw in ["class ", "struct ", "union ", "enum "] {
        let mut from = 0;
        while let Some(at) = text[from..].find(kw) {
            let i = from + at;
            from = i + kw.len();
            if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
                continue;
            }
            let rest = &text[from..];
            let name: String = rest.trim_start().chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
            if name.is_empty() {
                continue;
            }
            // a definition: `{` comes before any `;` / `(`
            let after = &rest[rest.find(&name).unwrap_or(0) + name.len()..];
            let brace = after.find('{');
            let semi = after.find(';');
            let paren = after.find('(');
            if let Some(br) = brace {
                if semi.map_or(true, |s| br < s) && paren.map_or(true, |p| br < p) {
                    out.push(name);
                }
            }
        }
    }
    out
}

impl HeaderIndex {
    pub fn build(root: &Path) -> HeaderIndex {
        let inc = root.join("include");
        let mut files = Vec::new();
        walk(&inc, &inc, &mut files);
        files.sort();
        let mut defs: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for f in files {
            let Ok(t) = std::fs::read(inc.join(&f)) else { continue };
            let t = String::from_utf8_lossy(&t);
            for n in defined_types(&t) {
                let v = defs.entry(n).or_default();
                if !v.contains(&f) {
                    v.push(f.clone());
                }
            }
        }
        let cache_dir = super::harvest::harvest_dir().join("autoctx");
        let header_ok = std::fs::read_to_string(cache_dir.join("header_ok.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<HashMap<String, bool>>(&s).ok())
            .unwrap_or_default();
        HeaderIndex { defs, root: root.to_path_buf(), cache_dir, header_ok: Mutex::new(header_ok), rel_flags: Mutex::new(None) }
    }

    /// Best header for a type name: the one named after it, else the shortest path.
    pub fn header_for(&self, name: &str) -> Option<&String> {
        let v = self.defs.get(name)?;
        v.iter()
            .find(|h| Path::new(h).file_stem().and_then(|s| s.to_str()) == Some(name))
            .or_else(|| v.iter().min_by_key(|h| h.len()))
    }

    /// The most common flag set of REL game (C++) units, minus the per-unit source include dir.
    pub fn rel_flags(&self, p: &Project) -> Option<Vec<String>> {
        let mut g = self.rel_flags.lock().unwrap();
        if g.is_none() {
            let mut count: HashMap<Vec<String>, usize> = HashMap::new();
            for u in &p.units {
                let Some(src) = &u.source else { continue };
                if Project::module_of(&u.name) == "main" || !src.ends_with(".cpp") || src.contains("/REL/") || src.starts_with("src/REL/") || u.cflags.is_empty() {
                    continue;
                }
                let dir = Path::new(src).parent().map(|d| d.to_string_lossy().replace('\\', "/")).unwrap_or_default();
                let mut f = u.cflags.clone();
                if let Some(i) = f.windows(2).position(|w| w[0] == "-i" && w[1] == dir) {
                    f.drain(i..i + 2);
                }
                *count.entry(f).or_default() += 1;
            }
            *g = count.into_iter().max_by_key(|(f, n)| (*n, f.len())).map(|(f, _)| f);
        }
        g.clone()
    }

    fn save_header_ok(&self) {
        let g = self.header_ok.lock().unwrap();
        let _ = std::fs::create_dir_all(&self.cache_dir);
        let _ = std::fs::write(self.cache_dir.join("header_ok.json"), serde_json::to_string(&*g).unwrap_or_default());
    }
}

/// Identifier tokens of a demangled name (`A::f(B&) const` -> A, f, B, const).
fn idents(s: &str) -> Vec<&str> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|t| !t.is_empty() && !t.as_bytes()[0].is_ascii_digit()).collect()
}

/// Type names referenced by a symbol name (mangled C++ names only).
pub fn type_names_of(sym: &str, out: &mut BTreeSet<String>) {
    let s = mwdec_lift::sig::strip_dtk_suffix(sym);
    let Some(d) = mwdec_lift::sig::demangle(s) else { return };
    for t in idents(&d) {
        out.insert(t.to_string());
    }
}

/// (unit with synthesized flags, context TU) for a unit without a source file.
pub fn auto_unit(p: &Project, u: &Unit, idx: &HeaderIndex) -> Result<(Unit, String)> {
    let Some(flags) = idx.rel_flags(p) else { bail!("no REL flag set found") };
    let mut unit = u.clone();
    unit.cflags = flags.clone();
    let cache = idx.cache_dir.join(format!("{}.txt", super::search_cmds::sanitize(&u.name)));
    if let Ok(c) = std::fs::read_to_string(&cache) {
        return Ok((unit, c));
    }
    // referenced type names: the unit's code and the module's other objects (data: vtables)
    let mut names = BTreeSet::new();
    let module = Project::module_of(&u.name);
    for rel in p.module_target_paths(module) {
        let Ok(o) = mwdec_obj::load_object(&p.path(rel).to_string_lossy()) else { continue };
        let own = rel == u.target_obj;
        for f in &o.functions {
            if own {
                type_names_of(&f.name, &mut names);
            }
            for r in &f.relocs {
                type_names_of(&r.target, &mut names);
            }
        }
        for (n, d) in &o.data {
            type_names_of(n, &mut names);
            for r in &d.relocs {
                type_names_of(&r.target, &mut names);
            }
        }
    }
    let mut headers: Vec<String> = base_headers().into_iter().filter(|h| idx.root.join("include").join(h).exists()).collect();
    for n in &names {
        if let Some(h) = idx.header_for(n) {
            if !headers.contains(h) {
                headers.push(h.clone());
            }
        }
    }
    // headers that compile alone (cached across units)
    let m = Mwcc::new(&p.root, &idx.cache_dir.join("work"), 6);
    let tmp = idx.cache_dir.join("tmp");
    let try_compile = |ctx: &str| -> std::result::Result<(), String> {
        match m.compile_tu(ctx, &flags, None, &tmp) {
            Ok(c) => {
                let _ = std::fs::remove_file(&c.obj_path);
                Ok(())
            }
            Err(e) => Err(e.messages().to_string()),
        }
    };
    let unknown: Vec<String> = {
        let g = idx.header_ok.lock().unwrap();
        headers.iter().filter(|h| !g.contains_key(*h)).cloned().collect()
    };
    if !unknown.is_empty() {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
        let res: Vec<(String, bool)> = pool.install(|| unknown.par_iter().map(|h| (h.clone(), try_compile(&format!("#include \"{h}\"\n")).is_ok())).collect());
        idx.header_ok.lock().unwrap().extend(res);
        idx.save_header_ok();
    }
    {
        let g = idx.header_ok.lock().unwrap();
        headers.retain(|h| g.get(h).copied().unwrap_or(false));
    }
    // combined: drop headers named in errors until it compiles
    let render = |hs: &[String]| hs.iter().map(|h| format!("#include \"{h}\"\n")).collect::<String>();
    let mut tries = 0;
    loop {
        let ctx = render(&headers);
        match try_compile(&ctx) {
            Ok(()) => break,
            Err(msg) => {
                tries += 1;
                // "#      In: include\X.hpp" / "#    From: ..." lines: drop the last included header
                // that appears in the messages, else the last header
                let culprit = headers
                    .iter()
                    .rposition(|h| msg.replace('\\', "/").contains(h.as_str()))
                    .unwrap_or(headers.len().saturating_sub(1));
                if headers.is_empty() || tries > 40 {
                    bail!("automatic context does not compile: {}", msg.lines().take(6).collect::<Vec<_>>().join(" | "));
                }
                headers.remove(culprit);
            }
        }
    }
    let ctx = render(&headers);
    let _ = std::fs::create_dir_all(&idx.cache_dir);
    let _ = std::fs::write(&cache, &ctx);
    Ok((unit, ctx))
}
