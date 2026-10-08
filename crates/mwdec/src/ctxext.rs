//! Context headers for classes a unit's code names but its include-only context lacks.
//!
//! The harness context of a unit is the `#include` lines of its source. Symbols of the unit's
//! target object (function names, relocation targets, data symbols such as vtables) can name
//! classes none of those headers (nor anything they include) defines; the drafter then has no
//! layout, member names, vtable or destructor for them. Headers under `include/` that define such
//! classes (the harvest header index) are appended to the context, as long as the extended
//! context still compiles (headers the compiler names in errors are dropped). The decision only
//! reads header texts and symbol names, and is cached on disk per (unit, context, flags).

use super::autoctx::{type_names_of, HeaderIndex};
use mwdec_core::*;
use mwdec_mwcc::Mwcc;
use mwdec_project::Project;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::sync::OnceLock;

struct Index {
    headers: HeaderIndex,
    /// header (relative to include/) -> headers it includes (resolved, relative to include/)
    includes: BTreeMap<String, Vec<String>>,
}

static INDEX: OnceLock<Index> = OnceLock::new();

fn include_targets(text: &str) -> Vec<String> {
    let mut out = vec![];
    for line in text.lines() {
        let t = line.trim_start();
        let Some(r) = t.strip_prefix('#') else { continue };
        let Some(r) = r.trim_start().strip_prefix("include") else { continue };
        let r = r.trim();
        let close = match r.chars().next() {
            Some('"') => '"',
            Some('<') => '>',
            _ => continue,
        };
        let inner = &r[1..];
        if let Some(e) = inner.find(close) {
            out.push(inner[..e].replace('\\', "/"));
        }
    }
    out
}

fn walk(dir: &Path, base: &Path, out: &mut BTreeSet<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, base, out);
        } else if matches!(p.extension().and_then(|x| x.to_str()), Some("h" | "hpp")) {
            if let Ok(rel) = p.strip_prefix(base) {
                out.insert(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

fn index(root: &Path) -> &'static Index {
    INDEX.get_or_init(|| {
        let headers = HeaderIndex::build(root);
        let inc = root.join("include");
        let mut files: BTreeSet<String> = BTreeSet::new();
        walk(&inc, &inc, &mut files);
        let mut includes = BTreeMap::new();
        for f in &files {
            let Ok(t) = std::fs::read(inc.join(f)) else { continue };
            let dir = Path::new(f).parent().map(|d| d.to_string_lossy().replace('\\', "/")).unwrap_or_default();
            let v: Vec<String> = include_targets(&String::from_utf8_lossy(&t))
                .into_iter()
                .map(|h| {
                    let rel = if dir.is_empty() { h.clone() } else { format!("{dir}/{h}") };
                    if files.contains(&rel) {
                        rel
                    } else {
                        h
                    }
                })
                .collect();
            includes.insert(f.clone(), v);
        }
        Index { headers, includes }
    })
}

/// Headers (relative to include/) reachable from the context's include lines.
fn closure(ix: &Index, context: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut q: VecDeque<String> = include_targets(context).into();
    while let Some(h) = q.pop_front() {
        if !seen.insert(h.clone()) {
            continue;
        }
        if let Some(v) = ix.includes.get(&h) {
            q.extend(v.iter().cloned());
        }
    }
    seen
}

/// The unit's context with the headers defining the classes its target object names that the
/// context doesn't (see the module docs). Opt-in (`MWDEC_AUTO_HEADERS=1`): on the train split it
/// changed 127 drafts without changing any exact result.
pub fn extended_context(p: &Project, u: &Unit, context: &str) -> String {
    if std::env::var_os("MWDEC_AUTO_HEADERS").is_none() || context.is_empty() || u.cflags.iter().any(|f| f == "-lang=c" || f == "-lang=c99") {
        return context.to_string();
    }
    let Ok(obj) = mwdec_obj::load_object(&p.path(&u.target_obj).to_string_lossy()) else { return context.to_string() };
    let ix = index(&p.root);
    let mut names = BTreeSet::new();
    for f in &obj.functions {
        type_names_of(&f.name, &mut names);
        for r in &f.relocs {
            type_names_of(&r.target, &mut names);
        }
    }
    for (n, d) in &obj.data {
        type_names_of(n, &mut names);
        for r in &d.relocs {
            type_names_of(&r.target, &mut names);
        }
    }
    let have = closure(ix, context);
    let mut extra: Vec<String> = vec![];
    for n in &names {
        let Some(defs) = ix.headers.defs.get(n) else { continue };
        if defs.iter().any(|h| have.contains(h)) {
            continue;
        }
        if let Some(h) = ix.headers.header_for(n) {
            if !extra.contains(h) {
                extra.push(h.clone());
            }
        }
    }
    if extra.is_empty() {
        return context.to_string();
    }
    extra.sort();
    let key = mwdec_mwcc::content_hash(&[context.as_bytes(), extra.join("\n").as_bytes(), u.cflags.join(" ").as_bytes(), b"ctxext-1"]);
    let dir = mwdec_core::paths::work_dir("mwdec-ctxext");
    let cache = dir.join(format!("{key:016x}.txt"));
    if let Ok(c) = std::fs::read_to_string(&cache) {
        return c;
    }
    let m = Mwcc::new(&p.root, &dir.join("work"), 2);
    let tmp = dir.join("tmp");
    let sep = if context.ends_with('\n') { "" } else { "\n" };
    let render = |hs: &[String]| format!("{context}{sep}{}", hs.iter().map(|h| format!("#include \"{h}\"\n")).collect::<String>());
    let mut tries = 0;
    let out = loop {
        if extra.is_empty() {
            break context.to_string();
        }
        let ctx = render(&extra);
        match m.compile_tu(&ctx, &u.cflags, None, &tmp) {
            Ok(c) => {
                let _ = std::fs::remove_file(&c.obj_path);
                break ctx;
            }
            Err(e) => {
                tries += 1;
                let msg = e.messages().replace('\\', "/");
                let culprit = extra.iter().rposition(|h| msg.contains(h.as_str())).unwrap_or(extra.len() - 1);
                extra.remove(culprit);
                if tries > 12 {
                    break context.to_string();
                }
            }
        }
    };
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(&cache, &out);
    out
}
