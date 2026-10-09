//! Project metadata: units and flags from objdiff.json + build.ninja, measures from report.json,
//! dataset/split.
//!
//! Nothing here reads source *bodies*. `source` paths are carried as bookkeeping only; the one
//! sanctioned read of a source file is [`harness::context_tu`] (include lines only), which exists
//! for the eval harness.
use anyhow::{bail, Context, Result};
use mwdec_core::*;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub mod ninja;
pub mod standalone;

pub struct Project {
    pub root: PathBuf,
    pub units: Vec<Unit>,
    /// Compiler version directory per unit name (e.g. "GC/2.7"), from build.ninja.
    pub mw_version: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct ObjdiffJson {
    units: Vec<ObjdiffUnit>,
}

#[derive(Deserialize)]
struct ObjdiffUnit {
    name: String,
    target_path: Option<String>,
    base_path: Option<String>,
    #[serde(default)]
    metadata: Option<ObjdiffMeta>,
}

#[derive(Deserialize, Default)]
struct ObjdiffMeta {
    source_path: Option<String>,
}

fn norm(p: &str) -> String {
    p.replace('\\', "/")
}

/// The real build compiles the TU in place, so `-gccinc` finds headers next to the source
/// (e.g. `src/Runtime/NMWException.h`). Our TUs live in a scratch dir, so make that directory an
/// explicit include path, searched first like the referencing file's directory. Headers only:
/// the compiler resolves `#include`s there; nothing here reads source files.
pub fn with_source_dir(cflags: &mut Vec<String>, source: &str) {
    let Some(dir) = Path::new(source).parent().map(|d| d.to_string_lossy().replace('\\', "/")) else { return };
    if dir.is_empty() || cflags.windows(2).any(|w| w[0] == "-i" && w[1] == dir) {
        return;
    }
    let at = cflags.iter().position(|f| f == "-i" || f == "-I" || f == "-ir").unwrap_or(cflags.len());
    cflags.splice(at..at, ["-i".to_string(), dir]);
}

impl Project {
    /// Load units from `objdiff.json` and per-unit compiler flags from `build.ninja`.
    pub fn load(root: &Path) -> Result<Project> {
        let od_path = root.join("objdiff.json");
        let od: ObjdiffJson = serde_json::from_str(
            &std::fs::read_to_string(&od_path).with_context(|| format!("reading {}", od_path.display()))?,
        )
        .context("parsing objdiff.json")?;
        let nj_path = root.join("build.ninja");
        let edges = ninja::mwcc_edges(
            &std::fs::read_to_string(&nj_path).with_context(|| format!("reading {}", nj_path.display()))?,
        )?;
        let by_out: HashMap<String, &ninja::MwccEdge> = edges.iter().map(|e| (norm(&e.output), e)).collect();
        let by_src: HashMap<String, &ninja::MwccEdge> = edges.iter().map(|e| (norm(&e.input), e)).collect();

        let mut units = Vec::new();
        let mut mw_version = BTreeMap::new();
        for u in od.units {
            let Some(target) = u.target_path else { continue };
            let source = u.metadata.and_then(|m| m.source_path).map(|s| norm(&s));
            let base = u.base_path.map(|s| norm(&s));
            let edge = base
                .as_ref()
                .and_then(|b| by_out.get(b))
                .or_else(|| source.as_ref().and_then(|s| by_src.get(s)));
            let mut cflags = edge.map(|e| e.cflags.clone()).unwrap_or_default();
            if let (Some(src), false) = (&source, cflags.is_empty()) {
                with_source_dir(&mut cflags, src);
            }
            if let Some(e) = edge {
                mw_version.insert(u.name.clone(), norm(&e.mw_version));
            }
            units.push(Unit { name: u.name, source, target_obj: norm(&target), base_obj: base, cflags });
        }
        Ok(Project { root: root.to_path_buf(), units, mw_version })
    }

    pub fn unit(&self, name: &str) -> Option<&Unit> {
        self.units.iter().find(|u| u.name == name)
    }

    /// Absolute path of a project-root-relative path.
    pub fn path(&self, rel: &str) -> PathBuf {
        self.root.join(rel)
    }

    /// Module of a unit: "main" or the REL name (the unit name's first path component).
    pub fn module_of(unit: &str) -> &str {
        unit.split('/').next().unwrap_or(unit)
    }

    /// Target objects of every unit in `module` (including auto-generated data splits).
    pub fn module_target_paths(&self, module: &str) -> Vec<&str> {
        self.units.iter().filter(|u| Self::module_of(&u.name) == module).map(|u| u.target_obj.as_str()).collect()
    }

    /// Load every target object of `module` in parallel (missing objects are skipped). Used to
    /// resolve literals and unnamed functions a target object only references (dtk puts pooled
    /// literals/strings in other splits).
    pub fn load_module_data(&self, module: &str) -> Vec<std::sync::Arc<ObjectFile>> {
        self.load_objects(&self.module_target_paths(module))
    }

    /// Objects as the project links them for `module`: our base object where the unit has one,
    /// else the target split. Resolves what *our* objects only reference.
    pub fn load_module_linked(&self, module: &str) -> Vec<std::sync::Arc<ObjectFile>> {
        let mut paths: Vec<&str> = self
            .units
            .iter()
            .filter(|u| Self::module_of(&u.name) == module)
            .map(|u| u.base_obj.as_deref().unwrap_or(&u.target_obj))
            .collect();
        paths.sort();
        paths.dedup();
        self.load_objects(&paths)
    }

    fn load_objects(&self, paths: &[&str]) -> Vec<std::sync::Arc<ObjectFile>> {
        paths
            .par_iter()
            .filter_map(|rel| {
                let o = mwdec_obj::load_object(&self.root.join(rel).to_string_lossy()).ok()?;
                Some(std::sync::Arc::new(o))
            })
            .collect()
    }

    /// Path of the unit's compiler executable (project-root relative), e.g. `build/compilers/GC/2.7/mwcceppc.exe`.
    pub fn compiler_rel(&self, unit: &str) -> String {
        let v = self.mw_version.get(unit).map(String::as_str).unwrap_or("GC/2.7");
        format!("build/compilers/{v}/mwcceppc.exe")
    }

    /// Functions currently at 100% fuzzy match in `build/G2ME01/report.json` whose symbol in the
    /// target object is not weak (weak = header inline / template instantiation, excluded),
    /// for units that have compiler flags. Compiler-generated `name$123` helpers are excluded. Sorted by (unit, symbol).
    pub fn dataset(&self) -> Result<Vec<DatasetEntry>> {
        let rp = self.root.join("build/G2ME01/report.json");
        let report: Report = serde_json::from_str(
            &std::fs::read_to_string(&rp).with_context(|| format!("reading {}", rp.display()))?,
        )
        .context("parsing report.json")?;
        let units: HashMap<&str, &Unit> = self.units.iter().map(|u| (u.name.as_str(), u)).collect();
        let mut out: Vec<DatasetEntry> = report
            .units
            .par_iter()
            .filter_map(|ru| {
                let u = units.get(ru.name.as_str())?;
                if u.cflags.is_empty() || u.base_obj.is_none() {
                    return None;
                }
                let full: Vec<&ReportFunction> =
                    ru.functions.iter().filter(|f| f.fuzzy_match_percent == Some(100.0)).collect();
                if full.is_empty() {
                    return None;
                }
                let tp = self.root.join(&u.target_obj);
                let obj = mwdec_obj::load_object(&tp.to_string_lossy()).ok()?;
                let bind: HashMap<&str, (SymBinding, usize)> =
                    obj.functions.iter().map(|f| (f.name.as_str(), (f.binding, f.code.len()))).collect();
                let v: Vec<DatasetEntry> = full
                    .into_iter()
                    .filter_map(|f| {
                        if is_compiler_generated(&f.name) {
                            return None;
                        }
                        let (b, len) = *bind.get(f.name.as_str())?;
                        if b == SymBinding::Weak {
                            return None;
                        }
                        let size = f.size.as_deref().and_then(|s| s.parse().ok()).unwrap_or(len as u32);
                        Some(DatasetEntry {
                            unit: u.name.clone(),
                            symbol: f.name.clone(),
                            size,
                            split: split_of(&u.name).to_string(),
                        })
                    })
                    .collect();
                Some(v)
            })
            .flatten()
            .collect();
        out.sort_by(|a, b| (&a.unit, &a.symbol).cmp(&(&b.unit, &b.symbol)));
        Ok(out)
    }
}

#[derive(Deserialize)]
struct Report {
    units: Vec<ReportUnit>,
}

#[derive(Deserialize)]
struct ReportUnit {
    name: String,
    #[serde(default)]
    functions: Vec<ReportFunction>,
}

#[derive(Deserialize)]
struct ReportFunction {
    name: String,
    size: Option<String>,
    fuzzy_match_percent: Option<f64>,
}

/// Compiler-generated helpers with TU-local numbered names (e.g. `__arraydtor$321`): they are
/// emitted as a side effect of other code, not written in source, so they are not dataset items.
pub fn is_compiler_generated(name: &str) -> bool {
    match name.rfind('$') {
        Some(i) => i + 1 < name.len() && name[i + 1..].bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

/// Size bucket label used by stats/eval: <=32, <=64, ..., <=1024, >1024 bytes.
pub fn size_bucket(size: u32) -> &'static str {
    match size {
        0..=32 => "<=32",
        33..=64 => "<=64",
        65..=128 => "<=128",
        129..=256 => "<=256",
        257..=512 => "<=512",
        513..=1024 => "<=1024",
        _ => ">1024",
    }
}

pub const SIZE_BUCKETS: [&str; 7] = ["<=32", "<=64", "<=128", "<=256", "<=512", "<=1024", ">1024"];

/// FNV-1a 32 of the unit name; `% 10 < 7` => "train", else "test".
pub fn split_of(unit: &str) -> &'static str {
    let mut h: u32 = 0x811c9dc5;
    for b in unit.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    if h % 10 < 7 { "train" } else { "test" }
}

/// Eval-harness bookkeeping. NOT for use by the decompiler.
pub mod harness {
    use super::*;

    /// The context TU for a unit: ONLY the `#include` lines of the unit's source file, in order,
    /// and nothing else. This is the single sanctioned read of a source file (DESIGN.md: "A
    /// context TU is the list of `#include` lines of the unit"); it never returns any other text.
    pub fn context_tu(project: &Project, unit: &Unit) -> Result<String> {
        let Some(src) = &unit.source else { bail!("unit {} has no source path", unit.name) };
        let text = std::fs::read(project.root.join(src)).with_context(|| format!("reading includes of {src}"))?;
        let text = String::from_utf8_lossy(&text);
        let mut out = include_lines(&text);
        // headers a `.inc` fragment of the file includes (its own includes are the TU's too):
        // only their `#include` lines, from the project's include directory
        let dir = project.root.join(src).parent().map(|d| d.to_path_buf()).unwrap_or_default();
        for frag in fragment_includes(&text) {
            for base in [project.root.join("include"), dir.clone()] {
                if let Ok(t) = std::fs::read(base.join(&frag)) {
                    for l in include_lines(&String::from_utf8_lossy(&t)).lines() {
                        if !out.lines().any(|o| o == l) {
                            out.push_str(l);
                            out.push('\n');
                        }
                    }
                    break;
                }
            }
        }
        Ok(out)
    }

    /// The `.inc` fragments a file includes (quoted names).
    fn fragment_includes(text: &str) -> Vec<String> {
        let mut out = vec![];
        for line in text.lines() {
            let t = line.trim_start();
            let Some(rest) = t.strip_prefix('#') else { continue };
            let Some(arg) = rest.trim_start().strip_prefix("include") else { continue };
            let arg = arg.trim();
            if let Some(r) = arg.strip_prefix('"') {
                if let Some(e) = r.find('"') {
                    let name = &r[..e];
                    if name.to_ascii_lowercase().ends_with(".inc") {
                        out.push(name.to_string());
                    }
                }
            }
        }
        out
    }

    /// Extract `#include` directives (one per line, trailing comments dropped).
    pub fn include_lines(text: &str) -> String {
        let mut out = String::new();
        for line in text.lines() {
            let t = line.trim_start();
            let Some(rest) = t.strip_prefix('#') else { continue };
            let rest = rest.trim_start();
            let Some(arg) = rest.strip_prefix("include") else { continue };
            let arg = arg.trim();
            // Keep only the header name token: "<...>" or "\"...\"".
            let tok = match arg.chars().next() {
                Some('"') => arg[1..].find('"').map(|e| &arg[..e + 2]),
                Some('<') => arg.find('>').map(|e| &arg[..e + 1]),
                _ => None,
            };
            // Headers only: `.inc`/`.c`/`.cpp` fragments are code pasted into function bodies
            // or the TU, not context.
            let is_header = tok.is_some_and(|t| {
                let name = t[1..t.len() - 1].to_ascii_lowercase();
                [".h", ".hpp", ".hh", ".hxx", ".h++"].iter().any(|e| name.ends_with(e)) || !name.contains('.')
            });
            if !is_header {
                continue;
            }
            if let Some(tok) = tok {
                out.push_str("#include ");
                out.push_str(tok);
                out.push('\n');
            }
        }
        out
    }
}

