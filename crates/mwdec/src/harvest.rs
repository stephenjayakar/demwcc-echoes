//! `mwdec harvest`: run draft + search over functions that are NOT yet matched in the project's
//! report.json (fuzzy < 100), smallest first, and keep every exact result.
//!
//! Inputs are the same as for `match`: the unit's target object, the module's other target
//! objects, a context TU of include lines only, and the compiler. Units with a source file use
//! their include lines (`harness::context_tu`); units without one get an automatic context built
//! from headers under `include/` only (`autoctx`). No source body is ever read.
//!
//! Output (under `--out-dir`, default `work_dir("harvest")`):
//! - `attempts.jsonl`: one line per finished attempt (any status), tagged with `--tag`;
//! - `exact.jsonl`: one line per exact result (unit, symbol, size, source, preamble, flags, ...);
//! - `funcs/<unit>/<symbol>.cpp`: context + final source of each exact result;
//! - `started.log`: a line per attempt start (crash detection: an attempt started twice under
//!   the same tag without finishing is skipped as `crashed`).
//!
//! Robustness: the work runs in a child process that the parent restarts after an abnormal exit
//! (allocation failure under the memory cap, compiler-driver panics); per-function panics are
//! caught. Compiler drivers (and their in-memory object caches) are per unit and dropped with
//! the unit, so memory stays bounded over long runs.
use super::search_cmds::{choose_draft, draft_with, externs_for, sanitize, unit_inputs_with_context, Compilers, NoDraft, UnitInputs};
use super::{load_project, autoctx};
use anyhow::{bail, Context, Result};
use mwdec_core::*;
use mwdec_mwcc::{ExternIndex, ObjIndex};
use mwdec_project::{harness, is_compiler_generated, size_bucket, Project, SIZE_BUCKETS};
use mwdec_search::{search, Scorer, SearchConfig};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Default results directory: `work_dir("harvest")`.
pub fn harvest_dir() -> PathBuf {
    mwdec_core::paths::work_dir("harvest")
}

#[derive(Clone)]
pub struct HarvestArgs {
    pub max_size: u32,
    pub min_size: u32,
    pub budget_secs: u64,
    pub max_compiles: Option<usize>,
    pub jobs: usize,
    pub workers: Option<usize>,
    pub unit: Option<String>,
    pub symbol: Option<String>,
    pub out_dir: PathBuf,
    pub limit: Option<usize>,
    /// Attempts with this tag are not repeated (bump it after decompiler improvements).
    pub tag: String,
    /// "sourced", "auto" or "all".
    pub scope: String,
    /// Report to read (default: <root>/build/G2ME01/report.json).
    pub report: Option<PathBuf>,
    pub no_supervise: bool,
    pub seed: u64,
    /// Only list the candidates.
    pub list: bool,
    /// Only functions whose best score in earlier attempts (any tag) is at least this (near
    /// misses worth a longer search).
    pub min_best: Option<f64>,
}

/// One unmatched function.
#[derive(Clone, Debug)]
pub struct Cand {
    pub unit: String,
    pub symbol: String,
    pub size: u32,
    pub fuzzy: f64,
    pub sourced: bool,
}

/// Unmatched (fuzzy < 100), non-weak, not compiler-generated functions of the report, smallest
/// first (grouped by unit within a size bucket, so unit inputs are reused).
pub fn candidates(p: &Project, report: &Path, a: &HarvestArgs) -> Result<Vec<Cand>> {
    let text = std::fs::read_to_string(report).with_context(|| format!("reading {}", report.display()))?;
    let r: serde_json::Value = serde_json::from_str(&text).context("parsing report.json")?;
    let units: HashMap<&str, &Unit> = p.units.iter().map(|u| (u.name.as_str(), u)).collect();
    let mut per_unit: Vec<(&Unit, Vec<(String, u32, f64)>)> = Vec::new();
    for ru in r["units"].as_array().into_iter().flatten() {
        let name = ru["name"].as_str().unwrap_or("");
        let Some(u) = units.get(name) else { continue };
        if a.unit.as_deref().is_some_and(|f| !u.name.contains(f)) {
            continue;
        }
        let sourced = u.source.is_some() && !u.cflags.is_empty();
        let ok_scope = match a.scope.as_str() {
            "sourced" => sourced,
            "auto" => !sourced,
            _ => true,
        };
        if !ok_scope {
            continue;
        }
        let mut fs = Vec::new();
        for f in ru["functions"].as_array().into_iter().flatten() {
            let fuzzy = f["fuzzy_match_percent"].as_f64().unwrap_or(0.0);
            if fuzzy >= 100.0 {
                continue;
            }
            let sym = f["name"].as_str().unwrap_or("").to_string();
            if is_compiler_generated(&sym) || a.symbol.as_deref().is_some_and(|s| s != sym) {
                continue;
            }
            let size: u32 = f["size"].as_str().and_then(|s| s.parse().ok()).unwrap_or(0);
            if size < a.min_size.max(1) || size > a.max_size {
                continue;
            }
            fs.push((sym, size, fuzzy));
        }
        if !fs.is_empty() {
            per_unit.push((u, fs));
        }
    }
    // Drop weak symbols (header inlines / template instances) using the target object.
    use rayon::prelude::*;
    let mut out: Vec<Cand> = per_unit
        .par_iter()
        .flat_map_iter(|(u, fs)| {
            let obj = mwdec_obj::load_object(&p.path(&u.target_obj).to_string_lossy()).ok();
            let bind: HashMap<&str, SymBinding> = obj.as_ref().map(|o| o.functions.iter().map(|f| (f.name.as_str(), f.binding)).collect()).unwrap_or_default();
            let sourced = u.source.is_some() && !u.cflags.is_empty();
            fs.iter()
                .filter(|(s, _, _)| bind.get(s.as_str()).is_some_and(|b| *b != SymBinding::Weak))
                .map(|(s, size, fz)| Cand { unit: u.name.clone(), symbol: s.clone(), size: *size, fuzzy: *fz, sourced })
                .collect::<Vec<_>>()
        })
        .collect();
    let bucket = |s: u32| SIZE_BUCKETS.iter().position(|b| *b == size_bucket(s)).unwrap_or(99);
    out.sort_by(|x, y| (bucket(x.size), &x.unit, x.size, &x.symbol).cmp(&(bucket(y.size), &y.unit, y.size, &y.symbol)));
    Ok(out)
}

/// Short file name for a symbol (Windows paths: keep it under ~100 chars, unique by hash).
pub fn file_stem(sym: &str) -> String {
    let s = sanitize(sym);
    if s.len() <= 90 {
        return s;
    }
    let h = mwdec_mwcc::content_hash(&[sym.as_bytes()]) as u32;
    format!("{}_{h:08x}", &s[..80])
}

/// Split a final candidate into (preamble, function definition): the definition is the last
/// top-level `{ ... }` block together with its header, which starts after the previous
/// top-level `;` / `}` (or a preprocessor line).
pub fn split_preamble(src: &str) -> (String, String) {
    let b = src.as_bytes();
    let (mut depth, mut last_end) = (0i32, 0usize);
    let mut boundaries = vec![0usize];
    let mut i = 0;
    let mut in_str: Option<u8> = None;
    while i < b.len() {
        let c = b[i];
        if let Some(q) = in_str {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == q {
                in_str = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' | b'\'' => in_str = Some(c),
            b'/' if b.get(i + 1) == Some(&b'/') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                if depth == 0 {
                    boundaries.push(i.min(b.len()));
                }
                continue;
            }
            b'#' if depth == 0 && (i == 0 || b[i - 1] == b'\n') => {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
                boundaries.push(i.min(b.len()));
                continue;
            }
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    last_end = i + 1;
                    boundaries.push(i + 1);
                }
            }
            b';' if depth == 0 => boundaries.push(i + 1),
            _ => {}
        }
        i += 1;
    }
    // the definition's block ends at last_end; its header starts at the last boundary before
    // the block's opening brace
    let mut d = 0i32;
    let mut open = None;
    for (j, &c) in b[..last_end].iter().enumerate().rev() {
        match c {
            b'}' => d += 1,
            b'{' => {
                d -= 1;
                if d == 0 {
                    open = Some(j);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(open) = open else { return (String::new(), src.to_string()) };
    let start = boundaries.iter().copied().filter(|&x| x <= open).max().unwrap_or(0);
    (src[..start].trim().to_string(), src[start..].trim().to_string())
}

#[derive(Default)]
struct Row {
    unit: String,
    symbol: String,
    size: u32,
    fuzzy: f64,
    sourced: bool,
    status: String,
    error: Option<String>,
    first_exact: bool,
    best_score: Option<f64>,
    compiles: u64,
    seconds: f64,
    source: Option<String>,
    flags: Vec<String>,
    compiler: String,
    context: String,
}

/// Per-unit state: inputs (context, PCH, TypeDb, drivers) built once and shared by the unit's
/// functions; dropped when the unit falls out of the small LRU.
struct UnitSlot {
    ui: Result<UnitInputs, String>,
    context: String,
}

fn unit_context(p: &Project, u: &Unit, idx: &autoctx::HeaderIndex) -> Result<(Unit, String)> {
    if u.source.is_some() && !u.cflags.is_empty() {
        return Ok((u.clone(), harness::context_tu(p, u)?));
    }
    autoctx::auto_unit(p, u, idx)
}

pub fn cmd_harvest(root: &Path, work: &Path, a: HarvestArgs) -> Result<()> {
    std::fs::create_dir_all(&a.out_dir)?;
    if !a.no_supervise && !a.list && std::env::var("MWDEC_HARVEST_CHILD").is_err() {
        return supervise(&a);
    }
    let p = load_project(root)?;
    let report = a.report.clone().unwrap_or_else(|| root.join("build/G2ME01/report.json"));
    let mut cands = candidates(&p, &report, &a)?;
    if a.list {
        for c in &cands {
            println!("{}\t{}\t{}\t{:.1}\t{}", c.size, if c.sourced { "src" } else { "auto" }, c.unit, c.fuzzy, c.symbol);
        }
        eprintln!("{} candidates", cands.len());
        return Ok(());
    }
    // Resume: skip exact results (any tag), attempts under this tag, and repeated crashes.
    let attempts_path = a.out_dir.join("attempts.jsonl");
    let exact_path = a.out_dir.join("exact.jsonl");
    let started_path = a.out_dir.join("started.log");
    let mut done: HashSet<(String, String)> = HashSet::new();
    let mut finished_tag: HashSet<(String, String)> = HashSet::new();
    for line in std::fs::read_to_string(&exact_path).unwrap_or_default().lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            done.insert((v["unit"].as_str().unwrap_or("").into(), v["symbol"].as_str().unwrap_or("").into()));
        }
    }
    let mut best: HashMap<(String, String), f64> = HashMap::new();
    for line in std::fs::read_to_string(&attempts_path).unwrap_or_default().lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(b) = v["best_score"].as_f64() {
                let e = best.entry((v["unit"].as_str().unwrap_or("").into(), v["symbol"].as_str().unwrap_or("").into())).or_insert(b);
                *e = e.max(b);
            }
            if v["tag"].as_str() == Some(a.tag.as_str()) {
                finished_tag.insert((v["unit"].as_str().unwrap_or("").into(), v["symbol"].as_str().unwrap_or("").into()));
            }
        }
    }
    let mut starts: HashMap<(String, String), usize> = HashMap::new();
    for line in std::fs::read_to_string(&started_path).unwrap_or_default().lines() {
        let mut it = line.split('\t');
        if let (Some(t), Some(u), Some(s)) = (it.next(), it.next(), it.next()) {
            if t == a.tag {
                *starts.entry((u.into(), s.into())).or_default() += 1;
            }
        }
    }
    let attempts_file = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&attempts_path)?);
    let exact_file = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&exact_path)?);
    let started_file = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&started_path)?);
    let n0 = cands.len();
    let mut crashed = Vec::new();
    cands.retain(|c| {
        let k = (c.unit.clone(), c.symbol.clone());
        if done.contains(&k) || finished_tag.contains(&k) {
            return false;
        }
        if let Some(mb) = a.min_best {
            if best.get(&k).map_or(true, |b| *b < mb) {
                return false;
            }
        }
        if starts.get(&k).copied().unwrap_or(0) >= 2 {
            crashed.push(c.clone());
            return false;
        }
        true
    });
    // Repeated crashes are recorded once (so the next restart doesn't see them again).
    for c in &crashed {
        let row = Row { unit: c.unit.clone(), symbol: c.symbol.clone(), size: c.size, fuzzy: c.fuzzy, sourced: c.sourced, status: "crashed".into(), ..Default::default() };
        let _ = attempts_file.lock().unwrap().write_all(format!("{}\n", row_json(&row, &a.tag)).as_bytes());
    }
    eprintln!("harvest: {} candidates ({} already done/attempted, {} crashed), budget {}s, tag {}", cands.len(), n0 - cands.len() - crashed.len(), crashed.len(), a.budget_secs, a.tag);
    if let Some(k) = a.limit {
        cands.truncate(k);
    }
    if cands.is_empty() {
        return Ok(());
    }
    let jobs = a.jobs.clamp(1, 6);
    let workers = a.workers.unwrap_or(6usize.div_ceil(jobs)).max(1);
    let idx = autoctx::HeaderIndex::build(&p.root);
    let t0 = Instant::now();

    // per-module extern indexes (main + one REL kept), per-unit inputs (LRU of 3)
    let externs: Mutex<Vec<(String, Arc<(ExternIndex, ExternIndex)>)>> = Mutex::new(Vec::new());
    let ext_lock = Mutex::new(());
    let units: Mutex<Vec<(String, Arc<Mutex<Option<Arc<UnitSlot>>>>)>> = Mutex::new(Vec::new());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let n_exact = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..jobs {
            let _ = std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(s, || loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(c) = cands.get(i) else { break };
                {
                    let mut f = started_file.lock().unwrap();
                    let _ = f.write_all(format!("{}\t{}\t{}\n", a.tag, c.unit, c.symbol).as_bytes());
                    let _ = f.flush();
                }
                let t = Instant::now();
                let module = Project::module_of(&c.unit).to_string();
                let ext = {
                    let _g = ext_lock.lock().unwrap();
                    let have = externs.lock().unwrap().iter().find(|(m, _)| *m == module).map(|(_, x)| x.clone());
                    match have {
                        Some(x) => x,
                        None => {
                            let x = Arc::new(externs_for(&p, &c.unit));
                            let mut g = externs.lock().unwrap();
                            g.push((module.clone(), x.clone()));
                            // keep main + the most recent other module
                            while g.len() > 2 {
                                let at = g.iter().position(|(m, _)| m != "main").unwrap_or(0);
                                g.remove(at);
                            }
                            x
                        }
                    }
                };
                let slot = {
                    let mut g = units.lock().unwrap();
                    let at = g.iter().position(|(n, _)| *n == c.unit);
                    let sl = match at {
                        Some(at) => g.remove(at).1,
                        None => Arc::new(Mutex::new(None)),
                    };
                    g.push((c.unit.clone(), sl.clone()));
                    while g.len() > 3 {
                        g.remove(0);
                    }
                    sl
                };
                let us = {
                    let mut g = slot.lock().unwrap();
                    if g.is_none() {
                        let r = (|| -> Result<(UnitInputs, String)> {
                            let u = p.unit(&c.unit).ok_or_else(|| anyhow::anyhow!("unknown unit"))?;
                            let (u, context) = unit_context(&p, u, &idx)?;
                            // one driver set per unit: its in-memory object cache and its fast-path
                            // compiler processes (2 workers; up to 3 units are kept) die with the unit
                            let cc = Compilers::new(&p.root, work, 6).with_fast_workers(2);
                            let ui = unit_inputs_with_context(&p, &u, &cc, &ext.0.objs, true, context.clone(), Some(&ext.0))?;
                            Ok((ui, context))
                        })();
                        *g = Some(Arc::new(match r {
                            Ok((ui, context)) => UnitSlot { ui: Ok(ui), context },
                            Err(e) => UnitSlot { ui: Err(format!("{e:#}")), context: String::new() },
                        }));
                    }
                    g.clone().unwrap()
                };
                let mut row = Row { unit: c.unit.clone(), symbol: c.symbol.clone(), size: c.size, fuzzy: c.fuzzy, sourced: c.sourced, context: us.context.clone(), ..Default::default() };
                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_one(&us, &ext, c, &a, workers, &mut row)));
                if res.is_err() {
                    row.status = "panic".into();
                }
                row.seconds = t.elapsed().as_secs_f64();
                if row.status == "exact" || row.status == "searched" {
                    n_exact.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if let Err(e) = write_exact(&a, &row, &exact_file) {
                        eprintln!("warning: writing result failed: {e}");
                    }
                }
                {
                    let mut f = attempts_file.lock().unwrap();
                    let _ = f.write_all(format!("{}\n", row_json(&row, &a.tag)).as_bytes());
                    let _ = f.flush();
                }
                eprintln!(
                    "[{:>5}/{}] {:<10} {:<5} {} {} ({:.1}s, {} compiles){}  [{} exact, {:.0}s]",
                    i + 1,
                    cands.len(),
                    row.status,
                    c.size,
                    c.unit,
                    c.symbol,
                    row.seconds,
                    row.compiles,
                    row.best_score.map(|s| format!(" best {s:.1}")).unwrap_or_default(),
                    n_exact.load(std::sync::atomic::Ordering::Relaxed),
                    t0.elapsed().as_secs_f64()
                );
            });
        }
    });
    eprintln!("harvest: {} exact of {} attempted in {:.0}s", n_exact.into_inner(), cands.len(), t0.elapsed().as_secs_f64());
    // Every result is written and flushed. Exit now instead of tearing down the cached unit
    // drivers: their persistent compilers (and any start still in flight on a fast-path worker)
    // could otherwise keep the process alive after the last candidate. Debugged compiler
    // processes end with their debugger, child compilers with the memory-cap job.
    for f in [&attempts_file, &exact_file, &started_file] {
        let _ = f.lock().map(|mut f| f.flush());
    }
    let _ = std::io::stderr().flush();
    let _ = std::io::stdout().flush();
    std::process::exit(0)
}

fn row_json(r: &Row, tag: &str) -> serde_json::Value {
    serde_json::json!({
        "unit": r.unit, "symbol": r.symbol, "size": r.size, "fuzzy_before": r.fuzzy, "sourced": r.sourced,
        "status": r.status, "error": r.error.as_ref().map(|e| e.chars().take(400).collect::<String>()),
        "first_exact": r.first_exact, "best_score": r.best_score, "compiles": r.compiles,
        "seconds": r.seconds, "tag": tag,
    })
}

fn write_exact(a: &HarvestArgs, r: &Row, file: &Mutex<std::fs::File>) -> Result<()> {
    let src = r.source.clone().unwrap_or_default();
    let (preamble, def) = split_preamble(&src);
    let dir = a.out_dir.join("funcs").join(sanitize(&r.unit));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.cpp", file_stem(&r.symbol)));
    std::fs::write(&path, format!("{}{}{}", r.context, src, if src.ends_with('\n') { "" } else { "\n" }))?;
    let v = serde_json::json!({
        "unit": r.unit, "symbol": r.symbol, "size": r.size, "fuzzy_before": r.fuzzy, "sourced": r.sourced,
        "status": r.status, "verdict": "EXACT", "first_exact": r.first_exact, "compiles": r.compiles,
        "seconds": r.seconds, "tag": a.tag, "source": src, "preamble": preamble, "definition": def,
        "flags": r.flags, "compiler": r.compiler, "context": r.context, "file": path.to_string_lossy(),
    });
    let mut f = file.lock().unwrap();
    f.write_all(format!("{v}\n").as_bytes())?;
    f.flush()?;
    Ok(())
}

fn run_one(us: &UnitSlot, ext: &(ExternIndex, ExternIndex), c: &Cand, a: &HarvestArgs, workers: usize, row: &mut Row) {
    let ui = match &us.ui {
        Ok(u) => u,
        Err(err) => {
            row.status = "unit-err".into();
            row.error = Some(err.clone());
            return;
        }
    };
    row.flags = ui.ctx.cflags.clone();
    row.compiler = ui.mwcc.compiler.clone();
    let Some(f) = mwdec_obj::find_function(&ui.target, &c.symbol) else {
        row.status = "missing".into();
        return;
    };
    let src = match draft_with(ui, f, false) {
        Ok(s) => s,
        Err(NoDraft::HeaderInline) => {
            row.status = "hdr-inline".into();
            return;
        }
        Err(NoDraft::Implicit(k)) => {
            row.status = "implicit".into();
            row.error = Some(k);
            return;
        }
        Err(NoDraft::Asm(k)) => {
            row.status = "asm".into();
            row.error = Some(k);
            return;
        }
        Err(NoDraft::Lift(err)) => {
            row.status = "lift-err".into();
            row.error = Some(err);
            return;
        }
    };
    let ti = ObjIndex::with_externs(&ui.target, &ext.0);
    let scorer = Scorer::new(&ui.mwcc, &ui.ctx, Some(&ui.plain), &ti, f, Some(&ext.1), &c.symbol);
    let src = super::search_cmds::repair_registers(&scorer, choose_draft(ui, f, &scorer, src), ui.tracer.as_deref());
    let cfg = SearchConfig {
        budget: Duration::from_secs(a.budget_secs),
        max_compiles: a.max_compiles,
        workers,
        seed: a.seed ^ mwdec_mwcc::content_hash(&[c.symbol.as_bytes()]) as u64,
        out_dir: None,
        tracer: ui.tracer.clone(),
        locate: true,
        ..Default::default()
    };
    let t0 = Instant::now();
    let mut r = search(&scorer, &src, &cfg);
    if r.initial.is_none() {
        // the draft doesn't compile: retry from the raw-offsets draft with the remaining budget
        if let Ok(raw) = super::search_cmds::draft_raw(ui, f) {
            if raw != src {
                let left = Duration::from_secs(a.budget_secs).saturating_sub(t0.elapsed()).max(Duration::from_secs(5));
                let r2 = search(&scorer, &raw, &SearchConfig { budget: left, ..cfg.clone() });
                if r2.initial.is_some() {
                    row.error = Some(format!("raw-offsets fallback (draft: {})", r.initial_error.clone().unwrap_or_default()));
                    let compiles = r.compiles;
                    r = r2;
                    r.compiles += compiles;
                }
            }
        }
    }
    row.first_exact = r.initial.as_ref().is_some_and(|f| f.exact);
    row.best_score = r.best.as_ref().map(|f| f.score);
    row.compiles = r.compiles;
    if row.error.is_none() {
        row.error = r.initial_error.clone();
    }
    row.status = if r.exact {
        row.source = Some(super::search_cmds::polish_exact(ui, &scorer, &r.best_src));
        if row.first_exact { "exact".into() } else { "searched".into() }
    } else if r.initial.is_some() {
        // keep the best candidate of a miss for later analysis / longer searches
        let dir = a.out_dir.join("miss").join(sanitize(&c.unit));
        if std::fs::create_dir_all(&dir).is_ok() {
            let _ = std::fs::write(dir.join(format!("{}.cpp", file_stem(&c.symbol))), &r.best_src);
        }
        "mismatch".into()
    } else {
        "cc-err".into()
    };
}

/// Parent mode: run the harvest in a child process and restart it after an abnormal exit
/// (the resume logic skips finished work and functions that crashed twice).
fn supervise(a: &HarvestArgs) -> Result<()> {
    let exe = std::env::current_exe()?;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut restarts = 0;
    loop {
        let st = std::process::Command::new(&exe).args(&args).env("MWDEC_HARVEST_CHILD", "1").status()?;
        if st.success() {
            return Ok(());
        }
        restarts += 1;
        eprintln!("harvest: child exited with {st}; restart {restarts}");
        if restarts > 200 {
            bail!("harvest: too many restarts");
        }
        std::thread::sleep(Duration::from_secs(2));
        let _ = &a.out_dir;
    }
}

/// Summary of the results directory: exact results by size bucket and unit kind.
pub fn summary(out_dir: &Path) -> Result<()> {
    let mut by: BTreeMap<(&str, bool), (usize, u64)> = BTreeMap::new();
    let text = std::fs::read_to_string(out_dir.join("exact.jsonl")).unwrap_or_default();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let size = v["size"].as_u64().unwrap_or(0);
        let e = by.entry((size_bucket(size as u32), v["sourced"].as_bool().unwrap_or(false))).or_default();
        e.0 += 1;
        e.1 += size;
    }
    let mut st: BTreeMap<String, usize> = BTreeMap::new();
    for line in std::fs::read_to_string(out_dir.join("attempts.jsonl")).unwrap_or_default().lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        *st.entry(v["status"].as_str().unwrap_or("?").to_string()).or_default() += 1;
    }
    println!("{:<8} {:>14} {:>14}", "size", "sourced n/B", "auto n/B");
    for b in SIZE_BUCKETS {
        let s = by.get(&(b, true)).copied().unwrap_or_default();
        let x = by.get(&(b, false)).copied().unwrap_or_default();
        if s.0 + x.0 > 0 {
            println!("{b:<8} {:>14} {:>14}", format!("{}/{}", s.0, s.1), format!("{}/{}", x.0, x.1));
        }
    }
    println!("attempt statuses: {st:?}");
    Ok(())
}

/// Strictly compare every function of each unit's built (base) object against its target
/// object; one JSON line per target function: {unit, symbol, size, exact, class, present}.
/// Used by the integration harness to prove harvested functions in the real build (and catch
/// regressions of the unit's other functions).
pub fn cmd_verify_units(root: &Path, units: &[String]) -> Result<()> {
    let p = load_project(root)?;
    let mut by_mod: BTreeMap<&str, Vec<&Unit>> = BTreeMap::new();
    for name in units {
        let Some(u) = p.unit(name) else {
            println!("{}", serde_json::json!({"unit": name, "error": "unknown unit"}));
            continue;
        };
        by_mod.entry(Project::module_of(&u.name)).or_default().push(u);
    }
    for (m, us) in by_mod {
        let ext = super::module_externs(&p, std::iter::once(m));
        let (text, oext) = &ext[m];
        for u in us {
            let target = match super::load_obj(&p, &u.target_obj) {
                Ok(t) => t,
                Err(e) => {
                    println!("{}", serde_json::json!({"unit": u.name, "error": format!("target: {e}")}));
                    continue;
                }
            };
            let ours = u.base_obj.as_deref().map(|b| super::load_obj(&p, b));
            let ours = match ours {
                Some(Ok(o)) => o,
                Some(Err(e)) => {
                    println!("{}", serde_json::json!({"unit": u.name, "error": format!("base: {e}")}));
                    continue;
                }
                None => {
                    println!("{}", serde_json::json!({"unit": u.name, "error": "no base object"}));
                    continue;
                }
            };
            let ti = ObjIndex::with_externs(&target, text);
            let oi = ObjIndex::with_externs(&ours, oext);
            for tf in &target.functions {
                let v = match oi.function(&tf.name) {
                    Some(of) => {
                        let d = mwdec_mwcc::compare_indexed(&ti, tf, &oi, of);
                        serde_json::json!({"unit": u.name, "symbol": tf.name, "size": tf.code.len(), "present": true,
                            "exact": d.result.exact, "class": d.class.label(), "weak": tf.binding == SymBinding::Weak})
                    }
                    None => serde_json::json!({"unit": u.name, "symbol": tf.name, "size": tf.code.len(), "present": false,
                        "exact": false, "class": "missing", "weak": tf.binding == SymBinding::Weak}),
                };
                println!("{v}");
            }
        }
    }
    Ok(())
}

/// Re-polish the exact results of a harvest directory (`exact.jsonl`) towards natural source:
/// each result is recompiled in its unit, polished with `mwdec_emit::tidy`, and written with
/// its naturalness before/after to `<out>` (JSONL, same fields as `exact.jsonl`).
pub fn cmd_repolish(root: &Path, work: &Path, dir: &Path, out: &Path, split: Option<&str>, limit: Option<usize>, unit_filter: Option<&str>) -> Result<()> {
    let p = load_project(root)?;
    let idx = autoctx::HeaderIndex::build(&p.root);
    let text = std::fs::read_to_string(dir.join("exact.jsonl")).with_context(|| format!("reading {}", dir.join("exact.jsonl").display()))?;
    let mut by_unit: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
    let mut n = 0;
    for l in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(l) else { continue };
        let Some(unit) = v.get("unit").and_then(|x| x.as_str()).map(String::from) else { continue };
        if split.map_or(false, |s| mwdec_project::split_of(&unit) != s) || unit_filter.map_or(false, |f| !unit.contains(f)) {
            continue;
        }
        if limit.map_or(false, |k| n >= k) {
            break;
        }
        n += 1;
        by_unit.entry(unit).or_default().push(v);
    }
    let mut outf = std::fs::File::create(out)?;
    let (mut before_sum, mut after_sum) = (mwdec_emit::tidy::Naturalness::default(), mwdec_emit::tidy::Naturalness::default());
    let (mut done, mut changed, mut lost, mut clean_before, mut clean_after) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (unit, rows) in by_unit {
        let r = (|| -> Result<UnitInputs> {
            let u = p.unit(&unit).ok_or_else(|| anyhow::anyhow!("unknown unit"))?;
            let (u, context) = unit_context(&p, u, &idx)?;
            let ext = externs_for(&p, &u.name);
            let cc = Compilers::new(&p.root, work, 3);
            unit_inputs_with_context(&p, &u, &cc, &ext.0.objs, true, context, Some(&ext.0))
        })();
        let ui = match r {
            Ok(ui) => ui,
            Err(e) => {
                eprintln!("{unit}: {e:#}");
                continue;
            }
        };
        let ext = externs_for(&p, &unit);
        for mut v in rows {
            let sym = v.get("symbol").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let src = v.get("source").and_then(|x| x.as_str()).unwrap_or("").to_string();
            let Some(f) = mwdec_obj::find_function(&ui.target, &sym) else { continue };
            let ti = ObjIndex::with_externs(&ui.target, &ext.0);
            let scorer = Scorer::new(&ui.mwcc, &ui.ctx, Some(&ui.plain), &ti, f, Some(&ext.1), &sym);
            // the stored result must still be exact here (same context and compiler)
            if !scorer.eval(&src).0.fitness().map_or(false, |x| x.exact) {
                lost += 1;
                eprintln!("{unit} {sym}: stored source not exact in this context; kept as is");
                continue;
            }
            let pol = super::search_cmds::polish_exact(&ui, &scorer, &src);
            let (b, a) = (mwdec_emit::tidy::measure(&src), mwdec_emit::tidy::measure(&pol));
            before_sum.add(&b);
            after_sum.add(&a);
            done += 1;
            changed += (pol != src) as usize;
            clean_before += (b.penalty() == 0) as usize;
            clean_after += (a.penalty() == 0) as usize;
            let (preamble, def) = split_preamble(&pol);
            v["source"] = serde_json::Value::String(pol.clone());
            v["preamble"] = serde_json::Value::String(preamble);
            v["definition"] = serde_json::Value::String(def);
            v["naturalness_before"] = serde_json::Value::String(b.summary());
            v["naturalness_after"] = serde_json::Value::String(a.summary());
            writeln!(outf, "{v}")?;
            println!("{unit} {sym} | {} -> {}", b.summary(), a.summary());
        }
    }
    let k = done.max(1) as f64;
    println!(
        "REPOLISH {done} results ({changed} changed, {lost} not reproducible): penalty per function {:.2} -> {:.2}; clean {clean_before} -> {clean_after}",
        before_sum.penalty() as f64 / k,
        after_sum.penalty() as f64 / k
    );
    println!("  before: {}", before_sum.summary());
    println!("  after:  {}", after_sum.summary());
    Ok(())
}
