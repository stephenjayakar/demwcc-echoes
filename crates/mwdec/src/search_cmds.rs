//! `mwdec match` and `mwdec eval`: first draft (lift + emit) followed by the compiler-in-the-loop
//! permuter (`mwdec-search`).
//!
//! Anti-cheat: the inputs given to the decompiler are the unit's target object, the module's
//! other target objects (literal values), the include-only context TU (`harness::context_tu`)
//! and the compiler. No source file body is ever opened here.
use super::{find_unit, load_obj, load_project, module_externs};
use anyhow::{anyhow, bail, Context, Result};
use mwdec_core::*;
use mwdec_mwcc::{ExternIndex, Mwcc, ObjIndex, UnitContext};
use mwdec_project::{harness, size_bucket, Project, SIZE_BUCKETS};
use mwdec_search::{search, Scorer, SearchConfig, SearchResult};
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub fn eval_dir() -> PathBuf {
    mwdec_core::paths::work_dir("eval")
}
/// Scratch (PCH, temporary TUs) for match/eval compiles.
pub fn search_work() -> PathBuf {
    mwdec_core::paths::work_dir("mwdec-search/work")
}
pub fn search_dir() -> PathBuf {
    mwdec_core::paths::work_dir("search")
}

fn split_ops(s: &Option<String>) -> Vec<String> {
    s.as_deref().map(|x| x.split(',').map(|o| o.trim().to_string()).filter(|o| !o.is_empty()).collect()).unwrap_or_default()
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

/// Compiler drivers per compiler version (Dolphin SDK / runtime units use GC/1.2.5n etc.).
/// Compiles in flight stay bounded by the number of search workers (each runs one at a time).
pub struct Compilers {
    root: PathBuf,
    work: PathBuf,
    jobs: usize,
    map: Mutex<HashMap<String, Arc<Mwcc>>>,
}

impl Compilers {
    pub fn new(root: &Path, work: &Path, jobs: usize) -> Compilers {
        Compilers { root: root.to_path_buf(), work: work.to_path_buf(), jobs, map: Mutex::new(HashMap::new()) }
    }

    /// Driver for mwdec-inline's probe TUs: the unit's compiler, with an on-disk cache (the
    /// same probe TU is compiled again for every function of the unit, across runs).
    pub fn probe_driver(&self, p: &Project, unit: &str) -> Arc<Mwcc> {
        let compiler = p.compiler_rel(unit);
        let key = format!("probes:{compiler}");
        self.map
            .lock()
            .unwrap()
            .entry(key)
            .or_insert_with(|| {
                let mut m = Mwcc::new(&self.root, &self.work.join("inline-probes").join(compiler.replace(['/', '.'], "_")), self.jobs);
                m.compiler = compiler.clone();
                Arc::new(m)
            })
            .clone()
    }

    /// Driver for the unit's own compiler (`Project::compiler_rel`).
    pub fn for_unit(&self, p: &Project, unit: &str) -> Arc<Mwcc> {
        let compiler = p.compiler_rel(unit);
        self.map
            .lock()
            .unwrap()
            .entry(compiler.clone())
            .or_insert_with(|| {
                let default = Mwcc::new(&self.root, &self.work, self.jobs);
                let mut m = if default.compiler == compiler {
                    default
                } else {
                    let mut m = Mwcc::new(&self.root, &self.work.join(compiler.replace(['/', '.'], "_")), self.jobs);
                    m.compiler = compiler.clone();
                    m
                };
                // Search candidates are rarely re-hit across runs: memory cache only (keeps the
                // disk small and time budgets honest). PCHs are still cached on disk.
                m.disk_cache = None;
                Arc::new(m)
            })
            .clone()
    }
}

/// Per-unit inputs shared by every function of the unit.
pub struct UnitInputs {
    pub target: ObjectFile,
    #[allow(dead_code)]
    pub context: String,
    /// Driver for the unit's compiler version.
    pub mwcc: Arc<Mwcc>,
    pub ctx: UnitContext,
    pub plain: UnitContext,
    pub db: Option<TypeDb>,
    /// `-lang=c` unit: emit C.
    pub c_mode: bool,
    /// Compiler tracer for register-only diffs (GC/2.7 units).
    pub tracer: Option<Arc<mwdec_search::trace::Tracer>>,
    /// Header inline templates of the context (mwdec-inline), folded back into calls.
    pub inlines: InlineLibs,
}

/// Why no draft exists.
#[derive(Debug, Clone)]
pub enum NoDraft {
    /// Defined inline in a context header: a standalone definition cannot exist.
    HeaderInline,
    /// Compiler-generated special member or header-defined template member: no standalone
    /// definition exists (`mwdec_project::standalone`).
    Implicit(String),
    Lift(String),
}

/// Inputs for one unit. `module_objs`: the module's target objects (main + the unit's REL), used
/// to recover vtables of every class the context knows.
pub fn unit_inputs(p: &Project, u: &Unit, cc: &Compilers, module_objs: &[Arc<ObjectFile>], with_db: bool) -> Result<UnitInputs> {
    if u.cflags.is_empty() {
        bail!("unit {} has no compiler flags", u.name);
    }
    let m = cc.for_unit(p, &u.name);
    let target = load_obj(p, &u.target_obj)?;
    let context = harness::context_tu(p, u)?;
    let plain = m.plain_context(&context, &u.cflags);
    let ctx = if context.is_empty() { plain.clone() } else { m.precompile(&context, &u.cflags).unwrap_or_else(|_| plain.clone()) };
    let c_mode = u.cflags.iter().any(|f| f == "-lang=c" || f == "-lang=c99");
    let db = if with_db {
        match mwdec_ctx::build_typedb(&context, &u.cflags, &mwdec_ctx::default_work_dir()) {
            Ok(mut db) => {
                // vtables of every class the context knows, from all objects of the module
                for o in module_objs {
                    let relevant = o.data.keys().any(|n| mwdec_ctx::vtable::vtable_class(n).is_some_and(|c| db.classes.contains_key(&c)));
                    if relevant {
                        let vt = mwdec_ctx::vtables_from_object(o, &db);
                        let vt: std::collections::BTreeMap<_, _> = vt.into_iter().filter(|(c, _)| db.classes.contains_key(c)).collect();
                        mwdec_ctx::apply_vtables(&mut db, &vt);
                    }
                }
                let vt = mwdec_ctx::vtables_from_object(&target, &db);
                mwdec_ctx::apply_vtables(&mut db, &vt);
                Some(db)
            }
            Err(e) => {
                eprintln!("warning: typedb for {} failed: {e}", u.name);
                None
            }
        }
    } else {
        None
    };
    let tracer = mwdec_search::trace::Tracer::new(&cc.root, &cc.work.join("trace"), &p.compiler_rel(&u.name), &u.cflags, &context).map(Arc::new);
    let inlines = InlineLibs {
        enabled: db.is_some() && !c_mode && std::env::var("MWDEC_NO_INLINE").is_err(),
        driver: cc.probe_driver(p, &u.name),
        lib: std::sync::OnceLock::new(),
    };
    Ok(UnitInputs { target, context, mwcc: m, ctx, plain, db, c_mode, tracer, inlines })
}

/// First draft from the lifter + emitter (implicit functions included: an explicit request).
pub fn draft(ui: &UnitInputs, f: &Function) -> std::result::Result<String, NoDraft> {
    draft_variant(ui, f, true, true)
}

/// First draft; `include_implicit` false refuses functions that can't exist as standalone
/// source (reported as their own eval column).
pub fn draft_with(ui: &UnitInputs, f: &Function, include_implicit: bool) -> std::result::Result<String, NoDraft> {
    draft_variant(ui, f, true, include_implicit)
}

/// Header inline templates of the unit context (mwdec-inline), built on first use; templates
/// are cached on disk across units and runs, so only unseen inlines are compiled.
pub struct InlineLibs {
    pub enabled: bool,
    driver: Arc<Mwcc>,
    lib: std::sync::OnceLock<Arc<mwdec_inline::InlineLib>>,
}

impl InlineLibs {
    pub fn get(&self, ui: &UnitInputs, u_flags: &str) -> Arc<mwdec_inline::InlineLib> {
        self.lib
            .get_or_init(|| {
                let Some(db) = (if self.enabled { ui.db.as_ref() } else { None }) else {
                    return Arc::new(mwdec_inline::InlineLib::default());
                };
                let cache = mwdec_inline::ProbeCache::new(u_flags, &ui.context);
                Arc::new(mwdec_inline::build_library_for(db, Some(&ui.target), Some(&cache), &|code| {
                    let mut r = self.driver.compile_in(&ui.ctx, code);
                    if let Err(mwdec_mwcc::MwccError::Compile { status, messages }) = &r {
                        if status.is_some_and(|s| s < 0) || messages.contains("Unhandled exception") {
                            r = self.driver.compile_in(&ui.plain, code);
                        }
                    }
                    match r {
                        Err(e) => Err(e.messages().to_string()),
                        Ok(c) => mwdec_obj::load_object_bytes(&c.obj_path.to_string_lossy(), &c.obj).map_err(|e| e.to_string()),
                    }
                }))
            })
            .clone()
    }
}

/// The compiler picks the better of the drafts with and without folded header inlines (they
/// differ for a minority of functions; one extra compile there).
pub fn choose_draft(ui: &UnitInputs, f: &Function, scorer: &Scorer, with_inlines: String) -> String {
    if !ui.inlines.enabled {
        return with_inlines;
    }
    let Ok(plain) = draft_variant(ui, f, false, true) else { return with_inlines };
    if plain == with_inlines {
        return with_inlines;
    }
    let (a, _) = scorer.eval(&with_inlines);
    let (b, _) = scorer.eval(&plain);
    match (a.fitness(), b.fitness()) {
        (Some(x), Some(y)) if y.better_than(x) => plain,
        (None, Some(_)) => plain,
        _ => with_inlines,
    }
}

fn draft_variant(ui: &UnitInputs, f: &Function, inlines: bool, include_implicit: bool) -> std::result::Result<String, NoDraft> {
    if let Some(db) = &ui.db {
        let sig = mwdec_lift::sig::sig_of(&f.name, Some(db));
        match mwdec_project::standalone::standalone(&sig, db) {
            mwdec_project::standalone::Standalone::HeaderInline => return Err(NoDraft::HeaderInline),
            mwdec_project::standalone::Standalone::Implicit(k) if !include_implicit => return Err(NoDraft::Implicit(k.to_string())),
            _ => {}
        }
    }
    let ir = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ir = mwdec_lift::lift_function(&ui.target, f, ui.db.as_ref())?;
        if let (Some(db), true) = (&ui.db, inlines && ui.inlines.enabled) {
            let lib = ui.inlines.get(ui, &format!("{}
{}", ui.mwcc.compiler, ui.ctx.cflags.join(" ")));
            mwdec_inline::apply(&mut ir, &lib, db);
        }
        Ok::<_, anyhow::Error>(ir)
    })) {
        Ok(Ok(ir)) => ir,
        Ok(Err(e)) => return Err(NoDraft::Lift(e.to_string())),
        Err(_) => return Err(NoDraft::Lift("lifter panic".into())),
    };
    let em = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mwdec_emit::emit_function(&ir, ui.db.as_ref(), &mwdec_emit::EmitOptions { c_mode: ui.c_mode, ..Default::default() }))) {
        Ok(em) => em,
        Err(_) => return Err(NoDraft::Lift("emitter panic".into())),
    };
    Ok(format!("{}{}", em.preamble, em.body))
}

fn externs_for(p: &Project, unit: &str) -> (ExternIndex, ExternIndex) {
    let mut m = module_externs(p, std::iter::once(unit));
    m.remove(Project::module_of(unit)).unwrap_or_else(|| (ExternIndex::new(vec![]), ExternIndex::new(vec![])))
}

pub struct MatchArgs {
    pub unit: String,
    pub symbol: String,
    pub budget_secs: u64,
    pub init: Option<PathBuf>,
    pub max_compiles: Option<usize>,
    pub workers: usize,
    pub seed: u64,
    pub no_db: bool,
    pub verbose: bool,
    pub out: Option<PathBuf>,
    pub no_locate: bool,
    pub disable_ops: Option<String>,
}

pub fn cmd_match(root: &Path, work: &Path, a: MatchArgs) -> Result<()> {
    let p = load_project(root)?;
    let u = find_unit(&p, &a.unit)?;
    let cc = Compilers::new(root, work, a.workers.clamp(1, 6));
    let t = Instant::now();
    let (t_ext, o_ext) = externs_for(&p, &u.name);
    let ui = unit_inputs(&p, u, &cc, &t_ext.objs, !a.no_db && a.init.is_none())?;
    let f = mwdec_obj::find_function(&ui.target, &a.symbol).ok_or_else(|| anyhow!("{} not in {}", a.symbol, u.target_obj))?;
    let init = match &a.init {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
        None => match draft(&ui, f) {
            Ok(s) => s,
            Err(NoDraft::HeaderInline) => bail!("{} is defined inline in a context header; nothing to decompile", a.symbol),
            Err(NoDraft::Implicit(k)) => bail!("{} can't exist as standalone source ({k})", a.symbol),
            Err(NoDraft::Lift(e)) => bail!("no first draft ({e}); pass --init <file.cpp>"),
        },
    };
    let ti = ObjIndex::with_externs(&ui.target, &t_ext);
    let scorer = Scorer::new(&ui.mwcc, &ui.ctx, Some(&ui.plain), &ti, f, Some(&o_ext), &a.symbol);
    let init = if a.init.is_none() { choose_draft(&ui, f, &scorer, init) } else { init };
    let out = a.out.clone().unwrap_or_else(|| search_dir().join(sanitize(&u.name)).join(sanitize(&a.symbol)));
    let cfg = SearchConfig {
        budget: Duration::from_secs(a.budget_secs),
        max_compiles: a.max_compiles,
        workers: a.workers.max(1),
        seed: a.seed,
        out_dir: Some(out.clone()),
        verbose: a.verbose,
        tracer: ui.tracer.clone(),
        locate: !a.no_locate,
        disabled_ops: split_ops(&a.disable_ops),
        ..Default::default()
    };
    eprintln!("setup {:.1}s; searching {} (budget {}s) ...", t.elapsed().as_secs_f64(), a.symbol, a.budget_secs);
    let r = search(&scorer, &init, &cfg);
    print!("{}", r.best_src);
    if !r.best_src.ends_with('\n') {
        println!();
    }
    print_result(&r);
    eprintln!("best source: {}", out.join("best.cpp").display());
    if !r.exact {
        std::process::exit(1);
    }
    Ok(())
}

fn print_result(r: &SearchResult) {
    let fmt = |f: &Option<mwdec_search::Fitness>| match f {
        Some(f) if f.exact => "EXACT".to_string(),
        Some(f) => format!("score {:.1} penalty {} ({})", f.score, f.penalty, f.class.label()),
        None => "-".into(),
    };
    eprintln!(
        "// {}: initial {} -> best {}; {} evals ({} compiles, {} compile errors, {} dups) in {:.1}s = {:.1}/s",
        if r.exact { "EXACT" } else { "MISMATCH" },
        r.initial_error.clone().unwrap_or_else(|| fmt(&r.initial)),
        fmt(&r.best),
        r.evals,
        r.compiles,
        r.compile_errors,
        r.duplicates,
        r.seconds,
        r.evals as f64 / r.seconds.max(1e-9)
    );
    let mut useful: Vec<&mwdec_search::search::OpStat> = r.op_stats.iter().filter(|s| s.improved > 0).collect();
    useful.sort_by(|a, b| b.new_best.cmp(&a.new_best).then(b.improved.cmp(&a.improved)));
    if !useful.is_empty() {
        eprintln!(
            "// useful operators: {}",
            useful.iter().map(|s| format!("{} {}/{}/{}", s.name, s.new_best, s.improved, s.tries)).collect::<Vec<_>>().join(", ")
        );
    }
}

pub struct EvalArgs {
    pub split: String,
    pub max_size: Option<u32>,
    pub min_size: u32,
    pub limit: Option<usize>,
    pub seed: u64,
    pub budget_secs: u64,
    pub max_compiles: Option<usize>,
    pub jobs: usize,
    pub workers: Option<usize>,
    pub no_db: bool,
    pub unit: Option<String>,
    pub out: Option<PathBuf>,
    /// Also evaluate functions that can't exist as standalone source (default: own column).
    pub include_implicit: bool,
    pub no_locate: bool,
    pub disable_ops: Option<String>,
}

#[derive(Default, Clone)]
struct Row {
    unit: String,
    symbol: String,
    size: u32,
    status: String,
    drafted: bool,
    compiled: bool,
    first_exact: bool,
    final_exact: bool,
    first_score: Option<f64>,
    best_score: Option<f64>,
    first_penalty: Option<u64>,
    best_penalty: Option<u64>,
    evals: u64,
    compiles: u64,
    seconds: f64,
    error: Option<String>,
    winning_ops: Vec<String>,
    /// op -> [tries, improved, new_best] (ops that were tried).
    ops: serde_json::Map<String, serde_json::Value>,
    polished: usize,
    /// kind of a function that can't exist as standalone source (`mwdec_project::standalone`)
    implicit: Option<String>,
    first_profile: Option<mwdec_search::score::DiffProfile>,
    best_profile: Option<mwdec_search::score::DiffProfile>,
    traces: u64,
}

impl Row {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "unit": self.unit, "symbol": self.symbol, "size": self.size, "status": self.status,
            "drafted": self.drafted, "first_draft_compiled": self.compiled, "first_draft_exact": self.first_exact,
            "final_exact": self.final_exact, "first_score": self.first_score, "best_score": self.best_score,
            "first_penalty": self.first_penalty, "best_penalty": self.best_penalty,
            "evals": self.evals, "compiles": self.compiles, "seconds": self.seconds, "error": self.error,
            "winning_ops": self.winning_ops, "ops": self.ops, "polished": self.polished, "implicit": self.implicit,
            "first_profile": self.first_profile, "best_profile": self.best_profile, "traces": self.traces,
        })
    }
}

/// Deterministic shuffle (SplitMix64-driven Fisher-Yates).
fn shuffle<T>(v: &mut [T], seed: u64) {
    let mut r = mwdec_search::rng::Rng::new(seed);
    for i in (1..v.len()).rev() {
        let j = r.below(i + 1);
        v.swap(i, j);
    }
}

pub fn cmd_eval(root: &Path, work: &Path, a: EvalArgs) -> Result<()> {
    let t0 = Instant::now();
    if a.split != "test" && a.split != "train" {
        bail!("--split must be test or train");
    }
    let p = load_project(root)?;
    let mut ds: Vec<DatasetEntry> = p
        .dataset()?
        .into_iter()
        .filter(|e| e.split == a.split && a.max_size.map_or(true, |m| e.size <= m) && e.size >= a.min_size)
        .filter(|e| a.unit.as_deref().map_or(true, |u| e.unit.contains(u)))
        .collect();
    shuffle(&mut ds, a.seed);
    if let Some(k) = a.limit {
        ds.truncate(k);
    }
    if ds.is_empty() {
        bail!("no dataset functions match");
    }
    let jobs = a.jobs.clamp(1, 16);
    let workers = a.workers.unwrap_or(6usize.div_ceil(jobs)).max(1);
    let cc = Compilers::new(root, work, 6);
    let out_path = a.out.clone().unwrap_or_else(|| {
        eval_dir().join(format!("eval_{}_s{}_b{}_{}.jsonl", a.split, a.seed, a.budget_secs, std::process::id()))
    });
    std::fs::create_dir_all(out_path.parent().unwrap())?;
    // Per-eval run directories (init/best sources), next to the jsonl: concurrent evals by
    // other worktrees do not overwrite each other.
    let runs_dir = out_path.with_extension("runs");
    let out_file = Mutex::new(std::fs::File::create(&out_path)?);
    eprintln!(
        "eval: {} functions from the {} split, budget {}s each, {jobs} in parallel x {workers} workers -> {}",
        ds.len(),
        a.split,
        a.budget_secs,
        out_path.display()
    );
    // Shared per-unit inputs and per-module extern indexes (built lazily, once).
    let units: Mutex<HashMap<String, Arc<Mutex<Option<Arc<Result<UnitInputs, String>>>>>>> = Mutex::new(HashMap::new());
    let externs: Mutex<HashMap<String, Arc<(ExternIndex, ExternIndex)>>> = Mutex::new(HashMap::new());
    let ext_lock = Mutex::new(());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs {
            let _ = std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(s, || loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(e) = ds.get(i) else { break };
                let t = Instant::now();
                let mut row = Row { unit: e.unit.clone(), symbol: e.symbol.clone(), size: e.size, ..Default::default() };
                let module = Project::module_of(&e.unit).to_string();
                let ext = {
                    let _g = ext_lock.lock().unwrap();
                    let have = externs.lock().unwrap().get(&module).cloned();
                    match have {
                        Some(x) => x,
                        None => {
                            let x = Arc::new(externs_for(&p, &e.unit));
                            externs.lock().unwrap().insert(module.clone(), x.clone());
                            x
                        }
                    }
                };
                let slot = units.lock().unwrap().entry(e.unit.clone()).or_default().clone();
                let ui = {
                    let mut g = slot.lock().unwrap();
                    if g.is_none() {
                        let r = p
                            .unit(&e.unit)
                            .ok_or_else(|| anyhow!("unknown unit"))
                            .and_then(|u| unit_inputs(&p, u, &cc, &ext.0.objs, !a.no_db));
                        *g = Some(Arc::new(r.map_err(|e| e.to_string())));
                    }
                    g.clone().unwrap()
                };
                run_one(&ui, &ext, e, &a, workers, &runs_dir, &mut row);
                row.seconds = t.elapsed().as_secs_f64();
                let line = row.json().to_string();
                {
                    let mut f = out_file.lock().unwrap();
                    let _ = writeln!(f, "{line}");
                    let _ = f.flush();
                }
                eprintln!(
                    "[{:>4}/{}] {:<9} {:<5} {} {} ({:.1}s, {} compiles){}",
                    i + 1,
                    ds.len(),
                    row.status,
                    e.size,
                    e.unit,
                    e.symbol,
                    row.seconds,
                    row.compiles,
                    row.best_score.map(|s| format!(" best {s:.1}")).unwrap_or_default()
                );
                rows.lock().unwrap().push(row);
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    print_table(&rows);
    println!("wrote {} ({:.0}s total)", out_path.display(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn run_one(ui: &Result<UnitInputs, String>, ext: &(ExternIndex, ExternIndex), e: &DatasetEntry, a: &EvalArgs, workers: usize, runs_dir: &Path, row: &mut Row) {
    let ui = match ui {
        Ok(u) => u,
        Err(err) => {
            row.status = "unit-err".into();
            row.error = Some(err.clone());
            return;
        }
    };
    let Some(f) = mwdec_obj::find_function(&ui.target, &e.symbol) else {
        row.status = "missing".into();
        return;
    };
    if let Some(db) = &ui.db {
        if let mwdec_project::standalone::Standalone::Implicit(k) = mwdec_project::standalone::standalone(&mwdec_lift::sig::sig_of(&f.name, Some(db)), db) {
            row.implicit = Some(k.to_string());
        }
    }
    let src = match draft_with(ui, f, a.include_implicit) {
        Ok(s) => s,
        Err(NoDraft::HeaderInline) => {
            row.status = "hdr-inline".into();
            return;
        }
        Err(NoDraft::Implicit(_)) => {
            row.status = "implicit".into();
            return;
        }
        Err(NoDraft::Lift(err)) => {
            row.status = "lift-err".into();
            row.error = Some(err);
            return;
        }
    };
    row.drafted = true;
    let ti = ObjIndex::with_externs(&ui.target, &ext.0);
    let scorer = Scorer::new(&ui.mwcc, &ui.ctx, Some(&ui.plain), &ti, f, Some(&ext.1), &e.symbol);
    let src = choose_draft(ui, f, &scorer, src);
    let cfg = SearchConfig {
        budget: Duration::from_secs(a.budget_secs),
        max_compiles: a.max_compiles,
        workers,
        seed: a.seed ^ mwdec_mwcc::content_hash(&[e.symbol.as_bytes()]) as u64,
        out_dir: Some(runs_dir.join(sanitize(&e.unit)).join(sanitize(&e.symbol))),
        tracer: ui.tracer.clone(),
        locate: !a.no_locate,
        disabled_ops: split_ops(&a.disable_ops),
        ..Default::default()
    };
    let r = search(&scorer, &src, &cfg);
    row.compiled = r.initial.is_some();
    row.first_exact = r.initial.as_ref().is_some_and(|f| f.exact);
    row.final_exact = r.exact;
    row.first_score = r.initial.as_ref().map(|f| f.score);
    row.first_penalty = r.initial.as_ref().map(|f| f.penalty);
    row.best_score = r.best.as_ref().map(|f| f.score);
    row.best_penalty = r.best.as_ref().map(|f| f.penalty);
    row.evals = r.evals;
    row.compiles = r.compiles;
    row.error = r.initial_error.clone();
    row.winning_ops = r.history.iter().flat_map(|h| h.ops.iter().map(|s| s.to_string())).collect();
    row.polished = r.polished;
    row.first_profile = r.initial.as_ref().map(|f| f.profile);
    row.best_profile = r.best.as_ref().map(|f| f.profile);
    row.traces = r.traces;
    for s in r.op_stats.iter().filter(|s| s.tries > 0) {
        row.ops.insert(s.name.to_string(), serde_json::json!([s.tries, s.improved, s.new_best]));
    }
    row.status = if r.exact {
        if row.first_exact { "exact".into() } else { "searched".into() }
    } else if row.compiled {
        "mismatch".into()
    } else {
        "cc-err".into()
    };
}

fn print_table(rows: &[Row]) {
    #[derive(Default)]
    struct B {
        n: usize,
        drafted: usize,
        compiled: usize,
        first: usize,
        fin: usize,
        score: f64,
        scored: usize,
        hdr: usize,
        imp: usize,
    }
    let mut by: BTreeMap<&str, B> = BTreeMap::new();
    let mut tot = B::default();
    for r in rows {
        for b in [by.entry(size_bucket(r.size)).or_default(), &mut tot] {
            if r.status == "hdr-inline" {
                b.hdr += 1;
                continue;
            }
            if r.status == "implicit" {
                b.imp += 1;
                continue;
            }
            b.n += 1;
            b.drafted += r.drafted as usize;
            b.compiled += r.compiled as usize;
            b.first += r.first_exact as usize;
            b.fin += r.final_exact as usize;
            if let Some(s) = r.best_score {
                b.score += s;
                b.scored += 1;
            }
        }
    }
    let pct = |a: usize, n: usize| if n == 0 { "-".to_string() } else { format!("{:.1}%", 100.0 * a as f64 / n as f64) };
    println!("{:<8} {:>5} {:>8} {:>9} {:>9} {:>9} {:>10} {:>6} {:>6}", "size", "n", "drafted", "compiled", "1st-exact", "final", "mean-best", "hdr", "impl");
    let line = |name: &str, b: &B| {
        println!(
            "{:<8} {:>5} {:>8} {:>9} {:>9} {:>9} {:>10} {:>6} {:>6}",
            name,
            b.n,
            pct(b.drafted, b.n),
            pct(b.compiled, b.n),
            pct(b.first, b.n),
            pct(b.fin, b.n),
            if b.scored > 0 { format!("{:.1}", b.score / b.scored as f64) } else { "-".into() },
            b.hdr,
            b.imp
        );
    };
    for name in SIZE_BUCKETS {
        if let Some(b) = by.get(name) {
            line(name, b);
        }
    }
    line("all", &tot);
}
