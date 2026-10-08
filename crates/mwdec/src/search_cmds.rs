//! `mwdec match` and `mwdec eval`: first draft (lift + emit) followed by the compiler-in-the-loop
//! permuter (`mwdec-search`).
//!
//! Anti-cheat: the inputs given to the decompiler are the unit's target object, the module's
//! other target objects (literal values), the include-only context TU (`harness::context_tu`)
//! and the compiler. No source file body is ever opened here.
use super::{find_unit, load_obj, load_project, module_externs};
use crate::draft_server::DraftReply;
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

pub fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect()
}

/// Compiler drivers per compiler version (Dolphin SDK / runtime units use GC/1.2.5n etc.).
/// Compiles in flight stay bounded by the number of search workers (each runs one at a time).
pub struct Compilers {
    root: PathBuf,
    work: PathBuf,
    jobs: usize,
    map: Mutex<HashMap<String, Arc<Mwcc>>>,
    /// Persistent-compiler workers per unit driver (`Mwcc::enable_fast`; 0 = off).
    fast_workers: usize,
}

impl Compilers {
    /// Unit drivers compile candidates through the fast path with up to `min(jobs, 4)` persistent
    /// compiler workers (`MWDEC_PERSIST=0` turns it off).
    pub fn new(root: &Path, work: &Path, jobs: usize) -> Compilers {
        Compilers { root: root.to_path_buf(), work: work.to_path_buf(), jobs, map: Mutex::new(HashMap::new()), fast_workers: jobs.min(4) }
    }

    /// Set the number of fast-path workers per unit driver (0: normal compiles only).
    pub fn with_fast_workers(mut self, n: usize) -> Compilers {
        self.fast_workers = n;
        self
    }

    /// End the fast-path workers of every driver; their counters summed (`None`: the fast path
    /// was never on).
    pub fn finish(&self) -> Option<mwdec_mwcc::FastStats> {
        let mut total: Option<mwdec_mwcc::FastStats> = None;
        for m in self.map.lock().unwrap().values() {
            if let Some(s) = m.fast_stats() {
                total.get_or_insert_with(Default::default).add(&s);
            }
            m.shutdown_fast();
        }
        total
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
                if self.fast_workers > 0 {
                    m.enable_fast(self.fast_workers);
                }
                Arc::new(m)
            })
            .clone()
    }
}

/// Per-unit inputs shared by every function of the unit.
pub struct UnitInputs {
    pub target: ObjectFile,
    /// What the lifter sees: `target` plus the values of literals it references that live in other
    /// objects of the module ([`with_extern_literals`]). `None` = same as `target`.
    pub lift_obj: Option<ObjectFile>,
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
    /// Machine code only assembly produces (`mwdec_lift::asmonly`): no C/C++ source can match.
    Asm(String),
    Lift(String),
}

/// `target` plus copies of the literals (dtk `lbl_` / compiler-local names in `.sdata2`, `.rodata`,
/// `.sdata`, `.data`) its code references but another target object of the module defines (shared
/// literal pools split off by dtk). Without them the lifter emits `extern float lbl_<addr>;`
/// instead of the value, which changes scheduling and register use. Each copied literal gets its
/// own section (`<section>@<name>`) so its bytes can't be confused with the target's own pools.
pub fn with_extern_literals(target: &ObjectFile, ext: &ExternIndex) -> Option<ObjectFile> {
    let mut want: Vec<&str> = target
        .functions
        .iter()
        .flat_map(|f| f.relocs.iter().map(|r| r.target.as_str()))
        .filter(|n| (n.starts_with("lbl_") || n.starts_with('@') || n.starts_with("...")) && !target.data.contains_key(*n))
        .filter(|n| !target.symbols.iter().any(|s| s.name == *n))
        .collect();
    want.sort();
    want.dedup();
    let mut out: Option<ObjectFile> = None;
    for name in want {
        let Some((def, sec, bytes)) = ext.defined(name, 4096) else { continue };
        if !matches!(sec.name.as_str(), ".sdata2" | ".rodata" | ".sdata" | ".data") || sec.bytes.is_empty() {
            continue;
        }
        let o = out.get_or_insert_with(|| target.clone());
        let sname = format!("{}@{name}", sec.name);
        o.sections.push(Section { name: sname.clone(), size: bytes.len() as u32, bytes: bytes.to_vec(), executable: false, relocs: vec![] });
        o.symbols.push(SymbolDef { name: name.to_string(), section: sname, address: 0, size: def.size, binding: SymBinding::Local, is_func: false });
        let n = (def.size as usize).min(bytes.len());
        o.data.insert(
            name.to_string(),
            DataSymbol {
                name: name.to_string(),
                binding: SymBinding::Local,
                section: sec.name.clone(),
                size: def.size,
                bytes: bytes[..n].to_vec(),
                relocs: vec![],
                address: 0,
            },
        );
    }
    out
}

/// Inputs for one unit. `module_objs`: the module's target objects (main + the unit's REL), used
/// to recover vtables of every class the context knows. `lit_ext`: the module's target-side
/// extern index, for literal values the target object only references.
pub fn unit_inputs(p: &Project, u: &Unit, cc: &Compilers, module_objs: &[Arc<ObjectFile>], with_db: bool, lit_ext: Option<&ExternIndex>) -> Result<UnitInputs> {
    if u.cflags.is_empty() {
        bail!("unit {} has no compiler flags", u.name);
    }
    let context = crate::ctxext::extended_context(p, u, &harness::context_tu(p, u)?);
    unit_inputs_with_context(p, u, cc, module_objs, with_db, context, lit_ext)
}

/// `unit_inputs` with an explicit context TU (include lines only), e.g. an automatic
/// header-only context for a unit without a source file (`mwdec harvest`).
pub fn unit_inputs_with_context(p: &Project, u: &Unit, cc: &Compilers, module_objs: &[Arc<ObjectFile>], with_db: bool, context: String, lit_ext: Option<&ExternIndex>) -> Result<UnitInputs> {
    if u.cflags.is_empty() {
        bail!("unit {} has no compiler flags", u.name);
    }
    let m = cc.for_unit(p, &u.name);
    let target = load_obj(p, &u.target_obj)?;
    let plain = m.plain_context(&context, &u.cflags).named(&u.name);
    let ctx = if context.is_empty() { plain.clone() } else { m.precompile(&context, &u.cflags).map(|c| c.named(&u.name)).unwrap_or_else(|_| plain.clone()) };
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
                for sym in target.functions.iter().map(|f| f.name.as_str()).chain(target.functions.iter().flat_map(|f| f.relocs.iter().map(|r| r.target.as_str()))) {
                    if sym.starts_with("__dt__") {
                        if let Some(c) = mwdec_lift::sig::sig_of(sym, None).this_class {
                            db.object_dtors.insert(c);
                        }
                    }
                }
                if std::env::var_os("MWDEC_NO_DECLARED_VTABLES").is_none() {
                    mwdec_ctx::vtable::declared_vtables(&mut db);
                    mwdec_ctx::resolve::fill_methods(&mut db);
                }
                if !c_mode && std::env::var("MWDEC_NO_INLINE").is_err() {
                    // (the probe driver caches its compiles on disk across runs)
                    mwdec_inline::complete::complete_in(&mut db, &context, &u.cflags, &cc.probe_driver(p, &u.name), &ctx, &plain);
                }
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
    let lift_obj = lit_ext.and_then(|e| with_extern_literals(&target, e));
    Ok(UnitInputs { target, lift_obj, context, mwcc: m, ctx, plain, db, c_mode, tracer, inlines })
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
    let plain = draft_variant(ui, f, false, true).ok();
    choose_between(scorer, with_inlines, plain)
}

/// Register repair of the chosen draft, then the draft variants (`mwdec_lift::variants`) when
/// that is not exact: the best variant, repaired too, replaces it only if strictly better (so a
/// variant never costs a match the default pipeline finds).
pub fn repair_or_variant(scorer: &Scorer, chosen: String, variants: Vec<String>, tracer: Option<&mwdec_search::trace::Tracer>) -> String {
    // an exact draft is kept whatever a persistent compiler said about it (its verdict must not
    // depend on the candidates compiled before it)
    if !scorer.eval(&chosen).0.fitness().is_some_and(|f| f.exact) && scorer.eval_normal(&chosen).0.fitness().is_some_and(|f| f.exact) {
        return chosen;
    }
    let best = repair_or_variant_only(scorer, chosen, variants, tracer);
    near_miss_pass(scorer, best, tracer)
}

/// Compiles the draft-time near-miss pass may spend.
pub const NEAR_MISS_COMPILES: usize = 40;

/// A small function whose best draft is a near miss (score >= 95) gets a bounded systematic
/// neighbourhood pass (`mwdec_search::search::quick_pass`, at most [`NEAR_MISS_COMPILES`]
/// compiles); an exact neighbour replaces it. `MWDEC_NO_NEAR_PASS` turns it off.
pub fn near_miss_pass(scorer: &Scorer, src: String, tracer: Option<&mwdec_search::trace::Tracer>) -> String {
    if std::env::var_os("MWDEC_NO_NEAR_PASS").is_some() || scorer.tf.code.len() > 128 {
        return src;
    }
    let (e, _) = scorer.eval(&src);
    let Some(f) = e.fitness().cloned() else { return src };
    if f.exact || f.score < 95.0 {
        return src;
    }
    scorer.mwcc.enable_fast(4);
    mwdec_search::search::quick_pass_with(scorer, &src, &f, NEAR_MISS_COMPILES, tracer).unwrap_or(src)
}

fn repair_or_variant_only(scorer: &Scorer, chosen: String, variants: Vec<String>, tracer: Option<&mwdec_search::trace::Tracer>) -> String {
    let base = repair_registers(scorer, chosen, tracer);
    if variants.is_empty() {
        return base;
    }
    let fit = |s: &str| scorer.eval(s).0.fitness().cloned();
    let bf = fit(&base);
    if bf.as_ref().is_some_and(|f| f.exact) {
        return base;
    }
    // (the best-scoring variant whose diff is register/order-only gets repaired too: its penalty
    // may rank it below another variant that no register edit can fix)
    let reg_only = variants
        .iter()
        .filter_map(|c| fit(c).filter(mwdec_search::regfix::register_only).map(|f| (c.clone(), f.score)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|x| x.0);
    let Some(v) = choose_among(scorer, variants) else { return base };
    let mut best = base;
    let mut best_fit = bf;
    let mut tried = vec![];
    for c in std::iter::once(v).chain(reg_only) {
        if tried.contains(&c) {
            continue;
        }
        tried.push(c.clone());
        let r = repair_registers(scorer, c, tracer);
        let rf = fit(&r);
        let better = match (&rf, &best_fit) {
            (Some(x), Some(y)) => x.draft_better_than(y),
            (Some(_), None) => true,
            _ => false,
        };
        if better {
            best = r;
            best_fit = rf;
        }
    }
    best
}

/// Kind of a function the compiler emits without a definition in the unit (`hdr-inline` or an
/// `mwdec_project::standalone` implicit kind); None for ordinary functions.
pub fn emitted_kind(ui: &UnitInputs, f: &Function) -> Option<String> {
    let db = ui.db.as_ref()?;
    match mwdec_project::standalone::standalone(&mwdec_lift::sig::sig_of(&f.name, Some(db)), db) {
        mwdec_project::standalone::Standalone::Yes => None,
        mwdec_project::standalone::Standalone::HeaderInline => Some("hdr-inline".into()),
        mwdec_project::standalone::Standalone::Implicit(k) => Some(k.to_string()),
    }
}

/// Sources that make the compiler emit `f` (an explicit instantiation or a use; see
/// `mwdec_emit::instantiate`), for header inlines, template instances and implicit members.
pub fn instantiation_drafts(ui: &UnitInputs, f: &Function) -> Vec<String> {
    if let Some(t) = thunk_target(&f.name) {
        return thunk_drafts(ui, t);
    }
    let db = ui.db.as_ref();
    let sig = mwdec_lift::sig::sig_of(&f.name, db);
    let is_class = |s: &str| {
        db.is_some_and(|d| mwdec_lift::sig::find_class(d, s).is_some())
            || mwdec_lift::sig::split_scope(s).1.chars().next().is_some_and(|c| !c.is_ascii_lowercase())
    };
    if mwdec_lift::sig::demangle(&f.name).is_none() {
        return mwdec_emit::instantiate::triggers_c(&sig);
    }
    let t = mwdec_emit::instantiate::triggers(&f.name, Some(&sig.ret), &is_class);
    // classes the context only declares whose destructor the function calls: a definition with
    // that destructor declared (`delete p` then calls it instead of freeing directly)
    let Some(db) = db else { return t };
    let mut stand_ins = String::new();
    let mut seen = std::collections::HashSet::new();
    for r in f.relocs.iter().filter(|r| r.target.starts_with("__dt__")) {
        let Some(c) = mwdec_lift::sig::sig_of(&r.target, Some(db)).this_class else { continue };
        if c.contains('<') || !seen.insert(c.clone()) || mwdec_lift::sig::find_class(db, &c).is_some_and(|k| !k.is_declaration) {
            continue;
        }
        let base = mwdec_lift::sig::split_scope(&c).1.to_string();
        stand_ins.push_str(&format!("struct {c} {{
    ~{base}();
}};
"));
    }
    if stand_ins.is_empty() {
        return t;
    }
    let with: Vec<String> = t.iter().map(|x| format!("{stand_ins}{x}")).collect();
    t.into_iter().chain(with).collect()
}

/// `@4@Method__5CFooFv` -> `Method__5CFooFv` (a `this`-adjusting thunk's target).
fn thunk_target(sym: &str) -> Option<&str> {
    let r = sym.strip_prefix('@')?;
    let (d, t) = r.split_once('@')?;
    (!d.is_empty() && d.chars().all(|c| c.is_ascii_digit())).then_some(t)
}

/// A `this`-adjusting thunk is emitted with its class's vtable, i.e. in the unit defining the
/// class's key function (its first non-inline virtual function). The drafts are the class's
/// virtual functions this object defines, each lifted alone (the destructor first; one of them
/// is the key function).
fn thunk_drafts(ui: &UnitInputs, target: &str) -> Vec<String> {
    let db = ui.db.as_ref();
    let Some(cls) = mwdec_lift::sig::sig_of(target, db).this_class else { return vec![] };
    let n = mwdec_lift::sig::norm_name(&cls);
    let vt: Vec<String> = db.and_then(|d| mwdec_lift::sig::find_class(d, &cls)).map(|c| c.vtable.iter().map(|v| v.symbol.clone()).collect()).unwrap_or_default();
    let mut cands: Vec<(usize, &Function)> = ui
        .target
        .functions
        .iter()
        .filter(|g| !g.name.starts_with('@'))
        .filter_map(|g| {
            let s = mwdec_lift::sig::sig_of(&g.name, db);
            if s.this_class.as_deref().map(mwdec_lift::sig::norm_name).as_deref() != Some(n.as_str()) {
                return None;
            }
            if mwdec_lift::sig::is_dtor(&s) {
                Some((0, g))
            } else {
                vt.iter().position(|v| *v == g.name).map(|i| (i + 1, g))
            }
        })
        .collect();
    cands.sort_by_key(|c| c.0);
    let mut out = Vec::new();
    for (k, g) in cands.into_iter().take(6) {
        out.extend(draft_with(ui, g, false).ok());
        if k == 0 {
            // (a destructor whose draft doesn't compile: the definition alone still emits the vtable)
            let base = mwdec_lift::sig::split_scope(&cls).1.split('<').next().unwrap_or("").to_string();
            out.push(format!("{}::~{base}() {{}}
", mwdec_emit::types::split_closers(&cls)));
        }
    }
    out
}

/// Lifted bodies of a function the compiler emits on demand (explicit specializations), each
/// alone and followed by each use from `alts`: a specialization of an inline member is itself
/// inline, emitted only when something needs it out of line.
pub fn specialization_drafts(alts: &[String], lifted: impl Iterator<Item = String>) -> Vec<String> {
    let mut out = Vec::new();
    for l in lifted {
        out.push(l.clone());
        for a in alts.iter().filter(|a| !a.starts_with("template ")) {
            out.push(format!("{l}
{a}"));
        }
    }
    out
}

/// The best (by compile + compare) of several drafts, in order; the first exact one wins.
/// [`Scorer::eval`] for draft-time decisions: a compile failure the fast path reported without
/// its messages (a trusted failure, `mwdec_mwcc` fast path) is confirmed with a normal compile,
/// so a draft is never dropped on an unconfirmed error.
pub fn eval_draft(scorer: &Scorer, src: &str) -> mwdec_search::Eval {
    match scorer.eval(src).0 {
        mwdec_search::Eval::CompileError(m) if m.contains("messages not collected") => scorer.eval_normal(src).0,
        e => e,
    }
}

pub fn choose_among(scorer: &Scorer, cands: Vec<String>) -> Option<String> {
    let mut best: Option<(String, Option<mwdec_search::Fitness>)> = None;
    let mut seen = std::collections::HashSet::new();
    for c in cands {
        if !seen.insert(c.clone()) {
            continue;
        }
        let r = eval_draft(scorer, &c);
        let fit = r.fitness().cloned();
        if std::env::var_os("MWDEC_SHOW_DRAFTS").is_some() {
            let what = match &fit {
                Some(f) if f.exact => "exact".to_string(),
                Some(f) => format!("score {:.1}", f.score),
                None => format!("{:?}", r).chars().take(300).collect(),
            };
            eprintln!("--- candidate: {what}
{c}");
        }
        if fit.as_ref().is_some_and(|f| f.exact) {
            return Some(c);
        }
        let better = match (&best, &fit) {
            (None, _) => true,
            (Some((_, None)), Some(_)) => true,
            (Some((_, Some(b))), Some(x)) => x.draft_better_than(b),
            _ => false,
        };
        if better {
            best = Some((c, fit));
        }
    }
    best.map(|b| b.0)
}

/// The better (by compile + compare) of the draft with folded inlines and the one without.
pub fn choose_between(scorer: &Scorer, with_inlines: String, plain: Option<String>) -> String {
    let Some(plain) = plain.filter(|p| *p != with_inlines) else { return with_inlines };
    let a = eval_draft(scorer, &with_inlines);
    let b = eval_draft(scorer, &plain);
    if std::env::var_os("MWDEC_SHOW_VARIANTS").is_some() {
        eprintln!("--- with inlines ({:?}):
{with_inlines}
--- plain ({:?}):
{plain}", a.fitness().map(|f| f.score), b.fitness().map(|f| f.score));
    }
    match (a.fitness(), b.fitness()) {
        (Some(x), Some(y)) if y.draft_better_than(x) => plain,
        (None, Some(_)) => plain,
        _ => with_inlines,
    }
}

/// Register-only mismatch of the chosen draft: a bounded deterministic repair over
/// variable-structure edits (`mwdec_search::regfix`), scored by the compiler. Part of drafting;
/// `MWDEC_NO_REGFIX` switches it off.
pub fn repair_registers(scorer: &Scorer, src: String, tracer: Option<&mwdec_search::trace::Tracer>) -> String {
    if std::env::var("MWDEC_NO_REGFIX").is_ok() {
        return src;
    }
    let (e, _) = scorer.eval(&src);
    let Some(f) = e.fitness() else { return src };
    if !mwdec_search::regfix::register_only(f) {
        return src;
    }
    // Many compiles of one context: persistent compilers pay off here (started on demand).
    scorer.mwcc.enable_fast(4);
    match mwdec_search::regfix::repair(scorer, &src, f, tracer, &mwdec_search::regfix::RepairConfig::default()) {
        Some(r) => r.src,
        None => src,
    }
}

pub fn draft_variant(ui: &UnitInputs, f: &Function, inlines: bool, include_implicit: bool) -> std::result::Result<String, NoDraft> {
    draft_opts(ui, f, inlines, include_implicit, false, false, 0)
}

/// Draft with member accesses as raw offsets (a fallback when field accesses don't compile,
/// e.g. private members used from a free function).
pub fn draft_raw(ui: &UnitInputs, f: &Function) -> std::result::Result<String, NoDraft> {
    draft_opts(ui, f, true, false, true, false, 0)
}

/// A static initializer's draft with the globals the context doesn't declare defined `const`.
pub fn draft_sinit_const(ui: &UnitInputs, f: &Function) -> std::result::Result<String, NoDraft> {
    draft_opts(ui, f, true, false, false, true, 0)
}

/// A static initializer's alternative draft `k` (objects built by their k-th scalar constructor).
pub fn draft_sinit_variant(ui: &UnitInputs, f: &Function, k: u8) -> std::result::Result<String, NoDraft> {
    draft_opts(ui, f, true, false, false, false, k)
}

fn draft_opts(ui: &UnitInputs, f: &Function, inlines: bool, include_implicit: bool, raw_offsets: bool, sinit_const: bool, sinit_variant: u8) -> std::result::Result<String, NoDraft> {
    draft_flipped(ui, f, inlines, include_implicit, raw_offsets, sinit_const, sinit_variant, &[]).map(|(s, _)| s)
}

/// Draft variant points tried per function (each costs one lift + emit and, when the source
/// differs, one compile).
pub const MAX_VARIANT_POINTS: usize = 16;

/// The draft variants of `f` (`mwdec_lift::variants`): the default draft (with folded inlines)
/// redrafted with each decision point it asked flipped, distinct sources other than the default.
/// The caller compiles them with the other candidates and keeps the best.
pub fn variant_drafts(ui: &UnitInputs, f: &Function, include_implicit: bool) -> Vec<String> {
    if std::env::var_os("MWDEC_NO_VARIANTS").is_some() {
        return vec![];
    }
    let Ok((base, points)) = draft_flipped(ui, f, true, include_implicit, false, false, 0, &[]) else { return vec![] };
    let mut out: Vec<String> = Vec::new();
    let points: Vec<&'static str> = points.into_iter().take(MAX_VARIANT_POINTS).collect();
    // each point alone, then (few points) every pair: independent source properties often
    // only match together
    let mut sets: Vec<Vec<&'static str>> = points.iter().map(|p| vec![*p]).collect();
    if (2..=4).contains(&points.len()) {
        for i in 0..points.len() {
            for j in i + 1..points.len() {
                sets.push(vec![points[i], points[j]]);
            }
        }
    }
    for set in &sets {
        if let Ok((s, _)) = draft_flipped(ui, f, true, include_implicit, false, false, 0, set) {
            if s != base && !out.contains(&s) {
                out.push(s);
            }
        }
    }
    // text variants: source rewrites of the search that settle an ambiguity of the draft's
    // form (a flag-if chain or `&&`, a comparison used as a number or a select of 0/1)
    for (name, k) in TEXT_VARIANT_OPS {
        let Some(op) = mwdec_search::ops::op_index(name) else { continue };
        let mut w = vec![0.0; mwdec_search::ops::OPS.len()];
        w[op] = 1.0;
        for n in mwdec_search::ops::neighbours(&base, &f.name, &w, None, None, 8, 1, *k) {
            if n.src != base && !out.contains(&n.src) {
                out.push(n.src);
            }
        }
    }
    out
}

/// Search operators whose rewrites are also draft variants (operator, at most this many sites).
pub const TEXT_VARIANT_OPS: &[(&str, usize)] = &[("flag_and", 2), ("cmp_select", 2)];

/// One draft with the given variant points flipped; also returns the points it asked.
fn draft_flipped(
    ui: &UnitInputs,
    f: &Function,
    inlines: bool,
    include_implicit: bool,
    raw_offsets: bool,
    sinit_const: bool,
    sinit_variant: u8,
    flipped: &[&'static str],
) -> std::result::Result<(String, Vec<&'static str>), NoDraft> {
    if !include_implicit {
        if let Some(why) = mwdec_lift::asmonly::requires_asm(&ui.target, f) {
            return Err(NoDraft::Asm(why));
        }
    }
    if let Some(db) = &ui.db {
        let sig = mwdec_lift::sig::sig_of(&f.name, Some(db));
        match mwdec_project::standalone::standalone(&sig, db) {
            mwdec_project::standalone::Standalone::HeaderInline if !include_implicit => return Err(NoDraft::HeaderInline),
            mwdec_project::standalone::Standalone::Implicit(k) if !include_implicit => return Err(NoDraft::Implicit(k.to_string())),
            _ => {}
        }
    }
    // the inline library lifts its probes: build it outside the variant scope
    if inlines && ui.inlines.enabled && ui.db.is_some() {
        let _ = ui.inlines.get(ui, &format!("{}
{}", ui.mwcc.compiler, ui.ctx.cflags.join(" ")));
    }
    let (res, points) = mwdec_lift::variants::draft(flipped, || draft_once(ui, f, inlines, raw_offsets, sinit_const, sinit_variant));
    res.map(|s| (s, points))
}

fn draft_once(ui: &UnitInputs, f: &Function, inlines: bool, raw_offsets: bool, sinit_const: bool, sinit_variant: u8) -> std::result::Result<String, NoDraft> {
    let ir = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // (`MWDEC_NO_UNIT_COMPILER=1`: lift as for the game compiler, to measure the compiler-specific rules)
        let compiler = if std::env::var_os("MWDEC_NO_UNIT_COMPILER").is_some() { None } else { Some(ui.mwcc.compiler.clone()) };
        let lopts = mwdec_lift::LiftOptions { compiler, ..Default::default() };
        let mut ir = mwdec_lift::lift_function_with(ui.lift_obj.as_ref().unwrap_or(&ui.target), f, ui.db.as_ref(), &lopts)?;
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
    let em = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mwdec_emit::emit_function(&ir, ui.db.as_ref(), &mwdec_emit::EmitOptions { c_mode: ui.c_mode, raw_offsets, sinit_const, sinit_variant, ..Default::default() }))) {
        Ok(em) => em,
        Err(_) => return Err(NoDraft::Lift("emitter panic".into())),
    };
    Ok(format!("{}{}", em.preamble, extern_c_definition(&em.body, &f.name, ui.c_mode)))
}

/// Rewrite an exact source towards natural code (names, casts, temporaries), keeping only the
/// changes that still match (see `mwdec_emit::tidy`).
pub fn polish_exact(ui: &UnitInputs, scorer: &Scorer, src: &str) -> String {
    if std::env::var("MWDEC_NO_POLISH").is_ok() {
        return src.to_string();
    }
    // a rewrite bug must never cost the exact result
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mwdec_emit::tidy::polish(src, ui.db.as_ref(), 100, &mut |s: &str| scorer.eval(s).0.fitness().map_or(false, |f| f.exact)).0)).unwrap_or_else(|_| src.to_string())
}

/// A function whose symbol is not a mangled C++ name (a placeholder like `fn_<module>_<addr>`)
/// needs C linkage in a C++ unit, or the compiler mangles the definition (`<name>__Fi`).
pub fn extern_c_definition(body: &str, symbol: &str, c_mode: bool) -> String {
    if c_mode || mwdec_lift::sig::demangle(symbol).is_some() || body.contains("extern \"C\"") {
        return body.to_string();
    }
    let needle = format!("{symbol}(");
    let mut out = String::with_capacity(body.len() + 12);
    let mut done = false;
    for line in body.split_inclusive('\n') {
        let t = line.trim_start();
        if !done && !t.starts_with("//") && !t.starts_with('#') && line.contains(&needle) {
            let indent = &line[..line.len() - t.len()];
            out.push_str(indent);
            out.push_str("extern \"C\" ");
            out.push_str(t);
            done = true;
        } else {
            out.push_str(line);
        }
    }
    out
}

pub fn externs_for(p: &Project, unit: &str) -> (ExternIndex, ExternIndex) {
    let mut m = module_externs(p, std::iter::once(unit));
    m.remove(Project::module_of(unit)).unwrap_or_else(|| (ExternIndex::new(vec![]), ExternIndex::new(vec![])))
}

type ExtPair = Arc<(ExternIndex, ExternIndex)>;

/// Extern indexes per module (target side, our side), the main module's built once and shared
/// as the base layer of every REL module's index.
pub struct ModuleExterns<'p> {
    p: &'p Project,
    main: std::sync::OnceLock<(Arc<ExternIndex>, Arc<ExternIndex>, ExtPair)>,
    mods: Mutex<HashMap<String, Arc<std::sync::OnceLock<ExtPair>>>>,
}

impl<'p> ModuleExterns<'p> {
    pub fn new(p: &'p Project) -> Self {
        ModuleExterns { p, main: Default::default(), mods: Default::default() }
    }

    fn main(&self) -> &(Arc<ExternIndex>, Arc<ExternIndex>, ExtPair) {
        self.main.get_or_init(|| {
            let t = Arc::new(ExternIndex::new(self.p.load_module_data("main")));
            let o = Arc::new(ExternIndex::new(self.p.load_module_linked("main")));
            let pair = Arc::new((ExternIndex::layered(t.clone(), vec![]), ExternIndex::layered(o.clone(), vec![])));
            (t, o, pair)
        })
    }

    pub fn get(&self, module: &str) -> ExtPair {
        let (t, o, pair) = self.main();
        if module == "main" {
            return pair.clone();
        }
        let cell = self.mods.lock().unwrap().entry(module.to_string()).or_default().clone();
        cell.get_or_init(|| {
            Arc::new((
                ExternIndex::layered(t.clone(), self.p.load_module_data(module)),
                ExternIndex::layered(o.clone(), self.p.load_module_linked(module)),
            ))
        })
        .clone()
    }

    /// Forget a module's index (in-flight users keep their `Arc`).
    pub fn drop_module(&self, module: &str) {
        self.mods.lock().unwrap().remove(module);
    }
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
    let ui = unit_inputs(&p, u, &cc, &t_ext.objs, !a.no_db && a.init.is_none(), Some(&t_ext))?;
    let f = mwdec_obj::find_function(&ui.target, &a.symbol).ok_or_else(|| anyhow!("{} not in {}", a.symbol, u.target_obj))?;
    let init = match &a.init {
        Some(path) => std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?,
        None => match draft(&ui, f) {
            Ok(s) if emitted_kind(&ui, f).is_some() || mwdec_emit::instantiate::is_template_instance(&f.name) => {
                let ti = ObjIndex::with_externs(&ui.target, &t_ext);
                let prover = crate::placeholders::UnitProver::new(&ui.mwcc, &ui.ctx, ui.db.as_ref());
                let scorer = Scorer::new(&ui.mwcc, &ui.ctx, Some(&ui.plain), &ti, f, Some(&o_ext), &a.symbol).with_prover(Some(&prover));
                let mut c = instantiation_drafts(&ui, f);
                let lifted = specialization_drafts(&c, std::iter::once(s));
                c.extend(lifted);
                choose_among(&scorer, c).unwrap_or_default()
            }
            Ok(s) => s,
            Err(NoDraft::HeaderInline) => bail!("{} is defined inline in a context header; nothing to decompile", a.symbol),
            Err(NoDraft::Implicit(k)) => bail!("{} can't exist as standalone source ({k})", a.symbol),
            Err(NoDraft::Asm(k)) => bail!("{} needs assembly ({k}); pass --init <file.cpp>", a.symbol),
            Err(NoDraft::Lift(e)) => bail!("no first draft ({e}); pass --init <file.cpp>"),
        },
    };
    let ti = ObjIndex::with_externs(&ui.target, &t_ext);
    let prover = crate::placeholders::UnitProver::new(&ui.mwcc, &ui.ctx, ui.db.as_ref());
    let scorer = Scorer::new(&ui.mwcc, &ui.ctx, Some(&ui.plain), &ti, f, Some(&o_ext), &a.symbol).with_prover(Some(&prover));
    let init = if a.init.is_none() && emitted_kind(&ui, f).is_none() && !mwdec_emit::instantiate::is_template_instance(&f.name) {
        repair_or_variant(&scorer, choose_draft(&ui, f, &scorer, init), variant_drafts(&ui, f, true), ui.tracer.as_deref())
    } else {
        init
    };
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
    let mut r = search(&scorer, &init, &cfg);
    if r.exact {
        let before = mwdec_emit::tidy::measure(&r.best_src);
        r.best_src = polish_exact(&ui, &scorer, &r.best_src);
        eprintln!("naturalness: {} -> {}", before.summary(), mwdec_emit::tidy::measure(&r.best_src).summary());
        let _ = std::fs::create_dir_all(&out);
        let _ = std::fs::write(out.join("polished.cpp"), &r.best_src);
    }
    print!("{}", r.best_src);
    if !r.best_src.ends_with('\n') {
        println!();
    }
    print_result(&r);
    eprintln!("best source: {}", out.join("best.cpp").display());
    if let Some(s) = cc.finish() {
        eprintln!("{}", s.line());
    }
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
    pub mem_report: bool,
    pub list: Option<PathBuf>,
    /// Draft only (no compiles): each row carries `draft_hash`, a hash of every draft text the
    /// compile stage would see (default, plain, raw, alternatives, variants). Two binaries with
    /// equal hashes on a function produce the same eval result for it (`tools/drafts_diff.py`).
    pub drafts_only: bool,
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
    /// Process commit (MB) when the row finished.
    mem_mb: u64,
    /// naturalness (`mwdec_emit::tidy`) of the draft and of the final source (polished when
    /// exact and MWDEC_EVAL_POLISH is set)
    nat_draft: Option<String>,
    nat_final: Option<String>,
    /// `--drafts-only`: hash of the draft texts.
    draft_hash: Option<String>,
    /// The draft carries the lifter's "loop condition never changes" warning: a wrong-meaning
    /// structure (counted across lists).
    wrong_meaning: bool,
    /// Exact result whose object defines data the target unit lacks (`Scorer::extra_data`).
    extra_data: Vec<String>,
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
            "first_profile": self.first_profile, "best_profile": self.best_profile, "traces": self.traces, "mem_mb": self.mem_mb,
            "nat_draft": self.nat_draft, "nat_final": self.nat_final, "draft_hash": self.draft_hash, "wrong_meaning": self.wrong_meaning, "extra_data": self.extra_data,
        })
    }
}

/// Hash of every draft text of a reply (`--drafts-only`).
fn draft_hash(d: &DraftReply) -> String {
    let mut parts: Vec<&[u8]> = vec![d.status.as_bytes(), b"\0src", d.src.as_deref().unwrap_or("").as_bytes(), b"\0plain", d.plain.as_deref().unwrap_or("").as_bytes(), b"\0raw", d.raw.as_deref().unwrap_or("").as_bytes()];
    parts.push(b"\0alts");
    parts.extend(d.alts.iter().map(|s| s.as_bytes()));
    parts.push(b"\0variants");
    parts.extend(d.variants.iter().map(|s| s.as_bytes()));
    format!("{:032x}", mwdec_mwcc::content_hash(&parts))
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
    if let Some(l) = &a.list {
        let text = std::fs::read_to_string(l).with_context(|| format!("reading {}", l.display()))?;
        let want: std::collections::HashSet<(String, String)> = text
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .filter_map(|v| Some((v.get("unit")?.as_str()?.to_string(), v.get("symbol")?.as_str()?.to_string())))
            .collect();
        ds.retain(|e| want.contains(&(e.unit.clone(), e.symbol.clone())));
    }
    shuffle(&mut ds, a.seed);
    if let Some(k) = a.limit {
        ds.truncate(k);
    }
    if ds.is_empty() {
        bail!("no dataset functions match");
    }
    let jobs = a.jobs.clamp(1, 16);
    let workers = a.workers.unwrap_or(6usize.div_ceil(jobs)).max(1);
    // Drafts alone compile each unit context once or twice per function: starting persistent
    // compilers costs more than it saves there (measured), so the fast path is for searches.
    let cc = Compilers::new(root, work, 6).with_fast_workers(if a.budget_secs == 0 { 0 } else { mwdec_mwcc::fast_workers_from_env(4) });
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
    // Shared per-unit inputs and per-module extern indexes, built lazily and dropped after the
    // unit's (module's) last function: keeping every unit's TypeDb/inline library and a fresh
    // index of all main objects per REL module made a 300-function eval grow past 3 GB.
    // Functions are processed grouped by unit (in shuffled order of first appearance).
    {
        let mut first: HashMap<String, usize> = HashMap::new();
        for (i, e) in ds.iter().enumerate() {
            first.entry(e.unit.clone()).or_insert(i);
        }
        ds.sort_by_key(|e| first[&e.unit]);
    }
    let unit_left: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::new());
    let module_left: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::new());
    for e in &ds {
        *unit_left.lock().unwrap().entry(e.unit.clone()).or_default() += 1;
        *module_left.lock().unwrap().entry(Project::module_of(&e.unit).to_string()).or_default() += 1;
    }
    let units: Mutex<HashMap<String, Arc<Mutex<Option<Arc<Result<UnitInputs, String>>>>>>> = Mutex::new(HashMap::new());
    let externs = ModuleExterns::new(&p);
    // Drafts in a child process (bounded memory; a pathological function becomes a row).
    let drafter = if std::env::var_os("MWDEC_INPROC_DRAFT").is_some() { None } else { Some(crate::draft_server::DraftClient::new(root, work, a.no_db)) };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..jobs {
            let _ = std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(s, || loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(e) = ds.get(i) else { break };
                let t = Instant::now();
                let mut row = Row { unit: e.unit.clone(), symbol: e.symbol.clone(), size: e.size, ..Default::default() };
                if a.drafts_only {
                    let d = match &drafter {
                        Some(c) => c.draft(&e.unit, &e.symbol, a.include_implicit, false),
                        None => DraftReply::err("unit-err", "--drafts-only needs the draft server"),
                    };
                    row.status = d.status.clone();
                    row.error = d.error.clone();
                    row.implicit = d.implicit.clone();
                    row.drafted = d.src.is_some() || !d.alts.is_empty();
                    row.draft_hash = Some(draft_hash(&d));
                    row.wrong_meaning = d.src.as_deref().is_some_and(|s| s.contains(mwdec_lift::structure::WARN_INVARIANT_LOOP));
                    row.seconds = t.elapsed().as_secs_f64();
                    let mut f = out_file.lock().unwrap();
                    let _ = writeln!(f, "{}", row.json());
                    let _ = f.flush();
                    drop(f);
                    if (i + 1) % 200 == 0 {
                        eprintln!("[{:>4}/{}] drafts", i + 1, ds.len());
                    }
                    continue;
                }
                let module = Project::module_of(&e.unit).to_string();
                let ext = externs.get(&module);
                let slot = units.lock().unwrap().entry(e.unit.clone()).or_default().clone();
                let ui = {
                    let mut g = slot.lock().unwrap();
                    if g.is_none() {
                        let r = p
                            .unit(&e.unit)
                            .ok_or_else(|| anyhow!("unknown unit"))
                            .and_then(|u| unit_inputs(&p, u, &cc, &ext.0.objs, !a.no_db && drafter.is_none(), drafter.is_none().then_some(&ext.0)));
                        *g = Some(Arc::new(r.map_err(|e| e.to_string())));
                    }
                    g.clone().unwrap()
                };
                drop(slot);
                run_one(&ui, &ext, e, &a, workers, &runs_dir, drafter.as_ref(), &mut row);
                drop(ui);
                drop(ext);
                {
                    let mut l = unit_left.lock().unwrap();
                    let n = l.get_mut(&e.unit).unwrap();
                    *n -= 1;
                    if *n == 0 {
                        units.lock().unwrap().remove(&e.unit);
                    }
                }
                {
                    let mut l = module_left.lock().unwrap();
                    let n = l.get_mut(&module).unwrap();
                    *n -= 1;
                    if *n == 0 {
                        externs.drop_module(&module);
                    }
                }
                row.seconds = t.elapsed().as_secs_f64();
                row.mem_mb = mwdec_core::memcap::commit_mb();
                let line = row.json().to_string();
                {
                    let mut f = out_file.lock().unwrap();
                    let _ = writeln!(f, "{line}");
                    let _ = f.flush();
                }
                eprintln!(
                    "[{:>4}/{}] {:<9} {:<5} {} {} ({:.1}s, {} compiles, {} MB){}",
                    i + 1,
                    ds.len(),
                    row.status,
                    e.size,
                    e.unit,
                    e.symbol,
                    row.seconds,
                    row.compiles,
                    row.mem_mb,
                    row.best_score.map(|s| format!(" best {s:.1}")).unwrap_or_default()
                );
                rows.lock().unwrap().push(row);
            });
        }
    });
    let rows = rows.into_inner().unwrap();
    print_table(&rows);
    if let Some(s) = cc.finish() {
        println!("{}", s.line());
    }
    println!("wrote {} ({:.0}s total)", out_path.display(), t0.elapsed().as_secs_f64());
    if a.mem_report {
        println!("{}", mwdec_core::memcap::report_line());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    ui: &Result<UnitInputs, String>,
    ext: &(ExternIndex, ExternIndex),
    e: &DatasetEntry,
    a: &EvalArgs,
    workers: usize,
    runs_dir: &Path,
    drafter: Option<&crate::draft_server::DraftClient>,
    row: &mut Row,
) {
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
    let d = match drafter {
        Some(c) => c.draft(&e.unit, &e.symbol, a.include_implicit, false),
        None => crate::draft_server::draft_local(ui, &e.symbol, a.include_implicit, false),
    };
    row.implicit = d.implicit.clone();
    let (src, plain) = match (d.status.as_str(), d.src) {
        ("ok", Some(s)) => (s, d.plain),
        (_, _) if !d.alts.is_empty() => (d.alts[0].clone(), None),
        (st, _) => {
            row.status = if st == "ok" { "lift-err".into() } else { st.to_string() };
            row.error = d.error;
            return;
        }
    };
    row.drafted = true;
    let ti = ObjIndex::with_externs(&ui.target, &ext.0);
    let prover = crate::placeholders::UnitProver::new(&ui.mwcc, &ui.ctx, ui.db.as_ref());
    let scorer = Scorer::new(&ui.mwcc, &ui.ctx, Some(&ui.plain), &ti, f, Some(&ext.1), &e.symbol).with_prover(Some(&prover));
    let src = if d.alts.is_empty() {
        choose_between(&scorer, src, plain)
    } else {
        // (instantiations first for functions emitted on demand, else the lifted drafts first)
        let mut c = if d.implicit.is_some() { d.alts.clone() } else { vec![] };
        c.push(src);
        c.extend(plain);
        if d.implicit.is_none() {
            c.extend(d.alts.iter().cloned());
        }
        let best = choose_among(&scorer, c.clone()).unwrap_or_default();
        if d.implicit.is_some() && !scorer.eval(&best).0.fitness().is_some_and(|f| f.exact) {
            // none of the instantiations matches (the header's body differs from the target's):
            // the lifted body as an explicit specialization
            let l = match drafter {
                Some(dc) => dc.draft(&e.unit, &e.symbol, true, true),
                None => crate::draft_server::draft_local(ui, &e.symbol, true, true),
            };
            c.extend(specialization_drafts(&d.alts, l.src.into_iter().chain(l.plain)));
            choose_among(&scorer, c).unwrap_or_default()
        } else {
            best
        }
    };
    let src = repair_or_variant(&scorer, src, d.variants.clone(), ui.tracer.as_deref());
    // Compile the draft before the search clock starts: a compiler crash with the unit's PCH is
    // repaired here (split PCH, once per unit context, cached on disk), not inside the budget.
    let mut first = eval_draft(&scorer, &src);
    // The verdict on the draft must not depend on which candidates of the unit ran before: a
    // non-exact result is re-checked with a normal compile (a persistent compiler's state is
    // shaped by earlier candidates; the normal object also replaces it in the memory cache).
    if !first.fitness().is_some_and(|f| f.exact) {
        first = scorer.eval_normal(&src).0;
    }
    // a draft that doesn't compile (e.g. a private member) is replaced by the raw-offsets draft
    let src = match (first.fitness(), d.raw) {
        (None, Some(raw)) if scorer.eval(&raw).0.fitness().is_some() => raw,
        _ => src,
    };
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
    if r.exact {
        row.extra_data = scorer.extra_data(&r.best_src);
    }
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
    row.nat_draft = Some(mwdec_emit::tidy::measure(&src).summary());
    row.wrong_meaning = src.contains(mwdec_lift::structure::WARN_INVARIANT_LOOP);
    let fin = if r.exact && std::env::var("MWDEC_EVAL_POLISH").is_ok() { polish_exact(ui, &scorer, &r.best_src) } else { r.best_src.clone() };
    row.nat_final = Some(mwdec_emit::tidy::measure(&fin).summary());
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
        asm: usize,
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
            if r.status == "asm" {
                b.asm += 1;
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
    println!("{:<8} {:>5} {:>8} {:>9} {:>9} {:>9} {:>10} {:>6} {:>6} {:>6}", "size", "n", "drafted", "compiled", "1st-exact", "final", "mean-best", "hdr", "impl", "asm");
    let line = |name: &str, b: &B| {
        println!(
            "{:<8} {:>5} {:>8} {:>9} {:>9} {:>9} {:>10} {:>6} {:>6} {:>6}",
            name,
            b.n,
            pct(b.drafted, b.n),
            pct(b.compiled, b.n),
            pct(b.first, b.n),
            pct(b.fin, b.n),
            if b.scored > 0 { format!("{:.1}", b.score / b.scored as f64) } else { "-".into() },
            b.hdr,
            b.imp,
            b.asm
        );
    };
    for name in SIZE_BUCKETS {
        if let Some(b) = by.get(name) {
            line(name, b);
        }
    }
    line("all", &tot);
    let wrong: Vec<&Row> = rows.iter().filter(|r| r.wrong_meaning).collect();
    if !wrong.is_empty() {
        println!("wrong-meaning drafts (loop condition never changes): {}", wrong.len());
        for r in wrong.iter().take(20) {
            println!("  {} {}", r.unit, r.symbol);
        }
    }
    // functions emitted without a definition (evaluated with --include-implicit), by kind
    let mut kinds: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.status != "implicit" && r.status != "hdr-inline") {
        let k = kinds.entry(r.implicit.as_deref().unwrap_or("standalone")).or_default();
        k.0 += 1;
        k.1 += r.compiled as usize;
        k.2 += r.final_exact as usize;
    }
    if kinds.len() > 1 {
        println!("{:<24} {:>5} {:>9} {:>9}", "kind", "n", "compiled", "exact");
        for (k, (n, c, x)) in &kinds {
            println!("{:<24} {:>5} {:>9} {:>9}", k, n, pct(*c, *n), pct(*x, *n));
        }
    }
}
