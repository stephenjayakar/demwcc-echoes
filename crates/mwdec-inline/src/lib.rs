//! mwdec-inline: recognise expanded header inline functions in lifted IR and fold them back
//! into calls (`a - b`, `v.MagSquared()`, `CVector3f::Dot(a, b)`, accessors...).
//!
//! 1. [`probe`]: for the inline functions of the unit's context headers, generate one probe
//!    function per inline (calling it with parameters as operands), compile them all in the unit
//!    context with the unit flags, and lift each probe with mwdec-lift: the lifted body is the
//!    inline's canonical expansion as IR, with parameters as holes ([`template::Template`]).
//! 2. [`matcher`]: find occurrences of those templates in a target function's IR (tree matching
//!    modulo temps, so scheduling and register allocation don't matter) and replace them with
//!    the call.

pub mod addr;
pub mod buffers;
pub mod cflow;
pub mod complete;
pub mod composed;
pub mod constfold;
pub mod ctors;
pub mod defctor;
pub mod groups;
pub mod iterloops;
pub mod matcher;
pub mod objlocals;
pub mod post;
pub mod probe;
pub mod reflocal;
pub mod relevance;
pub mod safety;
pub mod scalarinl;
pub mod ser;
pub mod session;
pub mod stmtinl;
pub mod stmts;
pub mod template;
pub mod util;
pub mod walk;
pub mod walkptr;

use mwdec_core::{ObjectFile, TypeDb};
use mwdec_lift::IrFunction;
pub use template::Template;

#[derive(Clone, Debug, Default)]
pub struct InlineLib {
    pub templates: Vec<Template>,
    /// (probe name, reason) of probes that compiled but gave no template.
    pub rejected: Vec<(String, String)>,
    pub probes_total: usize,
    pub probes_compiled: usize,
    /// Names of inlines with side effects (statement and mutator templates); folded calls of
    /// the others are pure values.
    pub effectful: std::collections::HashSet<String>,
    /// Default constructions of the classes the target's constructors hold as members/bases.
    pub default_ctors: defctor::DefCtors,
    /// Copy constructions of the same classes.
    pub copy_ctors: defctor::CopyCtors,
}

/// Template libraries of other lifter builds unused for a day are deleted (each build has its own
/// directory; the compiled probe objects they derive from are kept).
/// Marker file a template generation's directory gets on every use (its mtime = last use).
const LAST_USED: &str = "last_used";

fn prune_generations(root: &std::path::Path, gen: &str, keep: &std::path::Path) {
    // once per process, in the background (deleting a generation can take minutes; a draft
    // request must not wait for it)
    static ONCE: std::sync::Once = std::sync::Once::new();
    let (root, gen, keep) = (root.to_path_buf(), gen.to_string(), keep.to_path_buf());
    ONCE.call_once(move || {
        // (marks this generation as in use: a generation only read from is pruned by its last use)
        let _ = std::fs::create_dir_all(&keep).and_then(|_| std::fs::write(keep.join(LAST_USED), b""));
        let _ = std::thread::Builder::new().name("tcache-prune".into()).spawn(move || prune_old(&root, &gen, &keep));
    });
}

fn prune_old(root: &std::path::Path, gen: &str, keep: &std::path::Path) {
    let Ok(rd) = std::fs::read_dir(root) else { return };
    let prefix = format!("{gen}_");
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with(&prefix) || p == keep {
            continue;
        }
        // unused for a day (each lifter / context change starts a generation; disk is tight)
        let old = std::fs::metadata(p.join(LAST_USED)).or_else(|_| std::fs::metadata(p.join("packs"))).or_else(|_| e.metadata()).and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|d| d.as_secs() > 86400);
        if old {
            let _ = std::fs::remove_dir_all(&p);
        }
    }
}

/// Hash of the lifter / context / template-extraction sources (build time, see build.rs).
pub const TEMPLATE_INPUTS_HASH: &str = env!("MWDEC_TEMPLATE_INPUTS_HASH");

/// Version of the probe generator / template extraction (part of the cache key).
pub const TEMPLATE_VERSION: &str = "mwdec-inline templates v5";

/// Caches shared across units and runs: templates by probe text (+ compiler and flags), and
/// probes that failed to compile in one context (+ context hash).
pub struct ProbeCache {
    packs: ser::PackCache,
    objects: ser::ObjCache,
    failures: ser::Cache,
    defctors: defctor::DefCache,
}

impl ProbeCache {
    /// `flags` identifies compiler + flags; `context` the unit's context TU.
    pub fn new(flags: &str, _context: &str) -> ProbeCache {
        // one directory per generator version and template-input sources (lifter, context,
        // extraction): stale generations can be deleted whole
        let gen: String = TEMPLATE_VERSION.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let base = session::work_dir().join("tcache").join(&gen);
        // (`MWDI_TCACHE_SALT`: a separate template generation, for measurements)
        let salt = std::env::var("MWDI_TCACHE_SALT").map(|s| format!("_{s}")).unwrap_or_default();
        let dir = session::work_dir().join("tcache").join(format!("{gen}_{}{salt}", &TEMPLATE_INPUTS_HASH[..12]));
        prune_generations(&session::work_dir().join("tcache"), &gen, &dir);
        ProbeCache {
            packs: ser::PackCache::new(dir.join("packs"), format!("{TEMPLATE_VERSION}\n{flags}")),
            // a probe failing in one context (access, instantiation errors) fails elsewhere too
            // compiled probes and compile failures don't depend on the lifter
            objects: ser::ObjCache::new(base.join("objects"), format!("{TEMPLATE_VERSION}
{flags}")),
            failures: ser::Cache::new(base.join("failures"), format!("{TEMPLATE_VERSION}\n{flags}")),
            defctors: defctor::DefCache::new(dir.join("defctors"), format!("{TEMPLATE_VERSION}\n{flags}")),
        }
    }
}

/// Build the template library for a context: `compile` compiles a TU body in the unit context.
/// With `target`, only inlines of classes the target object refers to are probed.
pub fn build_library_for(db: &TypeDb, target: Option<&ObjectFile>, cache: Option<&ProbeCache>, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> InlineLib {
    let rel = target.map(|o| relevance::relevant_classes(o, db));
    let mut lib = build_library_rel(db, rel.as_ref(), cache, compile);
    if let Some(o) = target {
        let mut classes = defctor::wanted(o.functions.iter().map(|f| f.name.as_str()), db);
        for c in defctor::wanted_arg_classes(o, db) {
            if !classes.contains(&c) {
                classes.push(c);
            }
        }
        lib.default_ctors = defctor::build(db, &classes, cache.map(|c| &c.defctors), compile);
        let pairs = defctor::wanted_pairs(o.functions.iter().map(|f| f.name.as_str()), db);
        lib.copy_ctors = defctor::build_copies(db, &pairs, cache.map(|c| &c.defctors), compile);
    }
    lib
}

/// Library for the classes one function can touch (see [`relevance::relevant_for_function`]).
pub fn build_library_for_function(db: &TypeDb, f: &mwdec_core::Function, cache: Option<&ProbeCache>, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> InlineLib {
    let rel = relevance::relevant_for_function(f, db);
    let mut lib = build_library_rel(db, Some(&rel), cache, compile);
    let classes = defctor::wanted(std::iter::once(f.name.as_str()), db);
    lib.default_ctors = defctor::build(db, &classes, cache.map(|c| &c.defctors), compile);
    let pairs = defctor::wanted_pairs(std::iter::once(f.name.as_str()), db);
    lib.copy_ctors = defctor::build_copies(db, &pairs, cache.map(|c| &c.defctors), compile);
    lib
}

fn probe_key(p: &probe::Probe) -> String {
    p.decl.inline_body.as_deref().unwrap_or("").replacen(&p.name, "__P", 1)
}

/// Template of probe `p` from its lifted function.
fn template_of(p: &probe::Probe, obj: &ObjectFile, f: &mwdec_core::Function, db: &TypeDb, db2: &TypeDb) -> Result<Template, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mwdec_lift::lift_function(obj, f, Some(db2)))) {
        Ok(Ok(ir)) => template::from_probe(p, &ir, db).and_then(|mut t| {
            t.fixed = p.fixed.clone();
            // a specialisation whose expansion no longer reads anything but constants and the
            // fixed holes' neighbours is too generic to name (`f(x, false)` == `x`)
            if !t.fixed.is_empty() && t.ops == 0 && matches!(t.shape, template::Shape::Scalar(_)) {
                return Err("trivial specialisation".into());
            }
            // (only trivial accessors depend on them: other inlines are told apart by their
            // code already)
            if p.needs_dead {
                t.dead = template::dead_patterns(&ir).unwrap_or_default();
                if t.dead.is_empty() {
                    return Err("accessor without dead stores".into());
                }
            }
            template::name_literals(&mut t, &ir);
            Ok(t)
        }),
        _ => Err("lift".into()),
    }
}

fn build_library_rel(db: &TypeDb, rel: Option<&std::collections::HashSet<String>>, cache: Option<&ProbeCache>, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> InlineLib {
    let cache = if std::env::var("MWDI_NO_TCACHE").is_ok() { None } else { cache };
    let probes = probe::generate(db, rel);
    let total = probes.len();
    let t0 = std::time::Instant::now();
    // outcomes known from earlier units / runs: templates (this lifter), else the compiled
    // probe function to lift again, else compile
    let mut known: Vec<(usize, Result<Template, String>)> = vec![];
    let keys: Vec<String> = probes.iter().map(probe_key).collect();
    // the whole library for this lifter, if built before
    let pack = cache.and_then(|c| c.packs.get(&keys));
    let from_pack = pack.is_some();
    if let Some(pack) = &pack {
        for (k, key) in keys.iter().enumerate() {
            if let Some(r) = pack.get(key) {
                known.push((k, r.clone()));
            }
        }
    }
    let mut relift: Vec<(usize, probe::Probe, ObjectFile)> = vec![];
    let mut todo = vec![];
    for (k, p) in probes.into_iter().enumerate().filter(|_| !from_pack) {
        let key = keys[k].clone();
        if let Some(c) = cache {
            match c.objects.get(&key) {
                Some(Some(o)) => {
                    relift.push((k, p, o));
                    continue;
                }
                Some(None) => {
                    known.push((k, Err(format!("{}: no function", p.sig.qualified_name))));
                    continue;
                }
                None => {}
            }
            if c.failures.get(&key).is_some() {
                continue;
            }
        }
        todo.push((k, p));
    }
    let order: std::collections::HashMap<String, usize> = todo.iter().map(|(k, p)| (p.name.clone(), *k)).collect();
    let todo_probes: Vec<probe::Probe> = todo.into_iter().map(|(_, p)| p).collect();
    let attempted: Vec<probe::Probe> = todo_probes.clone();
    let ncomp = std::sync::atomic::AtomicUsize::new(0);
    let counted = |code: &str| {
        ncomp.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        compile(code)
    };
    let chunks = if todo_probes.is_empty() { vec![] } else { probe::compile(todo_probes, &counted) };
    let t_compile = t0.elapsed().as_secs_f64();
    // the lifter needs the probes' declarations: a copy of the TypeDb, only when something is
    // lifted (everything from the template cache: no copy)
    let db2 = (!chunks.is_empty() || !relift.is_empty()).then(|| {
        let mut d = db.clone();
        for (_, ps) in &chunks {
            probe::inject_decls(&mut d, ps);
        }
        let ps: Vec<probe::Probe> = relift.iter().map(|(_, p, _)| p.clone()).collect();
        probe::inject_decls(&mut d, &ps);
        d
    });
    let t_db = t0.elapsed().as_secs_f64();
    let compiled: std::collections::HashSet<String> = chunks.iter().flat_map(|c| c.1.iter().map(|p| p.name.clone())).collect();
    if let Some(c) = cache {
        for p in attempted.iter().filter(|p| !compiled.contains(&p.name)) {
            c.failures.put(&probe_key(p), &Err("compile".into()));
        }
    }
    let mut lib = InlineLib { probes_total: total, probes_compiled: compiled.len(), ..Default::default() };
    for (obj, probes) in &chunks {
        for p in probes {
            let prefix = format!("{}__", p.name);
            let f = obj.functions.iter().find(|f| f.name.starts_with(&prefix));
            if let Some(c) = cache {
                c.objects.put(&probe_key(p), &f.map(|f| ser::minimal_object(obj, f, &p.name)));
            }
            let r: Result<Template, String> = match f {
                None => Err("no function".into()),
                Some(f) => template_of(p, obj, f, db, db2.as_ref().unwrap()),
            };
            known.push((order[&p.name], r.map_err(|e| format!("{}: {e}", p.sig.qualified_name))));
        }
    }
    let t_lift0 = t0.elapsed().as_secs_f64();
    let mut t_l = 0.0;
    // compiled before, lifted again (the probe function is stored as `__P...`)
    for (k, p, mut o) in relift {
        for f in o.functions.iter_mut() {
            if let Some(rest) = f.name.strip_prefix("__P") {
                f.name = format!("{}{rest}", p.name);
            }
        }
        for s in o.symbols.iter_mut() {
            if let Some(rest) = s.name.strip_prefix("__P") {
                s.name = format!("{}{rest}", p.name);
            }
        }
        let tl = std::time::Instant::now();
        let r: Result<Template, String> = match o.functions.first() {
            None => Err("no function".into()),
            Some(f) => template_of(&p, &o, f, db, db2.as_ref().unwrap()),
        };
        t_l += tl.elapsed().as_secs_f64();
        known.push((k, r.map_err(|e| format!("{}: {e}", p.sig.qualified_name))));
    }
    // probe order (deterministic whatever came from the cache)
    known.sort_by_key(|(k, _)| *k);
    if let (Some(c), false) = (cache, from_pack) {
        let outcomes: Vec<(String, Result<Template, String>)> = known.iter().map(|(k, r)| (keys[*k].clone(), r.clone())).collect();
        c.packs.put(&keys, &outcomes);
    }
    if from_pack {
        lib.probes_compiled = known.len();
    }
    for (_, r) in known {
        match r {
            Ok(t) => lib.templates.push(t),
            Err(e) => lib.rejected.push((e.split(':').next().unwrap_or("").to_string(), e)),
        }
    }
    // bool inlines also in their condition form (`a && (b || c)` inside an `if`)
    let conds: Vec<Template> = lib.templates.iter().filter_map(cflow::bool_template).collect();
    lib.templates.extend(conds);
    lib.effectful = lib.templates.iter().filter(|t| matches!(t.shape, template::Shape::Stmts { .. } | template::Shape::Mutate { .. })).map(|t| t.name.clone()).collect();
    if let Ok(f) = std::env::var("MWDI_LIST") {
        for t in lib.templates.iter().filter(|t| t.name.contains(f.as_str())) {
            eprintln!("template {} ret_ref={} holes={:?} {:?}", t.name, t.ret_ref, t.holes, t.shape);
        }
        for (n, e) in lib.rejected.iter().filter(|(n, _)| n.contains(f.as_str())) {
            eprintln!("rejected {n}: {}", e.chars().take(200).collect::<String>());
        }
    }
    if std::env::var("MWDI_DEBUG").is_ok() {
        eprintln!("probe library: {} probes, {} to compile, {} compiles {:.1}s, total {:.1}s", total, attempted.len(), ncomp.load(std::sync::atomic::Ordering::Relaxed), t_compile, t0.elapsed().as_secs_f64());
        eprintln!("  db copy at {t_db:.1}s, relift from {t_lift0:.1}s, lifting {t_l:.1}s");
    }
    lib
}

/// Build the template library for a context (every inline of the context headers).
pub fn build_library(db: &TypeDb, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> InlineLib {
    build_library_for(db, None, None, compile)
}

pub fn library_for(s: &session::UnitSession) -> InlineLib {
    match &s.db {
        Some(db) => {
            let cache = ProbeCache::new(&format!("{}\n{}", s.mwcc.compiler, s.cflags.join(" ")), &s.context);
            build_library_for(db, Some(&s.obj), Some(&cache), &|code| s.compile(code))
        }
        None => InlineLib::default(),
    }
}

/// Rewrite recognised inline expansions in `ir`; returns the number of rewrites.
pub fn apply(ir: &mut IrFunction, lib: &InlineLib, db: &TypeDb) -> usize {
    let t0 = std::time::Instant::now();
    let n = matcher::apply(ir, lib, db);
    if std::env::var("MWDI_DEBUG").is_ok() {
        eprintln!("inline apply {}: {n} rewrites, {} templates, {} steps, {:.2}s [{}]", ir.symbol, lib.templates.len(), matcher::steps(), t0.elapsed().as_secs_f64(), util::prof::report());
    }
    n
}
