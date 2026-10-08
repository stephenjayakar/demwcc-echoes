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
pub mod ctors;
pub mod defctor;
pub mod groups;
pub mod iterloops;
pub mod matcher;
pub mod objlocals;
pub mod post;
pub mod probe;
pub mod relevance;
pub mod safety;
pub mod ser;
pub mod session;
pub mod stmts;
pub mod template;
pub mod util;
pub mod walk;

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

/// Version of the probe generator / template extraction (part of the cache key).
pub const TEMPLATE_VERSION: &str = "mwdec-inline templates v5";

/// Caches shared across units and runs: templates by probe text (+ compiler and flags), and
/// probes that failed to compile in one context (+ context hash).
pub struct ProbeCache {
    templates: ser::Cache,
    failures: ser::Cache,
    defctors: defctor::DefCache,
}

impl ProbeCache {
    /// `flags` identifies compiler + flags; `context` the unit's context TU.
    pub fn new(flags: &str, _context: &str) -> ProbeCache {
        // one directory per generator version: stale generations can be deleted whole
        let gen: String = TEMPLATE_VERSION.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let dir = session::work_dir().join("tcache").join(gen);
        ProbeCache {
            templates: ser::Cache::new(dir.join("templates"), format!("{TEMPLATE_VERSION}\n{flags}")),
            // a probe failing in one context (access, instantiation errors) fails elsewhere too
            failures: ser::Cache::new(dir.join("failures"), format!("{TEMPLATE_VERSION}\n{flags}")),
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
        let classes = defctor::wanted(o.functions.iter().map(|f| f.name.as_str()), db);
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
    let key = p.decl.inline_body.as_deref().unwrap_or("").replacen(&p.name, "__P", 1);
    // probes with stack-passed arguments: lifted again since float stack slots became one word
    let floats = p.params.iter().filter(|t| matches!(util::strip(t), mwdec_core::Type::Float { .. })).count();
    if floats > 8 || p.params.len() - floats > 8 {
        return format!("{key}
stack-args 2");
    }
    key
}

fn build_library_rel(db: &TypeDb, rel: Option<&std::collections::HashSet<String>>, cache: Option<&ProbeCache>, compile: &(dyn Fn(&str) -> Result<ObjectFile, String> + Sync)) -> InlineLib {
    let cache = if std::env::var("MWDI_NO_TCACHE").is_ok() { None } else { cache };
    let probes = probe::generate(db, rel);
    let total = probes.len();
    let t0 = std::time::Instant::now();
    // outcomes known from earlier units / runs
    let mut known: Vec<(usize, Result<Template, String>)> = vec![];
    let mut todo = vec![];
    for (k, p) in probes.into_iter().enumerate() {
        let key = probe_key(&p);
        if let Some(c) = cache {
            if let Some(r) = c.templates.get(&key) {
                known.push((k, r.map_err(|e| format!("{}: {e}", p.sig.qualified_name))));
                continue;
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
    // the lifter needs the probes' declarations: a copy of the TypeDb, only when something
    // was compiled (everything from the cache: no copy)
    let db2 = (!chunks.is_empty()).then(|| {
        let mut d = db.clone();
        for (_, ps) in &chunks {
            probe::inject_decls(&mut d, ps);
        }
        d
    });
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
            let r: Result<Template, String> = match obj.functions.iter().find(|f| f.name.starts_with(&prefix)) {
                None => Err("no function".into()),
                Some(f) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| mwdec_lift::lift_function(obj, f, db2.as_ref()))) {
                    Ok(Ok(ir)) => template::from_probe(p, &ir, db),
                    _ => Err("lift".into()),
                },
            };
            if let Some(c) = cache {
                c.templates.put(&probe_key(p), &r);
            }
            known.push((order[&p.name], r.map_err(|e| format!("{}: {e}", p.sig.qualified_name))));
        }
    }
    // probe order (deterministic whatever came from the cache)
    known.sort_by_key(|(k, _)| *k);
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
    if std::env::var("MWDI_DEBUG").is_ok() {
        eprintln!("probe library: {} probes, {} to compile, {} compiles {:.1}s, total {:.1}s", total, attempted.len(), ncomp.load(std::sync::atomic::Ordering::Relaxed), t_compile, t0.elapsed().as_secs_f64());
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
        eprintln!("inline apply {}: {n} rewrites, {} templates, {:.2}s [{}]", ir.symbol, lib.templates.len(), t0.elapsed().as_secs_f64(), util::prof::report());
    }
    n
}
