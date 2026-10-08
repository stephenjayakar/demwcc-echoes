//! Inline-fold diagnostics (`mwdec inline-why`): for a function whose draft keeps header inline
//! expansions unfolded, which inline templates *almost* match what is left, and why they don't.
//!
//! The matcher itself (mwdec-inline) is not consulted for reasons; this is an independent,
//! approximate comparison of memory-access fingerprints, so the diagnosis can't disturb it:
//!
//! - a template's fingerprint: the accesses its pattern makes through each object parameter
//!   (path of member offsets from the object, access size, load or store) and the functions it
//!   calls;
//! - the draft's fingerprint, after inline folding: the same accesses grouped by root (a
//!   variable, `this`, a global, a stack object), each with the index of its statement;
//! - every template whose object parameter's class can sit at some offset of a root's class is
//!   compared with that root at that offset; the best candidates are classified:
//!
//! | reason | meaning |
//! |---|---|
//! | `complete:type` | every access is there, one with another access type (float vs int, width) |
//! | `complete:interleaved` | every access is there, other statements sit between them |
//! | `complete:nested` | every access is there, the object is a member at a nonzero offset |
//! | `complete:other` | every access is there; the matcher rejected it for another reason (temporaries, holes, safety) |
//! | `partial:missing-store` | some stores of the template are missing (elided or dead stores) |
//! | `partial:missing-load` | some loads are missing (a value reused / CSE'd, a composed accessor) |
//! | `partial:missing-call` | the accesses match but a call of the template isn't there |
//!
//! TRAIN-split diagnostics only; nothing here feeds the draft.
use crate::search_cmds::{externs_for, unit_inputs, Compilers, UnitInputs};
use anyhow::{anyhow, Result};
use mwdec_core::{Function, TypeDb};
use mwdec_inline::template::{HoleKind, Shape, Template};
use mwdec_lift::{Callee, Expr, Stmt};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

/// One memory access: offsets from the root (outer first; all but the last dereference a
/// pointer member), size, store.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Access {
    path: Vec<i32>,
    size: u32,
    store: bool,
    float: bool,
}

/// `(root, path)` of a load/store address: `Load { base: Load { base: root, o1 }, o2 }` ->
/// (root, [o1, o2]).
fn access_path(e: &Expr) -> Option<(&Expr, Vec<i32>)> {
    match e {
        Expr::Load { base, offset, .. } | Expr::Member { base, offset, .. } => {
            let mut path = vec![*offset];
            let mut b: &Expr = base;
            while let Expr::Load { base: bb, offset: o, .. } = b {
                path.insert(0, *o);
                b = bb;
                if path.len() > 3 {
                    return None;
                }
            }
            let b = match b {
                Expr::Cast { e, .. } => e,
                x => x,
            };
            Some((b, path))
        }
        _ => None,
    }
}

fn size_of(t: &mwdec_core::Type) -> (u32, bool) {
    (mwdec_lift::scalar_size(t).unwrap_or(0), mwdec_lift::is_float(t))
}

fn callee_key(c: &Callee) -> Option<String> {
    match c {
        // (by name: probes and targets spell the symbol differently or not at all)
        Callee::Direct { sig, .. } | Callee::Method { sig, .. } => Some(sig.qualified_name.clone()),
        Callee::Virtual { vtable_offset, .. } => Some(format!("virtual+{vtable_offset}")),
        Callee::Indirect(_) => None,
    }
}

/// Accesses of `body` grouped by root (keyed by the root's debug text), with statement indexes;
/// and the callees.
#[derive(Default)]
struct Fingerprint {
    roots: BTreeMap<String, (Expr, Vec<(Access, usize)>)>,
    calls: HashSet<String>,
}

fn collect_expr(e: &Expr, store: bool, seq: usize, fp: &mut Fingerprint) {
    if let Some((root, path)) = access_path(e) {
        let (size, float) = match e {
            Expr::Load { ty, .. } | Expr::Member { ty, .. } => size_of(ty),
            _ => (0, false),
        };
        let key = format!("{root:?}");
        let ent = fp.roots.entry(key).or_insert_with(|| (root.clone(), vec![]));
        ent.1.push((Access { path, size, store, float }, seq));
    }
    if let Expr::Call { callee, .. } = e {
        if let Some(k) = callee_key(callee) {
            fp.calls.insert(k);
        }
    }
}

fn collect_stmts(body: &[Stmt], seq: &mut usize, fp: &mut Fingerprint) {
    for s in body {
        *seq += 1;
        let here = *seq;
        match s {
            Stmt::Assign { dst, src } => {
                collect_expr(dst, true, here, fp);
                // the destination's address computation and the source are reads
                let mut inner = vec![];
                if let Expr::Load { base, .. } | Expr::Member { base, .. } = dst {
                    base.walk(&mut |x| inner.push(x.clone()));
                }
                src.walk(&mut |x| inner.push(x.clone()));
                for x in inner {
                    collect_expr(&x, false, here, fp);
                }
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => e.walk(&mut |x| collect_expr(x, false, here, fp)),
            Stmt::If { cond, then, els } => {
                cond.walk(&mut |x| collect_expr(x, false, here, fp));
                collect_stmts(then, seq, fp);
                collect_stmts(els, seq, fp);
            }
            Stmt::While { cond, body } | Stmt::DoWhile { body, cond } => {
                cond.walk(&mut |x| collect_expr(x, false, here, fp));
                collect_stmts(body, seq, fp);
            }
            Stmt::For { init, cond, step, body } => {
                collect_stmts(init, seq, fp);
                cond.walk(&mut |x| collect_expr(x, false, here, fp));
                collect_stmts(body, seq, fp);
                collect_stmts(step, seq, fp);
            }
            Stmt::Switch { e, cases } => {
                e.walk(&mut |x| collect_expr(x, false, here, fp));
                for c in cases {
                    collect_stmts(&c.body, seq, fp);
                }
            }
            _ => {}
        }
    }
}

/// A template's accesses through one object parameter, and its callees.
struct TemplatePrint {
    hole: usize,
    class: String,
    accesses: Vec<Access>,
    calls: Vec<String>,
}

fn template_prints(t: &Template) -> Vec<TemplatePrint> {
    let mut fp = Fingerprint::default();
    let mut seq = 0;
    match &t.shape {
        Shape::Scalar(e) => e.walk(&mut |x| collect_expr(x, false, 0, &mut fp)),
        Shape::Object { comps, .. } => {
            for c in comps {
                c.pat.walk(&mut |x| collect_expr(x, false, 0, &mut fp));
            }
        }
        Shape::Mutate { hole, comps } => {
            for c in comps {
                let dst = Expr::Load { base: Box::new(Expr::Var(*hole)), offset: c.off, ty: c.ty.clone() };
                collect_expr(&dst, true, 0, &mut fp);
                c.pat.walk(&mut |x| collect_expr(x, false, 0, &mut fp));
            }
        }
        Shape::Stmts { stmts, result } => {
            collect_stmts(stmts, &mut seq, &mut fp);
            if let Some(r) = result {
                r.walk(&mut |x| collect_expr(x, false, 0, &mut fp));
            }
        }
    }
    let calls: Vec<String> = fp.calls.iter().cloned().collect();
    let mut out = vec![];
    for (h, k) in t.holes.iter().enumerate() {
        let HoleKind::Obj { class, .. } = k else { continue };
        let key = format!("{:?}", Expr::Var(h));
        let empty = vec![];
        let acc = fp.roots.get(&key).map_or(&empty, |(_, a)| a);
        let mut accesses: Vec<Access> = acc.iter().map(|(a, _)| a.clone()).collect();
        accesses.sort();
        accesses.dedup();
        if accesses.len() + calls.len() >= 2 || !calls.is_empty() {
            out.push(TemplatePrint { hole: h, class: class.clone(), accesses, calls: calls.clone() });
        }
    }
    out
}

/// One near match.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub template: String,
    pub root: String,
    pub offset: i32,
    pub matched: usize,
    pub total: usize,
    pub reason: String,
    pub detail: String,
}

impl Candidate {
    fn json(&self) -> serde_json::Value {
        serde_json::json!({"template": self.template, "root": self.root, "offset": self.offset, "matched": self.matched, "total": self.total, "reason": self.reason, "detail": self.detail})
    }
}

fn root_class(root: &Expr, vars: &[mwdec_lift::Var], db: &TypeDb) -> Option<String> {
    let t = mwdec_lift::types::ty_of(root, vars);
    let p = mwdec_lift::pointee(&t).cloned().unwrap_or(t);
    let r = mwdec_lift::types::resolve(Some(db), &p).into_owned();
    mwdec_lift::named(&r).map(|s| s.to_string())
}

/// Near matches of the library's templates in `ir` (after folding), best first.
pub fn diagnose(ir: &mwdec_lift::IrFunction, lib: &mwdec_inline::InlineLib, db: &TypeDb, max: usize) -> Vec<Candidate> {
    let mut fp = Fingerprint::default();
    let mut seq = 0;
    collect_stmts(&ir.body, &mut seq, &mut fp);
    let mut cands: Vec<(f64, usize, Candidate)> = vec![];
    let dbg = std::env::var("MWDEC_IW_TEMPLATE").ok();
    for t in &lib.templates {
        if dbg.as_deref().is_some_and(|n| t.name.contains(n)) {
            eprintln!("template {}: holes {:?}
  shape {:?}", t.name, t.holes, t.shape);
            for tp in template_prints(t) {
                eprintln!("  print hole {} {}: accesses {:?} calls {:?}", tp.hole, tp.class, tp.accesses, tp.calls);
            }
            eprintln!("  target calls {:?}", fp.calls);
        }
        for tp in template_prints(t) {
            if tp.accesses.is_empty() {
                // a template that only calls (`out.WriteReal32(x)` -> `out.DoPut(&tmp, 4)`):
                // its callees are all there, but the expansion wasn't recognised
                if tp.calls.iter().all(|c| fp.calls.contains(c)) && !tp.calls.is_empty() {
                    let total = tp.calls.len();
                    cands.push((1.0, total, Candidate { template: t.name.clone(), root: String::new(), offset: 0, matched: total, total, reason: "complete:calls-only".into(), detail: tp.calls.join(",") }));
                }
                continue;
            }
            for (key, (root, acc)) in &fp.roots {
                let rc = root_class(root, &ir.vars, db);
                // offsets where an object of the template's class can sit in this root
                let mut deltas: Vec<i32> = acc.iter().filter(|(a, _)| a.path.len() == tp.accesses[0].path.len()).map(|(a, _)| a.path[0] - tp.accesses[0].path[0]).collect();
                deltas.sort();
                deltas.dedup();
                for d in deltas {
                    // untyped roots (raw pointers): only exact access shapes, no type differences
                    let typed = rc.is_some();
                    let class_ok = match &rc {
                        Some(r) => (d == 0 && sig_eq(r, &tp.class)) || mwdec_inline::matcher::class_at_pub(db, r, d, &tp.class),
                        None => d >= 0 && tp.accesses.len() >= 2,
                    };
                    if !class_ok {
                        continue;
                    }
                    let shifted = |a: &Access| {
                        let mut p = a.path.clone();
                        p[0] += d;
                        p
                    };
                    let mut matched = 0;
                    let mut type_diff = 0;
                    let mut seqs = vec![];
                    let (mut miss_store, mut miss_load) = (0, 0);
                    for a in &tp.accesses {
                        let p = shifted(a);
                        match acc.iter().find(|(b, _)| b.path == p && b.store == a.store) {
                            Some((b, s)) => {
                                if b.size != a.size || b.float != a.float {
                                    if !typed {
                                        continue;
                                    }
                                    type_diff += 1;
                                }
                                matched += 1;
                                seqs.push(*s);
                            }
                            None if a.store => miss_store += 1,
                            None => miss_load += 1,
                        }
                    }
                    let calls_missing: Vec<&String> = tp.calls.iter().filter(|c| !fp.calls.contains(*c)).collect();
                    let total = tp.accesses.len() + tp.calls.len();
                    let got = matched + tp.calls.len() - calls_missing.len();
                    if got * 2 < total || matched == 0 {
                        continue;
                    }
                    seqs.sort();
                    seqs.dedup();
                    let span = seqs.last().copied().unwrap_or(0) - seqs.first().copied().unwrap_or(0) + 1;
                    let (reason, detail) = if got == total {
                        if type_diff > 0 {
                            ("complete:type", format!("{type_diff} access(es) of another type"))
                        } else if span > seqs.len() + 1 {
                            ("complete:interleaved", format!("{} statements span {span}", seqs.len()))
                        } else if d != 0 {
                            ("complete:nested", format!("object at +0x{d:x}"))
                        } else if !typed {
                            ("complete:untyped-root", "the object is reached through an untyped pointer".to_string())
                        } else {
                            ("complete:other", String::new())
                        }
                    } else if miss_store > 0 {
                        ("partial:missing-store", format!("{miss_store} store(s) missing"))
                    } else if miss_load > 0 {
                        ("partial:missing-load", format!("{miss_load} load(s) missing"))
                    } else {
                        ("partial:missing-call", calls_missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","))
                    };
                    let score = got as f64 / total as f64;
                    cands.push((
                        score,
                        total,
                        Candidate { template: t.name.clone(), root: short(key), offset: d, matched: got, total, reason: reason.into(), detail },
                    ));
                }
            }
        }
    }
    // best: highest share, then the most specific (largest fingerprint)
    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(b.1.cmp(&a.1)));
    let mut seen = HashSet::new();
    cands.into_iter().filter(|c| seen.insert((c.2.template.clone(), c.2.root.clone()))).take(max).map(|c| c.2).collect()
}

fn sig_eq(a: &str, b: &str) -> bool {
    mwdec_lift::sig::norm_name(a) == mwdec_lift::sig::norm_name(b)
}

fn short(s: &str) -> String {
    s.chars().take(60).collect()
}

fn diagnose_fn(ui: &UnitInputs, f: &Function, max: usize) -> Result<Vec<Candidate>> {
    let db = ui.db.as_ref().ok_or_else(|| anyhow!("no TypeDb"))?;
    let mut ir = mwdec_lift::lift_function(ui.lift_obj.as_ref().unwrap_or(&ui.target), f, Some(db))?;
    let lib = ui.inlines.get(ui, &format!("{}\n{}", ui.mwcc.compiler, ui.ctx.cflags.join(" ")));
    mwdec_inline::apply(&mut ir, &lib, db);
    Ok(diagnose(&ir, &lib, db, max))
}

/// `mwdec inline-why`: one function (`unit`, `symbol`), or every function of a `--list` JSONL
/// (with `--out` rows and a ranked summary of the best candidate's reason per function).
pub fn cmd_inline_why(root: &Path, work: &Path, unit: Option<String>, symbol: Option<String>, list: Option<std::path::PathBuf>, out: Option<std::path::PathBuf>) -> Result<()> {
    let p = crate::load_project(root)?;
    let mut items: Vec<(String, String)> = vec![];
    if let (Some(u), Some(s)) = (unit, symbol) {
        items.push((u, s));
    }
    if let Some(l) = list {
        for line in std::fs::read_to_string(&l)?.lines() {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                if let (Some(u), Some(s)) = (v["unit"].as_str(), v["symbol"].as_str()) {
                    items.push((u.into(), s.into()));
                }
            }
        }
    }
    let mut by_unit: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (u, s) in items {
        by_unit.entry(u).or_default().push(s);
    }
    let cc = Compilers::new(root, work, 4).with_fast_workers(0);
    let mut outf = match &out {
        Some(o) => Some(std::fs::File::create(o)?),
        None => None,
    };
    let single = outf.is_none();
    let mut summary: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
    let mut templates: HashMap<String, usize> = HashMap::new();
    let mut n = 0;
    for (uname, syms) in by_unit {
        let Some(u) = p.unit(&uname) else { continue };
        let (t_ext, _) = externs_for(&p, &u.name);
        let ui = match unit_inputs(&p, u, &cc, &t_ext.objs, true, Some(&t_ext)) {
            Ok(ui) => ui,
            Err(e) => {
                eprintln!("{uname}: {e:#}");
                continue;
            }
        };
        for s in syms {
            let Some(f) = mwdec_obj::find_function(&ui.target, &s) else { continue };
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| diagnose_fn(&ui, f, 5)));
            let cands = match res {
                Ok(Ok(c)) => c,
                _ => continue,
            };
            n += 1;
            let best = cands.first();
            let reason = best.map_or("none".to_string(), |c| c.reason.clone());
            let e = summary.entry(reason).or_default();
            e.0 += 1;
            if e.1.len() < 6 {
                e.1.push(format!("{} {} [{}]", uname, s, best.map_or(String::new(), |c| c.template.clone())));
            }
            if let Some(c) = best {
                *templates.entry(c.template.clone()).or_default() += 1;
            }
            if single {
                println!("{uname} {s}");
                for c in &cands {
                    println!("  {:.0}% {} on {} +0x{:x}: {} {}", 100.0 * c.matched as f64 / c.total as f64, c.template, c.root, c.offset, c.reason, c.detail);
                }
                if cands.is_empty() {
                    println!("  no template matches half of its fingerprint");
                }
            }
            if let Some(f) = outf.as_mut() {
                use std::io::Write;
                let _ = writeln!(f, "{}", serde_json::json!({"unit": uname, "symbol": s, "candidates": cands.iter().map(|c| c.json()).collect::<Vec<_>>()}));
            }
        }
    }
    if !single {
        let mut v: Vec<_> = summary.into_iter().collect();
        v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        println!("best near match per function ({n} functions):");
        for (r, (k, ex)) in v {
            println!("{k:>5}  {r}");
            for e in ex {
                println!("         {e}");
            }
        }
        let mut t: Vec<_> = templates.into_iter().collect();
        t.sort_by(|a, b| b.1.cmp(&a.1));
        println!("templates most often nearly matched:");
        for (name, k) in t.into_iter().take(20) {
            println!("{k:>5}  {name}");
        }
    }
    Ok(())
}
