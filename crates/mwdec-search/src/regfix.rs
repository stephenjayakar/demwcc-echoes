//! Register-only repair: a bounded, deterministic neighbourhood search for candidates whose code
//! differs from the target only in register numbers (same instructions otherwise).
//!
//! Register choice follows from the variable structure of the source (regalloc.md): which values
//! are named locals and which are temporaries, declaration order, evaluation order of operands
//! and the live ranges the statements give each value. Those are exactly the knobs a small set of
//! operators turns: naming / inlining a temporary, reordering declarations, swapping commutative
//! operands, moving a statement, `const` on a by-value parameter, the type of a local. Instead of
//! sampling them at random (the main search), every site of every such operator is enumerated
//! (level 1), the real compiler scores each neighbour, and the best few neighbours are expanded
//! once more (level 2), until an exact match or the compile budget. Directed edits from the
//! compiler tracer (the real colouring of the candidate, [`crate::trace`]) go first when a tracer
//! is available.
//!
//! The result is deterministic for a given source and target (no random sampling beyond fixed
//! enumeration seeds), so it can run as part of drafting.
use crate::cst::normalize;
use crate::hints::RegHints;
use crate::ops::{self, Parsed};
use crate::rng::Rng;
use crate::score::{DiffProfile, Eval, Fitness, Scorer};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Operators that change register choice without changing the instruction mix much, in
/// priority order (most often the deciding edit first; train-split eval statistics).
pub const REG_OPS: &[&str] = &[
    "extract_temp",
    "commutative",
    "inline_var_all",
    "reorder_decls",
    "hoist_decl",
    "param_const",
    "local_type",
    "inline_temp",
    "swap_stmts",
    "move_stmt",
    "cse_temp",
    "merge_decl",
    "flip_compare",
    "compound_assign",
    "swap_stores",
    "chain_assign",
    "refer_to_var",
    "ref_local",
    "split_update",
    "swap_args",
    "vec_op",
    "associative",
    "if_to_ternary",
    "ternary_to_if",
    "hint_order",
    "hint_temp",
];

#[derive(Clone, Debug)]
pub struct RepairConfig {
    /// Real compiles at most (cache hits don't count).
    pub max_compiles: usize,
    /// Neighbours of level 1 expanded at level 2.
    pub beam: usize,
    /// Concurrent compiles.
    pub threads: usize,
    /// Enumeration seeds tried per operator (each picks a site at random; duplicates dropped).
    pub seeds: u64,
    /// Wall-clock cap (a safety net for slow contexts; the compile cap normally ends first).
    pub max_time: std::time::Duration,
}

impl Default for RepairConfig {
    fn default() -> Self {
        RepairConfig { max_compiles: 160, beam: 4, threads: 4, seeds: 48, max_time: std::time::Duration::from_secs(20) }
    }
}

#[derive(Clone, Debug)]
pub struct Repair {
    pub src: String,
    pub fitness: Fitness,
    /// Operators applied, in order.
    pub ops: Vec<&'static str>,
    pub compiles: usize,
}

/// The diff is register-only (or close: instructions moved by a different register choice), so
/// variable-structure edits are the right tools.
pub fn register_only(f: &Fitness) -> bool {
    let p: &DiffProfile = &f.profile;
    !f.exact && f.size_delta == 0 && p.inserted + p.deleted + p.substituted == 0 && p.stack == 0 && p.branch == 0 && p.reloc == 0 && p.reg > 0
}

/// Every distinct neighbour of `src` under [`REG_OPS`] (one operator application each), in
/// operator priority order. Deduplicated by normalized text against `seen`.
pub fn neighbours(src: &str, symbol: &str, hints: Option<&RegHints>, seeds: u64, seen: &mut HashSet<String>) -> Vec<(String, &'static str)> {
    let Some(p) = Parsed::new(src, symbol) else { return vec![] };
    let mut out = vec![];
    for name in REG_OPS {
        let Some(op) = ops::op_index(name) else { continue };
        if (*name == "hint_order" || *name == "hint_temp") && hints.is_none() {
            continue;
        }
        let mut dry = 0;
        for s in 0..seeds {
            let mut rng = Rng::new(0x5eed ^ (s * 0x9e37));
            let Some(c) = p.apply(op, &mut rng, hints) else {
                dry += 1;
                if dry > 6 && s >= 8 {
                    break;
                }
                continue;
            };
            if seen.insert(normalize(&c)) {
                out.push((c, ops::OPS[op].name));
                dry = 0;
            } else {
                dry += 1;
                if dry > 12 {
                    break;
                }
            }
        }
    }
    out
}

/// `src` with every split declaration (`T x; ... x = v;`) merged into its first assignment
/// (`T x = v;`): the drafts' style, which keeps the temp-inlining operators from applying.
pub fn merge_all_decls(src: &str, symbol: &str) -> Option<String> {
    let op = ops::op_index("merge_decl")?;
    let mut cur = src.to_string();
    for i in 0..64u64 {
        let Some(p) = Parsed::new(&cur, symbol) else { break };
        let mut rng = Rng::new(i);
        match p.apply(op, &mut rng, None) {
            Some(n) => cur = n,
            None => break,
        }
    }
    (cur != src).then_some(cur)
}

struct Scored {
    src: String,
    fit: Fitness,
    ops: Vec<&'static str>,
}

fn interleave<T: Clone>(lists: Vec<Vec<T>>) -> Vec<T> {
    let longest = lists.iter().map(|l| l.len()).max().unwrap_or(0);
    let mut out = vec![];
    for i in 0..longest {
        for l in &lists {
            if let Some(c) = l.get(i) {
                out.push(c.clone());
            }
        }
    }
    out
}

/// Compile `cands` (in order) with `threads` workers until one is exact or the budget runs out.
fn score_all(scorer: &Scorer, cands: Vec<(String, Vec<&'static str>)>, threads: usize, compiles: &AtomicUsize, max: usize, deadline: std::time::Instant) -> Vec<Scored> {
    let next = AtomicUsize::new(0);
    let done = AtomicBool::new(false);
    let out = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                if done.load(Ordering::Relaxed) || compiles.load(Ordering::Relaxed) >= max || std::time::Instant::now() >= deadline {
                    break;
                }
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some((src, ops)) = cands.get(i) else { break };
                let (e, ran) = scorer.eval(src);
                if ran {
                    compiles.fetch_add(1, Ordering::Relaxed);
                }
                if let Eval::Ok(f) = e {
                    if f.exact {
                        done.store(true, Ordering::Relaxed);
                    }
                    out.lock().unwrap().push((i, Scored { src: src.clone(), fit: f, ops: ops.clone() }));
                }
            });
        }
    });
    let mut v = out.into_inner().unwrap();
    v.sort_by_key(|x| x.0);
    v.into_iter().map(|x| x.1).collect()
}

/// Repair a register-only mismatch of `src` (already scored as `fit`). Returns the best
/// candidate found if it is better than `src` (exact when possible); `None` otherwise or when the
/// diff is not register-only.
pub fn repair(scorer: &Scorer, src: &str, fit: &Fitness, tracer: Option<&crate::trace::Tracer>, cfg: &RepairConfig) -> Option<Repair> {
    if !register_only(fit) {
        return None;
    }
    let symbol = scorer.symbol.clone();
    let hints = crate::hints::target_hints(scorer.tf);
    let hints = (!hints.is_empty()).then_some(&hints);
    let compiles = AtomicUsize::new(0);
    let deadline = std::time::Instant::now() + cfg.max_time;
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(normalize(src));
    // Level 1: tracer-directed edits first, then every operator site.
    let mut cands: Vec<(String, Vec<&'static str>)> = vec![];
    if let Some(tr) = tracer {
        if let Ok(fixes) = tr.fixes(src, &symbol, scorer.tf) {
            for fx in &fixes {
                for c in crate::trace::apply_fix(src, &symbol, fx) {
                    if seen.insert(normalize(&c)) {
                        cands.push((c, vec!["trace_fix"]));
                    }
                }
            }
        }
    }
    let merged = merge_all_decls(src, &symbol).filter(|m| seen.insert(normalize(m)));
    if let Some(m) = &merged {
        cands.push((m.clone(), vec!["merge_decl"]));
    }
    // Neighbours of the draft and of its merged form, interleaved.
    let a: Vec<(String, Vec<&'static str>)> = neighbours(src, &symbol, hints, cfg.seeds, &mut seen).into_iter().map(|(c, o)| (c, vec![o])).collect();
    let b: Vec<(String, Vec<&'static str>)> = match &merged {
        Some(m) => neighbours(m, &symbol, hints, cfg.seeds, &mut seen).into_iter().map(|(c, o)| (c, vec!["merge_decl", o])).collect(),
        None => vec![],
    };
    cands.extend(interleave(vec![a, b]));
    let verbose = std::env::var("MWDEC_REGFIX_VERBOSE").is_ok();
    if verbose {
        eprintln!("regfix: level 1: {} candidates", cands.len());
    }
    let mut level = score_all(scorer, cands, cfg.threads, &compiles, cfg.max_compiles, deadline);
    if verbose {
        for s in &level {
            eprintln!("  {:?} penalty {} {:?}", s.ops, s.fit.penalty, s.fit.profile);
        }
    }
    let mut best: Option<Scored> = None;
    let consider = |best: &mut Option<Scored>, v: &[Scored]| {
        for s in v {
            if best.as_ref().map_or(true, |b| s.fit.better_than(&b.fit)) {
                *best = Some(Scored { src: s.src.clone(), fit: s.fit.clone(), ops: s.ops.clone() });
            }
        }
    };
    consider(&mut best, &level);
    // Level 2 (and 3 while budget remains): expand the best few register-only neighbours.
    for _depth in 0..2 {
        if best.as_ref().is_some_and(|b| b.fit.exact) || compiles.load(Ordering::Relaxed) >= cfg.max_compiles {
            break;
        }
        level.retain(|s| register_only(&s.fit) && s.fit.penalty <= fit.penalty);
        level.sort_by(|a, b| a.fit.cmp_better(&b.fit));
        // Best first, one parent per last operator (equal-fitness neighbours of one operator are
        // usually the same kind of change at different sites).
        let mut parents: Vec<Scored> = vec![];
        let mut rest = vec![];
        for s in level.drain(..) {
            if parents.len() < cfg.beam && !parents.iter().any(|p| p.ops.last() == s.ops.last()) {
                parents.push(s);
            } else {
                rest.push(s);
            }
        }
        for s in rest {
            if parents.len() >= cfg.beam {
                break;
            }
            parents.push(s);
        }
        if parents.is_empty() {
            break;
        }
        let mut cands = vec![];
        // Interleave the parents' neighbours so each gets a share of the budget.
        let lists: Vec<Vec<(String, Vec<&'static str>)>> = parents
            .iter()
            .map(|p| {
                neighbours(&p.src, &symbol, hints, cfg.seeds, &mut seen)
                    .into_iter()
                    .map(|(c, o)| {
                        let mut ops = p.ops.clone();
                        ops.push(o);
                        (c, ops)
                    })
                    .collect()
            })
            .collect();
        cands.extend(interleave(lists));
        if verbose {
            eprintln!("regfix: next level: {} candidates from {} parents, {} compiles so far", cands.len(), parents.len(), compiles.load(Ordering::Relaxed));
        }
        level = score_all(scorer, cands, cfg.threads, &compiles, cfg.max_compiles, deadline);
        consider(&mut best, &level);
    }
    let b = best?;
    b.fit.better_than(fit).then(|| Repair { src: b.src, fitness: b.fit, ops: b.ops, compiles: compiles.load(Ordering::Relaxed) })
}
