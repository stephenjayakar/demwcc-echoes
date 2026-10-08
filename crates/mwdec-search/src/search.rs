//! Parallel randomized beam / hill-climb over source mutations, scored by the real compiler.
//!
//! Workers (default 6, matching the compile pool) loop: pick a parent from the beam (biased to
//! the best), apply 1..k random operators, dedupe by normalized text, compile + compare, and
//! update the beam. Equal-fitness children are accepted (plateau walking, as decomp-permuter
//! does). After `stagnation` evaluations without a new best the beam restarts from
//! {best, initial}. Operator weights = base x diff-category multiplier (from the parent's diff
//! profile) x adaptive success rate. Stops on exact, time budget, or compile budget.
use crate::cst::normalize;
use crate::ops::{self, Cat, OPS};
use crate::rng::Rng;
use crate::score::{DiffProfile, Eval, Fitness, Scorer};
use serde::Serialize;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub budget: Duration,
    /// Stop after this many real (non-cached) compiles.
    pub max_compiles: Option<usize>,
    /// Concurrent candidate evaluations (the compile pool still caps processes).
    pub workers: usize,
    pub seed: u64,
    pub beam: usize,
    /// Evaluations without a new best before restarting from {best, initial}.
    pub stagnation: usize,
    /// Max operators chained per child.
    pub max_chain: usize,
    /// Diff-guided category weighting.
    pub guided: bool,
    /// Adapt operator weights to observed success.
    pub adaptive: bool,
    /// Where to keep `best.cpp` / `result.json` (updated on every improvement).
    pub out_dir: Option<PathBuf>,
    pub verbose: bool,
    /// Restrict to these operators (names); empty = all.
    pub only_ops: Vec<String>,
    /// After an exact match, spend up to this many evaluations on readability cleanups that
    /// keep the match (0 = off).
    pub polish_evals: usize,
    /// Use target register-allocation hints (mwdec-oracle) for hint_order / hint_temp.
    pub use_hints: bool,
    /// Before the clock starts, wait (up to this long) for the compile fast path to hold a
    /// persistent compiler for the context on as many workers as the search uses (setup, like
    /// the precompiled header; zero = start them on demand inside the budget).
    pub warm_fast: Duration,
    /// Compiler tracer (GC/2.7 units): on register-only diffs, directed fixes from the real
    /// colouring are tried before random mutations.
    pub tracer: Option<Arc<crate::trace::Tracer>>,
    /// Localise differences on every new best (`-sym on` diagnostic compiles): focus mutations
    /// on the differing statements and apply schedcheck statement moves for reorder diffs.
    pub locate: bool,
    /// Probability that a child's first mutation is restricted to the differing statements.
    pub focus_prob: f64,
    /// Operators switched off (ablations).
    pub disabled_ops: Vec<String>,
    /// Systematic neighbourhood search for targets up to this many bytes of code (0 = off):
    /// best-first over evaluated candidates, each expanded into all of its distinct one-step
    /// neighbours ([`ops::neighbours`]), evaluated best operator first. Random children keep
    /// running alongside (a share of the evaluations).
    pub systematic_max_bytes: usize,
    /// Seeds per operator when enumerating a neighbourhood.
    pub enum_tries: usize,
    /// Neighbours kept per expansion.
    pub enum_cap: usize,
    /// Scheduler advice (`mwdec_oracle::advice`) on reorder diffs, at most this many calls per
    /// search (each is a compile under the debugger plus a `-sym on` compile; 0 = off).
    pub max_advice: u64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            budget: Duration::from_secs(60),
            max_compiles: None,
            workers: 6,
            seed: 1,
            beam: 8,
            stagnation: 400,
            max_chain: 3,
            guided: true,
            adaptive: true,
            out_dir: None,
            verbose: false,
            only_ops: vec![],
            polish_evals: 60,
            use_hints: true,
            warm_fast: Duration::from_secs(20),
            tracer: None,
            locate: true,
            focus_prob: 0.6,
            disabled_ops: vec![],
            systematic_max_bytes: 128,
            enum_tries: 12,
            enum_cap: 800,
            max_advice: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OpStat {
    pub name: &'static str,
    /// Children this operator took part in (evaluated, not deduped).
    pub tries: u64,
    pub compile_errors: u64,
    /// Children strictly better than their parent.
    pub improved: u64,
    /// Children that became the global best.
    pub new_best: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Improvement {
    pub secs: f64,
    pub evals: u64,
    pub penalty: u64,
    pub score: f64,
    pub ops: Vec<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SearchResult {
    pub exact: bool,
    pub best: Option<Fitness>,
    pub initial: Option<Fitness>,
    pub initial_error: Option<String>,
    #[serde(skip)]
    pub best_src: String,
    /// Candidates evaluated (compiled or served from the cache), excluding duplicates.
    pub evals: u64,
    /// Real compiler runs.
    pub compiles: u64,
    pub compile_errors: u64,
    pub duplicates: u64,
    pub gen_failures: u64,
    pub restarts: u64,
    pub seconds: f64,
    pub op_stats: Vec<OpStat>,
    pub history: Vec<Improvement>,
    /// Cleanups accepted by the polish pass.
    pub polished: usize,
    /// Compiler traces run for register-only diffs.
    pub traces: u64,
    /// Diff localisations (line-attributed diagnostic compiles) run on new bests.
    pub locates: u64,
    /// Children whose first mutation was restricted to the differing statements.
    pub focused: u64,
    pub focused_improved: u64,
    /// Systematic search: neighbourhoods enumerated, and evaluations taken from them.
    pub expansions: u64,
    pub systematic: u64,
    /// Scheduler advice calls (reorder diffs).
    pub advices: u64,
}

#[derive(Clone)]
struct Cand {
    src: Arc<String>,
    fit: Fitness,
    /// Normalized source length (parsimony tie-break).
    len: usize,
    /// [`text_hash`] of `src`.
    h: u128,
}

impl Cand {
    fn new(src: Arc<String>, fit: Fitness) -> Cand {
        let h = text_hash(&src);
        Cand::with_hash(src, fit, h)
    }
    fn with_hash(src: Arc<String>, fit: Fitness, h: u128) -> Cand {
        let len = normalize(&src).len();
        Cand { src, fit, len, h }
    }
    /// Better fitness, or equal fitness and shorter source.
    fn better_than(&self, o: &Cand) -> bool {
        match self.fit.cmp_better(&o.fit) {
            std::cmp::Ordering::Less => true,
            std::cmp::Ordering::Equal => self.len < o.len,
            _ => false,
        }
    }
}

struct State {
    beam: Vec<Cand>,
    init: Option<Cand>,
    best: Option<Cand>,
    seen: HashSet<u128>,
    stats: Vec<OpStat>,
    evals: u64,
    compiles: u64,
    compile_errors: u64,
    duplicates: u64,
    gen_failures: u64,
    since_best: usize,
    dup_streak: usize,
    /// Directed candidates (source, operator id for stats) evaluated before random children.
    priority: Vec<(String, usize)>,
    /// Byte ranges of the differing statements, per candidate text hash.
    focus: std::collections::HashMap<u128, Arc<Vec<(usize, usize)>>>,
    /// Differing source lines of the latest analysed best (applied to other parents too: edits
    /// shift lines only a little).
    focus_lines: Option<Arc<Vec<u32>>>,
    locates: u64,
    focused: u64,
    focused_improved: u64,
    traced: HashSet<u128>,
    traces: u64,
    restarts: u64,
    history: Vec<Improvement>,
    /// Systematic search: neighbours of the candidate being expanded (popped from the end).
    enum_q: Vec<(String, usize)>,
    /// Evaluated candidates not expanded yet (bounded, best kept).
    open: Vec<Cand>,
    expanded: HashSet<u128>,
    enum_busy: bool,
    expansions: u64,
    systematic: u64,
    advices: u64,
    /// Latest new best waiting for diagnostics (older ones are superseded).
    diag_q: Option<(Arc<String>, Fitness, u128)>,
}

/// Time spent generating children / evaluating them, all searches of the process (µs):
/// `MWDEC_SEARCH_PROFILE` prints them after each search.
pub static PROF_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PROF_EVAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Open candidates kept for systematic expansion.
const OPEN_CAP: usize = 512;
/// Share of evaluations that stay random children while a neighbourhood is being evaluated.
const RANDOM_SHARE: f64 = 0.5;

fn text_hash(s: &str) -> u128 {
    mwdec_mwcc::content_hash(&[normalize(s).as_bytes()])
}

/// Weight multiplier for an operator with diff-class affinity `aff` given what differs
/// (catalog scheduling hint: classify the diff as R/S/L/C/M, prefer the matching knobs).
pub fn affinity_mult(p: &DiffProfile, aff: u8) -> f64 {
    let f = [
        (ops::R, p.reg),
        (ops::S, p.reorder),
        (ops::L, p.branch),
        (ops::C, p.inserted + p.deleted + p.substituted + p.other + p.reloc),
        (ops::M_, p.stack),
    ];
    let tot: u32 = f.iter().map(|x| x.1).sum();
    if tot == 0 {
        return 1.0;
    }
    let share: f64 = f.iter().filter(|x| aff & x.0 != 0).map(|x| x.1 as f64).sum::<f64>() / tot as f64;
    0.3 + 3.0 * share
}

/// Category multipliers from what differs (coarse; superseded by [`affinity_mult`]).
pub fn category_mult(p: &DiffProfile) -> impl Fn(Cat) -> f64 {
    let structural = p.inserted + p.deleted + p.substituted;
    let reorder = p.reorder;
    let regish = p.reg + p.stack;
    let branch = p.branch;
    move |c: Cat| {
        let mut m = 1.0;
        if structural > 0 {
            m *= match c {
                Cat::Types => 2.0,
                Cat::Control => 2.0,
                Cat::Temp => 1.5,
                Cat::Expr => 1.3,
                Cat::Order => 0.7,
            };
        } else if reorder > 0 {
            m *= match c {
                Cat::Order => 2.5,
                Cat::Temp => 1.5,
                Cat::Expr => 1.5,
                _ => 0.8,
            };
        } else if regish > 0 {
            m *= match c {
                Cat::Order => 3.0,
                Cat::Temp => 2.0,
                Cat::Expr => 1.5,
                _ => 0.5,
            };
        }
        if branch > 0 && c == Cat::Control {
            m *= 1.5;
        }
        m
    }
}

impl State {
    fn weights(&self, base: &[f64], parent: &Fitness, cfg: &SearchConfig) -> Vec<f64> {
        let (ti, tt): (u64, u64) = self.stats.iter().fold((0, 0), |a, s| (a.0 + s.improved, a.1 + s.tries));
        let global = (ti as f64 + 0.5) / (tt as f64 + 2.0);
        base.iter()
            .zip(OPS)
            .enumerate()
            .map(|(i, (&b, op))| {
                if b == 0.0 {
                    return 0.0;
                }
                let mut w = b;
                if cfg.guided {
                    w *= affinity_mult(&parent.profile, op.aff);
                    w *= crate::score::residue_mult(op.name, parent.profile.residue);
                }
                if cfg.adaptive {
                    let s = &self.stats[i];
                    let rate = (s.improved as f64 + 0.5) / (s.tries as f64 + 2.0);
                    w *= (rate / global).clamp(0.25, 4.0);
                }
                w
            })
            .collect()
    }

    /// Remember an evaluated candidate for systematic expansion.
    fn add_open(&mut self, c: Cand) {
        if self.expanded.contains(&c.h) || self.open.iter().any(|o| o.h == c.h) {
            return;
        }
        self.open.push(c);
        if self.open.len() > OPEN_CAP {
            if let Some(w) = (0..self.open.len()).max_by(|&a, &b| self.open[a].fit.cmp_better(&self.open[b].fit).then(self.open[a].len.cmp(&self.open[b].len))) {
                self.open.remove(w);
            }
        }
    }

    /// The best open candidate, marked expanded. Equal fitness: the first evaluated (breadth
    /// first over a plateau, in the order the neighbourhoods were ranked).
    fn pop_open(&mut self) -> Option<Cand> {
        let i = (0..self.open.len()).min_by(|&a, &b| self.open[a].fit.cmp_better(&self.open[b].fit))?;
        let c = self.open.remove(i);
        self.expanded.insert(c.h);
        Some(c)
    }

    fn insert_beam(&mut self, c: Cand, cap: usize, rng: &mut Rng) {
        if self.beam.iter().any(|b| Arc::ptr_eq(&b.src, &c.src) || *b.src == *c.src) {
            return;
        }
        if self.beam.len() < cap {
            self.beam.push(c);
        } else {
            // Replace the worst if not worse than it (ties: random among the worst).
            let worst = (0..self.beam.len()).max_by(|&a, &b| self.beam[a].fit.cmp_better(&self.beam[b].fit).then(self.beam[a].len.cmp(&self.beam[b].len))).unwrap();
            if !self.beam[worst].fit.better_than(&c.fit) {
                let wf = self.beam[worst].fit.clone();
                let ties: Vec<usize> = (0..self.beam.len()).filter(|&k| self.beam[k].fit.cmp_better(&wf).is_eq()).collect();
                let k = ties[rng.below(ties.len())];
                self.beam[k] = c;
            }
        }
    }
}

fn save(cfg: &SearchConfig, best: &Cand, extra: &str) {
    if let Some(d) = &cfg.out_dir {
        let _ = std::fs::create_dir_all(d);
        let _ = std::fs::write(d.join("best.cpp"), best.src.as_bytes());
        let _ = std::fs::write(d.join("best.json"), format!("{}\n", serde_json::to_string(&best.fit).unwrap_or_default()));
        if !extra.is_empty() {
            let _ = std::fs::write(d.join("result.json"), extra);
        }
    }
}

/// Directed candidates for candidate `src` with fitness `fit`: tracer register fixes (register-only
/// diffs) and schedcheck statement moves (reorder diffs), plus the focus ranges of the differing
/// statements. Runs diagnostic compiles; call outside the state lock.
struct Directed {
    cands: Vec<(String, usize)>,
    focus: Option<Arc<Vec<(usize, usize)>>>,
    lines: Option<Arc<Vec<u32>>>,
    traced: bool,
    located: bool,
    /// The scheduler advice ran (`mwdec_oracle::advice::statement_moves_in`).
    advised: bool,
}

fn directed_for(
    scorer: &Scorer,
    cfg: &SearchConfig,
    locator: Option<&crate::locate::Locator>,
    src: &str,
    fit: &Fitness,
    ops_ids: (usize, usize, usize),
    advice_left: bool,
) -> Directed {
    let (trace_op, sched_op, swap_op) = ops_ids;
    let symbol = &scorer.symbol;
    let mut d = Directed { cands: vec![], focus: None, lines: None, traced: false, located: false, advised: false };
    if fit.exact {
        return d;
    }
    let mut temp_fixes: Vec<crate::trace::Fix> = Vec::new();
    if let Some(tr) = &cfg.tracer {
        if fit.profile.reg > 0 && fit.profile.structural() == 0 {
            if let Ok(fixes) = tr.fixes(src, symbol, scorer.tf) {
                temp_fixes = fixes
                    .iter()
                    .filter(|f| f.kind == mwdec_oracle::tracer::VregKind::Temp && f.holder.as_ref().is_some_and(|h| h.2 == mwdec_oracle::tracer::VregKind::Temp))
                    .cloned()
                    .collect();
                d.traced = true;
                if cfg.verbose {
                    for fx in &fixes {
                        eprintln!("  trace: {}{} -> {}{}: {}", fx.class, fx.from, fx.class, fx.to, fx.suggestion);
                    }
                }
                for fx in &fixes {
                    for c in crate::trace::apply_fix(src, symbol, fx) {
                        if !d.cands.iter().any(|x| x.0 == c) {
                            d.cands.push((c, trace_op));
                        }
                    }
                }
            }
        }
    }
    if let Some(loc) = locator {
        if let Some(an) = loc.analyze(src, symbol, fit.profile.reorder > 0) {
            d.located = true;
            if cfg.verbose {
                eprintln!("  locate: diff lines {:?}, moves {:?}, swaps {:?}", an.lines, an.moves, an.swaps);
            }
            // Focus on the statements with the most differing instructions.
            let lines: Vec<u32> = an.lines.iter().copied().take(4).collect();
            let ranges = crate::locate::line_ranges(src, &lines);
            if !ranges.is_empty() {
                d.focus = Some(Arc::new(ranges));
                d.lines = Some(Arc::new(lines));
            }
            for &(x, y, _) in &an.moves {
                for c in crate::locate::move_before(src, symbol, x, y) {
                    if !d.cands.iter().any(|v| v.0 == c) {
                        d.cands.push((c, sched_op));
                    }
                }
            }
            // Temp creation order (tracer temp/temp fixes): the statements defining the two
            // registers in the differing code, evaluated in the other order.
            for fx in &temp_fixes {
                let fpr = fx.class == crate::trace::RegClass::Fpr;
                let ln = an.def_lines(fpr, fx.from);
                let lh = an.def_lines(fpr, fx.to);
                let mut k = 0;
                for &a in &ln {
                    for &b in &lh {
                        if a == b || k >= 6 {
                            continue;
                        }
                        // earlier: n must be created after the holder
                        let (x, y) = if fx.earlier { (b, a) } else { (a, b) };
                        for c in crate::locate::move_before(src, symbol, x, y) {
                            if !d.cands.iter().any(|v| v.0 == c) {
                                d.cands.push((c, trace_op));
                                k += 1;
                            }
                        }
                    }
                }
            }
            for &(x, y) in an.swaps.iter().take(6) {
                if let Some(c) = crate::locate::swap_lines(src, symbol, x, y) {
                    if !d.cands.iter().any(|v| v.0 == c) {
                        d.cands.push((c, swap_op));
                    }
                }
            }
        }
    }
    // Reorder diffs: the real scheduler says which statement to move (program-order ties and
    // operands waited for, besides the forced moves found above).
    if fit.profile.reorder > 0 && advice_left {
        if let Some(tr) = &cfg.tracer {
            let tf = crate::locate::to_asm(scorer.tf);
            let tobj = mwdec_oracle::asm::Obj { funcs: vec![tf.clone()], ..Default::default() };
            let cand = mwdec_oracle::advice::Candidate { context: &tr.context, src, symbol };
            if let Ok(diag) = mwdec_oracle::advice::statement_moves_in(&tr.comp, &cand, &tobj, &tf) {
                d.advised = true;
                if cfg.verbose {
                    let mv: Vec<_> = diag.moves.iter().map(|m| (m.line, m.before, m.votes)).collect();
                    eprintln!("  advice: moves {mv:?}");
                }
                for m in diag.moves.iter().take(6) {
                    for c in crate::locate::move_before(src, symbol, m.line, m.before) {
                        if !d.cands.iter().any(|v| v.0 == c) {
                            d.cands.push((c, sched_op));
                        }
                    }
                }
            }
        }
    }
    // Popped from the end: put the most specific (tracer, then forced moves) last.
    d.cands.reverse();
    d
}

/// Run the search from `init` (candidate source containing the function for `scorer.symbol`).
pub fn search(scorer: &Scorer, init: &str, cfg: &SearchConfig) -> SearchResult {
    let mut t0 = Instant::now();
    let mut base = ops::base_weights();
    let hints = if cfg.use_hints { crate::hints::target_hints(scorer.tf) } else { Default::default() };
    if !hints.is_empty() {
        for (n, w) in ops::HINT_OPS {
            if let Some(i) = ops::op_index(n) {
                base[i] = *w;
            }
        }
    }
    let hints_ref = (!hints.is_empty()).then_some(&hints);
    let trace_op = ops::op_index("trace_fix").unwrap();
    let ops_ids = (trace_op, ops::op_index("sched_move").unwrap(), ops::op_index("sched_swap").unwrap());
    for (i, o) in OPS.iter().enumerate() {
        if cfg.disabled_ops.iter().any(|n| n == o.name) {
            base[i] = 0.0;
        }
    }
    if !cfg.only_ops.is_empty() {
        for (i, o) in OPS.iter().enumerate() {
            if !cfg.only_ops.iter().any(|n| n == o.name) {
                base[i] = 0.0;
            }
        }
    }
    let mut st = State {
        beam: vec![],
        init: None,
        best: None,
        seen: HashSet::new(),
        stats: OPS.iter().map(|o| OpStat { name: o.name, ..Default::default() }).collect(),
        evals: 0,
        compiles: 0,
        compile_errors: 0,
        duplicates: 0,
        gen_failures: 0,
        since_best: 0,
        dup_streak: 0,
        priority: vec![],
        focus: Default::default(),
        focus_lines: None,
        locates: 0,
        focused: 0,
        focused_improved: 0,
        traced: HashSet::new(),
        traces: 0,
        restarts: 0,
        history: vec![],
        enum_q: vec![],
        open: vec![],
        expanded: HashSet::new(),
        enum_busy: false,
        expansions: 0,
        systematic: 0,
        advices: 0,
        diag_q: None,
    };
    let systematic = cfg.systematic_max_bytes > 0 && scorer.tf.code.len() <= cfg.systematic_max_bytes;
    st.seen.insert(text_hash(init));
    if let Some(d) = &cfg.out_dir {
        let _ = std::fs::create_dir_all(d);
        let _ = std::fs::write(d.join("init.cpp"), init);
    }
    let (mut e0, mut ran) = scorer.eval(init);
    // the starting point's verdict: never an unconfirmed fast-path failure, never a fast-path
    // object that a normal compile would judge differently
    if !e0.fitness().is_some_and(|f| f.exact) {
        let (e, r) = scorer.eval_normal(init);
        e0 = e;
        ran |= r;
    }
    st.evals += 1;
    st.compiles += ran as u64;
    let init_src = Arc::new(init.to_string());
    let mut initial_error = None;
    match &e0 {
        Eval::Ok(f) => {
            let c = Cand::new(init_src.clone(), f.clone());
            st.beam.push(c.clone());
            st.init = Some(c.clone());
            st.best = Some(c.clone());
            if systematic {
                st.add_open(c.clone());
            }
            save(cfg, &c, "");
            st.history.push(Improvement { secs: 0.0, evals: 1, penalty: f.penalty, score: f.score, ops: vec![] });
        }
        Eval::CompileError(m) => initial_error = Some(format!("compile error: {m}")),
        Eval::Missing(fs) => initial_error = Some(format!("symbol not defined; candidate defines {fs:?}")),
        Eval::Io(m) => initial_error = Some(format!("io: {m}")),
    }
    let initial = e0.fitness().cloned();
    let exact0 = initial.as_ref().is_some_and(|f| f.exact);
    if !exact0 && initial.is_some() && cfg.budget > Duration::ZERO && cfg.warm_fast > Duration::ZERO {
        let ctx = if scorer.pch_broken.load(Ordering::Relaxed) { scorer.plain.unwrap_or(scorer.ctx) } else { scorer.ctx };
        scorer.mwcc.warm_fast(ctx, cfg.workers.clamp(1, 4), cfg.warm_fast);
        t0 = Instant::now();
    }
    // Diagnostic compiles with line info: only set up when there is something to search.
    let want_locator = cfg.locate && !exact0 && initial.is_some() && cfg.budget > Duration::ZERO;
    let locator_cell: std::sync::OnceLock<Option<crate::locate::Locator>> = std::sync::OnceLock::new();
    let locator = || -> Option<&crate::locate::Locator> {
        locator_cell
            .get_or_init(|| {
                if !want_locator {
                    return None;
                }
                let ctx = if scorer.pch_broken.load(Ordering::Relaxed) { scorer.plain.unwrap_or(scorer.ctx) } else { scorer.ctx };
                crate::locate::Locator::new(scorer.mwcc, ctx, scorer.tf)
            })
            .as_ref()
    };
    // Diagnostics (initial candidate, then new bests) run in their own thread while the workers
    // search.
    let init_diag = AtomicBool::new(initial.is_some() && cfg.budget > Duration::ZERO);
    // Set once the initial diagnostics are in (the generation-failure stop waits for them).
    let init_diag_done = AtomicBool::new(!init_diag.load(Ordering::Relaxed));
    st.traced.insert(text_hash(init));
    let state = Mutex::new(st);
    let stop = AtomicBool::new(exact0 || initial.is_none());
    let symbol = scorer.symbol.clone();
    let locator = &locator;

    std::thread::scope(|s| {
        // diagnostics thread
        {
            let state = &state;
            let stop = &stop;
            let init_src = init_src.clone();
            let init_diag = &init_diag;
            let init_diag_done = &init_diag_done;
            let initial = &initial;
            let _ = std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(s, move || {
                let merge = |d: Directed, h: u128| {
                    let mut g = state.lock().unwrap();
                    g.advices += d.advised as u64;
                    g.traces += d.traced as u64;
                    g.locates += d.located as u64;
                    if let Some(fr) = d.focus {
                        g.focus.insert(h, fr);
                    }
                    if d.lines.is_some() {
                        g.focus_lines = d.lines;
                    }
                    g.priority.extend(d.cands);
                };
                if init_diag.swap(false, Ordering::Relaxed) {
                    if let Some(f) = &initial {
                        merge(directed_for(scorer, cfg, locator(), &init_src, f, ops_ids, cfg.max_advice > 0), text_hash(&init_src));
                    }
                }
                init_diag_done.store(true, Ordering::Relaxed);
                while !stop.load(Ordering::Relaxed) {
                    let (job, left) = {
                        let mut g = state.lock().unwrap();
                        (g.diag_q.take(), g.advices < cfg.max_advice)
                    };
                    match job {
                        Some((src, fit, h)) => merge(directed_for(scorer, cfg, locator(), &src, &fit, ops_ids, left), h),
                        None => std::thread::sleep(Duration::from_millis(5)),
                    }
                }
            });
        }
        for w in 0..cfg.workers.max(1) {
            let state = &state;
            let stop = &stop;
            let base = &base;
            let symbol = &symbol;
            let init_src = init_src.clone();
            let init_diag_done = &init_diag_done;
            let _ = std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(s, move || {
                let mut rng = Rng::new(cfg.seed.wrapping_mul(0x1000_0000_01b3).wrapping_add(w as u64 * 7919 + 1));
                while !stop.load(Ordering::Relaxed) {
                    if t0.elapsed() >= cfg.budget {
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                    // Systematic search: expand the best open candidate once the current
                    // neighbourhood is used up.
                    if systematic {
                        let job = {
                            let mut g = state.lock().unwrap();
                            if g.priority.is_empty() && g.enum_q.is_empty() && !g.enum_busy {
                                g.pop_open().map(|c| {
                                    g.enum_busy = true;
                                    let w = g.weights(base, &c.fit, cfg);
                                    let focus = g.focus.get(&c.h).cloned().or_else(|| g.focus_lines.as_ref().map(|ls| Arc::new(crate::locate::line_ranges(&c.src, ls))));
                                    (c, w, focus, g.expansions)
                                })
                            } else {
                                None
                            }
                        };
                        if let Some((c, w, focus, k)) = job {
                            let te = Instant::now();
                            let nb = ops::neighbours(&c.src, symbol, &w, hints_ref, focus.as_deref().map(|v| v.as_slice()), cfg.enum_tries, cfg.seed ^ (k + 1).wrapping_mul(0x9e37_79b9), cfg.enum_cap);
                            let hs: Vec<u128> = nb.iter().map(|n| text_hash(&n.src)).collect();
                            let total = nb.len();
                            let mut g = state.lock().unwrap();
                            g.enum_busy = false;
                            g.expansions += 1;
                            let mut q: Vec<(String, usize)> = nb.into_iter().zip(hs).filter(|(_, h)| !g.seen.contains(h)).map(|(n, _)| (n.src, n.op)).collect();
                            q.reverse();
                            if cfg.verbose {
                                eprintln!("  expand [{:.1}s]: penalty {} -> {} new of {} neighbours ({} ms)", t0.elapsed().as_secs_f64(), c.fit.penalty, q.len(), total, te.elapsed().as_millis());
                            }
                            g.enum_q = q;
                            continue;
                        }
                    }
                    // Choose parent and weights; directed candidates first.
                    let (parent, weights, directed, focus, chain) = {
                        let mut g = state.lock().unwrap();
                        // Neighbourhood exhausted (mostly duplicates): chain more operators.
                        let gen = (g.evals + g.duplicates).max(1) as f64;
                        let dup = g.duplicates as f64 / gen;
                        let chain = cfg.max_chain + if gen > 200.0 && dup > 0.6 { 2 } else if gen > 200.0 && dup > 0.4 { 1 } else { 0 };
                        if let Some(mc) = cfg.max_compiles {
                            if g.compiles as usize >= mc {
                                stop.store(true, Ordering::Relaxed);
                                break;
                            }
                        }
                        if g.since_best >= cfg.stagnation {
                            g.since_best = 0;
                            g.restarts += 1;
                            let best = g.best.clone().unwrap();
                            let init = g.init.clone().unwrap();
                            g.beam = vec![best, init];
                        }
                        let mut directed = g.priority.pop();
                        if directed.is_none() && systematic && !g.enum_q.is_empty() && !rng.chance(RANDOM_SHARE) {
                            directed = g.enum_q.pop();
                            g.systematic += 1;
                        }
                        let p = if directed.is_some() || rng.chance(0.5) || g.beam.len() == 1 {
                            g.best.clone().unwrap()
                        } else {
                            g.beam[rng.below(g.beam.len())].clone()
                        };
                        let w = g.weights(base, &p.fit, cfg);
                        let focus = g.focus.get(&p.h).cloned().or_else(|| {
                            g.focus_lines.as_ref().map(|ls| Arc::new(crate::locate::line_ranges(&p.src, ls)))
                        });
                        (p, w, directed, focus, chain)
                    };
                    // Generate a child.
                    let t_gen = Instant::now();
                    let k = {
                        let mut k = 1;
                        while k < chain && rng.chance(if chain > cfg.max_chain { 0.5 } else { 0.35 }) {
                            k += 1;
                        }
                        k
                    };
                    let mut src = (*parent.src).clone();
                    let mut used: Vec<usize> = Vec::new();
                    let mut was_focused = false;
                    if let Some((d, op)) = directed {
                        src = d;
                        used.push(op);
                    } else {
                        for step in 0..k {
                            let m = match &focus {
                                Some(fr) if step == 0 && rng.chance(cfg.focus_prob) => {
                                    let m = ops::mutate_in(&src, symbol, &weights, &mut rng, hints_ref, fr);
                                    was_focused = m.is_some();
                                    m.or_else(|| ops::mutate_with(&src, symbol, &weights, &mut rng, hints_ref))
                                }
                                _ => ops::mutate_with(&src, symbol, &weights, &mut rng, hints_ref),
                            };
                            if let Some(m) = m {
                                src = m.src;
                                used.push(m.op);
                            }
                        }
                    }
                    if used.is_empty() {
                        let mut g = state.lock().unwrap();
                        g.gen_failures += 1;
                        if g.gen_failures > 200 && g.evals < 2 && init_diag_done.load(Ordering::Relaxed) && g.priority.is_empty() {
                            stop.store(true, Ordering::Relaxed);
                        }
                        continue;
                    }
                    let h = text_hash(&src);
                    {
                        let mut g = state.lock().unwrap();
                        if !g.seen.insert(h) {
                            g.duplicates += 1;
                            g.dup_streak += 1;
                            if g.dup_streak > 5000 {
                                stop.store(true, Ordering::Relaxed);
                            }
                            continue;
                        }
                        g.dup_streak = 0;
                    }
                    let t_eval = Instant::now();
                    PROF_GEN.fetch_add((t_eval - t_gen).as_micros() as u64, Ordering::Relaxed);
                    let (ev, ran) = scorer.eval(&src);
                    PROF_EVAL.fetch_add(t_eval.elapsed().as_micros() as u64, Ordering::Relaxed);
                    let mut g = state.lock().unwrap();
                    g.evals += 1;
                    g.compiles += ran as u64;
                    g.since_best += 1;
                    g.focused += was_focused as u64;
                    for &o in &used {
                        g.stats[o].tries += 1;
                    }
                    let fit = match ev {
                        Eval::Ok(f) => f,
                        other => {
                            if cfg.verbose && std::env::var_os("MWDEC_SEARCH_LOG_ERRORS").is_some() {
                                if let Eval::CompileError(m) = &other {
                                    eprintln!("  compile error via {:?}: {m}", used.iter().map(|&o| OPS[o].name).collect::<Vec<_>>());
                                }
                            }
                            g.compile_errors += 1;
                            for &o in &used {
                                g.stats[o].compile_errors += 1;
                            }
                            continue;
                        }
                    };
                    if fit.better_than(&parent.fit) {
                        g.focused_improved += was_focused as u64;
                        for &o in &used {
                            g.stats[o].improved += 1;
                        }
                    }
                    let cand = Cand::with_hash(Arc::new(src), fit.clone(), h);
                    let strictly = g.best.as_ref().map_or(true, |b| fit.better_than(&b.fit));
                    if !strictly && g.best.as_ref().is_some_and(|b| cand.better_than(b)) {
                        // Same fitness, simpler source: keep as best without resetting stagnation.
                        save(cfg, &cand, "");
                        g.best = Some(cand.clone());
                    }
                    let is_best = strictly;
                    if is_best {
                        for &o in &used {
                            g.stats[o].new_best += 1;
                        }
                        g.since_best = 0;
                        let secs = t0.elapsed().as_secs_f64();
                        let evals = g.evals;
                        g.history.push(Improvement {
                            secs,
                            evals,
                            penalty: fit.penalty,
                            score: fit.score,
                            ops: used.iter().map(|&o| OPS[o].name).collect(),
                        });
                        if cfg.verbose {
                            eprintln!(
                                "[{secs:7.1}s {evals:6} evals] penalty {} score {:.1}{} via {:?}",
                                fit.penalty,
                                fit.score,
                                if fit.exact { " EXACT" } else { "" },
                                used.iter().map(|&o| OPS[o].name).collect::<Vec<_>>()
                            );
                        }
                        save(cfg, &cand, "");
                        g.best = Some(cand.clone());
                        if fit.exact {
                            stop.store(true, Ordering::Relaxed);
                        }
                        // first improvement: continue from the new best's neighbourhood
                        g.enum_q.clear();
                    }
                    if systematic {
                        g.add_open(cand.clone());
                    }
                    let _ = &init_src;
                    // New best: tracer fixes / schedcheck moves / diff focus.
                    let diag_src = (is_best && !fit.exact && g.traced.insert(h)).then(|| cand.src.clone());
                    if let Some(s) = diag_src {
                        g.diag_q = Some((s, fit.clone(), h));
                    }
                    g.insert_beam(cand, cfg.beam, &mut rng);
                }
            });
        }
    });

    let g = state.into_inner().unwrap();
    if std::env::var_os("MWDEC_SEARCH_PROFILE").is_some() {
        eprintln!("search profile: generate {:.1}s, evaluate {:.1}s (cumulative, all threads)", PROF_GEN.load(Ordering::Relaxed) as f64 / 1e6, PROF_EVAL.load(Ordering::Relaxed) as f64 / 1e6);
    }
    let mut best = g.best.clone();
    let mut polished = 0;
    if let Some(b) = &mut best {
        if b.fit.exact && cfg.polish_evals > 0 {
            let (src, n) = polish(scorer, &b.src, cfg.polish_evals, cfg.seed);
            if n > 0 {
                b.src = Arc::new(src);
                polished = n;
            }
        }
    }
    let res = SearchResult {
        exact: best.as_ref().is_some_and(|b| b.fit.exact),
        best: best.as_ref().map(|b| b.fit.clone()),
        initial,
        initial_error,
        best_src: best.as_ref().map(|b| (*b.src).clone()).unwrap_or_else(|| init.to_string()),
        evals: g.evals,
        compiles: g.compiles,
        compile_errors: g.compile_errors,
        duplicates: g.duplicates,
        gen_failures: g.gen_failures,
        restarts: g.restarts,
        seconds: t0.elapsed().as_secs_f64(),
        op_stats: g.stats,
        history: g.history,
        polished,
        traces: g.traces,
        locates: g.locates,
        focused: g.focused,
        focused_improved: g.focused_improved,
        expansions: g.expansions,
        systematic: g.systematic,
        advices: g.advices,
    };
    if let Some(b) = &best {
        save(cfg, b, &serde_json::to_string_pretty(&res).unwrap_or_default());
    }
    res
}

/// Bounded systematic pass for near misses at draft time: the distinct one-step neighbours of
/// `src` (`ops::neighbours`, operators weighted by what differs), best first, at most
/// `max_compiles` real compiles, the first exact one returned. Deterministic for a given source
/// and target.
pub fn quick_pass(scorer: &Scorer, src: &str, fit: &Fitness, max_compiles: usize) -> Option<String> {
    quick_pass_with(scorer, src, fit, max_compiles, None)
}

/// [`quick_pass`]; with a tracer, an instruction-order difference first asks the real scheduler
/// which statements to move (`mwdec_oracle::advice::statement_moves_in`, one traced compile and
/// one `-sym on` compile) and tries those moves before the neighbourhood.
pub fn quick_pass_with(scorer: &Scorer, src: &str, fit: &Fitness, max_compiles: usize, tracer: Option<&crate::trace::Tracer>) -> Option<String> {
    if fit.exact {
        return None;
    }
    let mut compiles = 0;
    if let (Some(tr), true) = (tracer, fit.profile.reorder > 0 && std::env::var_os("MWDEC_NO_NEAR_ADVICE").is_none()) {
        let tf = crate::locate::to_asm(scorer.tf);
        let tobj = mwdec_oracle::asm::Obj { funcs: vec![tf.clone()], ..Default::default() };
        let cand = mwdec_oracle::advice::Candidate { context: &tr.context, src, symbol: &scorer.symbol };
        if let Ok(diag) = mwdec_oracle::advice::statement_moves_in(&tr.comp, &cand, &tobj, &tf) {
            for mv in diag.moves.iter().take(4) {
                for c in crate::locate::move_before(src, &scorer.symbol, mv.line, mv.before) {
                    let (e, ran) = scorer.eval(&c);
                    compiles += ran as usize;
                    if e.fitness().is_some_and(|f| f.exact) {
                        return Some(c);
                    }
                }
            }
        }
    }
    let weights: Vec<f64> = ops::base_weights().iter().zip(OPS).map(|(w, o)| w * affinity_mult(&fit.profile, o.aff)).collect();
    let hints = crate::hints::target_hints(scorer.tf);
    let hints_ref = (!hints.is_empty()).then_some(&hints);
    let nb = ops::neighbours(src, &scorer.symbol, &weights, hints_ref, None, 12, 1, max_compiles * 3);
    for n in nb {
        if compiles >= max_compiles {
            break;
        }
        let (e, ran) = scorer.eval(&n.src);
        compiles += ran as usize;
        if e.fitness().is_some_and(|f| f.exact) {
            return Some(n.src);
        }
    }
    None
}

/// Readability pass over an exact match: random cleanup edits ([`ops::POLISH_OPS`]) are kept
/// when the result is still exact and shorter. Returns (source, accepted edits).
pub fn polish(scorer: &Scorer, src: &str, max_evals: usize, seed: u64) -> (String, usize) {
    let mut weights = vec![0.0; OPS.len()];
    for n in ops::POLISH_OPS {
        if let Some(i) = ops::op_index(n) {
            weights[i] = 1.0;
        }
    }
    let mut rng = Rng::new(seed ^ 0x5eed_0f_b0b);
    let mut cur = src.to_string();
    let mut cur_len = normalize(&cur).len();
    let mut seen = HashSet::new();
    let (mut evals, mut accepted, mut fails) = (0, 0, 0);
    while evals < max_evals && fails < 200 {
        let Some(mt) = ops::mutate(&cur, &scorer.symbol, &weights, &mut rng) else {
            fails += 1;
            continue;
        };
        let n = normalize(&mt.src);
        if n.len() >= cur_len || !seen.insert(n.clone()) {
            fails += 1;
            continue;
        }
        evals += 1;
        if let (Eval::Ok(f), _) = scorer.eval(&mt.src) {
            if f.exact {
                cur = mt.src;
                cur_len = n.len();
                accepted += 1;
            }
        }
    }
    (cur, accepted)
}
