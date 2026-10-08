//! Fitness of a candidate: the strict comparator decides exactness; a permuter-style penalty
//! over an opcode-level alignment ranks non-exact candidates and profiles what differs
//! (register allocation vs. stack offsets vs. inserted/deleted/reordered instructions), which
//! drives operator weighting.
use mwdec_core::Function;
use mwdec_mwcc::compare::masked_words;
use mwdec_mwcc::{compare_indexed, DiffClass, ExternIndex, MwccError, Mwcc, ObjIndex, UnitContext};
use serde::Serialize;
use std::cmp::Ordering;

pub const PEN_REG: u64 = 5;
pub const PEN_STACK: u64 = 1;
pub const PEN_BRANCH: u64 = 1;
pub const PEN_OTHER: u64 = 5;
pub const PEN_REORDER: u64 = 60;
pub const PEN_INSDEL: u64 = 100;
/// A deleted and an inserted instruction in the same alignment gap (an instruction changed
/// into another opcode, e.g. `cmpwi` -> `cmplwi`).
pub const PEN_SUBST: u64 = 100;
pub const PEN_RELOC: u64 = 30;

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct DiffProfile {
    /// Aligned instructions with the same opcode differing only in registers.
    pub reg: u32,
    /// Same opcode, different r1-relative displacement.
    pub stack: u32,
    /// Branch displacement / condition differences.
    pub branch: u32,
    /// Same opcode, other immediate differences.
    pub other: u32,
    /// Instructions present in both but at different places.
    pub reorder: u32,
    /// Instructions replaced by a different opcode at the same place.
    pub substituted: u32,
    pub inserted: u32,
    pub deleted: u32,
    /// Code identical but relocations/literals differ.
    pub reloc: u32,
    /// Categories of inserted/deleted/substituted instructions ([`res`] bits).
    pub residue: u32,
}

impl DiffProfile {
    pub fn structural(&self) -> u32 {
        self.inserted + self.deleted + self.reorder + self.substituted
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Fitness {
    pub exact: bool,
    /// Permuter-style penalty (0 iff exact).
    pub penalty: u64,
    /// Strict comparator score (0..100).
    pub score: f64,
    #[serde(serialize_with = "ser_class")]
    pub class: DiffClass,
    /// ours - target, in instructions.
    pub size_delta: i64,
    pub profile: DiffProfile,
}

fn ser_class<S: serde::Serializer>(c: &DiffClass, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(c.label())
}

impl Fitness {
    /// Total order: exact first, then lower penalty, then higher score, then smaller size delta.
    pub fn cmp_better(&self, o: &Fitness) -> Ordering {
        o.exact
            .cmp(&self.exact)
            .then(self.penalty.cmp(&o.penalty))
            .then(o.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal))
            .then(self.size_delta.unsigned_abs().cmp(&o.size_delta.unsigned_abs()))
    }
    pub fn better_than(&self, o: &Fitness) -> bool {
        self.cmp_better(o) == Ordering::Less
    }
}

/// Opcode identity: primary opcode plus extended opcode where it exists, plus Rc.
fn opkey(w: u32) -> u32 {
    let p = w >> 26;
    match p {
        19 | 31 | 63 => (p << 16) | ((w >> 1) & 0x3ff) << 1 | (w & 1),
        59 => (p << 16) | ((w >> 1) & 0x1f) << 1 | (w & 1),
        4 => (p << 16) | ((w >> 1) & 0x3ff),
        _ => p << 16,
    }
}

fn is_dform_mem(p: u32) -> bool {
    (32..=55).contains(&p) || p == 14 || p == 15 || p == 56 || p == 57 || p == 60 || p == 61
}

fn classify_pair(a: u32, b: u32, prof: &mut DiffProfile) -> u64 {
    if a == b {
        return 0;
    }
    let p = a >> 26;
    let x = a ^ b;
    if p == 16 || p == 18 {
        prof.branch += 1;
        return PEN_BRANCH;
    }
    if is_dform_mem(p) && x & 0xffff != 0 {
        let ra = (a >> 16) & 31;
        if ra == 1 && (b >> 16) & 31 == 1 {
            prof.stack += 1;
            return PEN_STACK;
        }
        prof.other += 1;
        return PEN_OTHER;
    }
    let imm_differs = match p {
        7 | 8 | 10 | 11 | 12 | 13 | 24..=29 => x & 0xffff != 0,
        // rlwimi/rlwinm/rlwnm: SH/MB/ME fields
        20 | 21 | 23 => x & 0xfffe != 0,
        _ => false,
    };
    if imm_differs {
        prof.other += 1;
        return PEN_OTHER;
    }
    prof.reg += 1;
    PEN_REG
}

/// Opcode-level LCS alignment; returns matched index pairs.
fn align(a: &[u32], b: &[u32]) -> Vec<(usize, usize)> {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return vec![];
    }
    if n.saturating_mul(m) > 12_000_000 {
        // Positional fallback for huge functions.
        return (0..n.min(m)).filter(|&i| a[i] == b[i]).map(|i| (i, i)).collect();
    }
    let w = m + 1;
    let mut dp = vec![0u16; (n + 1) * w];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            dp[i * w + j] = if a[i] == b[j] { dp[(i + 1) * w + j + 1] + 1 } else { dp[(i + 1) * w + j].max(dp[i * w + j + 1]) };
        }
    }
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n && j < m {
        if a[i] == b[j] {
            out.push((i, j));
            i += 1;
            j += 1;
        } else if dp[(i + 1) * w + j] >= dp[i * w + j + 1] {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Opcode-level alignment of target words `t` and ours `o` (matched index pairs).
pub fn align_pairs(t: &[u32], o: &[u32]) -> Vec<(usize, usize)> {
    let tk: Vec<u32> = t.iter().map(|&w| opkey(w)).collect();
    let ok: Vec<u32> = o.iter().map(|&w| opkey(w)).collect();
    align(&tk, &ok)
}

/// Penalty and profile for target words `t` vs ours `o` (masked).
pub fn penalty(t: &[u32], o: &[u32]) -> (u64, DiffProfile) {
    let mut prof = DiffProfile::default();
    let tk: Vec<u32> = t.iter().map(|&w| opkey(w)).collect();
    let ok: Vec<u32> = o.iter().map(|&w| opkey(w)).collect();
    let pairs = align(&tk, &ok);
    let mut pen = 0;
    let mut tm = vec![false; t.len()];
    let mut om = vec![false; o.len()];
    for &(i, j) in &pairs {
        tm[i] = true;
        om[j] = true;
        pen += classify_pair(t[i], o[j], &mut prof);
    }
    // Unmatched instructions per alignment gap: pairs of different opcodes in the same gap are
    // substitutions; the rest are paired by opcode across gaps as reorderings, else ins/del.
    let mut del: Vec<u32> = Vec::new();
    let mut ins: Vec<u32> = Vec::new();
    let mut bounds: Vec<(usize, usize)> = pairs.clone();
    bounds.push((t.len(), o.len()));
    let (mut pi, mut pj) = (0usize, 0usize);
    for &(bi, bj) in &bounds {
        let gd: Vec<u32> = (pi..bi).map(|i| tk[i]).collect();
        let gi: Vec<u32> = (pj..bj).map(|j| ok[j]).collect();
        let k = gd.len().min(gi.len());
        // Same opcode in one gap would have been aligned unless order differs: keep those for
        // reorder pairing; substitute the rest pairwise.
        let mut gd_rest = Vec::new();
        let mut gi_rest = gi.clone();
        let mut subst = 0;
        for &d in &gd {
            if let Some(pos) = gi_rest.iter().position(|&x| x == d) {
                // equal opcode within the gap: a local reordering
                gi_rest.remove(pos);
                prof.reorder += 1;
                pen += PEN_REORDER;
            } else {
                gd_rest.push(d);
            }
        }
        let k2 = gd_rest.len().min(gi_rest.len());
        let _ = k;
        subst += k2;
        prof.substituted += k2 as u32;
        pen += k2 as u64 * PEN_SUBST;
        del.extend(gd_rest.into_iter().skip(k2));
        ins.extend(gi_rest.into_iter().skip(k2));
        let _ = subst;
        pi = bi + 1;
        pj = bj + 1;
    }
    del.sort_unstable();
    ins.sort_unstable();
    let (mut i, mut j, mut r) = (0, 0, 0u32);
    while i < del.len() && j < ins.len() {
        match del[i].cmp(&ins[j]) {
            Ordering::Equal => {
                r += 1;
                i += 1;
                j += 1;
            }
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
        }
    }
    prof.reorder += r;
    prof.deleted = del.len() as u32 - r;
    prof.inserted = ins.len() as u32 - r;
    prof.residue = residue(t, o, &pairs);
    pen += r as u64 * PEN_REORDER + (prof.deleted + prof.inserted) as u64 * PEN_INSDEL;
    (pen, prof)
}

// ------------------------------------------------------------------ residue signature

/// Residue categories (bits of [`DiffProfile::residue`]): what kind of instructions are inserted,
/// deleted or substituted. Derived from a train-split analysis of mismatches left by the search;
/// they steer operator weights ([`residue_mult`]).
pub mod res {
    /// zero/sign extension: extsb, extsh, clrlwi, lha/lhz
    pub const EXT: u32 = 1;
    /// signedness: cmpw/cmplw, cmpwi/cmplwi, srawi/srwi, divw/divwu, addze
    pub const SIGN: u32 = 2;
    /// bool normalisation: neg, cntlzw, `srwi rX,rY,31`
    pub const BOOL: u32 = 4;
    /// double precision: frsp, fadd/fsub/fmul/fdiv (double), lfd
    pub const DOUBLE: u32 = 8;
    /// fused multiply-add vs separate multiply/add
    pub const FMA: u32 = 16;
    /// int<->float conversions: xoris, fctiwz, stfd
    pub const CONV: u32 = 32;
    /// branches: b, bc, cror
    pub const BRANCH: u32 = 64;
    /// calls: bl, bctrl
    pub const CALL: u32 = 128;
    /// stack traffic: loads/stores through r1
    pub const STACK: u32 = 256;
    /// register moves: mr, fmr
    pub const MOVE: u32 = 512;
    /// constants / addresses: li, lis, addi
    pub const CONST: u32 = 1024;
}

/// Residue categories of one instruction word.
pub fn word_class(w: u32) -> u32 {
    use res::*;
    let p = w >> 26;
    let xo10 = (w >> 1) & 0x3ff;
    let xo5 = (w >> 1) & 0x1f;
    let ra = (w >> 16) & 31;
    let rb = (w >> 11) & 31;
    let rs = (w >> 21) & 31;
    match p {
        10 | 11 => SIGN,
        14 => {
            if ra == 1 {
                STACK
            } else {
                CONST
            }
        }
        15 => CONST,
        16 => BRANCH,
        18 => {
            if w & 1 == 1 {
                CALL
            } else {
                BRANCH
            }
        }
        19 => match xo10 {
            449 | 417 | 193 | 225 | 257 | 289 | 129 | 33 => BRANCH,
            528 if w & 1 == 1 => CALL,
            _ => BRANCH,
        },
        21 => {
            let sh = (w >> 11) & 31;
            let mb = (w >> 6) & 31;
            let me = (w >> 1) & 31;
            if sh == 0 && me == 31 && mb >= 16 {
                EXT
            } else if me == 31 && sh != 0 && mb == 32 - sh {
                if mb == 31 {
                    BOOL | SIGN
                } else {
                    SIGN
                }
            } else {
                0
            }
        }
        27 => CONV,
        31 => match xo10 {
            954 | 922 => EXT,
            0 | 32 | 824 | 792 | 536 | 491 | 459 | 202 | 75 | 11 => SIGN,
            104 | 26 => BOOL,
            444 if rs == rb => MOVE,
            _ => 0,
        },
        42 | 43 => EXT,
        40 | 41 => EXT,
        32..=39 | 44..=47 | 48..=49 | 52..=53 => {
            if ra == 1 {
                STACK
            } else {
                0
            }
        }
        50 | 51 => DOUBLE | if ra == 1 { STACK } else { 0 },
        54 | 55 => CONV | if ra == 1 { STACK } else { 0 },
        59 => match xo5 {
            28 | 29 | 30 | 31 => FMA,
            25 | 21 | 20 => FMA,
            _ => 0,
        },
        63 => match xo5 {
            18 | 20 | 21 | 25 | 28 | 29 | 30 | 31 => DOUBLE,
            _ => match xo10 {
                12 => DOUBLE,
                15 | 14 => CONV,
                72 => MOVE,
                _ => 0,
            },
        },
        _ => 0,
    }
}

/// Residue signature of target words `t` vs ours `o` given the opcode alignment `pairs`:
/// categories of unmatched instructions whose opcode does not also appear unmatched on the other
/// side (those are reorderings).
pub fn residue(t: &[u32], o: &[u32], pairs: &[(usize, usize)]) -> u32 {
    let mut tm = vec![false; t.len()];
    let mut om = vec![false; o.len()];
    for &(i, j) in pairs {
        tm[i] = true;
        om[j] = true;
    }
    let tu: Vec<u32> = (0..t.len()).filter(|&i| !tm[i]).map(|i| t[i]).collect();
    let ou: Vec<u32> = (0..o.len()).filter(|&j| !om[j]).map(|j| o[j]).collect();
    let mut ok: Vec<u32> = ou.iter().map(|&w| opkey(w)).collect();
    let mut sig = 0;
    for &w in &tu {
        let k = opkey(w);
        if let Some(p) = ok.iter().position(|&x| x == k) {
            ok.swap_remove(p);
            continue;
        }
        sig |= word_class(w);
    }
    let mut tk: Vec<u32> = tu.iter().map(|&w| opkey(w)).collect();
    for &w in &ou {
        let k = opkey(w);
        if let Some(p) = tk.iter().position(|&x| x == k) {
            tk.swap_remove(p);
            continue;
        }
        sig |= word_class(w);
    }
    sig
}

/// Operator weight multiplier for a residue signature (train-split analysis: which knobs remove
/// which kinds of inserted/deleted instructions; catalog rows 5, 7, 10, 16, 22).
pub fn residue_mult(op: &str, sig: u32) -> f64 {
    use res::*;
    if sig == 0 {
        return 1.0;
    }
    let table: &[(u32, &[(&str, f64)])] = &[
        (EXT, &[("local_type", 3.0), ("sign_cast", 2.5), ("add_cast", 2.0), ("remove_cast", 2.0), ("decl_type", 2.5), ("bool_literal", 1.5)]),
        (SIGN, &[("sign_cast", 3.0), ("local_type", 3.0), ("cmp_const", 2.5), ("decl_type", 2.0), ("add_cast", 1.5)]),
        (BOOL, &[("decl_type", 4.0), ("bool_literal", 3.0), ("bool_return", 3.0), ("local_type", 2.0), ("cond_zero", 1.5)]),
        (DOUBLE, &[("float_literal", 5.0), ("fold_magic", 3.0), ("add_cast", 2.0), ("remove_cast", 2.0), ("decl_type", 2.0)]),
        (FMA, &[("extract_temp", 2.0), ("inline_temp", 2.0), ("associative", 2.5), ("commutative", 1.5), ("compound_assign", 2.0)]),
        (CONV, &[("fold_magic", 4.0), ("add_cast", 2.0), ("remove_cast", 2.0), ("local_type", 2.0), ("decl_type", 2.0)]),
        (BRANCH, &[("ternary_self", 3.0), ("negate_if", 2.0), ("if_to_ternary", 2.0), ("ternary_to_if", 2.0), ("guard_split", 2.0), ("push_not", 1.5), ("loop_exit", 1.5), ("switch_to_if", 1.5), ("if_to_switch", 1.5), ("select_init", 2.0)]),
        (MOVE, &[("extract_temp", 1.5), ("inline_temp", 1.5), ("cse_temp", 1.5), ("inline_var_all", 1.5), ("refer_to_var", 1.5), ("chain_assign", 1.5)]),
        (STACK, &[("forward_stack", 3.0), ("ref_local", 2.0), ("inline_var_all", 1.5), ("inline_temp", 1.5), ("addr_form", 1.5)]),
        (CONST, &[("addr_form", 2.0), ("cse_temp", 1.5), ("extract_temp", 1.3), ("associative", 1.5), ("index_loop", 1.5)]),
    ];
    let mut m = 1.0f64;
    for (bit, ops) in table {
        if sig & bit != 0 {
            if let Some((_, f)) = ops.iter().find(|(n, _)| *n == op) {
                m = m.max(*f);
            }
        }
    }
    m
}

/// Fitness of our function vs the target function.
pub fn fitness(t: &ObjIndex, tf: &Function, o: &ObjIndex, of: &Function) -> Fitness {
    let d = compare_indexed(t, tf, o, of);
    let tw = masked_words(tf);
    let ow = masked_words(of);
    let (mut pen, mut prof) = penalty(&tw, &ow);
    if !d.result.exact && pen == 0 {
        prof.reloc = d.classes.len().max(1) as u32;
        pen = PEN_RELOC * prof.reloc as u64;
    }
    if d.result.exact {
        pen = 0;
    }
    Fitness {
        exact: d.result.exact,
        penalty: pen,
        score: d.result.score,
        class: d.class,
        size_delta: ow.len() as i64 - tw.len() as i64,
        profile: prof,
    }
}

/// Outcome of evaluating one candidate.
#[derive(Clone, Debug)]
pub enum Eval {
    Ok(Fitness),
    /// The compiler rejected the candidate (first error line).
    CompileError(String),
    /// Compiled, but the symbol is not defined (name/signature mismatch).
    Missing(Vec<String>),
    Io(String),
}

impl Eval {
    pub fn fitness(&self) -> Option<&Fitness> {
        match self {
            Eval::Ok(f) => Some(f),
            _ => None,
        }
    }
}

/// Everything needed to score candidates for one target function.
pub struct Scorer<'a> {
    pub mwcc: &'a Mwcc,
    pub ctx: &'a UnitContext,
    /// Context without PCH, used when the compiler crashes with the PCH (negative exit status).
    pub plain: Option<&'a UnitContext>,
    pub target: &'a ObjIndex<'a>,
    pub tf: &'a Function,
    /// Extern index for our side (literal resolution); `None` = our object only.
    pub ours_ext: Option<&'a ExternIndex>,
    pub symbol: String,
    /// Set once the PCH crashed the compiler: use `plain` from then on.
    pub pch_broken: std::sync::atomic::AtomicBool,
    /// Proves target placeholders (`fn_<addr>`) equal to our named functions.
    pub prover: Option<&'a dyn mwdec_mwcc::PlaceholderProver>,
}

impl<'a> Scorer<'a> {
    pub fn new(
        mwcc: &'a Mwcc,
        ctx: &'a UnitContext,
        plain: Option<&'a UnitContext>,
        target: &'a ObjIndex<'a>,
        tf: &'a Function,
        ours_ext: Option<&'a ExternIndex>,
        symbol: &str,
    ) -> Scorer<'a> {
        Scorer { mwcc, ctx, plain, target, tf, ours_ext, symbol: symbol.to_string(), pch_broken: Default::default(), prover: None }
    }

    /// Resolve target placeholders through `p` (see `mwdec_mwcc::placeholder`).
    pub fn with_prover(mut self, p: Option<&'a dyn mwdec_mwcc::PlaceholderProver>) -> Self {
        self.prover = p;
        self
    }
}

/// The compiler crashed (access violation, seen with some units' PCH) rather than rejecting the
/// input. Exit status is negative or 1 with an "Unhandled exception" report.
pub fn is_crash(e: &MwccError) -> bool {
    // Text-based so it also covers a dedicated crash variant in mwdec-mwcc.
    let m = e.messages().to_ascii_lowercase();
    let d = e.to_string().to_ascii_lowercase();
    m.contains("unhandled exception")
        || m.contains("access violation")
        || m.contains("internal compiler error")
        || d.starts_with("mwcc crashed")
        || matches!(e, MwccError::Compile { status: Some(s), .. } if *s < 0)
}

/// First meaningful line of a compiler error.
pub fn first_error(msg: &str) -> String {
    let lines: Vec<&str> = msg.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        if l.trim_start().starts_with("#   Error:") {
            if let Some(n) = lines.get(i + 1) {
                return n.trim_start_matches('#').trim().to_string();
            }
        }
    }
    lines.iter().rev().find(|l| !l.trim().is_empty()).map(|s| s.trim().to_string()).unwrap_or_default()
}

impl Scorer<'_> {
    /// Compile + compare. Second value: whether the compiler actually ran (not a cache hit).
    pub fn eval(&self, src: &str) -> (Eval, bool) {
        use std::sync::atomic::Ordering::Relaxed;
        let mut r = match self.plain {
            Some(plain) if self.pch_broken.load(Relaxed) => self.mwcc.compile_in(plain, src),
            _ => self.mwcc.compile_in(self.ctx, src),
        };
        if let (Err(e), Some(plain)) = (&r, self.plain) {
            if is_crash(e) && !self.pch_broken.load(Relaxed) {
                // MWCC crashes (access violation) with some units' PCH; the plain context is
                // slower but codegen-identical.
                self.pch_broken.store(true, Relaxed);
                r = self.mwcc.compile_in(plain, src);
            }
        }
        let c = match r {
            Ok(c) => c,
            Err(e) if is_crash(&e) => return (Eval::CompileError(format!("compiler crash: {}", first_error(e.messages()))), true),
            Err(e @ MwccError::Compile { .. }) => return (Eval::CompileError(first_error(e.messages())), true),
            Err(e) => return (Eval::Io(e.to_string()), true),
        };
        let ran = !c.cache_hit;
        let ours = match mwdec_obj::load_object_bytes("candidate.o", &c.obj) {
            Ok(o) => o,
            Err(e) => return (Eval::Io(e.to_string()), ran),
        };
        let Some(of) = mwdec_obj::find_function(&ours, &self.symbol) else {
            return (Eval::Missing(ours.functions.iter().map(|f| f.name.clone()).collect()), ran);
        };
        let oi = match self.ours_ext {
            Some(e) => ObjIndex::with_externs(&ours, e),
            None => ObjIndex::new(&ours),
        }
        .with_prover(self.prover);
        let fit = fitness(self.target, self.tf, &oi, of);
        // An exact match from the fast path (a persistent compiler) only counts once a normal
        // compile of the same candidate confirms it.
        if fit.exact && c.fast {
            let ctx = match self.plain {
                Some(p) if self.pch_broken.load(Relaxed) => p,
                _ => self.ctx,
            };
            let mut n = self.mwcc.compile_in_normal(ctx, src);
            if let (Err(e), Some(plain)) = (&n, self.plain) {
                if is_crash(e) {
                    n = self.mwcc.compile_in_normal(plain, src);
                }
            }
            match n.ok().and_then(|c| self.fitness_of(&c.obj)) {
                Some(f) => {
                    self.mwcc.note_fast_confirm(f.exact);
                    if !f.exact {
                        return (Eval::Ok(f), true);
                    }
                }
                None => {
                    self.mwcc.note_fast_confirm(false);
                    return (Eval::CompileError("normal-compile confirmation failed".into()), true);
                }
            }
        }
        // Candidates of a context with a split PCH (mwdec-mwcc works around a compiler crash by
        // moving a few headers out of the PCH): confirm an exact match with the plain context.
        if fit.exact && !self.pch_broken.load(Relaxed) && self.mwcc.uses_split(self.ctx) {
            if let Some(plain) = self.plain {
                let confirmed = self.mwcc.compile_in_normal(plain, src).ok().and_then(|c| self.fitness_of(&c.obj));
                match confirmed {
                    Some(f) if f.exact => {}
                    Some(f) => {
                        eprintln!("mwdec-search: split-PCH exact match of {} not confirmed by the plain context", self.symbol);
                        return (Eval::Ok(f), true);
                    }
                    None => return (Eval::CompileError("plain-context confirmation failed".into()), true),
                }
            }
        }
        (Eval::Ok(fit), ran)
    }

    /// Fitness of the target function in a compiled object (`None`: unreadable or missing).
    fn fitness_of(&self, obj: &[u8]) -> Option<Fitness> {
        let o = mwdec_obj::load_object_bytes("candidate.o", obj).ok()?;
        let of = mwdec_obj::find_function(&o, &self.symbol)?;
        let oi = match self.ours_ext {
            Some(e) => ObjIndex::with_externs(&o, e),
            None => ObjIndex::new(&o),
        }
        .with_prover(self.prover);
        Some(fitness(self.target, self.tf, &oi, of))
    }
}

