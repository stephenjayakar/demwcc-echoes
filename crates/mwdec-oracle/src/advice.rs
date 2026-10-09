//! Stable entry points for the match search: questions about one candidate, answered by the real
//! compiler (tracer probes) and the dependence checker.
//!
//! - [`statement_moves`]: which source statement must move before which, and why; or that the
//!   remaining order differences are not steerable by statement order (priority / register reuse).
//! - [`peepholes_fired`]: which post-RA peephole rules changed the candidate's code, and whether the
//!   target kept the instruction form the rule removed (the rule did not fire there).
//! - [`copies`]: the copy-propagation decision for every copy/use pair (temp vs named class).
//!
//! Compatibility: the result types are `#[non_exhaustive]` and only ever gain fields or variants;
//! function signatures stay as they are (new behaviour gets new functions). [`API_VERSION`] counts
//! additions. Lines are 1-based lines of [`Candidate::src`] (the context is subtracted), `None` when
//! the instruction comes from the context (inline functions) or has no line.
//!
//! All of this is diagnostic: a match is always decided by a plain compile.

use crate::asm;
use crate::compile::Compiler;
use crate::explain::{explain_sched_diff, OrderAdvice, Verdict};
use crate::sched::{EdgeKind, WhyKind};
use crate::tracer::{trace_source, TraceOptions};
use anyhow::{anyhow, Result};
use serde::Serialize;

/// Bumped when something is added to this module.
pub const API_VERSION: u32 = 1;

/// One candidate: the unit context (include lines and declarations) and the candidate text that
/// follows it. The translation unit compiled is `context + "\n" + src`, as in the search's tracer.
#[derive(Clone, Copy, Debug)]
pub struct Candidate<'a> {
    pub context: &'a str,
    pub src: &'a str,
    /// mangled symbol of the function of interest
    pub symbol: &'a str,
}

impl Candidate<'_> {
    pub fn tu(&self) -> String {
        format!("{}\n{}", self.context, self.src)
    }
    /// Number of translation-unit lines before the first line of `src`.
    pub fn line_offset(&self) -> u32 {
        self.context.matches('\n').count() as u32 + 1
    }
    /// 1-based line range of the definition of `symbol` in `src` (signature line to closing brace),
    /// found by the function's name followed by `(` and a body; None if not found.
    pub fn body_lines(&self) -> Option<(u32, u32)> {
        let name = self.symbol.split("__").find(|s| !s.is_empty())?;
        let text = self.src;
        let bytes = text.as_bytes();
        let mut from = 0;
        while let Some(k) = text[from..].find(name).map(|k| k + from) {
            from = k + name.len();
            let before_ok = k == 0 || !(bytes[k - 1].is_ascii_alphanumeric() || bytes[k - 1] == b'_');
            let after = text[from..].trim_start();
            if !before_ok || !after.starts_with('(') {
                continue;
            }
            // a definition: '{' comes before any ';' after the parameter list
            let rest = &text[from..];
            let (Some(open), semi) = (rest.find('{'), rest.find(';')) else { continue };
            if semi.is_some_and(|s| s < open) {
                continue;
            }
            let open = from + open;
            let mut depth = 0i32;
            let mut i = open;
            let mut in_str: Option<u8> = None;
            while i < bytes.len() {
                let c = bytes[i];
                const BACKSLASH: u8 = 92;
                const QUOTE: u8 = 34;
                const APOS: u8 = 39;
                const NL: u8 = 10;
                match in_str {
                    Some(q) => {
                        if c == BACKSLASH {
                            i += 1;
                        } else if c == q {
                            in_str = None;
                        }
                    }
                    None => match c {
                        QUOTE | APOS => in_str = Some(c),
                        b'/' if bytes.get(i + 1) == Some(&b'/') => {
                            while i < bytes.len() && bytes[i] != NL {
                                i += 1;
                            }
                        }
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                let line_of = |pos: usize| bytes[..pos].iter().filter(|&&b| b == NL).count() as u32 + 1;
                                return Some((line_of(k), line_of(i)));
                            }
                        }
                        _ => {}
                    },
                }
                i += 1;
            }
            return None;
        }
        None
    }

    fn src_line(&self, tu_line: Option<u32>) -> Option<u32> {
        tu_line.and_then(|l| l.checked_sub(self.line_offset())).filter(|&l| l > 0)
    }
}

/// Why a statement has to move.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub enum MoveReason {
    /// The candidate's order is forced by a memory-order dependence (store/load through possibly
    /// aliasing addresses): the scheduler can never swap them, the statements must be swapped.
    MemoryDependence,
    /// Both instructions were ready with equal priority and program order decided.
    ProgramOrder,
    /// The wanted instruction waited for an operand: the statement computing it must come earlier.
    WaitFor { instr: String },
}

/// Move source line `line` so that it comes before line `before`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct StatementMove {
    pub line: u32,
    pub before: u32,
    pub reason: MoveReason,
    /// how many order differences ask for this move
    pub votes: u32,
}

/// Every instruction-order difference of a candidate against the target, classified.
#[derive(Clone, Debug, Default, Serialize)]
#[non_exhaustive]
pub struct OrderDiagnosis {
    /// statement moves, most votes first
    pub moves: Vec<StatementMove>,
    /// differences ordered by a true data dependence: rewrite the computation, not the order
    pub data_dependent: u32,
    /// differences ordered only by register reuse: fix the registers first
    pub regalloc: u32,
    /// differences decided by the scheduler's priorities: statement order is irrelevant
    pub priority: u32,
    /// differences the recorded schedule could not place (different blocks, not found)
    pub unexplained: u32,
    /// entry-block differences decided by the const qualification of a parameter (register
    /// saves against loads through it): change the parameter's type, not the order
    pub qualifier: u32,
    /// per-difference details (the full `explain` output)
    pub details: Vec<OrderAdvice>,
}

impl OrderDiagnosis {
    /// No order difference at all (the diff is registers/operands only).
    pub fn in_order(&self) -> bool {
        self.details.is_empty()
    }
    /// Differences exist but none of them can be fixed by moving a statement.
    pub fn order_not_steerable(&self) -> bool {
        !self.details.is_empty() && self.moves.is_empty()
    }
}

fn memory_path(a: &OrderAdvice) -> Option<bool> {
    [&a.pre_ra, &a.post_ra].into_iter().flatten().find_map(|e| match &e.why {
        WhyKind::Dependence(p) => Some(p.contains(&EdgeKind::Memory)),
        _ => None,
    })
}

/// Which source statements must move so that the candidate's instruction order becomes the
/// target's. `target_obj` is the target object file, `target_symbol` the target function.
pub fn statement_moves(comp: &Compiler, cand: &Candidate, target_obj: &[u8], target_symbol: &str) -> Result<OrderDiagnosis> {
    let tobj = asm::parse(target_obj)?;
    let tf = tobj.funcs.iter().find(|f| f.name == target_symbol).ok_or_else(|| anyhow!("{target_symbol} not in the target"))?.clone();
    let details = explain_sched_diff(comp, &cand.tu(), cand.symbol, &tobj, &tf)?;
    Ok(diagnose(cand, details))
}

/// [`statement_moves`] with the target function already parsed (`target_obj` gives the data
/// symbols its relocations refer to; a single-function object is enough).
pub fn statement_moves_in(comp: &Compiler, cand: &Candidate, target_obj: &asm::Obj, target: &asm::Func) -> Result<OrderDiagnosis> {
    let details = explain_sched_diff(comp, &cand.tu(), cand.symbol, target_obj, target)?;
    Ok(diagnose(cand, details))
}

fn diagnose(cand: &Candidate, details: Vec<OrderAdvice>) -> OrderDiagnosis {
    let mut d = OrderDiagnosis::default();
    // the line table of inlined header code holds header line numbers: keep moves inside the body
    let body = cand.body_lines();
    let inside = |l: u32| body.map_or(true, |(a, b)| l > a && l < b);
    let add = |line: Option<u32>, before: Option<u32>, reason: MoveReason, moves: &mut Vec<StatementMove>| -> bool {
        let (Some(line), Some(before)) = (cand.src_line(line), cand.src_line(before)) else { return false };
        if line == before || !inside(line) || !inside(before) {
            return false;
        }
        match moves.iter_mut().find(|m| m.line == line && m.before == before && m.reason == reason) {
            Some(m) => m.votes += 1,
            None => moves.push(StatementMove { line, before, reason, votes: 1 }),
        }
        true
    };
    let mut moves = vec![];
    for a in &details {
        match &a.verdict {
            Verdict::Forced(_) => {
                if memory_path(a) == Some(true) {
                    if !add(a.second.line, a.first.line, MoveReason::MemoryDependence, &mut moves) {
                        d.unexplained += 1;
                    }
                } else {
                    d.data_dependent += 1;
                }
            }
            Verdict::SwapStatements(l, b) => {
                if !add(*l, *b, MoveReason::ProgramOrder, &mut moves) {
                    d.unexplained += 1;
                }
            }
            Verdict::WaitFor(instr, l) => {
                if !add(*l, a.first.line, MoveReason::WaitFor { instr: instr.clone() }, &mut moves) {
                    d.unexplained += 1;
                }
            }
            Verdict::RegAlloc => d.regalloc += 1,
            Verdict::Priority(_) => d.priority += 1,
            Verdict::ConstBase(_) | Verdict::NonConstBase(_) => d.qualifier += 1,
            Verdict::Unknown => d.unexplained += 1,
        }
    }
    moves.sort_by(|a, b| b.votes.cmp(&a.votes).then(a.line.cmp(&b.line)).then(a.before.cmp(&b.before)));
    d.moves = moves;
    d.details = details;
    d
}

/// A post-RA peephole rule that changed one of the candidate's instructions.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct PeepholeFired {
    /// rule name (`tracer::PEEPHOLE_RULES`), e.g. `make_record_form`
    pub rule: String,
    /// the instruction the rule matched (compiler PCode text)
    pub before: String,
    /// the instruction after the rule (None: removed)
    pub after: Option<String>,
    /// instructions of the rule's input form (same mnemonic family and immediates, any registers)
    /// in the candidate's final code / in the target; the target count is None without a target
    pub cand_before: u32,
    pub target_before: Option<u32>,
    /// the same for the output form (None when the rule removed the instruction or no target)
    pub cand_after: Option<u32>,
    pub target_after: Option<u32>,
}

impl PeepholeFired {
    /// The target keeps more instructions of the rule's input form than the candidate: the rule
    /// fired in the candidate where the target's code did not trigger it, so the source pattern
    /// that triggers it must change.
    pub fn missed_in_target(&self) -> bool {
        self.target_before.is_some_and(|t| t > self.cand_before)
    }
}

/// Register-insensitive key of an instruction, from either a disassembly line (`cmpwi r0, 0x0`,
/// `lwz r3, 0x4(r31)`) or compiler PCode text (`cmpi     cr0=, r0, #0`): the mnemonic family and
/// the integer immediates.
fn loose(text: &str) -> String {
    let mut it = text.split_whitespace();
    let m = it.next().unwrap_or("");
    let fam = match m.trim_end_matches('.') {
        "li" | "addi" | "subi" | "la" => "addi",
        "lis" | "addis" | "subis" => "addis",
        "mr" | "or" => "or",
        "slwi" | "srwi" | "clrlwi" | "clrrwi" | "rotlwi" | "rotrwi" | "extlwi" | "extrwi" | "rlwinm" | "clrlslwi" => "rlwinm",
        "cmpwi" | "cmpi" => "cmpi",
        "cmplwi" | "cmpli" => "cmpli",
        "cmpw" | "cmp" => "cmp",
        "cmplw" | "cmpl" => "cmpl",
        "sub" | "subf" => "subf",
        "fmr" => "fmr",
        x => x,
    };
    let rec = if m.ends_with('.') { "." } else { "" };
    let mut imms = vec![];
    for o in it.collect::<Vec<_>>().join(" ").split(',') {
        let o = o.trim().trim_end_matches('=');
        let o = o.split('(').next().unwrap_or("").trim().trim_start_matches('#');
        let reg = |p: &str| o.strip_prefix(p).is_some_and(|n| !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()));
        if o.is_empty() || reg("r") || reg("f") || reg("cr") {
            continue;
        }
        let (neg, a) = match o.strip_prefix('-') {
            Some(a) => (true, a),
            None => (false, o),
        };
        let v = match a.strip_prefix("0x") {
            Some(h) => i64::from_str_radix(h, 16).ok(),
            None => a.parse::<i64>().ok(),
        };
        match v {
            Some(v) => imms.push(if neg { -v } else { v }.to_string()),
            None => imms.push(o.to_string()),
        }
    }
    format!("{fam}{rec} {}", imms.join(","))
}

fn func_keys(bytes: &[u8], sym: &str) -> Result<Vec<String>> {
    let o = asm::parse(bytes)?;
    let base = sym.split("__").find(|s| !s.is_empty()).unwrap_or(sym);
    let f = o
        .funcs
        .iter()
        .find(|f| f.name == sym)
        .or_else(|| o.funcs.iter().find(|f| f.name.split("__").find(|s| !s.is_empty()) == Some(base)))
        .ok_or_else(|| anyhow!("{sym} not in the object"))?;
    Ok(asm::disasm_func(&o, f, asm::AsmOpts { offsets: false, literals: false })
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.starts_with('.'))
        .map(loose)
        .collect())
}

/// Peephole rules that fired in `cand.symbol`. With `target` = (object bytes, symbol), each hit
/// also counts the rule's input/output forms in the target (see [`PeepholeFired::missed_in_target`]).
pub fn peepholes_fired(comp: &Compiler, cand: &Candidate, target: Option<(&[u8], &str)>) -> Result<Vec<PeepholeFired>> {
    let t = trace_source(comp, &cand.tu(), &TraceOptions { peephole: true, ..Default::default() })?;
    let ck = func_keys(&t.object, cand.symbol).unwrap_or_default();
    let tk = match target {
        Some((bytes, sym)) => Some(func_keys(bytes, sym)?),
        None => None,
    };
    let count = |keys: &[String], text: &str| -> u32 {
        let k = loose(text);
        keys.iter().filter(|x| **x == k).count() as u32
    };
    let base = cand.symbol.split("__").find(|s| !s.is_empty()).unwrap_or(cand.symbol);
    Ok(t
        .peephole
        .iter()
        .filter(|h| h.function == cand.symbol || h.function == base)
        .map(|h| PeepholeFired {
            rule: h.rule.clone(),
            before: h.before.clone(),
            after: h.after.clone(),
            cand_before: count(&ck, &h.before),
            target_before: tk.as_ref().map(|k| count(k, &h.before)),
            cand_after: h.after.as_ref().map(|a| count(&ck, a)),
            target_after: match (&tk, &h.after) {
                (Some(k), Some(a)) => Some(count(k, a)),
                _ => None,
            },
        })
        .collect())
}

/// Copy propagation of one copy into one use.
#[derive(Clone, Debug, Serialize)]
#[non_exhaustive]
pub struct CopyDecision {
    pub copy: String,
    pub use_text: String,
    pub accepted: bool,
    /// the compiler's reason (see probes.md section 3)
    pub reason: String,
}

/// Every copy-propagation decision in `cand.symbol` (which copies stay and so keep a value
/// named-class, which are propagated away into a temp).
pub fn copies(comp: &Compiler, cand: &Candidate) -> Result<Vec<CopyDecision>> {
    let t = trace_source(comp, &cand.tu(), &TraceOptions { copyprop: true, ..Default::default() })?;
    let base = cand.symbol.split("__").find(|s| !s.is_empty()).unwrap_or(cand.symbol);
    Ok(t
        .copyprop
        .iter()
        .filter(|c| c.function == cand.symbol || c.function == base)
        .map(|c| CopyDecision { copy: c.copy.clone(), use_text: c.use_text.clone(), accepted: c.accepted, reason: c.reason.clone() })
        .collect())
}

