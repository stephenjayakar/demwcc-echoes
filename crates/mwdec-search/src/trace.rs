//! Register-only diffs: ask the real compiler (via the tracer, `mwdec_oracle::tracer`)
//! how it coloured the candidate, derive which registers disagree with the target, and turn the
//! colouring rules (regalloc.md) into concrete source edits ("declare x before y", "make x
//! temp-class"). Diagnostic only: candidates are still judged by the plain compiler.
//!
//! Covers both register classes (GPR and FPR) and every allocatable register: callee-saved ones
//! (r14-r31 / f14-f31, obtained from the top down) and volatile ones (r0, r3-r12 / f0-f13, lowest
//! free first). In both cases a value whose wanted register is *more preferred* than its current
//! one must be coloured earlier, i.e. have a higher virtual register: declared earlier (named
//! locals), created later (temps), or be temp-class rather than named.
use crate::cst::Cst;
use crate::func::{self, FuncInfo};
use crate::ops;
use crate::rng::Rng;
use mwdec_core::Function;
use mwdec_mwcc::compare::masked_words;
use mwdec_oracle::compile::Compiler;
use mwdec_oracle::tracer::{trace_source, ColorRound, IgNode, TraceOptions, VregKind};
use ppc750cl::{Argument, Ins};
use std::collections::HashMap;

/// Tracing setup for one unit (GC/2.7 only).
#[derive(Clone, Debug)]
pub struct Tracer {
    pub comp: Compiler,
    /// Context TU text (include lines) prepended to the candidate.
    pub context: String,
}

/// Register class of a fix.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegClass {
    Gpr,
    Fpr,
}

impl std::fmt::Display for RegClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            RegClass::Gpr => "r",
            RegClass::Fpr => "f",
        })
    }
}

/// One register disagreement and what the colouring says about it.
#[derive(Clone, Debug)]
pub struct Fix {
    pub class: RegClass,
    /// The candidate value sitting in `from` that the target has in `to`.
    pub vreg: u16,
    pub name: Option<String>,
    pub kind: VregKind,
    pub from: u8,
    pub to: u8,
    /// The candidate value currently holding `to` (an interfering neighbour), if any.
    pub holder: Option<(u16, Option<String>, VregKind)>,
    /// `n` must be coloured before the holder (wanted register is more preferred).
    pub earlier: bool,
    pub suggestion: String,
}

impl Tracer {
    /// Tracer for a unit with these flags, or `None` if its compiler is not GC/2.7.
    pub fn new(root: &std::path::Path, work: &std::path::Path, compiler_rel: &str, cflags: &[String], context: &str) -> Option<Tracer> {
        if !compiler_rel.contains("GC/2.7/") {
            return None;
        }
        let comp = Compiler { root: root.to_path_buf(), work: work.to_path_buf(), cflags: Some(cflags.to_vec()), ..Default::default() };
        Some(Tracer { comp, context: context.to_string() })
    }

    /// Register fixes for candidate `src` of `symbol` against the target function.
    pub fn fixes(&self, src: &str, symbol: &str, target: &Function) -> anyhow::Result<Vec<Fix>> {
        let tu = format!("{}\n{}", self.context, src);
        let t = trace_source(&self.comp, &tu, &TraceOptions { coloring: true, ..Default::default() })?;
        let obj = mwdec_obj::load_object_bytes("traced.o", &t.object)?;
        let Some(of) = mwdec_obj::find_function(&obj, symbol) else { anyhow::bail!("traced object lacks {symbol}") };
        let moves = reg_moves(target, of);
        if moves.is_empty() {
            return Ok(vec![]);
        }
        let base = symbol.split("__").find(|s| !s.is_empty()).unwrap_or(symbol);
        let mut out = Vec::new();
        for class in [RegClass::Gpr, RegClass::Fpr] {
            let cname = if class == RegClass::Gpr { "GPR" } else { "FPR" };
            let mv: Vec<(u8, u8)> = moves.iter().filter(|m| m.0 == class).map(|m| (m.1, m.2)).collect();
            if mv.is_empty() {
                continue;
            }
            let round = t
                .rounds
                .iter()
                .find(|r| r.class == cname && r.function == symbol)
                .or_else(|| t.rounds.iter().find(|r| r.class == cname && r.function.contains(base)));
            if let Some(r) = round {
                out.extend(colour_fixes(r, class, &mv));
            }
        }
        Ok(out)
    }
}

/// Preference rank of a register in the colouring: volatile registers lowest number first, then
/// the callee-saved registers in the order they are obtained (31 downward).
pub fn pref_rank(r: u8) -> u32 {
    if r <= 13 {
        r as u32
    } else {
        14 + (31 - r as u32)
    }
}

fn allocatable(class: RegClass, r: u8) -> bool {
    match class {
        RegClass::Gpr => !matches!(r, 1 | 2 | 13),
        RegClass::Fpr => true,
    }
}

/// Register operands of an instruction in order (class, number).
fn reg_args(w: u32) -> (String, Vec<(RegClass, u8)>) {
    let p = Ins::new(w).simplified();
    let regs = p
        .args_iter()
        .filter_map(|a| match a {
            Argument::GPR(g) => Some((RegClass::Gpr, g.0)),
            Argument::FPR(f) => Some((RegClass::Fpr, f.0)),
            _ => None,
        })
        .collect();
    (p.mnemonic.to_string(), regs)
}

/// (class, candidate register, target register) for registers that disagree, by majority over
/// position-aligned instructions with equal opcodes (register-only diffs keep positions).
pub fn reg_moves(target: &Function, ours: &Function) -> Vec<(RegClass, u8, u8)> {
    let tw = masked_words(target);
    let ow = masked_words(ours);
    if tw.len() != ow.len() {
        return vec![];
    }
    let mut votes: HashMap<(RegClass, u8, u8), usize> = HashMap::new();
    for (&a, &b) in tw.iter().zip(&ow) {
        if a == b || a >> 26 != b >> 26 {
            continue;
        }
        let (ma, ra) = reg_args(a);
        let (mb, rb) = reg_args(b);
        if ma != mb || ra.len() != rb.len() {
            continue;
        }
        for (&(ca, t), &(cb, o)) in ra.iter().zip(&rb) {
            if ca == cb && t != o && allocatable(ca, t) && allocatable(ca, o) {
                *votes.entry((ca, o, t)).or_default() += 1;
            }
        }
    }
    // One target register per candidate register (highest vote).
    let mut best: HashMap<(RegClass, u8), (u8, usize)> = HashMap::new();
    for ((c, o, t), n) in votes {
        let e = best.entry((c, o)).or_insert((t, 0));
        if n > e.1 || (n == e.1 && t < e.0) {
            *e = (t, n);
        }
    }
    let mut v: Vec<(RegClass, u8, u8)> = best.into_iter().map(|((c, o), (t, _))| (c, o, t)).collect();
    v.sort();
    v
}

/// Callee-saved GPR moves only (the original tracer scope); kept for callers/tests.
pub fn gpr_moves(target: &Function, ours: &Function) -> Vec<(u8, u8)> {
    reg_moves(target, ours).into_iter().filter(|m| m.0 == RegClass::Gpr && m.1 >= 14 && m.2 >= 14).map(|m| (m.1, m.2)).collect()
}

fn label(n: &IgNode) -> String {
    match &n.name {
        Some(s) => format!("`{s}` (v{})", n.vreg),
        None => format!("temp v{}", n.vreg),
    }
}

/// Colouring-rule fixes for `moves` = (candidate register, target register) in one class.
pub fn colour_fixes(round: &ColorRound, class: RegClass, moves: &[(u8, u8)]) -> Vec<Fix> {
    let mut out = vec![];
    let by_reg = |r: u8| -> Vec<&IgNode> {
        let mut v: Vec<&IgNode> = round.nodes.iter().filter(|n| n.reg == Some(r) && n.coalesced_into.is_none() && n.degree > 0).collect();
        // Named values first (volatile registers hold many short temps), then colouring order.
        v.sort_by_key(|n| (n.name.is_none(), n.order.unwrap_or(usize::MAX)));
        v
    };
    for &(from, to) in moves {
        let earlier = pref_rank(to) < pref_rank(from);
        let mut found = false;
        for n in by_reg(from) {
            let holder = by_reg(to).into_iter().find(|h| h.neighbours.contains(&(n.vreg as i16)));
            if holder.is_none() && n.name.is_none() {
                continue;
            }
            let s = if n.blocked {
                format!("{} is K-blocked (degree {}): coloured before everything else", label(n), n.degree)
            } else if let Some(h) = holder {
                let hn = label(h);
                match (n.kind, h.kind, earlier) {
                    (VregKind::Named, VregKind::Named, true) => format!("declare {} before {}", label(n), hn),
                    (VregKind::Named, VregKind::Named, false) => format!("declare {} after {}", label(n), hn),
                    (VregKind::Temp, VregKind::Temp, true) => format!("create {} after {}", label(n), hn),
                    (VregKind::Temp, VregKind::Temp, false) => format!("create {} before {}", label(n), hn),
                    (VregKind::Named, VregKind::Temp, true) => format!("make {} temp-class or {} named", label(n), hn),
                    (VregKind::Temp, VregKind::Named, false) => format!("make {} named or {} temp-class", label(n), hn),
                    _ => format!("{} vs {}: reverse their class order", label(n), hn),
                }
            } else {
                format!("{} reuses {}{}: the value holding {}{} must overlap it", label(n), class, from, class, to)
            };
            out.push(Fix {
                class,
                vreg: n.vreg,
                name: n.name.clone(),
                kind: n.kind,
                from,
                to,
                holder: holder.map(|h| (h.vreg, h.name.clone(), h.kind)),
                earlier,
                suggestion: s,
            });
            found = true;
            break;
        }
        let _ = found;
    }
    out
}

/// Source edits implementing a fix. Returns candidate sources.
pub fn apply_fix(src: &str, symbol: &str, fix: &Fix) -> Vec<String> {
    let mut out = Vec::new();
    let earlier = fix.earlier;
    let Some((_, hname, hkind)) = fix.holder.clone() else {
        // No interfering holder: a named value that should get a more preferred register can be
        // made to start earlier by declaring it first (stack of declarations at the top).
        if let (Some(n), VregKind::Named, true) = (&fix.name, fix.kind, earlier) {
            out.extend(move_decl_first(src, symbol, n));
        }
        return out;
    };
    match (fix.kind, hkind, &fix.name, &hname) {
        (VregKind::Named, VregKind::Named, Some(n), Some(h)) => {
            let (a, b) = if earlier { (n, h) } else { (h, n) };
            out.extend(move_decl_before(src, symbol, a, b));
        }
        (VregKind::Named, VregKind::Temp, Some(n), _) if earlier => out.extend(unname(src, symbol, n)),
        (VregKind::Temp, VregKind::Named, _, Some(h)) if !earlier => out.extend(unname(src, symbol, h)),
        (VregKind::Param, _, Some(n), _) | (_, VregKind::Param, _, Some(n)) => {
            // const on a by-value param passed as a call argument turns it into an FE temp.
            out.extend(const_param(src, symbol, n));
        }
        _ => {}
    }
    out
}

/// Move the declaration of local `a` to just before the declaration of local `b` (same block),
/// splitting an initializer off into an assignment at the old place.
pub fn move_decl_before(src: &str, symbol: &str, a: &str, b: &str) -> Option<String> {
    let cst = Cst::parse(src);
    let def = func::find_target(&cst, symbol)?;
    let info = FuncInfo::analyze(&cst, def)?;
    let da = info.vars.get(a).filter(|v| !v.is_param)?.decl;
    let db = info.vars.get(b).filter(|v| !v.is_param)?.decl;
    if cst.parent(da) != cst.parent(db) || cst.nodes[da].start <= cst.nodes[db].start {
        return None;
    }
    move_decl_to(&cst, src, da, a, cst.nodes[db].start)
}

/// Move the declaration of local `a` to the top of its block.
pub fn move_decl_first(src: &str, symbol: &str, a: &str) -> Option<String> {
    let cst = Cst::parse(src);
    let def = func::find_target(&cst, symbol)?;
    let info = FuncInfo::analyze(&cst, def)?;
    let da = info.vars.get(a).filter(|v| !v.is_param)?.decl;
    let p = cst.parent(da)?;
    let first = cst.named(p).into_iter().find(|&c| func::is_stmt(cst.kind(c)))?;
    if first == da {
        return None;
    }
    move_decl_to(&cst, src, da, a, cst.nodes[first].start)
}

fn move_decl_to(cst: &Cst, src: &str, da: usize, a: &str, at: usize) -> Option<String> {
    let ds: Vec<usize> = cst.children_by_field(da, "declarator").collect();
    if ds.len() != 1 {
        return None;
    }
    let d = ds[0];
    let line_start = cst.src[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let ind: String = cst.src[line_start..at].chars().take_while(|c| c.is_whitespace()).collect();
    let prefix = &cst.src[cst.nodes[da].start..cst.nodes[d].start];
    let edits = if cst.kind(d) == "init_declarator" {
        let decl = cst.child(d, "declarator")?;
        let val = cst.child(d, "value")?;
        if cst.kind(val) == "initializer_list" || cst.kind(val) == "argument_list" {
            return None;
        }
        vec![
            crate::cst::Edit::insert(at, format!("{prefix}{};\n{ind}", cst.text(decl))),
            crate::cst::Edit::replace(cst, da, format!("{a} = {};", cst.text(val))),
        ]
    } else {
        vec![crate::cst::Edit::insert(at, format!("{}\n{ind}", cst.text(da))), crate::cst::Edit::replace(cst, da, String::new())]
    };
    let out = crate::cst::apply(src, &edits)?;
    (Cst::parse(&out).errors <= cst.errors).then_some(out)
}

/// Make local `name` temp-class by inlining it at its use sites.
pub fn unname(src: &str, symbol: &str, name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for op in ["inline_var_all", "inline_temp"] {
        let mut rng = Rng::new(7);
        if let Some(s) = ops::apply_op_focused(src, symbol, ops::op_index(op).unwrap(), &mut rng, name) {
            if !out.contains(&s) {
                out.push(s);
            }
        }
    }
    out
}

/// Toggle top-level `const` on by-value parameter `name`.
pub fn const_param(src: &str, symbol: &str, name: &str) -> Option<String> {
    let cst = Cst::parse(src);
    let def = func::find_target(&cst, symbol)?;
    let info = FuncInfo::analyze(&cst, def)?;
    let v = info.vars.get(name).filter(|v| v.is_param)?;
    if v.ty.contains('&') || v.ty.contains('*') {
        return None;
    }
    let p = v.decl;
    let t = cst.text(p);
    let new = match t.strip_prefix("const ") {
        Some(r) => r.to_string(),
        None => format!("const {t}"),
    };
    let out = crate::cst::apply(src, &[crate::cst::Edit::replace(&cst, p, new)])?;
    (Cst::parse(&out).errors <= cst.errors).then_some(out)
}

