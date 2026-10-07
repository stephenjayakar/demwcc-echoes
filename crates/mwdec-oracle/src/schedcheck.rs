//! Scheduling dependence checker for scheduling-only diffs.
//!
//! Given the target function and a candidate compile of the same function (same instructions, some
//! in a different order), it pairs instructions per basic block and classifies every pair whose
//! relative order differs:
//! - **Forced**: a true register dependence or a memory-order dependence (MWCC alias rules:
//!   pointer accesses all alias each other and globals; distinct globals and disjoint stack slots do
//!   not) orders the two instructions in one of the programs. MWCC's scheduler can never swap them,
//!   so the *source statements* producing them must be reordered (or the dependence removed, e.g. by
//!   caching a load in a local). With `-sym on` line info the report names the source lines.
//! - **RegAlloc**: only a register anti/output dependence (an artefact of register choice) orders
//!   them: fix the register assignment first.
//! - **Priority**: independent; the list scheduler chose by latency/critical path. Statement order
//!   matters only as a tie-break (swap statements as a low-confidence try).
//! Basic blocks end at branches, labels and calls (MWCC blocks end at calls; nothing moves across).

use crate::asm::{self, Func, Obj};
use ppc750cl::{Argument, Ins, Opcode};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct I {
    off: u32,
    key: String,  // exact text (registers included)
    loose: String, // mnemonic + non-register operands (relocs/immediates)
    defs: u64,
    uses: u64,
    mem: Option<Mem>,
    store: bool,
    call: bool,
    frame: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum Mem {
    Stack { off: i32, size: u32 },
    Global(String),
    Pointer,
}

fn mem_alias(a: &Mem, b: &Mem) -> bool {
    match (a, b) {
        (Mem::Stack { off: x, size: sx }, Mem::Stack { off: y, size: sy }) => {
            *x < *y + *sy as i32 && *y < *x + *sx as i32
        }
        (Mem::Global(x), Mem::Global(y)) => x == y,
        (Mem::Stack { .. }, _) | (_, Mem::Stack { .. }) => false,
        _ => true,
    }
}

fn access_size(op: Opcode) -> u32 {
    use Opcode::*;
    match op {
        Lbz | Lbzu | Lbzx | Stb | Stbu | Stbx => 1,
        Lhz | Lha | Lhzu | Lhau | Lhzx | Lhax | Sth | Sthu | Sthx => 2,
        Lfd | Lfdu | Lfdx | Stfd | Stfdu | Stfdx => 8,
        PsqL | PsqSt => 8,
        _ => 4,
    }
}

fn bit(r: u8) -> u64 {
    1u64 << r
}

fn decode(obj: &Obj, f: &Func) -> Vec<I> {
    let lines = asm::disasm_func(obj, f, asm::AsmOpts { offsets: true, literals: false });
    let texts: BTreeMap<u32, String> = lines
        .iter()
        .filter_map(|l| {
            let (o, t) = l.trim().split_once(": ")?;
            Some((u32::from_str_radix(o.trim(), 16).ok()?, t.trim().to_string()))
        })
        .collect();
    let rel_at: BTreeMap<u32, &asm::Rel> = f.relocs.iter().map(|r| (r.offset & !3, r)).collect();
    let frame_re = regex::Regex::new(
        r"^(stwu r1, |mflr r0$|mtlr r0$|addi r1, r1, |(stw|lwz) r0, 0x[0-9a-f]+\(r1\)$|(stw|lwz) r(1[4-9]|2[0-9]|3[01]), 0x[0-9a-f]+\(r1\)$|(stmw|lmw) r[0-9]+, |(stfd|lfd) f(1[4-9]|2[0-9]|3[01]), 0x[0-9a-f]+\(r1\)$|psq_(st|l) f(1[4-9]|2[0-9]|3[01]), 0x[0-9a-f]+\(r1\))",
    )
    .unwrap();
    let lit_re = regex::Regex::new(r"@\d+|lbl_[0-9A-Fa-f]+|\.L_[0-9a-f]+").unwrap();
    let mut out = vec![];
    for (i, c) in f.code.chunks_exact(4).enumerate() {
        let off = (i * 4) as u32;
        let w = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        let ins = Ins::new(w);
        let s = ins.simplified();
        let key = texts.get(&off).cloned().unwrap_or_else(|| s.to_string());
        let key = lit_re.replace_all(&key, "@L").into_owned();
        let loose = {
            let mut parts = vec![s.mnemonic.to_string()];
            for a in s.args_iter() {
                match a {
                    Argument::GPR(_) | Argument::FPR(_) | Argument::CRField(_) => {}
                    _ => parts.push(a.to_string()),
                }
            }
            if let Some(r) = rel_at.get(&off) {
                parts.push(format!("{}+{}", lit_re.replace_all(&r.target, "@L"), r.addend));
            }
            parts.join(" ")
        };
        let mut defs = 0u64;
        let mut uses = 0u64;
        for a in ins.defs().iter() {
            match a {
                Argument::GPR(g) => defs |= bit(g.0),
                Argument::FPR(f) => defs |= bit(32 + f.0),
                _ => {}
            }
        }
        for a in ins.uses().iter() {
            match a {
                Argument::GPR(g) => uses |= bit(g.0),
                Argument::FPR(f) => uses |= bit(32 + f.0),
                _ => {}
            }
        }
        let call = ins.op == Opcode::B && (w & 1) == 1;
        // memory classification
        let mut mem = None;
        let mut store = false;
        let is_load = s.mnemonic.starts_with('l') && !matches!(s.mnemonic, "li" | "lis");
        let is_store = s.mnemonic.starts_with("st") || s.mnemonic.starts_with("psq_st");
        if is_load || is_store {
            store = is_store;
            let base = s.args_iter().filter_map(|a| if let Argument::GPR(g) = a { Some(g.0) } else { None }).last();
            let offset = s.args_iter().find_map(|a| if let Argument::Offset(o) = a { Some(o.0 as i32) } else { None });
            mem = Some(if let Some(r) = rel_at.get(&off) {
                Mem::Global(r.target.clone())
            } else if base == Some(1) {
                Mem::Stack { off: offset.unwrap_or(0), size: access_size(ins.op) }
            } else {
                Mem::Pointer
            });
        }
        let frame = frame_re.is_match(&key);
        out.push(I { off, key, loose, defs, uses, mem, store, call, frame });
    }
    out
}

/// Block boundaries: after branches/calls, before branch targets.
fn blocks(ins: &[I], f: &Func) -> Vec<(usize, usize)> {
    let n = ins.len();
    let mut leaders = std::collections::BTreeSet::new();
    leaders.insert(0usize);
    for (i, c) in f.code.chunks_exact(4).enumerate() {
        let w = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        let x = Ins::new(w);
        if x.is_branch() || ins[i].call {
            if i + 1 < n {
                leaders.insert(i + 1);
            }
            if !ins[i].call {
                if let Some(d) = x.branch_dest((i * 4) as u32) {
                    if (d as usize / 4) < n {
                        leaders.insert(d as usize / 4);
                    }
                }
            }
        }
    }
    let l: Vec<usize> = leaders.into_iter().collect();
    l.iter().enumerate().map(|(k, &s)| (s, l.get(k + 1).copied().unwrap_or(n))).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum DepKind {
    /// true register dependence (value flows from first to second)
    Data,
    /// memory order (store/load through possibly aliasing addresses)
    Memory,
    /// register anti/output dependence only (register-allocation artefact)
    RegAlloc,
}

/// Direct dependence of b on a (a earlier in program order), strongest kind.
fn dep(a: &I, b: &I) -> Option<DepKind> {
    if a.defs & b.uses != 0 {
        return Some(DepKind::Data);
    }
    if let (Some(ma), Some(mb)) = (&a.mem, &b.mem) {
        if (a.store || b.store) && mem_alias(ma, mb) {
            return Some(DepKind::Memory);
        }
    }
    if a.call || b.call {
        return Some(DepKind::Memory);
    }
    if a.uses & b.defs != 0 || a.defs & b.defs != 0 {
        return Some(DepKind::RegAlloc);
    }
    None
}

/// Transitive closure over the given instruction indices (program order): for each pair (i, j),
/// i before j, the best dependence kind on any path (Data/Memory if a path of real edges exists,
/// else RegAlloc), keyed by (i, j).
fn closure(ins: &[I], idx: &[usize]) -> std::collections::HashMap<(usize, usize), DepKind> {
    let n = idx.len();
    // real[a][b], any[a][b] over positions
    let mut real = vec![vec![false; n]; n];
    let mut any = vec![vec![false; n]; n];
    let mut first_kind: Vec<Vec<Option<DepKind>>> = vec![vec![None; n]; n];
    let mut has_mem = vec![vec![false; n]; n];
    for b in 0..n {
        for a in (0..b).rev() {
            if let Some(k) = dep(&ins[idx[a]], &ins[idx[b]]) {
                any[a][b] = true;
                if k != DepKind::RegAlloc {
                    real[a][b] = true;
                    first_kind[a][b] = Some(k);
                    has_mem[a][b] = k == DepKind::Memory;
                }
            }
        }
    }
    // propagate: process b ascending; reach(a, b) |= reach(a, m) && edge(m, b)
    let edge_real = real.clone();
    let edge_mem = has_mem.clone();
    let edge_any = any.clone();
    for b in 0..n {
        for m in 0..b {
            if !edge_any[m][b] {
                continue;
            }
            for a in 0..m {
                if any[a][m] {
                    any[a][b] = true;
                }
                if real[a][m] && edge_real[m][b] {
                    real[a][b] = true;
                    if has_mem[a][m] || edge_mem[m][b] {
                        has_mem[a][b] = true;
                    }
                    if first_kind[a][b].is_none() {
                        first_kind[a][b] = first_kind[a][m];
                    }
                }
            }
        }
    }
    let mut out = std::collections::HashMap::new();
    for a in 0..n {
        for b in a + 1..n {
            if real[a][b] {
                out.insert((idx[a], idx[b]), if has_mem[a][b] { DepKind::Memory } else { DepKind::Data });
            } else if any[a][b] {
                out.insert((idx[a], idx[b]), DepKind::RegAlloc);
            }
        }
    }
    out
}

#[derive(Clone, Debug, Serialize)]
pub struct InstrRef {
    pub text: String,
    pub cand_off: u32,
    pub target_off: u32,
    /// candidate source line (when the candidate was compiled with `-sym on`)
    pub line: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Inversion {
    /// earlier in the candidate, later in the target
    pub cand_first: InstrRef,
    pub cand_second: InstrRef,
    /// Some(kind) = ordered by a dependence (in the candidate or in the target); None = priority only
    pub forced: Option<DepKind>,
    /// which program's dependence forces it ("candidate" / "target")
    pub forced_in: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BlockReport {
    pub index: usize,
    pub cand_range: (u32, u32),
    pub target_range: (u32, u32),
    pub inversions: Vec<Inversion>,
    /// instructions present in only one of the two blocks (not a scheduling-only difference)
    pub unmatched_target: Vec<String>,
    pub unmatched_cand: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct SchedReport {
    pub blocks: Vec<BlockReport>,
    /// block structure differs (different count): not a scheduling-only diff
    pub structure_differs: bool,
    /// summary: (line that must move earlier, line it must precede, dependence kind)
    pub statement_moves: Vec<(u32, u32, DepKind)>,
}

/// Source line per candidate instruction, from an UNSCHEDULED compile of the same candidate
/// (`#pragma scheduling off` keeps statement order, so its `-sym on` line table is exact).
/// `line_offset` is subtracted (e.g. 1 when a pragma line was prepended).
pub fn attribute_lines(cand_obj: &Obj, cand: &Func, uns_obj: &Obj, uns: &Func, line_offset: u32) -> Vec<Option<u32>> {
    let ci = decode(cand_obj, cand);
    let ui = decode(uns_obj, uns);
    let cb = blocks(&ci, cand);
    let ub = blocks(&ui, uns);
    let mut out = vec![None; ci.len()];
    for (&(cs, ce), &(us, ue)) in cb.iter().zip(ub.iter()) {
        let mut used = vec![false; ui.len()];
        for pass in 0..2 {
            for c in cs..ce {
                if out[c].is_some() {
                    continue;
                }
                let m = (us..ue).find(|&u| {
                    !used[u] && if pass == 0 { ui[u].key == ci[c].key } else { ui[u].loose == ci[c].loose }
                });
                if let Some(u) = m {
                    used[u] = true;
                    out[c] = uns_obj.line_at(uns.address + ui[u].off).map(|l| l.saturating_sub(line_offset));
                }
            }
        }
    }
    out
}

/// Compare `cand` against `target`. `lines`: optional source line per candidate instruction
/// (see [`attribute_lines`]); otherwise `cand_obj`'s own line table is used if present.
pub fn check(target_obj: &Obj, target: &Func, cand_obj: &Obj, cand: &Func, lines: Option<&[Option<u32>]>) -> SchedReport {
    let ti = decode(target_obj, target);
    let ci = decode(cand_obj, cand);
    let tb = blocks(&ti, target);
    let cb = blocks(&ci, cand);
    let structure_differs = tb.len() != cb.len();
    let mut report = SchedReport { blocks: vec![], structure_differs, statement_moves: vec![] };
    for (k, (&(ts, te), &(cs, ce))) in tb.iter().zip(cb.iter()).enumerate() {
        // body instructions (prologue/epilogue excluded), indices within the function
        let tbody: Vec<usize> = (ts..te).filter(|&i| !ti[i].frame).collect();
        let cbody: Vec<usize> = (cs..ce).filter(|&i| !ci[i].frame).collect();
        // pair instructions: exact text first, then loose (register-insensitive), in order
        let mut t_of_c: BTreeMap<usize, usize> = BTreeMap::new();
        let mut used_t = vec![false; ti.len()];
        for pass in 0..2 {
            for &c in &cbody {
                if t_of_c.contains_key(&c) {
                    continue;
                }
                let found = tbody.iter().copied().find(|&t| {
                    !used_t[t] && if pass == 0 { ti[t].key == ci[c].key } else { ti[t].loose == ci[c].loose }
                });
                if let Some(t) = found {
                    used_t[t] = true;
                    t_of_c.insert(c, t);
                }
            }
        }
        let unmatched_target: Vec<String> = tbody.iter().filter(|&&t| !used_t[t]).map(|&t| ti[t].key.clone()).collect();
        let unmatched_cand: Vec<String> =
            cbody.iter().filter(|c| !t_of_c.contains_key(c)).map(|&c| ci[c].key.clone()).collect();
        let cclo = closure(&ci, &cbody);
        let tclo = closure(&ti, &tbody);
        let mut inversions = vec![];
        let pairs: Vec<(usize, usize)> = t_of_c.iter().map(|(&c, &t)| (c, t)).collect();
        for x in 0..pairs.len() {
            for y in x + 1..pairs.len() {
                let (ca, ta) = pairs[x];
                let (cb_, tb_) = pairs[y];
                // candidate order ca < cb_ (pairs sorted by candidate index); inversion if target reversed
                if ta < tb_ {
                    continue;
                }
                let fc = cclo.get(&(ca, cb_)).copied();
                let ft = tclo.get(&(tb_, ta)).copied();
                let pick = |k: Option<DepKind>| k.filter(|&d| d != DepKind::RegAlloc);
                let (forced, forced_in) = match (pick(fc), pick(ft)) {
                    (Some(k), _) => (Some(k), Some("candidate".to_string())),
                    (None, Some(k)) => (Some(k), Some("target".to_string())),
                    _ => match (fc, ft) {
                        (Some(DepKind::RegAlloc), _) => (Some(DepKind::RegAlloc), Some("candidate".into())),
                        (_, Some(DepKind::RegAlloc)) => (Some(DepKind::RegAlloc), Some("target".into())),
                        _ => (None, None),
                    },
                };
                let r = |c: usize, t: usize| InstrRef {
                    text: ci[c].key.clone(),
                    cand_off: ci[c].off,
                    target_off: ti[t].off,
                    line: match lines {
                        Some(l) => l.get(c).copied().flatten(),
                        None => cand_obj.line_at(cand.address + ci[c].off),
                    },
                };
                inversions.push(Inversion { cand_first: r(ca, ta), cand_second: r(cb_, tb_), forced, forced_in });
            }
        }
        // statement moves: forced data/memory inversions between different source lines
        for inv in &inversions {
            if let (Some(k @ (DepKind::Data | DepKind::Memory)), Some(l1), Some(l2)) =
                (inv.forced, inv.cand_first.line, inv.cand_second.line)
            {
                if l1 != l2 && !report.statement_moves.iter().any(|m| m.0 == l2 && m.1 == l1) {
                    report.statement_moves.push((l2, l1, k));
                }
            }
        }
        report.blocks.push(BlockReport {
            index: k,
            cand_range: (ci[cs].off, ci[ce - 1].off),
            target_range: (ti[ts].off, ti[te - 1].off),
            inversions,
            unmatched_target,
            unmatched_cand,
        });
    }
    report
}
